from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.schemas import Source
from app.prompts.templates import (
    RAG_ANSWER_PROMPT,
    ROUTER_PROMPT,
    SYSTEM_PROMPT,
    VALIDATION_PROMPT,
)
from app.services.compression_service import get_compressor
from app.services.llm_service import get_llm_service
from app.tools.query_tools import get_multi_query_generator, get_query_rewriter
from app.tools.tavily_search import get_tavily_tool
from app.utils.helpers import count_tokens, format_citations
from app.utils.logger import get_logger

log = get_logger(__name__)


# ── Node Functions ─────────────────────────────────────────────────────────────
# Each node receives the full AgentState dict and returns a partial dict
# with updated keys. LangGraph merges these updates.


async def query_understanding_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract intent and prepare for query rewriting."""
    query = state["original_query"]
    log.info("Query understanding", extra={"query": query[:80]})
    # Pass through — intent extraction is implicit in the rewrite step
    return {"rewritten_query": query}


async def query_rewriting_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite query + generate sub-queries."""
    rewriter = get_query_rewriter()
    gen = get_multi_query_generator()

    history = state.get("conversation_history", "")
    query = state.get("rewritten_query") or state["original_query"]

    rewritten = await rewriter.arewrite(query, history)
    sub_queries = await gen.agenerate(rewritten, n=3)

    return {
        "rewritten_query": rewritten,
        "sub_queries": sub_queries,
    }


async def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Decide retrieval strategy: vector_only | web_only | both | direct."""
    llm = get_llm_service()
    query = state["rewritten_query"]
    history = state.get("conversation_history", "")

    # Honour explicit override from API caller
    if state.get("force_route"):
        return {"route": state["force_route"]}

    prompt = ROUTER_PROMPT.format(query=query, history=history or "None")
    try:
        raw = await llm.ainvoke([HumanMessage(content=prompt)])
        route = raw.strip().lower()
        if route not in {"vector_only", "web_only", "both", "direct"}:
            route = "both"
    except Exception as e:
        log.warning("Router failed, defaulting to both", extra={"error": str(e)})
        route = "both"

    log.info("Router decision", extra={"route": route, "query": query[:60]})
    return {"route": route}


async def dense_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """FAISS dense retrieval for all sub-queries."""
    from app.ingestion.ingestion_pipeline import get_ingestion_pipeline
    from app.retrieval.dense_retrieval import DenseRetriever

    pipeline = get_ingestion_pipeline()
    store = pipeline.get_faiss_store()
    if store is None:
        return {"retrieved_docs": []}

    retriever = DenseRetriever(store)
    sub_queries = state.get("sub_queries") or [state["rewritten_query"]]

    all_docs: List[Document] = []
    for q in sub_queries:
        docs = await retriever.aretrieve(q)
        all_docs.extend(docs)

    return {"retrieved_docs": all_docs}


async def sparse_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """BM25 sparse retrieval."""
    from app.ingestion.ingestion_pipeline import get_ingestion_pipeline
    from app.retrieval.sparse_retrieval import SparseRetriever

    pipeline = get_ingestion_pipeline()
    bm25_docs = pipeline.get_bm25_docs()
    if not bm25_docs:
        return {"retrieved_docs": state.get("retrieved_docs", [])}

    retriever = SparseRetriever(bm25_docs)
    sub_queries = state.get("sub_queries") or [state["rewritten_query"]]

    all_sparse: List[Document] = []
    for q in sub_queries:
        docs = await retriever.aretrieve(q)
        all_sparse.extend(docs)

    existing = state.get("retrieved_docs", [])
    return {"retrieved_docs": existing + all_sparse}


async def web_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Tavily web search."""
    tavily = get_tavily_tool()
    sub_queries = state.get("sub_queries") or [state["rewritten_query"]]

    try:
        web_sources = await tavily.search_multiple(
            queries=sub_queries[:2],  # limit to 2 queries for cost
            max_results_per_query=3,
        )
    except Exception as e:
        log.error("Web search failed", extra={"error": str(e)})
        web_sources = []

    return {"web_results": web_sources}


async def hybrid_merge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """RRF-merge retrieved docs + convert web results to Documents."""
    from app.retrieval.hybrid_retrieval import HybridRetriever
    from app.retrieval.dense_retrieval import DenseRetriever
    from app.retrieval.sparse_retrieval import SparseRetriever
    from app.ingestion.ingestion_pipeline import get_ingestion_pipeline

    pipeline = get_ingestion_pipeline()
    store = pipeline.get_faiss_store()
    bm25_docs = pipeline.get_bm25_docs()

    sub_queries = state.get("sub_queries") or [state["rewritten_query"]]
    route = state.get("route", "both")

    merged: List[Document] = []

    if route in ("vector_only", "both") and store and bm25_docs:
        dense = DenseRetriever(store)
        sparse = SparseRetriever(bm25_docs)
        hybrid = HybridRetriever(dense, sparse)
        merged = await hybrid.aretrieve(sub_queries)
    elif route in ("vector_only", "both"):
        merged = state.get("retrieved_docs", [])

    # Convert web results to Documents so reranker can process them uniformly
    web_results = state.get("web_results", [])
    for src in web_results:
        merged.append(
            Document(
                page_content=src.snippet,
                metadata={
                    "source": src.title,
                    "url": src.url,
                    "source_type": "web",
                    "web_score": src.score,
                },
            )
        )

    return {"merged_docs": merged}


async def reranking_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Rerank merged docs with BGE cross-encoder."""
    from app.rerank.reranker import get_reranker

    docs = state.get("merged_docs", [])
    query = state["rewritten_query"]

    if not docs:
        return {"reranked_docs": []}

    reranker = get_reranker()
    reranked = await reranker.arerank(query, docs)
    return {"reranked_docs": reranked}


async def compression_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compress context and build the context string."""
    compressor = get_compressor()
    docs = state.get("reranked_docs", [])
    query = state["rewritten_query"]

    compressed = compressor.compress(docs, query)
    context_str = compressor.to_context_string(compressed)

    # Build sources list from compressed docs
    sources: List[Source] = []
    for i, doc in enumerate(compressed, start=1):
        meta = doc.metadata
        src_type = meta.get("source_type", "document")
        sources.append(
            Source(
                title=meta.get("source", f"Document {i}"),
                url=meta.get("url", ""),
                snippet=doc.page_content[:300],
                score=meta.get("rerank_score", meta.get("rrf_score", 0.0)),
                source_type=src_type,
                metadata=meta,
            )
        )

    return {
        "compressed_context": context_str,
        "sources": sources,
    }


async def answer_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate answer from compressed context using Groq."""
    llm = get_llm_service()
    context = state.get("compressed_context", "No context available.")
    question = state["rewritten_query"]
    history = state.get("conversation_history", "")
    sources = state.get("sources", [])

    citation_block = format_citations([s.model_dump() for s in sources])
    full_context = f"{context}\n\nSOURCES:\n{citation_block}" if citation_block else context

    rag_prompt = RAG_ANSWER_PROMPT.format(
        context=full_context,
        history=history or "None",
        question=question,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=rag_prompt),
    ]

    try:
        answer = await llm.ainvoke(messages)
        tokens = count_tokens(answer) + count_tokens(rag_prompt)
    except Exception as e:
        log.error("Answer generation failed", extra={"error": str(e)})
        answer = "I encountered an error generating a response. Please try again."
        tokens = 0

    return {"answer": answer, "tokens_used": tokens}


async def validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if the answer is grounded. Mark invalid for retry if not."""
    llm = get_llm_service()
    answer = state.get("answer", "")
    context = state.get("compressed_context", "")
    question = state["rewritten_query"]
    retry_count = state.get("retry_count", 0)

    # Skip validation on retries to avoid infinite loops
    if retry_count >= 2 or not context:
        return {"is_valid": True}

    prompt = VALIDATION_PROMPT.format(
        question=question,
        context=context[:2000],
        answer=answer[:1000],
    )
    try:
        raw = await llm.ainvoke([HumanMessage(content=prompt)])
        # Parse JSON response
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        result = json.loads(raw)
        is_valid = result.get("is_valid", True)
    except Exception as e:
        log.warning("Validation parse failed, assuming valid", extra={"error": str(e)})
        is_valid = True

    return {"is_valid": is_valid, "retry_count": retry_count + 1}


async def memory_update_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Store this turn in conversation memory."""
    from app.memory.memory_manager import get_memory_manager

    manager = get_memory_manager()
    session_id = state.get("session_id", "default")
    mem = manager.get_or_create(session_id)
    mem.add_turn(
        human=state["original_query"],
        ai=state.get("answer", ""),
    )
    log.info("Memory updated", extra={"session_id": session_id})
    return {}
