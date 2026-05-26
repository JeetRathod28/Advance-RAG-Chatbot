from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.schemas import Source
from app.prompts.templates import RAG_ANSWER_PROMPT, SYSTEM_PROMPT
from app.services.compression_service import get_compressor
from app.services.llm_service import get_llm_service
from app.tools.query_tools import get_multi_query_generator, get_query_rewriter
from app.tools.tavily_search import get_tavily_tool
from app.utils.helpers import format_citations
from app.utils.logger import get_logger

log = get_logger(__name__)

# Patterns indicating the query genuinely needs real-time / external info.
# NOTE: Years (20xx) intentionally EXCLUDED — document questions about specific
# years should NOT trigger web search. Years are common in resume/report content
# and were causing false positives that returned wrong-entity web results.
_FRESH_INFO_RE = re.compile(
    r"\b(current|latest|today|yesterday|this\s+week|breaking|live|"
    r"price|cost|stock\s+price|share\s+price|"
    r"news|headlines|announcement)\b",
    re.IGNORECASE,
)


def _needs_fresh_info(query: str) -> bool:
    """
    True when the query asks for genuinely time-sensitive / real-world current data.
    Applied to the ORIGINAL user message (not the LLM-rewritten version) to prevent
    the query rewriter from injecting temporal terms that falsely trigger web search.
    """
    return bool(_FRESH_INFO_RE.search(query))


def _scope_web_queries(sub_queries: List[str], indexed_sources: List[str]) -> List[str]:
    """
    When doing web supplementation for a document query, anchor each sub-query
    to the document's entity names to avoid returning information about wrong people.

    Extracts meaningful name tokens from indexed filenames (e.g. "Jeet_Rathod_GenAI.pdf"
    → "Jeet Rathod") and prepends them if not already present in the query.
    """
    if not indexed_sources:
        return sub_queries

    # Build a combined entity prefix from all indexed doc filenames
    entity_tokens: List[str] = []
    for src in indexed_sources:
        # Strip extension, split on underscore/dash/space, take first 2–3 non-generic tokens
        stem = re.sub(r"\.[^.]+$", "", src)  # remove extension
        parts = re.split(r"[_\-\s]+", stem)
        # Filter out purely numeric, very short, or common generic words
        _GENERIC = {"resume", "cv", "doc", "report", "file", "pdf", "final", "v1", "v2"}
        meaningful = [p for p in parts if len(p) >= 2 and p.lower() not in _GENERIC]
        entity_tokens.extend(meaningful[:3])  # cap at 3 tokens per doc

    if not entity_tokens:
        return sub_queries

    entity_prefix = " ".join(entity_tokens)

    scoped: List[str] = []
    for q in sub_queries:
        # Only prepend if the entity is not already in the query
        if entity_prefix.lower() not in q.lower():
            scoped.append(f"{entity_prefix} {q}")
        else:
            scoped.append(q)
    return scoped


# ── Internal helpers ───────────────────────────────────────────────────────────

def _decide_route(force_route: Optional[str], has_docs: bool) -> str:
    """
    - Forced route from UI pill → honour exactly.
    - Auto + docs indexed  → "vector_first": docs first, web only if needed.
    - Auto + no docs       → "web_only".
    """
    if force_route in {"vector_only", "web_only", "both", "direct"}:
        return force_route
    return "vector_first" if has_docs else "web_only"


async def _retrieve_vector(sub_queries: List[str], has_docs: bool) -> List[Document]:
    """FAISS + BM25 hybrid retrieval."""
    if not has_docs:
        return []

    from app.ingestion.ingestion_pipeline import get_ingestion_pipeline
    from app.retrieval.dense_retrieval import DenseRetriever
    from app.retrieval.hybrid_retrieval import HybridRetriever
    from app.retrieval.sparse_retrieval import SparseRetriever

    pipeline = get_ingestion_pipeline()
    store = pipeline.get_faiss_store()
    sparse_retriever = pipeline.get_sparse_retriever()  # Cached retriever (Bug #15 fix)

    if not store or sparse_retriever is None:
        return []

    dense = DenseRetriever(store)
    hybrid = HybridRetriever(dense, sparse_retriever)
    docs = await hybrid.aretrieve(sub_queries)
    log.info("Vector retrieval done", extra={"docs": len(docs)})
    return docs


async def _retrieve_web(
    sub_queries: List[str],
    indexed_sources: Optional[List[str]] = None,
    scope_to_docs: bool = False,
) -> tuple[List[Document], List[Source]]:
    """
    Tavily web search -> (Documents, Source metadata).

    When scope_to_docs=True, queries are entity-scoped to the indexed document
    filenames to prevent returning information about wrong people/entities.
    """
    tavily = get_tavily_tool()
    web_sources: List[Source] = []
    web_docs: List[Document] = []

    # Scope queries to document entities when supplementing document queries
    queries_to_search = sub_queries[:2]
    if scope_to_docs and indexed_sources:
        queries_to_search = _scope_web_queries(queries_to_search, indexed_sources)
        log.info("Web queries scoped to document entities", extra={"queries": queries_to_search})

    try:
        web_sources = await tavily.search_multiple(
            queries_to_search, max_results_per_query=4
        )
        for src in web_sources:
            web_docs.append(
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
        log.info("Web retrieval done", extra={"results": len(web_sources)})
    except Exception as e:
        log.warning("Web search failed", extra={"error": str(e)})
    return web_docs, web_sources


async def _build_context(
    all_docs: List[Document],
    rewritten_query: str,
) -> tuple[List[Document], List[Source], str]:
    """Rerank + compress -> sources list + context string."""
    from app.rerank.reranker import get_reranker

    if not all_docs:
        return [], [], ""

    reranker = get_reranker()
    reranked = await reranker.arerank(rewritten_query, all_docs)

    compressor = get_compressor()
    compressed = compressor.compress(reranked, rewritten_query)
    context_str = compressor.to_context_string(compressed)

    sources: List[Source] = []
    for i, doc in enumerate(compressed, start=1):
        meta = doc.metadata
        sources.append(
            Source(
                title=meta.get("source", f"Source {i}"),
                url=meta.get("url", ""),
                snippet=doc.page_content[:300],
                score=float(
                    meta.get(
                        "rerank_score",
                        meta.get("rrf_score", meta.get("web_score", 0.0)),
                    )
                ),
                source_type=meta.get("source_type", "document"),
                metadata=meta,
            )
        )
    return compressed, sources, context_str


def _build_messages(
    context: str,
    history: str,
    question: str,
    sources: List[Source],
) -> List:
    citation_block = format_citations([s.model_dump() for s in sources])
    full_ctx = f"{context}\n\nSOURCES:\n{citation_block}" if citation_block else context

    if not full_ctx.strip():
        full_ctx = (
            "No relevant context found. Answer based on your general knowledge "
            "if possible, and be clear about uncertainty."
        )

    prompt = RAG_ANSWER_PROMPT.format(
        context=full_ctx,
        history=history or "None",
        question=question,
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]


# ── Public API ─────────────────────────────────────────────────────────────────

async def stream_agent(
    session_id: str,
    message: str,
    conversation_history: str = "",
    force_route: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Streaming pipeline:
      Rewrite -> Route -> Retrieve -> Rerank -> Compress -> Stream -> Memory

    Auto-routing logic (when user selects "Auto" in the UI):
      * No documents indexed            -> web search
      * Documents indexed, normal query -> answer from documents only (no web)
      * Documents indexed, query needs  -> supplement docs with web search
        real-time info (price/news/live data) — checked on ORIGINAL message only
      * Documents indexed, no results   -> fall back to web search
    """
    from app.ingestion.ingestion_pipeline import get_ingestion_pipeline
    from app.memory.memory_manager import get_memory_manager

    llm = get_llm_service()
    pipeline = get_ingestion_pipeline()
    has_docs = pipeline.vector_store_size() > 0

    route = _decide_route(force_route, has_docs)

    # Check fresh-info on the ORIGINAL message (before rewriting) — Bug #4 fix
    original_needs_web = _needs_fresh_info(message)

    # Get indexed source names for query scoping — Bug #21 fix
    indexed_sources = list(pipeline.get_indexed_files().keys())

    # Step 1: Rewrite query + sub-queries
    try:
        rewriter = get_query_rewriter()
        gen = get_multi_query_generator()
        rewritten = await rewriter.arewrite(message, conversation_history)
        sub_queries = await gen.agenerate(rewritten, n=3)
    except Exception:
        rewritten = message
        sub_queries = [message]

    _label = {
        "vector_only":  "documents",
        "web_only":     "web",
        "both":         "documents + web",
        "direct":       "knowledge base",
        "vector_first": "documents",
    }.get(route, "auto")
    yield {"type": "status", "content": f"Searching {_label}..."}

    # Step 2: Retrieve
    all_docs: List[Document] = []
    used_web = False

    if route == "direct":
        pass

    elif route == "vector_only":
        all_docs = await _retrieve_vector(sub_queries, has_docs)

    elif route == "web_only":
        all_docs, _ = await _retrieve_web(sub_queries)
        used_web = True

    elif route == "both":
        v_docs = await _retrieve_vector(sub_queries, has_docs)
        w_docs, _ = await _retrieve_web(sub_queries, indexed_sources, scope_to_docs=False)
        all_docs = v_docs + w_docs
        used_web = True

    elif route == "vector_first":
        v_docs = await _retrieve_vector(sub_queries, has_docs)

        if not v_docs:
            # Nothing retrieved from vector store -> fall back to web
            yield {"type": "status", "content": "No document matches found, searching web..."}
            all_docs, _ = await _retrieve_web(sub_queries)
            used_web = True
            log.info("Vector returned nothing, using web fallback")

        else:
            # Check document quality by reranking top candidates
            from app.rerank.reranker import get_reranker
            from app.config.settings import get_settings
            settings = get_settings()
            reranker = get_reranker()

            reranked_v_docs = await reranker.arerank(rewritten, v_docs)
            best_score = -999.0
            if reranked_v_docs:
                best_score = reranked_v_docs[0].metadata.get("rerank_score", -999.0)

            if best_score < settings.doc_quality_threshold:
                # Document quality too low -> supplement with web
                yield {"type": "status", "content": f"Searching web (document match score {best_score:.2f} < threshold)..."}
                w_docs, _ = await _retrieve_web(
                    sub_queries, indexed_sources, scope_to_docs=True
                )
                all_docs = v_docs + w_docs
                used_web = True
                log.info(f"Supplementing docs with web (quality low: {best_score:.2f} < {settings.doc_quality_threshold})")
            elif original_needs_web:
                # Only supplement with web if ORIGINAL message asks for real-time data
                yield {"type": "status", "content": "Searching web for up-to-date information..."}
                w_docs, _ = await _retrieve_web(
                    sub_queries, indexed_sources, scope_to_docs=True
                )
                all_docs = v_docs + w_docs
                used_web = True
                log.info("Supplementing docs with web (time-sensitive original query)")
            else:
                # Normal document question -> use docs only, skip web entirely
                all_docs = v_docs
                log.info(
                    "Using documents only (skipping web search)",
                    extra={"chunks": len(v_docs), "best_score": best_score},
                )

    # Step 3: Rerank + compress
    compressed, sources, context = await _build_context(all_docs, rewritten)

    if compressed:
        src_tag = ""
        if used_web and has_docs:
            src_tag = " from documents + web"
        elif used_web:
            src_tag = " from web"
        yield {
            "type": "status",
            "content": (
                f"Found {len(compressed)} relevant "
                f"source{'s' if len(compressed) != 1 else ''}{src_tag}"
            ),
        }

    # Step 4: Stream answer
    messages = _build_messages(context, conversation_history, rewritten, sources)
    full_answer = ""
    try:
        async for token in llm.astream(messages):
            full_answer += token
            yield {"type": "token", "content": token}
    except Exception as e:
        log.error("LLM streaming failed", extra={"error": str(e)})
        yield {"type": "error", "content": f"LLM error: {str(e)}"}
        return

    # Step 5: Emit sources + done
    yield {"type": "sources", "content": [s.model_dump() for s in sources]}

    # Step 6: Store in memory
    try:
        manager = get_memory_manager()
        mem = manager.get_or_create(session_id)
        mem.add_turn(human=message, ai=full_answer)
    except Exception as e:
        log.warning("Memory update failed", extra={"error": str(e)})

    yield {"type": "done", "content": ""}


async def run_agent(
    session_id: str,
    message: str,
    conversation_history: str = "",
    force_route: Optional[str] = None,
) -> Dict[str, Any]:
    """Non-streaming version: same smart routing pipeline."""
    from app.ingestion.ingestion_pipeline import get_ingestion_pipeline
    from app.memory.memory_manager import get_memory_manager
    from app.utils.helpers import count_tokens

    llm = get_llm_service()
    pipeline = get_ingestion_pipeline()
    has_docs = pipeline.vector_store_size() > 0

    route = _decide_route(force_route, has_docs)

    # Check fresh-info on ORIGINAL message (not rewritten) — Bug #4 fix
    original_needs_web = _needs_fresh_info(message)
    indexed_sources = list(pipeline.get_indexed_files().keys())

    try:
        rewriter = get_query_rewriter()
        gen = get_multi_query_generator()
        rewritten = await rewriter.arewrite(message, conversation_history)
        sub_queries = await gen.agenerate(rewritten, n=3)
    except Exception:
        rewritten = message
        sub_queries = [message]

    all_docs: List[Document] = []

    if route == "direct":
        pass

    elif route == "vector_only":
        all_docs = await _retrieve_vector(sub_queries, has_docs)

    elif route == "web_only":
        all_docs, _ = await _retrieve_web(sub_queries)

    elif route == "both":
        v = await _retrieve_vector(sub_queries, has_docs)
        w, _ = await _retrieve_web(sub_queries, indexed_sources, scope_to_docs=False)
        all_docs = v + w

    elif route == "vector_first":
        v_docs = await _retrieve_vector(sub_queries, has_docs)
        if not v_docs:
            all_docs, _ = await _retrieve_web(sub_queries)
            log.info("Vector returned nothing, using web fallback")
        else:
            # Check document quality by reranking top candidates
            from app.rerank.reranker import get_reranker
            from app.config.settings import get_settings
            settings = get_settings()
            reranker = get_reranker()

            reranked_v_docs = await reranker.arerank(rewritten, v_docs)
            best_score = -999.0
            if reranked_v_docs:
                best_score = reranked_v_docs[0].metadata.get("rerank_score", -999.0)

            if best_score < settings.doc_quality_threshold:
                w_docs, _ = await _retrieve_web(
                    sub_queries, indexed_sources, scope_to_docs=True
                )
                all_docs = v_docs + w_docs
                log.info(f"Supplementing docs with web (quality low: {best_score:.2f} < {settings.doc_quality_threshold})")
            elif original_needs_web:
                w_docs, _ = await _retrieve_web(
                    sub_queries, indexed_sources, scope_to_docs=True
                )
                all_docs = v_docs + w_docs
                log.info("Supplementing docs with web (time-sensitive original query)")
            else:
                all_docs = v_docs
                log.info("Using documents only (skipping web search)")

    compressed, sources, context = await _build_context(all_docs, rewritten)
    messages = _build_messages(context, conversation_history, rewritten, sources)

    try:
        answer = await llm.ainvoke(messages)
        tokens = count_tokens(answer)
    except Exception as e:
        log.error("LLM call failed", extra={"error": str(e)})
        answer = f"Error generating response: {str(e)}"
        tokens = 0

    try:
        manager = get_memory_manager()
        mem = manager.get_or_create(session_id)
        mem.add_turn(human=message, ai=answer)
    except Exception:
        pass

    return {
        "answer": answer,
        "sources": sources,
        "tokens_used": tokens,
        "route": route,
    }


# ── Legacy/Test compatibility functions ────────────────────────────────────────

def route_after_router(state: Dict[str, Any]) -> str:
    """Routing condition after router node (legacy / test support)."""
    return "direct_answer" if state.get("route") == "direct" else "retrieve"


def route_after_validation(state: Dict[str, Any]) -> str:
    """Routing condition after validation node (legacy / test support)."""
    if state.get("is_valid") is True:
        return "memory_update"
    return "query_rewriting" if state.get("retry_count", 0) < 3 else "memory_update"


def get_agent() -> Any:
    """Dummy agent loader (legacy / test support)."""
    return None

