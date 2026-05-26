from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts.templates import MULTI_QUERY_PROMPT, QUERY_REWRITE_PROMPT
from app.services.llm_service import get_llm_service
from app.utils.logger import get_logger

log = get_logger(__name__)


class QueryRewriter:
    """Rewrite ambiguous queries using conversation history."""

    def __init__(self) -> None:
        self._llm = get_llm_service()

    def _parse_rewrite(self, raw: str, original: str) -> str:
        """
        Bug #19 fix: validate the LLM rewriter output.
        - Take only the FIRST non-empty line (prevents multi-line preamble output).
        - If the result is too long (>500 chars) or empty, fall back to original.
        - Strip common preamble patterns LLMs emit despite instructions.
        """
        # Take first non-empty line only
        first_line = next(
            (line.strip() for line in raw.split("\n") if line.strip()), ""
        )
        if not first_line or len(first_line) > 500:
            return original
        # Remove common preamble prefixes the LLM might emit
        for prefix in ("REWRITTEN QUERY:", "Rewritten query:", "QUERY:", "Query:"):
            if first_line.startswith(prefix):
                first_line = first_line[len(prefix):].strip()
        return first_line or original

    def rewrite(self, query: str, history: str = "") -> str:
        prompt = QUERY_REWRITE_PROMPT.format(query=query, history=history or "None")
        messages = [HumanMessage(content=prompt)]
        try:
            raw = self._llm.invoke(messages).strip()
            rewritten = self._parse_rewrite(raw, query)
            log.info(
                "Query rewritten",
                extra={"original": query[:80], "rewritten": rewritten[:80]},
            )
            return rewritten
        except Exception as e:
            log.warning("Query rewrite failed, using original", extra={"error": str(e)})
            return query

    async def arewrite(self, query: str, history: str = "") -> str:
        prompt = QUERY_REWRITE_PROMPT.format(query=query, history=history or "None")
        messages = [HumanMessage(content=prompt)]
        try:
            raw = (await self._llm.ainvoke(messages)).strip()
            rewritten = self._parse_rewrite(raw, query)
            log.info(
                "Query rewritten (async)",
                extra={"original": query[:80], "rewritten": rewritten[:80]},
            )
            return rewritten
        except Exception as e:
            log.warning("Async query rewrite failed", extra={"error": str(e)})
            return query


class MultiQueryGenerator:
    """Generate N diverse sub-queries from a single question."""

    def __init__(self) -> None:
        self._llm = get_llm_service()

    def _parse_queries(self, raw: str, original: str, n: int) -> List[str]:
        """
        Bug #18 fix: insert original FIRST, then slice to [:n].
        Previously it sliced to [:n] BEFORE inserting the original, yielding n+1 queries.
        """
        generated = [q.strip() for q in raw.split("\n") if q.strip()]
        # Remove any lines that are clearly preamble (e.g. "QUERIES:", numbered items)
        cleaned = []
        for line in generated:
            # Strip leading numbering like "1.", "1)", "-", "•"
            line = line.lstrip("0123456789.-•) ").strip()
            if line and len(line) > 5:  # skip empty or near-empty items
                cleaned.append(line)

        # Build final list: original first, then up to (n-1) generated unique ones
        queries: List[str] = [original]
        seen = {original.lower()}
        for q in cleaned:
            if q.lower() not in seen:
                queries.append(q)
                seen.add(q.lower())
            if len(queries) >= n:
                break
        return queries

    def generate(self, query: str, n: int = 3) -> List[str]:
        prompt = MULTI_QUERY_PROMPT.format(query=query, n=n)
        messages = [HumanMessage(content=prompt)]
        try:
            raw = self._llm.invoke(messages).strip()
            queries = self._parse_queries(raw, query, n)
            log.info(
                "Multi-queries generated",
                extra={"original": query[:60], "count": len(queries)},
            )
            return queries
        except Exception as e:
            log.warning("Multi-query generation failed", extra={"error": str(e)})
            return [query]

    async def agenerate(self, query: str, n: int = 3) -> List[str]:
        prompt = MULTI_QUERY_PROMPT.format(query=query, n=n)
        messages = [HumanMessage(content=prompt)]
        try:
            raw = (await self._llm.ainvoke(messages)).strip()
            queries = self._parse_queries(raw, query, n)
            log.info(
                "Multi-queries generated (async)",
                extra={"original": query[:60], "count": len(queries)},
            )
            return queries
        except Exception as e:
            log.warning("Async multi-query generation failed", extra={"error": str(e)})
            return [query]


_rewriter: QueryRewriter | None = None
_multi_gen: MultiQueryGenerator | None = None


def get_query_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter


def get_multi_query_generator() -> MultiQueryGenerator:
    global _multi_gen
    if _multi_gen is None:
        _multi_gen = MultiQueryGenerator()
    return _multi_gen
