from __future__ import annotations

import asyncio
import re
from typing import List

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config.settings import get_settings
from app.models.schemas import Source
from app.utils.logger import get_logger

log = get_logger(__name__)


def _clean_snippet(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _deduplicate_urls(sources: List[Source]) -> List[Source]:
    seen: set[str] = set()
    unique: List[Source] = []
    for src in sources:
        if src.url not in seen:
            seen.add(src.url)
            unique.append(src)
    return unique


# Bug #20 fix: Only retry on network-level errors, NOT on CancelledError,
# quota errors, or other non-retriable exceptions.
_RETRIABLE_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class TavilySearchTool:
    """
    Async Tavily web search with result cleaning and citation formatting.
    Uses search_depth="advanced" for higher-quality results.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    @retry(
        retry=retry_if_exception_type(_RETRIABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Source]:
        try:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=self._settings.tavily_api_key)
            response = await client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
            )
        except asyncio.CancelledError:
            # Propagate cancellation — do NOT retry
            raise
        except _RETRIABLE_ERRORS:
            # These are network errors — let tenacity handle retry
            raise
        except Exception as e:
            # Auth errors, quota exceeded, etc. — log and raise without retry
            log.error("Tavily search failed (non-retriable)", extra={"error": str(e), "query": query})
            raise

        sources: List[Source] = []
        for result in response.get("results", []):
            snippet = _clean_snippet(result.get("content", ""))[:500]
            sources.append(
                Source(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=snippet,
                    score=float(result.get("score", 0.0)),
                    source_type="web",
                    metadata={"published_date": result.get("published_date", "")},
                )
            )

        sources = _deduplicate_urls(sources)
        sources.sort(key=lambda s: s.score, reverse=True)

        log.info(
            "Web search complete",
            extra={"query": query[:80], "results": len(sources)},
        )
        return sources

    async def search_multiple(
        self,
        queries: List[str],
        max_results_per_query: int = 3,
    ) -> List[Source]:
        """Run multiple searches concurrently and deduplicate."""
        tasks = [self.search(q, max_results_per_query) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_sources: List[Source] = []
        for r in results:
            if isinstance(r, Exception):
                log.warning("One web search query failed", extra={"error": str(r)})
            elif isinstance(r, list):
                all_sources.extend(r)

        return _deduplicate_urls(all_sources)


_tavily_tool: TavilySearchTool | None = None


def get_tavily_tool() -> TavilySearchTool:
    global _tavily_tool
    if _tavily_tool is None:
        _tavily_tool = TavilySearchTool()
    return _tavily_tool
