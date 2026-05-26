from __future__ import annotations

import threading
from typing import List

from langchain_core.documents import Document

from app.config.settings import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class Reranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-large.
    Pipeline: retrieve top-20 → rerank → return top-5.
    Lazy-initialized on first use to avoid slowing startup.
    """

    _instance: Reranker | None = None
    _lock = threading.Lock()

    def __new__(cls) -> Reranker:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._model = None
        return cls._instance

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        settings = get_settings()
        log.info(
            "Loading reranker model",
            extra={"model": settings.reranker_model},
        )
        self._model = CrossEncoder(
            settings.reranker_model,
            max_length=512,
            device="cpu",
        )
        log.info("Reranker model loaded")

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int | None = None,
    ) -> List[Document]:
        if not docs:
            return docs

        self._load()
        settings = get_settings()
        keep = top_k or settings.rerank_top_k

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs, batch_size=16, show_progress_bar=False)

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        results: List[Document] = []
        for score, doc in scored[:keep]:
            doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": float(score)},
            )
            results.append(doc)

        log.info(
            "Reranking done",
            extra={"input": len(docs), "output": len(results)},
        )
        return results

    async def arerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int | None = None,
    ) -> List[Document]:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.rerank(query, docs, top_k)
        )


def get_reranker() -> Reranker:
    return Reranker()
