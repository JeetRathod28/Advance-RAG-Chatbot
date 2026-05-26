from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config.settings import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class DenseRetriever:
    """
    FAISS-backed dense retrieval.
    Returns top-k candidates WITHOUT pre-filtering so the reranker
    can make the final relevance judgement.
    """

    def __init__(self, faiss_store: FAISS) -> None:
        self._store = faiss_store
        self._settings = get_settings()

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: Optional[Dict] = None,
    ) -> List[Document]:
        top_k = k or self._settings.max_retrieval_k

        if self._store is None:
            log.warning("FAISS store not initialized — returning empty")
            return []

        try:
            # Use relevance_score_fn-aware method so scores are in [0, 1]
            # where 1 = most similar (works correctly for both L2 and IP indexes)
            results = self._store.similarity_search_with_relevance_scores(
                query, k=top_k, filter=filter
            )
        except Exception as e:
            log.warning(
                "similarity_search_with_relevance_scores failed, falling back",
                extra={"error": str(e)},
            )
            # Fallback: plain search without scores
            try:
                plain = self._store.similarity_search(query, k=top_k, filter=filter)
                for doc in plain:
                    doc.metadata["dense_score"] = 1.0
                    doc.metadata["retrieval_type"] = "dense"
                log.info("Dense retrieval done (fallback)", extra={"results": len(plain)})
                return plain
            except Exception as e2:
                log.error("Dense retrieval failed", extra={"error": str(e2)})
                return []

        docs: List[Document] = []
        for doc, score in results:
            # Accept all results — no threshold. Let the reranker decide.
            doc.metadata["dense_score"] = float(score)
            doc.metadata["retrieval_type"] = "dense"
            docs.append(doc)

        log.info("Dense retrieval done", extra={"results": len(docs)})
        return docs

    async def aretrieve(
        self,
        query: str,
        k: int | None = None,
        filter: Optional[Dict] = None,
    ) -> List[Document]:
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.retrieve(query, k, filter)
        )
