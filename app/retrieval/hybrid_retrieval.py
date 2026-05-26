from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from app.config.settings import get_settings
from app.retrieval.dense_retrieval import DenseRetriever
from app.retrieval.sparse_retrieval import SparseRetriever
from app.utils.helpers import content_hash
from app.utils.logger import get_logger

log = get_logger(__name__)


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    """
    Combines dense (FAISS) + sparse (BM25) retrieval via
    Reciprocal Rank Fusion (RRF).  Each sub-query is retrieved
    independently, then all results are merged and fused.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._settings = get_settings()

    def retrieve(
        self,
        queries: List[str],
        k: int | None = None,
        filter: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Retrieve for multiple queries, then RRF-fuse all results.
        """
        top_k = k or self._settings.max_retrieval_k
        rrf_k = self._settings.rrf_k

        # Accumulate RRF scores per document hash
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_registry: Dict[str, Document] = {}

        for query in queries:
            dense_docs = self._dense.retrieve(query, k=top_k, filter=filter)
            sparse_docs = self._sparse.retrieve(query, k=top_k)

            for rank, doc in enumerate(dense_docs):
                h = content_hash(doc.page_content)
                doc_registry[h] = doc
                rrf_scores[h] += _rrf_score(rank, rrf_k)

            for rank, doc in enumerate(sparse_docs):
                h = content_hash(doc.page_content)
                if h not in doc_registry:
                    doc_registry[h] = doc
                else:
                    # Bug #13 fix: keep whichever version has the higher individual score
                    existing = doc_registry[h]
                    existing_score = existing.metadata.get("dense_score", 0.0)
                    new_score = doc.metadata.get("sparse_score", 0.0)
                    if new_score > existing_score:
                        doc_registry[h] = doc
                rrf_scores[h] += _rrf_score(rank, rrf_k)

        # Sort by RRF score descending
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results: List[Document] = []
        for h, score in ranked[:top_k]:
            doc = doc_registry[h]
            doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "rrf_score": score,
                    "retrieval_type": "hybrid",
                },
            )
            results.append(doc)

        log.info(
            "Hybrid retrieval done",
            extra={
                "queries": len(queries),
                "unique_docs": len(rrf_scores),
                "returned": len(results),
            },
        )
        return results

    async def aretrieve(
        self,
        queries: List[str],
        k: int | None = None,
        filter: Optional[Dict] = None,
    ) -> List[Document]:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.retrieve(queries, k, filter)
        )
