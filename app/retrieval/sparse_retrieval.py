from __future__ import annotations

import asyncio
import re
from typing import List

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config.settings import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# Common English stop-words — removing them improves BM25 precision
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "that", "this", "these", "those", "it", "its",
    "as", "into", "than", "then", "so", "if", "about", "up", "out", "not",
})


def _tokenize(text: str) -> List[str]:
    """
    Improved tokenizer:
    - Lowercases text
    - Splits on whitespace and punctuation but keeps hyphens inside words
      so 'full-stack' stays as one token AND produces 'full' and 'stack'
    - Removes stop-words
    - Ignores tokens shorter than 2 chars
    """
    text = text.lower()
    # Split on whitespace and most punctuation, but keep hyphens mid-word
    raw_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)

    tokens: List[str] = []
    for tok in raw_tokens:
        if len(tok) < 2 or tok in _STOP_WORDS:
            continue
        tokens.append(tok)
        # Also add hyphen sub-parts so 'full-stack' matches 'stack'
        if "-" in tok:
            for part in tok.split("-"):
                if len(part) >= 2 and part not in _STOP_WORDS:
                    tokens.append(part)

    return tokens


class SparseRetriever:
    """
    BM25-based sparse keyword retrieval.
    Returns all docs with positive BM25 scores (no arbitrary top-k cut)
    so the hybrid reranker has more candidates.
    """

    def __init__(self, docs: List[Document]) -> None:
        self._docs = docs
        self._settings = get_settings()
        self._bm25: BM25Okapi | None = None

        if docs:
            self._build_index(docs)

    def _build_index(self, docs: List[Document]) -> None:
        corpus = [_tokenize(doc.page_content) for doc in docs]
        self._bm25 = BM25Okapi(corpus)
        log.info("BM25 index built", extra={"docs": len(docs)})

    def update(self, new_docs: List[Document]) -> None:
        self._docs = new_docs
        if new_docs:
            self._build_index(new_docs)

    def retrieve(self, query: str, k: int | None = None) -> List[Document]:
        top_k = k or self._settings.max_retrieval_k

        if self._bm25 is None or not self._docs:
            log.warning("BM25 index empty — returning empty")
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        raw_scores = np.array(self._bm25.get_scores(query_tokens))

        # Only keep docs with positive BM25 score (actual keyword match)
        positive_mask = raw_scores > 0
        if not positive_mask.any():
            # No keyword match at all — return empty so hybrid falls back to dense only
            log.info("Sparse retrieval: no keyword matches found")
            return []

        # Normalize to [0, 1] among positive-scoring docs
        max_score = raw_scores.max()
        norm_scores = raw_scores / max_score

        # Sort and take top_k
        top_indices = np.argsort(norm_scores)[::-1][:top_k]

        docs: List[Document] = []
        for idx in top_indices:
            score = float(norm_scores[idx])
            if score <= 0:
                continue
            doc = self._docs[idx]
            docs.append(Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "sparse_score": score,
                    "retrieval_type": "sparse",
                },
            ))

        log.info("Sparse retrieval done", extra={"results": len(docs)})
        return docs

    async def aretrieve(self, query: str, k: int | None = None) -> List[Document]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.retrieve, query, k
        )
