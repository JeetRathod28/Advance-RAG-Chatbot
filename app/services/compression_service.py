from __future__ import annotations

from typing import List

import numpy as np
from langchain_core.documents import Document

from app.config.settings import get_settings
from app.utils.helpers import cosine_similarity, count_tokens
from app.utils.logger import get_logger

log = get_logger(__name__)


class ContextCompressor:
    """
    Trim retrieved & reranked docs to fit the LLM context window.

    Strategy (in order):
    1. Near-duplicate removal  — drop chunks that are >92% similar to a
       higher-ranked chunk (avoids sending the same text twice).
    2. Token-budget truncation — keep as many full chunks as fit within
       max_context_tokens; the last chunk is trimmed if needed.

    NOTE: We deliberately do NOT re-score by cosine similarity here.
    The reranker already produced a quality ordering; re-filtering with
    embeddings would discard reranker-approved chunks.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Lazy-import embedding service to avoid circular imports."""
        from app.services.embedding_service import get_embedding_service
        svc = get_embedding_service()
        return svc.embed_documents(texts)

    def compress(
        self,
        docs: List[Document],
        query: str,
        max_tokens: int | None = None,
    ) -> List[Document]:
        if not docs:
            return docs

        settings = self._settings
        max_tok = max_tokens or settings.max_context_tokens
        dedup_thresh = settings.dedup_threshold  # 0.92 — only near-identical

        # ── Step 1: Near-duplicate removal ────────────────────────────────────
        # Only run if we have more than 1 doc (skip expensive embedding call otherwise)
        if len(docs) > 1:
            texts = [d.page_content for d in docs]
            try:
                embs = self._get_embeddings(texts)
                kept_docs: List[Document] = []
                kept_embs: List[np.ndarray] = []

                for doc, emb in zip(docs, embs):
                    # Check against already-kept docs
                    is_dup = any(
                        cosine_similarity(emb, ke) >= dedup_thresh
                        for ke in kept_embs
                    )
                    if not is_dup:
                        kept_docs.append(doc)
                        kept_embs.append(emb)

                docs = kept_docs
            except Exception as e:
                log.warning("Dedup embedding failed, skipping dedup", extra={"error": str(e)})

        # ── Step 2: Token-budget truncation ───────────────────────────────────
        result: List[Document] = []
        used_tokens = 0

        for doc in docs:
            tok_count = count_tokens(doc.page_content)

            if used_tokens + tok_count <= max_tok:
                result.append(doc)
                used_tokens += tok_count
            else:
                # Try to fit a trimmed version of this chunk
                remaining = max_tok - used_tokens
                if remaining >= 60:   # Only worth including if ≥60 tokens remain
                    words = doc.page_content.split()
                    # Rough estimate: 1 token ≈ 0.75 words
                    word_limit = int(remaining * 0.75)
                    trimmed_text = " ".join(words[:word_limit])
                    result.append(Document(
                        page_content=trimmed_text + " ...",
                        metadata=doc.metadata,
                    ))
                break

        log.info(
            "Context compressed",
            extra={
                "input_chunks": len(docs),
                "output_chunks": len(result),
                "tokens": used_tokens,
            },
        )
        return result

    def to_context_string(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata
            source = meta.get("source", "unknown")
            page   = meta.get("page", "")
            section = meta.get("section", "")

            ref = f"[{i}] {source}"
            if page != "":
                ref += f" · page {page}"
            if section:
                ref += f" · {section}"

            parts.append(f"{ref}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)


_compressor: ContextCompressor | None = None


def get_compressor() -> ContextCompressor:
    global _compressor
    if _compressor is None:
        _compressor = ContextCompressor()
    return _compressor
