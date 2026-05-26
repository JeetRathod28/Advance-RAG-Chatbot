from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

import numpy as np
import tiktoken

from app.config.settings import get_settings

_TOKENIZER: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


def count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


def clean_text(text: str) -> str:
    """Normalize Unicode, strip control chars, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters except newline/tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse multiple blank lines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def deduplicate_docs(
    docs: list[Any],
    embeddings: list[np.ndarray] | None = None,
    threshold: float | None = None,
) -> list[Any]:
    """Remove near-duplicate documents using content hash first, then cosine sim."""
    settings = get_settings()
    threshold = threshold or settings.dedup_threshold

    seen_hashes: set[str] = set()
    unique_docs: list[Any] = []
    unique_embeddings: list[np.ndarray] = []

    for i, doc in enumerate(docs):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        h = content_hash(content)
        if h in seen_hashes:
            continue

        # Check cosine similarity against already-kept embeddings
        if embeddings and i < len(embeddings):
            emb = embeddings[i]
            is_dup = any(
                cosine_similarity(emb, kept_emb) >= threshold
                for kept_emb in unique_embeddings
            )
            if is_dup:
                continue
            unique_embeddings.append(emb)

        seen_hashes.add(h)
        unique_docs.append(doc)

    return unique_docs


def format_citations(sources: list[dict[str, Any]]) -> str:
    """Produce a numbered citation block for prompt injection."""
    lines = []
    for i, src in enumerate(sources, start=1):
        title = src.get("title", "Untitled")
        url = src.get("url", "")
        snippet = src.get("snippet", "")[:200]
        ref = f"[{i}] {title}"
        if url:
            ref += f" — {url}"
        if snippet:
            ref += f"\n    {snippet}"
        lines.append(ref)
    return "\n".join(lines)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within max_tokens."""
    enc = _get_tokenizer()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
