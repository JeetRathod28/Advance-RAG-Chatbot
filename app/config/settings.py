from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (Ollama — local) ──────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout: int = 300

    # ── Web Search ────────────────────────────────────────────────────────────
    tavily_api_key: str = ""

    # ── Embedding & Reranker (lightweight CPU models) ─────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_batch_size: int = 32

    # ── Retrieval ─────────────────────────────────────────────────────────────
    max_retrieval_k: int = 30       # More candidates for the reranker to judge
    rerank_top_k: int = 8           # More docs survive into the final context
    score_threshold: float = 0.0    # No pre-rerank filtering — let reranker decide
    rrf_k: int = 60                 # RRF constant
    # Auto-routing: if best doc rerank_score < this, treat as "document doesn't answer"
    # and supplement with web search. ms-marco scores: >0 = relevant, <-3 = irrelevant.
    doc_quality_threshold: float = -1.0

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 1000          # Larger chunks preserve more context
    chunk_overlap: int = 200        # More overlap catches boundary information

    # ── Context Compression ───────────────────────────────────────────────────
    max_context_tokens: int = 3200  # 20% safety buffer for tiktoken vs llama tokenizer mismatch
    compression_threshold: float = 0.0   # No cosine filter — reranker already scored
    dedup_threshold: float = 0.92   # Only remove near-identical chunks

    # ── Paths ─────────────────────────────────────────────────────────────────
    faiss_index_path: str = "./vector_store"
    data_path: str = "./data"
    log_level: str = "INFO"

    # ── API ───────────────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    max_upload_mb: int = 50

    # ── Memory ────────────────────────────────────────────────────────────────
    memory_max_tokens: int = 2000
    session_ttl_hours: int = 24

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
