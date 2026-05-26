from __future__ import annotations

import threading
from typing import List

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from app.config.settings import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class EmbeddingService:
    """Singleton wrapper around all-MiniLM-L6-v2 (384-dim, ~90 MB)."""

    _instance: EmbeddingService | None = None
    _lock = threading.Lock()

    def __new__(cls) -> EmbeddingService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def _initialize(self) -> None:
        if self._initialized:
            return
        settings = get_settings()
        log.info("Loading embedding model", extra={"model": settings.embedding_model})
        self._model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": settings.embedding_batch_size,
            },
        )
        self._batch_size = settings.embedding_batch_size
        self._initialized = True
        log.info("Embedding model loaded")

    @property
    def model(self) -> HuggingFaceEmbeddings:
        self._initialize()
        return self._model

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        self._initialize()
        vecs = self._model.embed_documents([t.strip() for t in texts])
        return np.array(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        self._initialize()
        vec = self._model.embed_query(text.strip())
        return np.array(vec, dtype=np.float32)

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        self._initialize()
        vecs = self._model.embed_documents([t.strip() for t in texts])
        return np.array(vecs, dtype=np.float32)


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
