from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config.settings import get_settings
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.text_splitter import SemanticChunker
from app.services.embedding_service import get_embedding_service
from app.utils.helpers import content_hash
from app.utils.logger import get_logger

log = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates: Load → Split → Embed → FAISS index → BM25 index → Persist.

    Fixes applied:
    - Bug #2/#8: Re-uploading the same filename first removes its old chunks
      from both FAISS and BM25 before adding the new ones.
    - Bug #15: Caches a single SparseRetriever instance that is only rebuilt
      when the BM25 corpus actually changes.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._loader = DocumentLoader()
        self._splitter = SemanticChunker()
        self._embedder = get_embedding_service()

        self._faiss_path = Path(self._settings.faiss_index_path)
        self._faiss_path.mkdir(parents=True, exist_ok=True)

        self._faiss_store: FAISS | None = None
        self._bm25_docs: List[Document] = []
        self._indexed_files: Dict[str, Dict[str, Any]] = {}

        # Bug #15: cached sparse retriever — only rebuilt on corpus change
        self._sparse_retriever: Optional[Any] = None
        self._sparse_dirty: bool = False

        self._load_indices()

    # ── Public API ────────────────────────────────────────────────────────────

    async def ingest(self, file_path: str | Path) -> Dict[str, Any]:
        path = Path(file_path)

        # Bug #2 + #8: Remove old chunks for this filename before re-indexing
        if path.name in self._indexed_files:
            log.info(
                "Re-upload detected — removing old chunks for file",
                extra={"filename": path.name},
            )
            self._remove_file_chunks(path.name)

        # Load
        docs = await self._loader.aload(path)
        if not docs:
            return {"status": "error", "message": "No content extracted", "chunks": 0}

        # Split
        chunks = self._splitter.split(docs)

        # Embed + add to FAISS
        await self._add_to_faiss(chunks)

        # Add to BM25 corpus
        self._bm25_docs.extend(chunks)
        self._sparse_dirty = True  # Mark cache as stale (Bug #15)

        # Track file
        self._indexed_files[path.name] = {
            "filename": path.name,
            "chunks": len(chunks),
            "indexed_at": docs[0].metadata.get("indexed_at", ""),
        }

        # Persist
        self._save_indices()

        log.info(
            "Ingestion complete",
            extra={"doc_file": path.name, "chunks": len(chunks)},
        )
        return {
            "status": "success",
            "filename": path.name,
            "chunks_indexed": len(chunks),
        }

    def get_faiss_store(self) -> FAISS | None:
        return self._faiss_store

    def get_bm25_docs(self) -> List[Document]:
        return self._bm25_docs

    def get_sparse_retriever(self) -> Optional[Any]:
        """
        Returns a cached SparseRetriever instance, rebuilding only when needed.
        This is the Bug #15 fix — avoids rebuilding BM25 index on every request.
        """
        if not self._bm25_docs:
            return None

        if self._sparse_retriever is None or self._sparse_dirty:
            from app.retrieval.sparse_retrieval import SparseRetriever
            self._sparse_retriever = SparseRetriever(self._bm25_docs)
            self._sparse_dirty = False
            log.info(
                "SparseRetriever rebuilt",
                extra={"corpus_size": len(self._bm25_docs)},
            )

        return self._sparse_retriever

    def get_indexed_files(self) -> Dict[str, Any]:
        return self._indexed_files

    def vector_store_size(self) -> int:
        if self._faiss_store is None:
            return 0
        return self._faiss_store.index.ntotal

    def clear(self) -> int:
        """Delete all indexed documents and reset the stores. Returns number of files removed."""
        import shutil

        files_removed = len(self._indexed_files)

        self._faiss_store = None
        self._bm25_docs = []
        self._indexed_files = {}
        self._sparse_retriever = None
        self._sparse_dirty = False

        if self._faiss_path.exists():
            shutil.rmtree(self._faiss_path)
        self._faiss_path.mkdir(parents=True, exist_ok=True)

        log.info("Vector store cleared", extra={"files_removed": files_removed})
        return files_removed

    # ── Private helpers ───────────────────────────────────────────────────────

    def _remove_file_chunks(self, filename: str) -> None:
        """
        Remove all BM25 chunks belonging to the given filename.
        FAISS does not support selective deletion in this version of LangChain,
        so we rebuild the FAISS index from the remaining docs.
        """
        # Remove from BM25 corpus
        original_count = len(self._bm25_docs)
        self._bm25_docs = [
            d for d in self._bm25_docs
            if d.metadata.get("source", "") != filename
        ]
        removed_bm25 = original_count - len(self._bm25_docs)

        # Remove from indexed_files registry
        self._indexed_files.pop(filename, None)

        # Rebuild FAISS from remaining BM25 docs (only if we actually removed something)
        if removed_bm25 > 0:
            self._rebuild_faiss_from_bm25()
            self._sparse_dirty = True

        log.info(
            "Old chunks removed for re-upload",
            extra={"filename": filename, "bm25_removed": removed_bm25},
        )

    def _rebuild_faiss_from_bm25(self) -> None:
        """Rebuild FAISS index from the current BM25 corpus docs."""
        if not self._bm25_docs:
            self._faiss_store = None
            return

        embedding_fn = self._embedder.model
        try:
            from langchain_community.vectorstores.utils import DistanceStrategy
            self._faiss_store = FAISS.from_documents(
                self._bm25_docs,
                embedding_fn,
                distance_strategy=DistanceStrategy.COSINE,
            )
        except (ImportError, TypeError):
            self._faiss_store = FAISS.from_documents(self._bm25_docs, embedding_fn)

        log.info(
            "FAISS rebuilt after chunk removal",
            extra={"vectors": self._faiss_store.index.ntotal},
        )

    async def _add_to_faiss(self, chunks: List[Document]) -> None:
        embedding_fn = self._embedder.model

        if self._faiss_store is None:
            # Use COSINE distance so similarity_search_with_relevance_scores
            # returns proper [0, 1] cosine similarity values.
            try:
                from langchain_community.vectorstores.utils import DistanceStrategy
                self._faiss_store = FAISS.from_documents(
                    chunks,
                    embedding_fn,
                    distance_strategy=DistanceStrategy.COSINE,
                )
            except (ImportError, TypeError):
                # Older langchain-community without DistanceStrategy — fall back
                self._faiss_store = FAISS.from_documents(chunks, embedding_fn)
        else:
            self._faiss_store.add_documents(chunks)

    def _save_indices(self) -> None:
        if self._faiss_store is not None:
            self._faiss_store.save_local(str(self._faiss_path))

        bm25_path = self._faiss_path / "bm25_docs.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(self._bm25_docs, f)

        meta_path = self._faiss_path / "indexed_files.json"
        with open(meta_path, "w") as f:
            json.dump(self._indexed_files, f, indent=2)

        log.info("Indices persisted", extra={"path": str(self._faiss_path)})

    def _load_indices(self) -> None:
        faiss_index = self._faiss_path / "index.faiss"
        if faiss_index.exists():
            try:
                self._faiss_store = FAISS.load_local(
                    str(self._faiss_path),
                    self._embedder.model,
                    allow_dangerous_deserialization=True,
                )
                log.info(
                    "FAISS index loaded",
                    extra={"size": self._faiss_store.index.ntotal},
                )
            except Exception as e:
                log.error("Failed to load FAISS index", extra={"error": str(e)})

        bm25_path = self._faiss_path / "bm25_docs.pkl"
        if bm25_path.exists():
            try:
                with open(bm25_path, "rb") as f:
                    self._bm25_docs = pickle.load(f)
                log.info("BM25 corpus loaded", extra={"docs": len(self._bm25_docs)})
                self._sparse_dirty = True  # Will be built on first request
            except Exception as e:
                log.error("Failed to load BM25 corpus", extra={"error": str(e)})

        meta_path = self._faiss_path / "indexed_files.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    self._indexed_files = json.load(f)
            except Exception:
                pass


# Global singleton
_pipeline: IngestionPipeline | None = None


def get_ingestion_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline
