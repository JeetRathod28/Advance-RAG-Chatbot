from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document

from app.utils.helpers import clean_text
from app.utils.logger import get_logger

log = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


class DocumentLoader:
    """Load PDF, TXT, DOCX, and Markdown files into LangChain Documents."""

    def load(self, file_path: str | Path) -> List[Document]:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {ALLOWED_EXTENSIONS}"
            )

        log.info("Loading document", extra={"path": str(path), "type": ext})

        if ext == ".pdf":
            docs = self._load_pdf(path)
        elif ext in (".txt", ".md"):
            docs = self._load_text(path)
        elif ext == ".docx":
            docs = self._load_docx(path)
        else:
            docs = []

        # Enrich metadata and clean content
        enriched: List[Document] = []
        for doc in docs:
            content = clean_text(doc.page_content)
            if not content.strip():
                continue
            meta = {
                **doc.metadata,
                "source": path.name,
                "file_path": str(path),
                "file_type": ext.lstrip("."),
                "indexed_at": datetime.utcnow().isoformat(),
            }
            enriched.append(Document(page_content=content, metadata=meta))

        log.info(
            "Document loaded",
            extra={"path": str(path), "pages": len(enriched)},
        )
        return enriched

    def _load_pdf(self, path: Path) -> List[Document]:
        loader = PyMuPDFLoader(str(path))
        return loader.load()

    def _load_text(self, path: Path) -> List[Document]:
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        return loader.load()

    def _load_docx(self, path: Path) -> List[Document]:
        loader = Docx2txtLoader(str(path))
        return loader.load()

    async def aload(self, file_path: str | Path) -> List[Document]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.load, file_path
        )
