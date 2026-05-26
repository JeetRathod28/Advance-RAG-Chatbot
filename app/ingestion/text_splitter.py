from __future__ import annotations

import re
from collections import defaultdict
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class SemanticChunker:
    """
    Improved chunker:
    - Larger chunks (1000 chars) preserve more context per chunk
    - More overlap (200 chars) catches information at boundaries
    - Heading-aware: section headings are prepended to child chunks
      so every chunk knows which section it belongs to
    - Strips degenerate whitespace before splitting
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            # Prefer splitting at double-newlines, single newlines, then sentences
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            keep_separator=True,   # Keeps punctuation so sentences stay whole
            add_start_index=True,  # Records char offset in metadata
        )

    # ── Heading detection ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_heading(text: str) -> str | None:
        """
        Returns the first heading-like line in a chunk, if any.
        Detects: ALL-CAPS lines, lines ending with ':', Markdown headings (#).
        """
        for line in text.split("\n"):
            line = line.strip()
            if not line or len(line) > 120:
                continue
            if line.startswith("#"):                           # Markdown heading
                return line.lstrip("#").strip()
            if re.match(r"^[A-Z][A-Z\s\-/&]{3,}$", line):   # ALL CAPS header
                return line
            if re.match(r"^[A-Z][^.!?]{2,60}:$", line):     # "Section title:"
                return line.rstrip(":")
        return None

    # ── Main split ────────────────────────────────────────────────────────────

    def split(self, docs: List[Document]) -> List[Document]:
        # Pre-process: collapse excessive blank lines
        cleaned: List[Document] = []
        for doc in docs:
            text = re.sub(r"\n{3,}", "\n\n", doc.page_content)
            text = re.sub(r" {2,}", " ", text)
            cleaned.append(Document(page_content=text, metadata=doc.metadata))

        raw_chunks = self._splitter.split_documents(cleaned)

        # Group by source for chunk_id / total_chunks metadata
        source_groups: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(raw_chunks):
            src = chunk.metadata.get("source", "unknown")
            source_groups[src].append(i)

        final_chunks: List[Document] = []
        for src, indices in source_groups.items():
            total = len(indices)
            current_heading: str | None = None

            for position, idx in enumerate(indices):
                chunk = raw_chunks[idx]
                content = chunk.page_content.strip()

                if not content:
                    continue

                # Update heading context when we encounter a new heading
                detected = self._extract_heading(content)
                if detected:
                    current_heading = detected

                # Prepend section heading to chunk if it doesn't start with it
                if (
                    current_heading
                    and current_heading.lower() not in content[:80].lower()
                ):
                    content = f"[Section: {current_heading}]\n{content}"

                final_chunks.append(Document(
                    page_content=content,
                    metadata={
                        **chunk.metadata,
                        "chunk_id": position,
                        "total_chunks": total,
                        "section": current_heading or "",
                    },
                ))

        log.info(
            "Documents split into chunks",
            extra={"input_docs": len(docs), "output_chunks": len(final_chunks)},
        )
        return final_chunks
