from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    score: float = 0.0
    source_type: Literal["document", "web"] = "document"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=8192)
    use_web_search: Optional[bool] = None   # None = let router decide


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    session_id: str
    tokens_used: int = 0


class StreamChunk(BaseModel):
    type: Literal["token", "sources", "done", "error"]
    content: Any


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    status: Literal["success", "error"]
    message: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    vector_store_size: int = 0
    version: str = "1.0.0"
    models_loaded: dict[str, bool] = Field(default_factory=dict)


class IndexedDocument(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    indexed_at: str


class ClearResponse(BaseModel):
    status: Literal["success", "error"]
    message: str
    files_removed: int = 0
