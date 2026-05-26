from __future__ import annotations

from fastapi import APIRouter

from app.ingestion.ingestion_pipeline import get_ingestion_pipeline
from app.models.schemas import HealthResponse, IndexedDocument
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    pipeline = get_ingestion_pipeline()
    size = pipeline.vector_store_size()

    models_loaded: dict[str, bool] = {}
    try:
        from app.services.embedding_service import get_embedding_service
        svc = get_embedding_service()
        models_loaded["embedding"] = svc._initialized
    except Exception:
        models_loaded["embedding"] = False

    try:
        from app.rerank.reranker import get_reranker
        rr = get_reranker()
        models_loaded["reranker"] = rr._model is not None
    except Exception:
        models_loaded["reranker"] = False

    return HealthResponse(
        status="ok",
        vector_store_size=size,
        version="1.0.0",
        models_loaded=models_loaded,
    )


@router.get("/sources", response_model=list[IndexedDocument])
async def list_sources() -> list[IndexedDocument]:
    pipeline = get_ingestion_pipeline()
    files = pipeline.get_indexed_files()
    return [
        IndexedDocument(
            doc_id=name,
            filename=info["filename"],
            chunks=info["chunks"],
            indexed_at=info.get("indexed_at", ""),
        )
        for name, info in files.items()
    ]
