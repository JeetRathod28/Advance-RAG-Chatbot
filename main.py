from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.upload import router as upload_router
from app.config.settings import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models and load indices on startup."""
    settings = get_settings()
    log.info("Starting RAG Chatbot", extra={"model": settings.llm_model})

    # Ensure directories exist
    Path(settings.faiss_index_path).mkdir(parents=True, exist_ok=True)
    Path(settings.data_path).mkdir(parents=True, exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)

    # Load ingestion pipeline (loads FAISS index if it exists)
    from app.ingestion.ingestion_pipeline import get_ingestion_pipeline
    pipeline = get_ingestion_pipeline()
    log.info(
        "Vector store ready",
        extra={"size": pipeline.vector_store_size()},
    )

    # Warm up embedding model in background (don't block startup)
    async def _warm_embeddings():
        try:
            from app.services.embedding_service import get_embedding_service
            svc = get_embedding_service()
            svc.embed_query("warm up")
            log.info("Embedding model warmed up")
        except Exception as e:
            log.warning("Embedding warmup failed", extra={"error": str(e)})

    asyncio.create_task(_warm_embeddings())

    log.info("Startup complete")
    yield

    log.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Advanced RAG Chatbot",
        description=(
            "Production-grade RAG chatbot with hybrid retrieval, "
            "agentic reasoning, and streaming responses."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routers
    app.include_router(chat_router, prefix="/api")
    app.include_router(upload_router, prefix="/api")
    app.include_router(health_router, prefix="/api")

    # Serve React frontend if built
    frontend_dist = Path("./frontend/dist")
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
        log.info("Serving frontend from frontend/dist")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
