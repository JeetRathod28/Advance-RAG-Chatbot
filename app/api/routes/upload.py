from __future__ import annotations

import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.config.settings import get_settings
from app.ingestion.ingestion_pipeline import get_ingestion_pipeline
from app.ingestion.document_loader import ALLOWED_EXTENSIONS
from app.models.schemas import ClearResponse, UploadResponse
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> UploadResponse:
    try:
        settings = get_settings()

        # Validate filename
        filename = file.filename or "upload"
        suffix = Path(filename).suffix.lower()
        if not suffix:
            suffix = ".txt"

        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )

        # Read and validate size
        content = await file.read()
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max {settings.max_upload_mb} MB",
            )
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")

        # Save to data directory
        data_dir = Path(settings.data_path)
        data_dir.mkdir(parents=True, exist_ok=True)
        dest = data_dir / filename
        dest.write_bytes(content)

        log.info("File saved", extra={"doc_filename": filename, "bytes": len(content)})

        # Ingest into vector store
        pipeline = get_ingestion_pipeline()
        result = await pipeline.ingest(dest)

        if result.get("status") == "error":
            raise HTTPException(status_code=422, detail=result.get("message", "Ingestion error"))

        return UploadResponse(
            filename=filename,
            chunks_indexed=result["chunks_indexed"],
            status="success",
            message=f"Successfully indexed {result['chunks_indexed']} chunks.",
        )

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        log.error("Upload route failed", extra={"error": str(e), "traceback": tb})
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.delete("/clear-documents", response_model=ClearResponse)
async def clear_documents() -> ClearResponse:
    """Remove all indexed documents and reset the vector store."""
    try:
        pipeline = get_ingestion_pipeline()
        files_removed = pipeline.clear()
        log.info("Documents cleared via API", extra={"files_removed": files_removed})
        return ClearResponse(
            status="success",
            message=f"Cleared {files_removed} document{'s' if files_removed != 1 else ''} from the index.",
            files_removed=files_removed,
        )
    except Exception as e:
        log.error("Clear documents failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")
