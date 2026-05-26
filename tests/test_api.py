"""FastAPI endpoint tests using TestClient."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Patch heavy model loading before importing main
    with (
        patch("app.services.embedding_service.EmbeddingService._initialize"),
        patch("app.ingestion.ingestion_pipeline.IngestionPipeline._load_indices"),
        patch("app.agents.rag_agent.get_agent"),
    ):
        from main import app
        with TestClient(app) as c:
            yield c


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"ok", "degraded", "error"}
    assert "version" in data
    assert "vector_store_size" in data


def test_health_response_schema(client):
    response = client.get("/api/health")
    data = response.json()
    assert isinstance(data["vector_store_size"], int)
    assert isinstance(data["models_loaded"], dict)


# ── Sources ────────────────────────────────────────────────────────────────────

def test_sources_returns_list(client):
    response = client.get("/api/sources")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ── Upload ─────────────────────────────────────────────────────────────────────

def test_upload_rejects_invalid_extension(client):
    import io
    content = b"some content"
    response = client.post(
        "/api/upload",
        files={"file": ("malware.exe", io.BytesIO(content), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_rejects_oversized_file(client):
    import io
    # 51 MB file
    big_content = b"x" * (51 * 1024 * 1024)
    response = client.post(
        "/api/upload",
        files={"file": ("huge.txt", io.BytesIO(big_content), "text/plain")},
    )
    assert response.status_code == 413


# ── Chat (non-streaming) ───────────────────────────────────────────────────────

def test_chat_requires_session_id(client):
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 422  # Missing session_id


def test_chat_requires_message(client):
    response = client.post("/api/chat", json={"session_id": "s1"})
    assert response.status_code == 422  # Missing message


def test_chat_empty_message_rejected(client):
    response = client.post("/api/chat", json={"session_id": "s1", "message": ""})
    assert response.status_code == 422


@patch("app.api.routes.chat.run_agent")
def test_chat_returns_response(mock_run, client):
    mock_run.return_value = {
        "answer": "Test answer",
        "sources": [],
        "tokens_used": 42,
    }
    response = client.post(
        "/api/chat",
        json={"session_id": "test_s1", "message": "What is LangChain?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer"
    assert data["session_id"] == "test_s1"
    assert data["tokens_used"] == 42
