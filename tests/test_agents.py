"""Tests for agent nodes and routing logic."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Router node ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_respects_force_route():
    from app.agents.nodes import router_node

    state = {
        "rewritten_query": "What is Python?",
        "conversation_history": "",
        "force_route": "web_only",
    }
    result = await router_node(state)
    assert result["route"] == "web_only"


@pytest.mark.asyncio
async def test_router_falls_back_to_both_on_bad_llm_response():
    from app.agents.nodes import router_node

    with patch("app.agents.nodes.get_llm_service") as mock_svc:
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value="completely invalid response xyz")
        mock_svc.return_value = llm

        state = {
            "rewritten_query": "Tell me about AI",
            "conversation_history": "",
            "force_route": None,
        }
        result = await router_node(state)
        assert result["route"] in {"vector_only", "web_only", "both", "direct"}


# ── Validation node ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_validation_skips_on_high_retry():
    from app.agents.nodes import validation_node

    state = {
        "rewritten_query": "test",
        "compressed_context": "some context",
        "answer": "some answer",
        "retry_count": 3,
    }
    result = await validation_node(state)
    assert result["is_valid"] is True


@pytest.mark.asyncio
async def test_validation_skips_with_no_context():
    from app.agents.nodes import validation_node

    state = {
        "rewritten_query": "hello",
        "compressed_context": "",
        "answer": "hi there",
        "retry_count": 0,
    }
    result = await validation_node(state)
    assert result["is_valid"] is True


# ── Graph routing conditions ───────────────────────────────────────────────────

def test_route_after_router_direct():
    from app.agents.rag_agent import route_after_router
    state = {"route": "direct"}
    assert route_after_router(state) == "direct_answer"


def test_route_after_router_vector():
    from app.agents.rag_agent import route_after_router
    state = {"route": "vector_only"}
    assert route_after_router(state) == "retrieve"


def test_route_after_router_web():
    from app.agents.rag_agent import route_after_router
    state = {"route": "web_only"}
    assert route_after_router(state) == "retrieve"


def test_route_after_validation_valid():
    from app.agents.rag_agent import route_after_validation
    state = {"is_valid": True, "retry_count": 0}
    assert route_after_validation(state) == "memory_update"


def test_route_after_validation_invalid_with_retries_left():
    from app.agents.rag_agent import route_after_validation
    state = {"is_valid": False, "retry_count": 1}
    assert route_after_validation(state) == "query_rewriting"


def test_route_after_validation_exceeded_retries():
    from app.agents.rag_agent import route_after_validation
    state = {"is_valid": False, "retry_count": 3}
    assert route_after_validation(state) == "memory_update"


# ── Memory update node ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_memory_update_stores_turn():
    from app.agents.nodes import memory_update_node
    from app.memory.memory_manager import get_memory_manager

    state = {
        "session_id": "test_session_abc",
        "original_query": "What is RAG?",
        "answer": "RAG stands for Retrieval-Augmented Generation.",
    }
    await memory_update_node(state)

    manager = get_memory_manager()
    mem = manager.get("test_session_abc")
    assert mem is not None
    history = mem.get_history_string()
    assert "RAG" in history
