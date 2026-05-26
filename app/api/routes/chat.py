from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.rag_agent import run_agent, stream_agent
from app.memory.memory_manager import get_memory_manager
from app.models.schemas import ChatRequest, ChatResponse
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["chat"])


def _sse_encode(data: dict) -> str:
    """Format a dict as an SSE data event."""
    return f"data: {json.dumps(data)}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Events:
      {"type": "status",  "content": "..."}
      {"type": "token",   "content": "..."}
      {"type": "sources", "content": [...]}
      {"type": "done",    "content": ""}
      {"type": "error",   "content": "..."}
    """
    manager = get_memory_manager()
    mem = manager.get_or_create(request.session_id)
    history = mem.get_history_string()

    force_route: str | None = None
    if request.use_web_search is True:
        force_route = "web_only"
    elif request.use_web_search is False:
        force_route = "vector_only"

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for chunk in stream_agent(
                session_id=request.session_id,
                message=request.message,
                conversation_history=history,
                force_route=force_route,
            ):
                yield _sse_encode(chunk)
        except Exception as e:
            log.error("Streaming error", extra={"error": str(e)})
            yield _sse_encode({"type": "error", "content": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Non-streaming chat endpoint (full response in one JSON payload).
    """
    manager = get_memory_manager()
    mem = manager.get_or_create(request.session_id)
    history = mem.get_history_string()

    force_route: str | None = None
    if request.use_web_search is True:
        force_route = "web_only"
    elif request.use_web_search is False:
        force_route = "vector_only"

    try:
        final_state = await run_agent(
            session_id=request.session_id,
            message=request.message,
            conversation_history=history,
            force_route=force_route,
        )
    except Exception as e:
        log.error("Chat failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        answer=final_state.get("answer", ""),
        sources=final_state.get("sources", []),
        session_id=request.session_id,
        tokens_used=final_state.get("tokens_used", 0),
    )
