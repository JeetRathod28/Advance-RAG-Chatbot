from __future__ import annotations

import threading
from typing import Dict, List, Optional

from app.memory.conversation_memory import ConversationMemory
from app.utils.logger import get_logger

log = get_logger(__name__)


class MemoryManager:
    """
    Thread-safe registry of per-session ConversationMemory instances.
    Also handles TTL-based cleanup.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> ConversationMemory:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationMemory(session_id)
                log.info("New session created", extra={"session_id": session_id})
            else:
                # Evict if expired and start fresh
                mem = self._sessions[session_id]
                if mem.is_expired:
                    log.info(
                        "Session expired, resetting",
                        extra={"session_id": session_id},
                    )
                    self._sessions[session_id] = ConversationMemory(session_id)

            return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[ConversationMemory]:
        with self._lock:
            return self._sessions.get(session_id)

    def clear(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                log.info("Session cleared", extra={"session_id": session_id})

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())

    def evict_expired(self) -> int:
        with self._lock:
            expired = [
                sid for sid, mem in self._sessions.items() if mem.is_expired
            ]
            for sid in expired:
                del self._sessions[sid]
            if expired:
                log.info("Evicted expired sessions", extra={"count": len(expired)})
            return len(expired)


_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
