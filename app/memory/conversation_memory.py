from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple

from app.config.settings import get_settings
from app.utils.helpers import count_tokens, truncate_to_tokens
from app.utils.logger import get_logger

log = get_logger(__name__)

Turn = Tuple[str, str]  # (human, ai)


class ConversationMemory:
    """
    Per-session sliding-window memory with automatic summarization.
    Keeps recent turns as raw text; older turns are summarized.
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._settings = get_settings()
        self._turns: List[Turn] = []
        self._summary: str = ""
        self._last_accessed: datetime = datetime.utcnow()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_expired(self) -> bool:
        ttl = timedelta(hours=self._settings.session_ttl_hours)
        return datetime.utcnow() - self._last_accessed > ttl

    def touch(self) -> None:
        self._last_accessed = datetime.utcnow()

    def add_turn(self, human: str, ai: str) -> None:
        self._turns.append((human, ai))
        self.touch()
        self._maybe_summarize()

    def get_history_string(self) -> str:
        """Return conversation history as a formatted string."""
        parts: List[str] = []
        if self._summary:
            parts.append(f"[Earlier conversation summary]\n{self._summary}")
        for human, ai in self._turns:
            parts.append(f"Human: {human}\nAssistant: {ai}")
        return "\n\n".join(parts)

    def get_messages(self) -> List[dict]:
        """Return history as a list of role/content dicts."""
        messages: List[dict] = []
        for human, ai in self._turns:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": ai})
        return messages

    def clear(self) -> None:
        self._turns = []
        self._summary = ""

    # ── Internal ──────────────────────────────────────────────────────────────

    def _maybe_summarize(self) -> None:
        """Summarize and trim oldest turns when token budget is exceeded."""
        max_tokens = self._settings.memory_max_tokens
        history = self.get_history_string()

        if count_tokens(history) <= max_tokens:
            return

        # Keep last 3 turns as-is; summarize the rest
        if len(self._turns) <= 3:
            return

        to_summarize = self._turns[:-3]
        self._turns = self._turns[-3:]

        # Build a simple extractive summary
        lines = []
        for human, ai in to_summarize:
            lines.append(f"User asked: {human[:200]}")
            lines.append(f"Assistant replied: {ai[:300]}")

        new_summary_block = "\n".join(lines)
        if self._summary:
            combined = f"{self._summary}\n\n{new_summary_block}"
        else:
            combined = new_summary_block

        # Truncate summary itself to stay bounded
        self._summary = truncate_to_tokens(combined, max_tokens // 2)
        log.info(
            "Conversation memory summarized",
            extra={"session_id": self._session_id},
        )
