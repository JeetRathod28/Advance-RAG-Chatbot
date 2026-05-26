from __future__ import annotations

from typing import AsyncIterator, Iterator

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from app.config.settings import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class LLMService:
    """
    Ollama LLM client with streaming support.

    Bug #17 fix: cache the ChatOllama instance rather than creating
    a new object on every invoke/stream call.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._llm = self._build_llm()

    def _build_llm(self) -> ChatOllama:
        return ChatOllama(
            base_url=self._settings.ollama_base_url,
            model=self._settings.llm_model,
            temperature=self._settings.llm_temperature,
            num_predict=self._settings.llm_max_tokens,
            timeout=self._settings.llm_timeout,
        )

    def get_llm(self) -> ChatOllama:
        """Return the cached LLM instance."""
        return self._llm

    def invoke(self, messages: list[BaseMessage] | list[dict]) -> str:
        return self._llm.invoke(messages).content

    async def ainvoke(self, messages: list[BaseMessage] | list[dict]) -> str:
        return (await self._llm.ainvoke(messages)).content

    def stream(self, messages: list[BaseMessage] | list[dict]) -> Iterator[str]:
        for chunk in self._llm.stream(messages):
            if chunk.content:
                yield chunk.content

    async def astream(
        self, messages: list[BaseMessage] | list[dict]
    ) -> AsyncIterator[str]:
        async for chunk in self._llm.astream(messages):
            if chunk.content:
                yield chunk.content


_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
