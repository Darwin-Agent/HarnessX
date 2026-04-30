# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
from ....core.events import Message
from .base import compress_by_token_budget


class SummarizationMemory:
    """Compresses old messages using an LLM. Requires a model_provider."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", ratio: float = 0.7):
        self.model = model
        self.ratio = ratio
        self._messages: list[Message] = []

    async def add(self, messages: list[Message]) -> None:
        self._messages.extend(messages)

    async def retrieve(self, query: str, k: int = 10) -> list[Message]:
        return self._messages[-k:]

    async def compress(self, messages: list[Message], budget: int) -> list[Message]:
        return compress_by_token_budget(messages, budget)

    async def persist(self) -> None:
        pass

    async def load(self, run_id: str) -> list[Message]:
        return list(self._messages)
