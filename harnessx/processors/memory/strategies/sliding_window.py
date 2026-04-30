# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
from collections import deque
from ....core.events import Message
from .base import compress_by_token_budget


class SlidingWindowMemory:
    """Keeps the most recent n messages. No external dependencies."""

    def __init__(self, n: int = 20):
        self.n = n
        self._messages: deque[Message] = deque(maxlen=n)

    async def add(self, messages: list[Message]) -> None:
        for msg in messages:
            self._messages.append(msg)

    async def retrieve(self, query: str, k: int = 10) -> list[Message]:
        return list(self._messages)

    async def compress(self, messages: list[Message], budget: int) -> list[Message]:
        return compress_by_token_budget(messages, budget)

    async def persist(self) -> None:
        pass

    async def load(self, run_id: str) -> list[Message]:
        return list(self._messages)
