# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator, Protocol, runtime_checkable

from ...core.events import Message, StepStartEvent, rough_token_count
from ...core.processor import MultiHookProcessor

if TYPE_CHECKING:
    from .strategies.base import BaseMemory


@runtime_checkable
class BaseMemoryExtractor(Protocol):
    """Extract memories from a message list.

    Returns a list of ``Message`` objects to be written to the memory backend.
    Typically selects the oldest messages (those most at risk of eviction).
    """

    async def extract(self, messages: list[Message]) -> list[Message]: ...


class OldestMessagesExtractor:
    """Default extractor — returns the oldest ``n`` messages in the window.

    These are the messages most likely to be evicted by ``CompactionProcessor``.
    Storing them before compaction ensures they survive in the memory backend.

    Args:
        n: Number of oldest messages to extract (default: 20).
    """

    def __init__(self, n: int = 20) -> None:
        self.n = n

    async def extract(self, messages: list[Message]) -> list[Message]:
        # Skip system message (index 0) — it is already persisted in the system prompt
        non_system = [m for m in messages if m.role != "system"]
        return non_system[: self.n]


class MemoryExtractionProcessor(MultiHookProcessor):
    """Write oldest context messages to memory before compaction fires.

    No-op when the current token count is below ``threshold``.  When it fires,
    it extracts messages via the ``extractor`` strategy and writes them to the
    memory backend.  Both this processor and ``MemoryRetrievalProcessor`` share
    the same memory backend instance.

    Args:
        memory:    Memory backend (same instance as ``MemoryRetrievalProcessor``).
        threshold: Token count above which extraction is triggered (default: 80 000).
        extractor: Strategy choosing which messages to extract.
                   Defaults to ``OldestMessagesExtractor(n=20)``.
    """

    _singleton_group = "memory.extraction"
    _order = 3  # same bucket as MemoryRetrievalProcessor; Profile sets explicit order=

    def __init__(
        self,
        memory: "BaseMemory",
        threshold: int = 80_000,
        extractor: "BaseMemoryExtractor | None" = None,
    ) -> None:
        self._memory = memory
        self.threshold = threshold
        self.extractor: BaseMemoryExtractor = extractor or OldestMessagesExtractor()

    async def on_step_start(self, event: StepStartEvent) -> AsyncIterator[StepStartEvent]:
        token_count = rough_token_count(list(event.raw_messages))
        if token_count < self.threshold:
            yield event
            return
        try:
            extracted = await self.extractor.extract(list(event.raw_messages))
            if extracted:
                await self._memory.add(extracted)
        except Exception:
            pass
        yield event
