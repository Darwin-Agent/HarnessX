# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, AsyncIterator

from ...core.events import (
    Message,
    StepEndEvent,
    StepStartEvent,
    dict_to_message,
    rough_token_count,
    _extract_text,
)
from ...core.processor import MultiHookProcessor

if TYPE_CHECKING:
    from .strategies.base import BaseMemory
    from .strategies.policy import MemoryPolicy


class MemoryRetrievalProcessor(MultiHookProcessor):
    """Inject long-term memories into context and persist new messages.

    Args:
        memory:        Memory backend implementing ``retrieve`` / ``add``.
        memory_policy: Gate controlling retrieval, compression, and storage.
                       Defaults to ``AlwaysPolicy`` (always retrieve and store).
        top_k:         Maximum number of memory entries retrieved per step.
    """

    _singleton_group = "memory.retrieval"
    _order = 3

    def __init__(
        self,
        memory: "BaseMemory",
        memory_policy: "MemoryPolicy | None" = None,
        top_k: int = 10,
    ) -> None:
        if memory_policy is None:
            from .strategies.policy import AlwaysPolicy

            memory_policy = AlwaysPolicy()
        self._memory = memory
        self._memory_policy = memory_policy
        self._top_k = top_k
        self._step_start_message_count: int = 0

    async def on_step_start(self, event: StepStartEvent) -> AsyncIterator[StepStartEvent]:
        self._step_start_message_count = len(event.raw_messages)

        # Extract last user message as retrieval query
        query = ""
        for msg in reversed(event.raw_messages):
            if msg.role == "user":
                query = _extract_text(msg.content)
                break

        token_count = rough_token_count(list(event.raw_messages))
        retrieved: list[Message] = []
        try:
            if await self._memory_policy.should_retrieve(query, token_count, event.context_window or 0):
                retrieved = await self._memory.retrieve(query=query, k=self._top_k)
        except Exception:
            pass

        if not retrieved:
            yield event
            return

        # Drop tool-bound messages that cannot be injected without their paired context:
        # - role="tool": tool results need a preceding assistant tool_use block
        # - role="assistant" with tool_calls: would need following tool_results
        retrieved = [m for m in retrieved if m.role != "tool" and not (m.role == "assistant" and m.tool_calls)]

        if not retrieved:
            yield event
            return

        # Insert retrieved memories after system message (if present) to preserve invariant 1.
        if event.messages and event.messages[0].role == "system":
            merged = (event.messages[0],) + tuple(retrieved) + event.messages[1:]
        else:
            merged = tuple(retrieved) + event.messages
        yield dataclasses.replace(
            event,
            messages=merged,
            token_count=rough_token_count(list(merged)),
        )

    async def on_step_end(self, event: StepEndEvent) -> AsyncIterator[StepEndEvent]:
        if event.state_snapshot:
            snapshot = event.state_snapshot
            if isinstance(snapshot, dict):
                raw = snapshot.get("raw_messages")
                if raw is None:
                    raw = snapshot.get("messages", [])
            else:
                raw = list(snapshot.messages)
            all_msgs = [dict_to_message(m) if isinstance(m, dict) else m for m in raw]
            new_msgs = all_msgs[self._step_start_message_count :]
            if new_msgs:
                try:
                    if await self._memory_policy.should_store(new_msgs):
                        await self._memory.add(new_msgs)
                except Exception:
                    pass
        yield event
