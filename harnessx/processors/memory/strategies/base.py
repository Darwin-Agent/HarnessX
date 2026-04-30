# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Protocol, runtime_checkable
from ....core.events import Message, rough_token_count


def compress_by_token_budget(messages: "list[Message]", budget: int) -> "list[Message]":
    """Keep the most-recent messages that fit within budget tokens.

    Preserves tool_use/tool_result pairs: groups assistant+tool_calls with
    their immediately following tool-role messages so they are kept or dropped
    as a unit.
    """
    groups: list[list[Message]] = []
    i = len(messages) - 1
    while i >= 0:
        m = messages[i]
        if m.role == "tool":
            group = [m]
            i -= 1
            while i >= 0 and messages[i].role == "tool":
                group.insert(0, messages[i])
                i -= 1
            if i >= 0 and messages[i].role == "assistant" and messages[i].tool_calls:
                group.insert(0, messages[i])
                i -= 1
            groups.append(group)
        else:
            groups.append([m])
            i -= 1

    groups.reverse()

    result: list[Message] = []
    total = 0
    for group in reversed(groups):
        tokens = rough_token_count(group)
        if total + tokens > budget:
            break
        for msg in group:
            result.insert(0, msg)
        total += tokens
    return result or (messages[-1:] if messages else [])


@runtime_checkable
class BaseMemory(Protocol):
    """Long-term cross-session knowledge storage.

    Distinct from in-session context-window guards:
    BaseMemory persists facts across runs; context processors manage
    which messages from the current run fit in the model context window.
    """

    async def add(self, messages: list[Message]) -> None: ...
    async def retrieve(
        self,
        query: str,
        k: int = 10,
        *,
        query_blocks: list[dict] | None = None,
    ) -> list[Message]: ...
    async def compress(self, messages: list[Message], budget: int) -> list[Message]: ...
    async def persist(self) -> None: ...
    async def load(self, run_id: str) -> list[Message]: ...


@runtime_checkable
class MutableMemory(BaseMemory, Protocol):
    """BaseMemory + update/delete for fine-grained memory management."""

    async def update(self, message_id: str, message: Message) -> bool: ...
    async def delete(self, message_id: str) -> bool: ...
