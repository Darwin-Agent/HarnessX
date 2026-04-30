# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ....core.events import ToolSchema
    from ....core.harness import BaseTask
    from ....core.state import State


@runtime_checkable
class BaseToolFilter(Protocol):
    """Filters the tool schema list each step. Applied before the model call."""

    async def filter(
        self,
        schemas: "tuple[ToolSchema, ...]",
        task: "BaseTask",
        state: "State",
    ) -> "tuple[ToolSchema, ...]": ...


class AllowlistToolFilter:
    """Only tools whose name is in `names` pass through."""

    def __init__(self, names: list[str]) -> None:
        self._names = frozenset(names)

    async def filter(self, schemas, task, state) -> "tuple[ToolSchema, ...]":
        return tuple(s for s in schemas if s.name in self._names)


class BlocklistToolFilter:
    """Tools whose name is in `names` are hidden; all others pass through."""

    def __init__(self, names: list[str]) -> None:
        self._names = frozenset(names)

    async def filter(self, schemas, task, state) -> "tuple[ToolSchema, ...]":
        return tuple(s for s in schemas if s.name not in self._names)


class TagToolFilter:
    """Filter by tool tags stored in ToolSchema.metadata["tags"].

    Args:
        blocked_tags: Tools with any of these tags are hidden.
        allowed_tags: If specified, only tools with at least one of these tags pass.
    """

    def __init__(
        self,
        blocked_tags: list[str] | None = None,
        allowed_tags: list[str] | None = None,
    ) -> None:
        self._blocked = frozenset(blocked_tags or [])
        self._allowed = frozenset(allowed_tags or [])

    def _get_tags(self, schema) -> frozenset[str]:
        metadata = getattr(schema, "metadata", {}) or {}
        tags = metadata.get("tags", [])
        return frozenset(str(t) for t in tags) if isinstance(tags, (list, tuple)) else frozenset()

    async def filter(self, schemas, task, state) -> "tuple[ToolSchema, ...]":
        result = []
        for s in schemas:
            tags = self._get_tags(s)
            if self._blocked and tags & self._blocked:
                continue
            if self._allowed and not (tags & self._allowed):
                continue
            result.append(s)
        return tuple(result)


class CompositeToolFilter:
    """Chain multiple filters — all must pass (AND semantics)."""

    def __init__(self, filters: list[BaseToolFilter]) -> None:
        self._filters = filters

    async def filter(self, schemas, task, state) -> "tuple[ToolSchema, ...]":
        result = schemas
        for f in self._filters:
            result = await f.filter(result, task, state)
        return result
