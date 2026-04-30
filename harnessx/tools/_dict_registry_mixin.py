# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ..core.events import ToolSchema
from .base import Tool, ToolConflictError


class _DictRegistryMixin:
    """Provides _tools dict, register(), get_schemas(), and list_names()."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, replace: bool = False) -> None:
        """Register *tool* in this registry.

        Args:
            tool:    The :class:`~harnessx.tools.base.Tool` to register.
            replace: If ``True``, silently overwrite an existing tool with the
                     same name.  Default ``False`` — raises
                     :exc:`~harnessx.tools.base.ToolConflictError` on conflict
                     so accidental shadowing of built-in tools is caught early.

        Raises:
            ToolConflictError: A tool named ``tool.name`` is already registered
                               and *replace* is ``False``.
        """
        if not replace and tool.name in self._tools:
            existing = self._tools[tool.name]
            raise ToolConflictError(
                f"Tool name conflict: '{tool.name}' is already registered "
                f"(existing: {existing.fn.__module__}.{existing.fn.__qualname__}, "
                f"new: {tool.fn.__module__}.{tool.fn.__qualname__}). "
                f"Use register(tool, replace=True) to overwrite intentionally."
            )
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[ToolSchema]:
        return [t.to_schema() for t in self._tools.values()]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())
