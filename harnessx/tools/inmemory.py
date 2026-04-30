# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from .base import ToolResult, _execute_tool
from ._dict_registry_mixin import _DictRegistryMixin


class InMemoryToolRegistry(_DictRegistryMixin):
    """Simple in-memory tool registry. No external dependencies."""

    async def execute(self, name: str, input: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(output="", error=f"Tool '{name}' not found")
        return await _execute_tool(tool, input)
