# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from typing import Callable

from ...logging import logger

from ...core.events import ToolCallEvent
from ...core.processor import MultiHookProcessor


def _default_approval_callback(tool_name: str, tool_input: dict) -> bool:
    """Default: CLI confirmation for dangerous tools."""
    try:
        response = input(f"\n[ToolWhitelist] Allow dangerous tool '{tool_name}'? (y/N): ")
        return response.strip().lower() == "y"
    except EOFError:
        return False


class ToolWhitelistProcessor(MultiHookProcessor):
    """
    Hooks: before_tool
    Maintains allowed_tools and dangerous_tools lists.
    Dangerous tools require approval_callback confirmation.
    Unlisted tools are blocked (approved=False).
    """

    _singleton_group = "tool_whitelist"
    _order = 10

    def __init__(
        self,
        allowed_tools: list[str] | None = None,
        dangerous_tools: list[str] | None = None,
        approval_callback: Callable[[str, dict], bool] | None = None,
        allow_all: bool = False,
    ):
        self.allowed_tools = set(allowed_tools or [])
        self.dangerous_tools = set(dangerous_tools if dangerous_tools is not None else ["Bash", "Write"])
        self.approval_callback = approval_callback or _default_approval_callback
        self.allow_all = allow_all

    async def on_before_tool(self, event: ToolCallEvent):
        name = event.tool_name

        if name in self.dangerous_tools and not self.allow_all:
            approved = self.approval_callback(name, event.tool_input)
            yield dataclasses.replace(event, approved=approved)
            return

        if self.allow_all or name in self.allowed_tools:
            yield dataclasses.replace(event, approved=True)
        else:
            logger.warning(f"Tool '{name}' not in whitelist, blocking")
            yield dataclasses.replace(event, approved=False)
