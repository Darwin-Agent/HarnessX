# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, AsyncIterator

from ...core.events import StepStartEvent
from ...core.processor import MultiHookProcessor

if TYPE_CHECKING:
    from .strategies.tool_filter import BaseToolFilter


class ToolFilterProcessor(MultiHookProcessor):
    """Filter the tool set exposed to the model each step.

    Args:
        tool_filter: Strategy implementing
                     ``async filter(tools, task, context) -> tuple[ToolSchema, ...]``.
    """

    _singleton_group = "tools.filter"
    _order = 6

    def __init__(self, tool_filter: "BaseToolFilter") -> None:
        self.tool_filter = tool_filter

    async def on_step_start(self, event: StepStartEvent) -> AsyncIterator[StepStartEvent]:
        filtered = await self.tool_filter.filter(event.tools, event.task, event)
        yield dataclasses.replace(event, tools=filtered)
