# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, AsyncIterator

from ...core.events import TaskStartEvent
from ...core.processor import MultiHookProcessor

if TYPE_CHECKING:
    from .strategies.system_prompt.base import BaseSystemPromptBuilder


class SystemPromptProcessor(MultiHookProcessor):
    """Build the base system prompt from a strategy.

    Args:
        system_builder: Strategy implementing ``async build(workspace) -> str``.
                        Defaults to ``DefaultSystemPromptBuilder``.
    """

    _singleton_group = "context.system"
    _order = 1

    def __init__(self, system_builder: "BaseSystemPromptBuilder | None" = None) -> None:
        if system_builder is None:
            from .strategies.system_prompt.default import DefaultSystemPromptBuilder

            system_builder = DefaultSystemPromptBuilder()
        self.system_builder = system_builder

    async def on_task_start(self, event: TaskStartEvent) -> AsyncIterator[TaskStartEvent]:
        system = await self.system_builder.build(workspace=event.workspace)
        yield dataclasses.replace(event, system_prompt=system)
