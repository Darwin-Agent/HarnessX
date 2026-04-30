# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses

from ...core.events import BeforeModelEvent, Message, StepStartEvent, TaskEndEvent, ToolResultEvent
from ...core.processor import MultiHookProcessor


class ToolFailureLimitError(Exception):
    """Raised by ToolFailureGuard when failure count exceeds the configured limit."""


class ToolFailureGuard(MultiHookProcessor):
    """Stop or warn when too many tool calls fail in a single turn.

    Counts tool errors within a turn (reset at ``step_start``).  When the
    count reaches ``max_failures``:

    - ``raise_on_exceed=True``  → raises :exc:`ToolFailureLimitError`.
    - ``raise_on_exceed=False`` → injects a diagnostic warning into the system
      prompt, nudging the model to change strategy instead of retrying.

    Args:
        max_failures:    Maximum allowed tool failures per turn (default 3).
        raise_on_exceed: Raise instead of warn when the limit is reached.
    """

    _singleton_group = "tool_failure_guard"
    _order = 30

    def __init__(self, max_failures: int = 3, raise_on_exceed: bool = False) -> None:
        self.max_failures = max_failures
        self.raise_on_exceed = raise_on_exceed
        self._failure_count = 0
        self._pending_failure_count = 0

    async def on_after_tool(self, event: ToolResultEvent):
        if event.error:
            self._failure_count += 1
        yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._failure_count = 0
        yield event

    async def on_step_start(self, event: StepStartEvent):
        self._pending_failure_count = self._failure_count
        self._failure_count = 0
        yield event

    async def on_before_model(self, event: BeforeModelEvent):
        count = self._pending_failure_count
        if count >= self.max_failures:
            if self.raise_on_exceed:
                raise ToolFailureLimitError(
                    f"Too many tool failures this turn ({count} >= {self.max_failures}). "
                    "Halting to prevent runaway retry loops."
                )
            warning = (
                f"[ToolFailureGuard] {count} tool calls failed last turn "
                f"({self.max_failures} is the limit). "
                "Stop retrying the same failing approach. Diagnose the root cause "
                "and take a fundamentally different action, or report the failure."
            )
            yield dataclasses.replace(
                event,
                messages=event.messages + (Message(role="user", content=warning),),
            )
        else:
            yield event
