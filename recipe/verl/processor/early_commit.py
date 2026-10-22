# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""EarlyCommitProcessor — nudges model to commit answer when budget is low."""

from __future__ import annotations

import dataclasses
from typing import AsyncIterator

from harnessx.core.events import (
    Message,
    StepEndEvent,
    TaskStartEvent,
    BeforeModelEvent,
    StepStartEvent,
    rough_token_count,
)
from harnessx.core.processor import MultiHookProcessor


class EarlyCommitProcessor(MultiHookProcessor):
    """Injects budget-aware nudge messages to encourage early answer commit.

    At 50% of max_steps: gentle reminder to start converging.
    At last step or token_budget_ratio: strong ultimatum to answer now.
    """

    _order = 50

    def __init__(self, max_steps: int = 12, token_budget_ratio: float = 0.95):
        self._token_budget_ratio = token_budget_ratio
        self._max_steps: int = max_steps
        self._current_step: int = 0
        self._context_window: int = 64_000
        self._fired: set[str] = set()

    async def on_task_start(self, event: TaskStartEvent) -> AsyncIterator[TaskStartEvent]:
        self._current_step = 0
        self._context_window = 64_000
        self._fired.clear()
        yield event

    async def on_step_start(self, event: StepStartEvent) -> AsyncIterator[StepStartEvent]:
        self._context_window = event.context_window
        yield event

    async def on_step_end(self, event: StepEndEvent) -> AsyncIterator[StepEndEvent]:
        self._current_step += 1
        yield event

    async def on_before_model(self, event: BeforeModelEvent) -> AsyncIterator[BeforeModelEvent]:
        nudge = None
        half = self._max_steps // 2

        near_context_limit = rough_token_count(list(event.messages)) >= self._context_window * self._token_budget_ratio
        if (self._current_step >= self._max_steps - 1 or near_context_limit) and "final" not in self._fired:
            self._fired.add("final")
            nudge = (
                "[LAST CHANCE] You MUST provide your final answer NOW in "
                "<answer>...</answer> tags. Do NOT call any more tools."
            )
        elif self._current_step >= half and "half" not in self._fired:
            self._fired.add("half")
            nudge = (
                "[REMINDER] You are past the halfway point. Start converging "
                "toward a final answer. Only call a tool if absolutely necessary."
            )

        if nudge:
            msgs = list(event.messages)
            if msgs and msgs[-1].role == "user":
                last = msgs[-1]
                msgs[-1] = dataclasses.replace(last, content=last.content + "\n\n" + nudge)
            else:
                msgs.append(Message(role="user", content=nudge))
            yield dataclasses.replace(event, messages=tuple(msgs))
        else:
            yield event
