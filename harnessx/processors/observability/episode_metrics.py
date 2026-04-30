# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import AsyncIterator

from harnessx.core.events import StepEndEvent, TaskEndEvent, ToolResultEvent
from harnessx.core.processor import MultiHookProcessor


class EpisodeMetricsProcessor(MultiHookProcessor):
    """
    Accumulates per-episode RL training metrics.

    Instance variables (readable after Harness.run()):
        step_tokens:        list[int]    — cumulative_tokens at each step end
        step_costs:         list[float]  — cumulative_cost_usd at each step end
        tool_summaries:     list[str]    — tool_call_summary per step
        tool_success_count: int          — total successful tool executions
        tool_error_count:   int          — total failed tool executions
        episode_summary:    dict         — populated on task_end

    reward_func() uses result.task_end for the final episode summary
    (total_steps, total_tokens, exit_reason) — not this processor's fields.
    """

    def __init__(self) -> None:
        self.step_tokens: list[int] = []
        self.step_costs: list[float] = []
        self.tool_summaries: list[str] = []
        self.tool_success_count: int = 0
        self.tool_error_count: int = 0
        self.episode_summary: dict = {}

    async def on_after_tool(
        self,
        event: ToolResultEvent,
    ) -> AsyncIterator[ToolResultEvent]:
        """Count tool successes and errors as each tool result arrives.

        Two error channels:
          event.error — set when tool registry raises an exception (e.g. tool not found)
          event.result — code_interpreter returns all execution errors as "Error: ..." strings
                         (never raises, so event.error is None for sandbox errors)
        """
        is_error = bool(event.error) or (isinstance(event.result, str) and event.result.startswith("Error:"))
        if is_error:
            self.tool_error_count += 1
        else:
            self.tool_success_count += 1
        yield event

    async def on_step_end(
        self,
        event: StepEndEvent,
    ) -> AsyncIterator[StepEndEvent]:
        self.step_tokens.append(event.cumulative_tokens)
        self.step_costs.append(event.cumulative_cost_usd)
        self.tool_summaries.append(event.tool_call_summary or "")
        yield event

    async def on_task_end(
        self,
        event: TaskEndEvent,
    ) -> AsyncIterator[TaskEndEvent]:
        self.episode_summary = {
            "total_steps": event.total_steps,
            "total_tokens": event.total_tokens,
            "total_input_tokens": event.total_input_tokens,
            "total_output_tokens": event.total_output_tokens,
            "total_cost_usd": event.total_cost_usd,
            "exit_reason": event.exit_reason,
            "tool_success_count": self.tool_success_count,
            "tool_error_count": self.tool_error_count,
        }
        yield event
