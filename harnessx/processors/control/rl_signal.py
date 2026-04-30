# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import AsyncIterator

from harnessx.core.events import BeforeModelEvent, StepEndEvent
from harnessx.core.processor import MultiHookProcessor
from harnessx.core.runloop import LoopDetectedError


class RLSignalCollectorProcessor(MultiHookProcessor):
    """
    Tracks per-episode RL training signals and enforces loop termination.

    Instance variables (readable after Harness.run()):
        tool_call_fingerprints: list[str]   — per-step tool_call_summary strings
        loop_step_ids: list[int]            — step IDs where loop was detected
        total_tool_calls: int               — total tool calls across episode

    Loop termination:
        When the same tool_call pattern repeats >= 2 times in the last _WINDOW
        steps, on_before_model raises LoopDetectedError on the NEXT step.
        run_loop catches this → exit_reason = "loop_detected".
        EnhancedToolSuccessPRM.loop_penalty then fires.
    """

    _WINDOW = 5  # number of recent steps to check for loop pattern

    def __init__(self) -> None:
        self.tool_call_fingerprints: list[str] = []
        self.loop_step_ids: list[int] = []
        self.total_tool_calls: int = 0

    async def on_step_end(
        self,
        event: StepEndEvent,
    ) -> AsyncIterator[StepEndEvent]:
        fingerprint = event.tool_call_summary or ""

        # Count tool calls (non-empty fingerprint = at least one tool call)
        if fingerprint:
            self.total_tool_calls += fingerprint.count("|") + 1

        # Loop detection: same fingerprint appears >= 2 times in recent window
        if fingerprint:
            recent = self.tool_call_fingerprints[-self._WINDOW :]
            if recent.count(fingerprint) >= 2:
                self.loop_step_ids.append(event.step_id)

        self.tool_call_fingerprints.append(fingerprint)

        yield event

    async def on_before_model(
        self,
        event: BeforeModelEvent,
    ) -> AsyncIterator[BeforeModelEvent]:
        """Raise LoopDetectedError if a loop was confirmed on the previous step.

        run_loop catches LoopDetectedError → exit_reason = "loop_detected".
        """
        if self.loop_step_ids and self.loop_step_ids[-1] == event.step_id - 1:
            raise LoopDetectedError(f"Repeated tool calls detected at step {self.loop_step_ids[-1]}")
        yield event

    @property
    def loop_detected(self) -> bool:
        """True if any loop was detected during this episode."""
        return bool(self.loop_step_ids)
