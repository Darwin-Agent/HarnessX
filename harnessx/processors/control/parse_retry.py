# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ...core.events import ModelResponseEvent, TaskEndEvent
from ...core.processor import MultiHookProcessor
from ...core.runloop import ModelParseError


class ParseRetryProcessor(MultiHookProcessor):
    """Validate ModelResponseEvent structure and raise on malformed output.

    Checks that every tool call in the model response has a non-empty name and
    a dict input.  Raises :exc:`ModelParseError` on the first violation.

    Args:
        max_consecutive_errors: Raise only after this many consecutive bad
            responses (default 1 — fail on first).  Set higher to tolerate
            occasional model hiccups before aborting.
    """

    _singleton_group = "parse_retry"
    _order = 10

    def __init__(self, max_consecutive_errors: int = 1) -> None:
        self.max_consecutive_errors = max_consecutive_errors
        # Keyed by run_id so parallel workers (sharing the singleton via
        # _singleton_group) each track their own consecutive-error count.
        self._error_counts: dict[str, int] = {}

    def _is_valid(self, event: ModelResponseEvent) -> tuple[bool, str]:
        """Return (valid, error_message)."""
        for tc in event.tool_calls:
            if not tc.name:
                return False, "tool_call missing name"
            if not isinstance(tc.input, dict):
                return False, f"tool_call '{tc.name}' input is not a dict"
        return True, ""

    async def on_after_model(self, event: ModelResponseEvent):
        run_id = event.run_id
        valid, err = self._is_valid(event)
        if valid:
            self._error_counts.pop(run_id, None)
            yield event
            return

        count = self._error_counts.get(run_id, 0) + 1
        self._error_counts[run_id] = count
        if count >= self.max_consecutive_errors:
            self._error_counts.pop(run_id, None)
            raise ModelParseError(f"Malformed model response ({count} consecutive): {err}")
        yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._error_counts.pop(event.run_id, None)
        yield event
