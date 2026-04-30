# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import hashlib
from collections import deque

import json

from ...core.events import (
    StepStartEvent,
    TaskEndEvent,
    TaskStartEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ...core.processor import MultiHookProcessor
from ...core.runloop import LoopDetectedError


_WARN_TEMPLATE = (
    "\n\n[LoopDetection] ⚠️  The exact same tool call(s) have been issued {count} times "
    "in a row ({tools}). You are stuck in a loop. Stop and reconsider your approach: "
    "re-read the task, check what you have already tried, and try something fundamentally "
    "different."
)

_NAME_WARN_TEMPLATE = (
    "\n\n[LoopDetection] ⚠️  You have called `{tool}` {count} times consecutively with "
    "different arguments. This suggests you are stuck in a repetitive pattern "
    "({pattern}). Stop and ask yourself: is this work actually necessary, or are "
    "you going in circles? Re-read the task requirements and consider a completely "
    "different approach — or simply finish if the task is already done."
)


class LoopDetectionProcessor(MultiHookProcessor):
    """Detect repeating step patterns using two parallel fingerprint strategies.

    **Strategy 1 — exact** (name + inputs): high-precision, two-phase
    (warn then raise).  Catches identical tool calls repeated verbatim.

    **Strategy 2 — name-only** (tool names, inputs ignored): catches semantic
    loops where the agent repeatedly calls the same *type* of tool with varying
    arguments (document-generation loops, infinite analysis sampling, etc.).
    Warn-only — never raises, because name-only matching is noisier and
    terminating on a loose signal would cause false-positive task failures.

    Both strategies use the **consecutive run length** at the tail of their
    respective sliding windows, so interleaved steps break the count and avoid
    false positives on legitimate exploration.

    **Compaction awareness**: fingerprint windows are cleared when a
    message-count drop of ≥ ``compaction_drop_threshold`` is detected in
    ``on_step_start``, preventing stale pre-compaction fingerprints from
    contributing to spurious loop counts.

    Args:
        window_size:               Sliding window size for both strategies
                                   (default 12).
        warn_threshold:            Strategy 1 consecutive-repeat count that
                                   injects a warning (default 3).
        threshold:                 Strategy 1 consecutive-repeat count that
                                   raises :exc:`LoopDetectedError` (default 5).
        name_warn_threshold:       Strategy 2 consecutive-repeat count that
                                   injects a warning (default 8).  No raise.
        compaction_drop_threshold: Message-count drop that signals compaction
                                   and triggers a fingerprint reset (default 5).
    """

    _singleton_group = "loop_detection"
    _order = 20

    def __init__(
        self,
        window_size: int = 12,
        warn_threshold: int = 3,
        threshold: int = 5,
        name_warn_threshold: int = 8,
        compaction_drop_threshold: int = 5,
    ):
        self.window_size = window_size
        self.warn_threshold = warn_threshold
        self.threshold = threshold
        self.name_warn_threshold = name_warn_threshold
        self.compaction_drop_threshold = compaction_drop_threshold

        self._fingerprints: deque[str] = deque(maxlen=window_size)
        self._name_fingerprints: deque[str] = deque(maxlen=window_size)
        self._current_run_id: str = ""
        self._prev_message_count: int = 0
        # Maps tool_call_id → exact fingerprint (captured in on_before_tool,
        # consumed in on_after_tool to avoid a separate staging step).
        self._pending_fp: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_fingerprint(self, tool_call_summary: str) -> str:
        """sha256 of tool_call_summary; empty string if no tool calls."""
        if not tool_call_summary:
            return ""
        return hashlib.sha256(tool_call_summary.encode()).hexdigest()[:16]

    @staticmethod
    def _consecutive_tail(window: deque, fp: str) -> int:
        """Count how many entries at the *tail* of *window* equal *fp*."""
        count = 0
        for past in reversed(window):
            if past == fp:
                count += 1
            else:
                break
        return count

    def _reset(self) -> None:
        self._fingerprints.clear()
        self._name_fingerprints.clear()
        self._prev_message_count = 0
        self._pending_fp.clear()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    async def on_task_start(self, event: TaskStartEvent):
        """Reset all per-task state at the beginning of each task."""
        self._reset()
        self._current_run_id = event.run_id
        yield event

    async def on_step_start(self, event: StepStartEvent):
        """Detect compaction-driven message drops; clear stale fingerprints if needed."""
        if event.run_id != self._current_run_id:
            self._reset()
            self._current_run_id = event.run_id

        # CompactionProcessor (order=8) runs before us (order=20) so
        # event.messages already reflects any eviction done this step.
        current_count = len(event.messages)
        if self._prev_message_count > 0 and self._prev_message_count - current_count >= self.compaction_drop_threshold:
            self._fingerprints.clear()
            self._name_fingerprints.clear()
            self._pending_fp.clear()
        self._prev_message_count = current_count
        yield event

    async def on_before_tool(self, event: ToolCallEvent):
        """Capture the exact fingerprint (name + serialised inputs) for this call.

        Stored by tool_call_id so on_after_tool can look it up without needing
        the inputs again — ToolResultEvent does not carry them.
        """
        try:
            input_str = json.dumps(event.tool_input, sort_keys=True, ensure_ascii=False)
        except Exception:
            input_str = repr(event.tool_input)
        self._pending_fp[event.tool_call_id] = self._compute_fingerprint(f"{event.tool_name}\x00{input_str}")
        yield event

    async def on_after_tool(self, event: ToolResultEvent):
        """Detect loop and inject warning directly into this tool result if needed.

        Both strategies are evaluated here, so no staging step is required.
        The warning is appended to the result content so the system prompt is
        never touched by this processor.
        """
        fp = self._pending_fp.pop(event.tool_call_id, None)

        # Steps with no fingerprint (e.g. on_before_tool not seen) break the run.
        if not fp:
            self._fingerprints.append("")
            self._name_fingerprints.append("")
            yield event
            return

        # ── Strategy 1: exact fingerprint (name + inputs) ──────────────
        s1_run = self._consecutive_tail(self._fingerprints, fp) + 1
        self._fingerprints.append(fp)

        if s1_run >= self.threshold:
            raise LoopDetectedError(
                f"Loop detected: identical tool call '{event.tool_name}' repeated {s1_run} times consecutively"
            )

        warning = ""
        if s1_run >= self.warn_threshold:
            warning = _WARN_TEMPLATE.format(
                count=s1_run,
                tools=f"`{event.tool_name}`",
            )

        # ── Strategy 2: name-only fingerprint (warn-only, no raise) ────
        s2_run = self._consecutive_tail(self._name_fingerprints, event.tool_name) + 1
        self._name_fingerprints.append(event.tool_name)

        if not warning and s2_run >= self.name_warn_threshold:
            warning = _NAME_WARN_TEMPLATE.format(
                tool=event.tool_name,
                count=s2_run,
                pattern=event.tool_name,
            )

        if warning:
            yield dataclasses.replace(event, result=(event.result or "") + "\n\n" + warning)
        else:
            yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._reset()
        yield event
