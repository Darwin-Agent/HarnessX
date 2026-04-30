# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from collections import defaultdict

from ...core.events import TaskEndEvent, ToolResultEvent, ToolCallEvent
from ...core.processor import MultiHookProcessor

_EDIT_TOOLS = {"Write", "Edit"}

_SOFT_HINT_TEMPLATE = (
    "\n\n[RepeatedFileEditDetector] `{path}` has been written/edited {count} times. "
    "If the task is still not resolved, consider stepping back and trying a "
    "completely different approach rather than continuing to patch the same file."
)

_HARD_HINT_TEMPLATE = (
    "\n\n[RepeatedFileEditDetector] ⚠️  `{path}` has been written/edited {count} times. "
    "Your current approach is not working. STOP patching this file. "
    "Step back completely, re-read the task requirements from scratch, and design "
    "a fundamentally different solution."
)


class RepeatedFileEditDetector(MultiHookProcessor):
    """Inject a hint into the tool result when the same file is edited repeatedly.

    Tracks ``Write`` and ``Edit`` tool calls per file path.  When the edit count
    for a file reaches *soft_threshold*, a warning hint is appended.  At
    *hard_threshold*, a stronger message demanding a different approach is injected
    and the counter resets to zero.

    Args:
        soft_threshold:  Edit count that triggers the soft warning (default 7).
        hard_threshold:  Edit count that triggers the hard reflection message (default 12).
        edit_tools:      Tool names counted as edits (default ``{"Write", "Edit"}``).
    """

    _singleton_group = "repeated_edit_detector"
    _order = 25  # after ToolCallCorrectionLayer (5) and ParseRetryProcessor (10)

    def __init__(
        self,
        soft_threshold: int = 7,
        hard_threshold: int = 12,
        edit_tools: set[str] | None = None,
    ) -> None:
        self._soft_threshold = soft_threshold
        self._hard_threshold = hard_threshold
        self._edit_tools = edit_tools if edit_tools is not None else set(_EDIT_TOOLS)
        self._counts: dict[str, int] = defaultdict(int)
        # Staging dict: tool_call_id → file_path, populated in before_tool so
        # after_tool can recover the path without re-parsing the result string.
        self._pending_paths: dict[str, str] = {}

    async def on_before_tool(self, event: ToolCallEvent):
        # Only stage if the tool call is approved — a blocked call never reaches
        # after_tool, so staging it would leave a dangling entry in _pending_paths.
        if event.tool_name in self._edit_tools and event.approved:
            path = event.tool_input.get("file_path", "")
            if path:
                self._pending_paths[event.tool_call_id] = path
        yield event

    async def on_after_tool(self, event: ToolResultEvent):
        if event.tool_name not in self._edit_tools:
            yield event
            return

        path = self._pending_paths.pop(event.tool_call_id, None)
        if not path:
            yield event
            return

        self._counts[path] += 1
        count = self._counts[path]

        if count >= self._hard_threshold:
            hint = _HARD_HINT_TEMPLATE.format(path=path, count=count)
            self._counts[path] = 0
            yield dataclasses.replace(event, result=event.result + hint)
        elif count >= self._soft_threshold:
            hint = _SOFT_HINT_TEMPLATE.format(path=path, count=count)
            yield dataclasses.replace(event, result=event.result + hint)
        else:
            yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._counts.clear()
        self._pending_paths.clear()
        yield event
