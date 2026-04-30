# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from typing import Any

from ...core.events import (
    BeforeModelEvent,
    Message,
    StepStartEvent,
    TaskEndEvent,
    ToolCallEvent,
)
from ...core.processor import MultiHookProcessor


class TodoCheck(MultiHookProcessor):
    """Remind the model of remaining todos after a tool-free step.

    Activates only when the model has called ``todo_write`` at least once.
    After a step in which the model produced no tool calls (pure-text output),
    the processor injects a user message at the next ``step_start`` listing
    any items whose status is ``"pending"`` or ``"in_progress"``.

    Pair with :func:`make_todo_tool` to register the ``todo_write`` tool.

    Args:
        tool_name: Name of the todo tool to watch (default ``"todo_write"``).
    """

    _singleton_group = "todo_check"
    _order = 15

    def __init__(self, tool_name: str = "todo_write") -> None:
        self.tool_name = tool_name
        self._todos: list[dict] = []  # latest snapshot from todo_write
        self._step_had_tools = False  # did the current step call any tool?
        self._no_tools_last_step = False  # cached for before_model

    # ── hooks ────────────────────────────────────────────────────────────────

    async def on_step_start(self, event: StepStartEvent):
        self._no_tools_last_step = not self._step_had_tools
        self._step_had_tools = False  # reset for the new step
        yield event

    async def on_before_model(self, event: BeforeModelEvent):
        if self._todos and self._no_tools_last_step:
            pending = [t for t in self._todos if t.get("status") in ("pending", "in_progress")]
            if pending:
                mark = {"in_progress": "[-]", "pending": "[ ]"}
                lines = "\n".join(
                    f"  {mark.get(t.get('status', 'pending'), '[ ]')} {t.get('id', '?')}. {t.get('content', '')}"
                    for t in pending
                )
                reminder = f"[TodoCheck] Incomplete todos:\n{lines}"
                yield dataclasses.replace(
                    event,
                    messages=event.messages + (Message(role="user", content=reminder),),
                )
                return
        yield event

    async def on_before_tool(self, event: ToolCallEvent):
        self._step_had_tools = True
        if event.tool_name == self.tool_name:
            self._todos = list(event.tool_input.get("todos", []))
        yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._todos = []
        self._step_had_tools = False
        yield event


# ---------------------------------------------------------------------------
# Backwards-compatible alias so existing imports keep working
# ---------------------------------------------------------------------------
TodoWriteEnforcer = TodoCheck


def make_todo_tool(tool_name: str = "todo_write") -> "Any":
    """Return a :class:`~harnessx.tools.base.Tool` for creating and updating a todo list.

    The tool stores a per-closure in-memory list of todo items.  It is intended
    to be bundled with :class:`TodoCheck` via
    :meth:`~harnessx.core.builder.HarnessBuilder.add_tool`.

    Example model call::

        todo_write(todos=[
            {"id": "1", "content": "Research topic", "status": "in_progress"},
            {"id": "2", "content": "Write report",   "status": "pending"},
        ])
    """
    from ...tools.base import Tool

    _store: list[dict] = []

    def _write(todos: list) -> str:
        _store.clear()
        _store.extend(todos)
        mark_map = {"done": "x", "in_progress": "-", "pending": " "}
        lines = [
            f"  [{mark_map.get(item.get('status', 'pending'), ' ')}] {item.get('id', '?')}. {item.get('content', '')}"
            for item in _store
        ]
        return "Todo list updated:\n" + "\n".join(lines)

    return Tool(
        name=tool_name,
        description=(
            "Create or update the task todo list. Call this to track multiple "
            "sub-tasks. Provide the complete list each call — it replaces the "
            "previous list."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Complete todo list (replaces previous list)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "done"],
                            },
                        },
                        "required": ["id", "content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
        fn=_write,
    )
