# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Leakage guard processor for meta-agent evolve sessions.

When the meta-agent is about to end its turn (model response has no tool
calls), this processor fires once to inject a self-check user message asking
the model to review its authored template files for task-specific leakage.

Pattern mirrors CustomSelfVerifyProcessor in benchmarks/terminal_bench_2/harness.py:

1. on_after_model  — detect exit intent, stash the message, yield a fake
                     keepalive tool call so the run loop does NOT exit.
2. on_before_tool  — intercept the fake tool, block it with a synthetic ack.
3. on_before_model — inject the stashed user message (last msg is tool result,
                     role≠user, so appending +1 user is valid per the contract).
"""

from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path

from ...core.events import (
    BeforeModelEvent,
    Message,
    ModelResponseEvent,
    TaskEndEvent,
    TaskStartEvent,
    ToolCall,
    ToolCallEvent,
)
from ...core.processor import MultiHookProcessor

_LEAKAGE_GUARD_TOOL = "_meta_leakage_guard"
_LEAKAGE_GUARD_ACK = "Leakage self-check initiated. See the message above for instructions."

_SELF_CHECK_MSG = """\
[Leakage self-check] Before you end your turn, review every template file \
(.j2) you authored under your output_dir.

**What is task-specific leakage?**
Content that only makes sense for the specific tasks in THIS benchmark run, \
not for general unseen tasks. Ask: could this section have been written \
without ever seeing the trajectories? If not, it is leakage.

**Generalizability test**
For each section of your template ask: \
"Would this guidance help an agent solving a completely different task it \
has never encountered?" \
If the answer is no — rewrite it as a general strategy description or remove it.

If you find leakage: fix the template now, then end your turn.
If everything is already general: you may end your turn immediately.
"""


class LeakageGuardProcessor(MultiHookProcessor):
    """Inject a one-shot leakage self-check when the meta-agent tries to exit.

    Fires at most once per task run.  On the next no-tool-call turn it stays
    silent and the run loop exits normally.

    Args:
        output_dir: The evolve output directory.  Included in the message so
            the model knows where to look.
    """

    _singleton_group = "meta_leakage_guard"
    _order = 90

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir).resolve()
        self._checked: bool = False
        self._pending_message: str = ""

    async def on_task_start(self, event: TaskStartEvent):
        self._checked = False
        self._pending_message = ""
        yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._checked = False
        self._pending_message = ""
        yield event

    async def on_before_model(self, event: BeforeModelEvent):
        if not self._pending_message:
            yield event
            return
        msg = self._pending_message
        self._pending_message = ""
        # last message is a tool result (role≠user) → append exactly +1 user
        yield dataclasses.replace(
            event,
            messages=event.messages + (Message(role="user", content=msg),),
        )

    async def on_after_model(self, event: ModelResponseEvent):
        exit_intent = event.finish_reason in ("end_turn", "stop") and not event.tool_calls
        if exit_intent and not self._checked:
            self._checked = True
            self._pending_message = _SELF_CHECK_MSG
            keepalive = ToolCall(
                id=f"lgd-{uuid.uuid4().hex[:8]}",
                name=_LEAKAGE_GUARD_TOOL,
                input={},
            )
            yield dataclasses.replace(event, tool_calls=(keepalive,))
        else:
            yield event

    async def on_before_tool(self, event: ToolCallEvent):
        if event.tool_name == _LEAKAGE_GUARD_TOOL:
            yield dataclasses.replace(
                event,
                approved=False,
                synthetic_result=_LEAKAGE_GUARD_ACK,
            )
        else:
            yield event
