# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harnessx.core.events import ToolSchema
    from harnessx.core.harness import BaseTask
    from harnessx.core.state import State

# Tool name prefixes that indicate read-only / information-gathering operations.
# tau2 action tools (refuel_data, enable_roaming, toggle_*, disconnect_*, set_*)
# do NOT match these prefixes and are treated as mutation tools.
_READ_PREFIXES = ("get_", "list_", "search_", "find_", "check_", "look_", "think")


class PhaseAwareToolFilter:
    """Restrict tool exposure by task phase.

    - Steps 0 … read_only_steps-1: only read-only tools (prefix-matched).
    - Steps read_only_steps+: all tools unlocked.

    Rationale: tau2 agents often call action tools (refuel_data, enable_roaming)
    before fully diagnosing the issue, wasting turns and degrading task score.
    Restricting actions in early steps forces an information-gathering phase.

    Tool classification uses name-prefix convention:
      Read  — get_*, list_*, search_*, find_*, check_*, look_*, think*
      Action — everything else (refuel_data, enable_roaming, toggle_*, etc.)

    Args:
        read_only_steps: Number of initial steps limited to read tools (default 2).
    """

    def __init__(self, read_only_steps: int = 2) -> None:
        self.read_only_steps = read_only_steps

    def _is_read_tool(self, name: str) -> bool:
        return any(name.startswith(p) for p in _READ_PREFIXES)

    async def filter(
        self,
        schemas: "tuple[ToolSchema, ...]",
        task: "BaseTask",
        state: "State",
    ) -> "tuple[ToolSchema, ...]":
        # ToolFilterProcessor passes the StepStartEvent as `state`.
        # Read raw_messages from it to count completed agent turns — each
        # assistant message represents one turn.  Fallback to State.messages
        # for callers that pass a State directly.
        msgs = getattr(state, "raw_messages", None) or getattr(state, "messages", None) or []
        turn = sum(1 for m in msgs if getattr(m, "role", None) == "assistant")
        if turn < self.read_only_steps:
            return tuple(s for s in schemas if self._is_read_tool(s.name))
        return schemas
