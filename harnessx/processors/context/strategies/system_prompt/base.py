# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .....workspace.workspace import Workspace


@runtime_checkable
class BaseSystemPromptBuilder(Protocol):
    """Builds the role=system message content (called once per task, result is frozen).

    Dynamic additions (CWD, date, tool names, skill injection, reasoning budget)
    are the responsibility of step_start processors, not this builder.
    """

    async def build(self, workspace: "Workspace | None" = None) -> str: ...
