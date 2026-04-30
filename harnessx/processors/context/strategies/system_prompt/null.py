# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .....workspace.workspace import Workspace


class NullSystemPromptBuilder:
    """No system prompt. For benchmarks that manage their own prompts."""

    async def build(self, workspace: "Workspace | None" = None) -> str:
        return ""
