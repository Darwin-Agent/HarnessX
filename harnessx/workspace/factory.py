# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from ..core.harness import HarnessConfig


def inherit_workspace(config: "HarnessConfig", agent_id: str) -> "HarnessConfig":
    """Return a copy of config with a fresh child Workspace and bound tool registry.

    If config has no workspace, returns config unchanged.
    """
    if config.workspace is None:
        return config
    from ..tools.spawn_subagent import _resolve_workspace

    parent_ws = _resolve_workspace(config.workspace)
    if parent_ws is None:
        return config
    child_ws = parent_ws.child(agent_id=agent_id)
    from ..tools.inmemory import InMemoryToolRegistry
    from ..tools.builtin import (
        bash_tool,
        read_tool,
        write_tool,
        edit_tool,
        glob_tool,
        grep_tool,
    )

    registry = InMemoryToolRegistry()
    for t in [bash_tool, read_tool, write_tool, edit_tool, glob_tool, grep_tool]:
        registry.register(t)
    return config.copy(
        workspace=child_ws,
        tool_registry=registry,
    )


def build_spawn_tool(
    parent_model_config: "Any" = None,
    parent_harness_config: "HarnessConfig | None" = None,
    child_config_fn: "Any | None" = None,
    max_depth: int = 3,
) -> "Any":
    """Return the module-level ``spawn_subagent`` tool.

    The tool reads parent model/harness configs from the RunLoop's ContextVar
    at call time, so no binding is required at construction.  All arguments are
    accepted for backward-compatibility but are ignored.
    """
    from ..tools.spawn_subagent import spawn_subagent_tool

    return spawn_subagent_tool
