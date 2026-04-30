# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Pre-declared worker subagents the meta-agent can spawn.

Exposed via a single tool ``spawn_reflect_worker`` whose ``kind`` argument
selects a pre-declared worker spec (tool allow-list, static system prompt,
budget). See :mod:`trajectory_digester` for the current implementation.
"""

from .trajectory_digester import (
    SPAWN_REFLECT_WORKER_TOOL_NAME,
    WorkerSpec,
    make_spawn_reflect_worker_tool,
)

__all__ = [
    "SPAWN_REFLECT_WORKER_TOOL_NAME",
    "WorkerSpec",
    "make_spawn_reflect_worker_tool",
]
