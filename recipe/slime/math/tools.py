# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio

from harnessx.tools.code_execution import make_code_execution_tool
from harnessx.tools.inmemory import InMemoryToolRegistry

# ---------------------------------------------------------------------------
# Math task configuration
# ---------------------------------------------------------------------------

TOOL_CONFIGS: dict = {
    "tool_concurrency": 32,  # align with OpenClaw-RL (32 concurrent processes)
    "python_timeout": 120,  # 2 minutes — eval uses longer context, needs more headroom
    "max_output_chars": 1024,  # align with OpenClaw-RL max_obs_chars truncation
}

# Exposed for any caller that needs the semaphore directly (e.g. preflight checks)
SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])

_MATH_ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "math",
        "random",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "operator",
        "statistics",
        "decimal",
        "fractions",
        "json",
        "re",
        "string",
        "textwrap",
        "heapq",
        "bisect",
        "sympy",
        "numpy",
    }
)

# ---------------------------------------------------------------------------
# Tool instance
# ---------------------------------------------------------------------------

code_interpreter_tool = make_code_execution_tool(
    name="code_interpreter",
    description=(
        "A tool for executing Python code in a safe sandbox environment. "
        "Supports math, statistics, itertools, collections, sympy, numpy."
    ),
    allowed_modules=_MATH_ALLOWED_MODULES,
    timeout=TOOL_CONFIGS["python_timeout"],
    max_output_chars=TOOL_CONFIGS["max_output_chars"],
    concurrency=TOOL_CONFIGS["tool_concurrency"],
)


def get_registry() -> InMemoryToolRegistry:
    """Return an InMemoryToolRegistry with code_interpreter registered."""
    registry = InMemoryToolRegistry()
    registry.register(code_interpreter_tool)
    return registry
