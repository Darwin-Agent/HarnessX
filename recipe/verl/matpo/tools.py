# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""MATPO RL-aware subagent tools — pre-registered, synchronous, token-capturing."""

from __future__ import annotations

from harnessx.tools.base import Tool
from harnessx.tools.builtin import web_search_tool, bash_tool
from harnessx.tools.code_execution import make_code_execution_tool
from recipe.verl.research.prompt import (
    SEARCHER_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    SEARCHER_QUESTION_PROMPT,
    CODER_QUESTION_PROMPT,
)
from recipe.verl.research.config import cfg
from recipe.verl.matpo.spawn_rl import register_child_config, spawn_subagent_rl

# ─── Build code_interpreter for child registration ───────────────────────────

_GAIA_ALLOWED_MODULES = {
    "math",
    "statistics",
    "json",
    "re",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "csv",
    "decimal",
    "fractions",
    "operator",
    "string",
    "textwrap",
    "unicodedata",
    "hashlib",
    "base64",
    "struct",
    "calendar",
    "time",
    "random",
    "bisect",
    "heapq",
    "copy",
    "pprint",
    "difflib",
    "html",
    "urllib",
    "io",
    "typing",
    "dataclasses",
    "enum",
    "numbers",
    "cmath",
    "abc",
}

_code_interpreter_tool = make_code_execution_tool(
    name="code_interpreter",
    description="Execute Python code for calculations, data analysis, and text processing.",
    allowed_modules=_GAIA_ALLOWED_MODULES,
    timeout=cfg.code_timeout,
    max_output_chars=cfg.code_max_output_chars,
)

# ─── Pre-register child configs at import time ────────────────────────────────

register_child_config(
    name="searcher",
    tools=[web_search_tool],
    system_prompt=SEARCHER_SYSTEM_PROMPT,
    question_prompt=SEARCHER_QUESTION_PROMPT,
    max_steps=6,
)

register_child_config(
    name="coder",
    tools=[bash_tool, _code_interpreter_tool],
    system_prompt=CODER_SYSTEM_PROMPT,
    question_prompt=CODER_QUESTION_PROMPT,
    max_steps=6,
)


# ─── Tool wrappers (model-facing) ────────────────────────────────────────────


async def _spawn_searcher_rl(task: str) -> str:
    return await spawn_subagent_rl(task=task, agent_name="searcher")


async def _spawn_coder_rl(task: str) -> str:
    return await spawn_subagent_rl(task=task, agent_name="coder")


searcher_agent_rl_tool = Tool(
    name="searcher",
    description="Spawn a search sub-agent to search the web for relevant information.",
    input_schema={
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "The search task to delegate."},
        },
        "required": ["task"],
    },
    fn=_spawn_searcher_rl,
)

coder_agent_rl_tool = Tool(
    name="coder",
    description="Spawn a coding sub-agent to execute Python code or shell commands.",
    input_schema={
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "The coding/computation task to delegate."},
        },
        "required": ["task"],
    },
    fn=_spawn_coder_rl,
)
