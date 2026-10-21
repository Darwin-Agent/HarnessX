# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Research tools for verl backend — assembled from HarnessX builtins."""
from __future__ import annotations

from harnessx.tools.builtin import (
    web_search_tool,
    # web_fetch_tool,
    # browser_tool,
    # bash_tool,
    # read_tool,
    bash_tool,
)
from harnessx.tools.code_execution import make_code_execution_tool
from recipe.verl.research.config import cfg

_GAIA_ALLOWED_MODULES = {
    "math", "statistics", "json", "re", "datetime", "collections",
    "itertools", "functools", "csv", "decimal", "fractions", "operator",
    "string", "textwrap", "unicodedata", "hashlib", "base64", "struct",
    "calendar", "time", "random", "bisect", "heapq", "copy", "pprint",
    "difflib", "html", "urllib", "io", "typing", "dataclasses", "enum",
    "numbers", "cmath", "abc",
}

code_interpreter_tool = make_code_execution_tool(
    name="code_interpreter",
    description="Execute Python code for calculations, data analysis, and text processing.",
    allowed_modules=_GAIA_ALLOWED_MODULES,
    timeout=cfg.code_timeout,
    max_output_chars=cfg.code_max_output_chars,
)


# ─── Tool registry: only subagent tools exposed to model ─────────────────────

if cfg.matpo_enabled:
    from recipe.verl.matpo.tools import searcher_agent_rl_tool, coder_agent_rl_tool

    RESEARCH_TOOLS = [
        # web_search_tool,      # not directly exposed — only via searcher subagent
        # bash_tool,            # not directly exposed — only via coder subagent
        # code_interpreter_tool,# not directly exposed — only via coder subagent
        searcher_agent_rl_tool,
        coder_agent_rl_tool,
    ]
else:
    RESEARCH_TOOLS = [
        web_search_tool,
        bash_tool,
        code_interpreter_tool,
    ]
