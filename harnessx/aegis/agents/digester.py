# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Digester agent builder — Stage P.2 LLM per-task analysis agent.

Public API:
    DigesterInputs   — dataclass describing a single per-task digest job
    build_digester_harness(inputs) -> HarnessConfig
    _select_template(pattern) -> Path
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harnessx import HarnessConfig

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# Harness source root — blocked from Digester read scope to prevent
# prompt injection and unintended exfiltration of harness internals.
from harnessx.aegis._paths import HARNESSX_SRC_ROOT as _HARNESSX_SRC_ROOT

_PATTERN_TO_TEMPLATE = {
    "ALL_FAIL": "digester_all_fail.md",
    "ALL_PASS": "digester_all_pass.md",
    "PARTIAL_PASS": "digester_partial_pass.md",
    # aliases used in pattern_templates.md
    "FAIL": "digester_all_fail.md",
    "PASS": "digester_all_pass.md",
}


@dataclass
class DigesterInputs:
    task_id: str
    pattern: str  # one of ALL_FAIL, ALL_PASS, PARTIAL_PASS, PASS, FAIL
    trajectory_paths: list[Path]
    digest_out_path: Path
    # Layer A — pre-computed mechanical facts. Injected verbatim into the
    # system prompt; Digester is told to treat as ground truth and not rewrite.
    trace_facts_md: str = ""


def _select_template(pattern: str) -> Path:
    """Maps pattern to template file path."""
    key = pattern.upper()
    filename = _PATTERN_TO_TEMPLATE.get(key)
    if filename is None:
        raise ValueError(
            f"Unknown pattern {pattern!r}. "
            f"Valid patterns: {sorted(_PATTERN_TO_TEMPLATE)}"
        )
    return _TEMPLATES_DIR / filename


def build_digester_harness(inputs: DigesterInputs) -> "HarnessConfig":
    """Return a HarnessConfig for the Digester agent.

    Caller wraps with ``ModelConfig(main=...).agentic(cfg)`` to get a
    runnable Harness.  This function is pure — no file I/O is performed
    at call time.
    """
    from harnessx.core.builder import HarnessBuilder
    from harnessx.meta_harness.processors.read_scope_gate import ReadScopeGateProcessor
    from harnessx.meta_harness.processors.write_scope_gate import WriteScopeGateProcessor
    from harnessx.processors.context.system_prompt import SystemPromptProcessor
    from harnessx.aegis._prompt import StaticSystemPromptBuilder, render_template
    from harnessx.tools.builtin import bash_tool, grep_tool, read_tool, write_tool
    from harnessx.tools.inmemory import InMemoryToolRegistry

    template_path = _select_template(inputs.pattern)

    # Pre-render the Jinja2 template at build time. Using StaticSystemPromptBuilder
    # because TemplateSystemPromptBuilder.extra_context is silently dropped during
    # HarnessConfig serialization round-trip (only primitive-value dicts survive).
    # trajectory_paths: absolute paths (for the Read tool, read-scope allowlist).
    # trajectory_refs: relative-style `trajectories/<basename>` — for citation
    # anchors the structure gate IV-1 expects (absolute paths get rejected).
    trajectory_refs = [f"trajectories/{p.name}" for p in inputs.trajectory_paths]
    rendered = render_template(
        template_path,
        task_id=inputs.task_id,
        digest_out_path=str(inputs.digest_out_path),
        trajectory_paths=[str(p) for p in inputs.trajectory_paths],
        trajectory_refs=trajectory_refs,
        trace_facts_md=inputs.trace_facts_md or "",
    )
    system_builder = StaticSystemPromptBuilder(text=rendered)

    # Tool registry: Read (trajectories), Write (digest), + Bash/Grep for
    # ad-hoc trace analysis (e.g. `jq .exit_reason trajectories/*.jsonl` or
    # grepping for a specific step pattern across multiple rollouts). No
    # web tools — the Digester reasons from trace evidence only.
    tool_reg = InMemoryToolRegistry()
    for t in (read_tool, write_tool, bash_tool, grep_tool):
        tool_reg.register(t)

    # Write gate: agent may only write to the designated digest output file.
    write_gate = WriteScopeGateProcessor(
        allowed_files=(str(inputs.digest_out_path),),
    )

    # Read gate: block harnessx source (prevent prompt injection / source
    # exfil) + archived prior runs (cross-experiment leakage), matching the
    # Planner/Evolver/Critic gates. Explicitly allow the trajectory files
    # the Digester was handed.
    from harnessx.aegis._paths import archive_roots
    trajectory_allowed = tuple(str(p) for p in inputs.trajectory_paths)
    read_gate = ReadScopeGateProcessor(
        blocked_roots=(str(_HARNESSX_SRC_ROOT.resolve()), *archive_roots()),
        allowed_files=trajectory_allowed if trajectory_allowed else None,
    )

    cfg = (
        HarnessBuilder()
        .slot(tool_registry=tool_reg)
        .add(SystemPromptProcessor(system_builder))
        .add(write_gate)
        .add(read_gate)
        .build()
    )
    return cfg
