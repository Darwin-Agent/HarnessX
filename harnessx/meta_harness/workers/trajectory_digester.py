# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Pre-declared sub-agent workers for the reflect inner agent.

The reflect agent sees 50+ trajectory ``.md`` files per round and must also
author artifacts. Doing both in one agent thins its attention: cluster quality
and author quality both drop. Mirroring Claude Code's Task/subagent design,
this module exposes *one* tool — ``spawn_reflect_worker`` — whose ``kind``
argument selects a pre-declared worker type. Each worker type has:

- a fixed allow-list of tools (no artifact-writers ever),
- a fixed static system prompt (worker-specific, not inherited),
- a fixed step + cost budget.

Currently one kind is implemented: ``trajectory-digester`` (read a set of
``.md`` trajectory files and return a structured summary the reflect agent
can consume). ``artifact-auditor`` and ``processor-probe-dryrun`` are
reserved kinds but not yet available.

The tool deliberately does NOT expose free-form ``tools`` / ``system_prompt``
/ ``model`` overrides — those are what Claude Code classifies as "expert
mode" and would give the reflect agent enough rope to spawn a look-alike
artifact author with artifact-write permissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ...tools.base import Tool
from ...tools.inmemory import InMemoryToolRegistry

if TYPE_CHECKING:
    from ...core.harness import HarnessConfig
    from ...core.model_config import ModelConfig

SPAWN_REFLECT_WORKER_TOOL_NAME = "spawn_reflect_worker"


@dataclass
class WorkerSpec:
    kind: str
    allowed_tools: frozenset[str]
    system_prompt: str
    max_steps: int = 20
    max_cost_usd: float = 0.5
    description: str = ""


_TRAJECTORY_DIGESTER_PROMPT = """\
You are a trajectory digester worker. The reflect agent spawned you because a round has too many trajectory `.md` files to fit in one attention window. Your job is narrow and concrete:

**Input**: a list of trajectory `.md` paths (provided as a markdown list in the user message).

**Tools**: `Read`, `Glob`, `Grep`, `Bash`. You cannot write artifacts of any kind.

**Output (your `final_output`)**: a single markdown block with these sections, terse and evidence-quoting, total under ~1500 words:

1. `## By exit_reason` — for each distinct `exit_reason` (done / max_steps / error / budget_exceeded / loop_detected) seen, list the task_ids and a one-line symptom summary. Quote a short snippet (<=120 chars) from the body that shows the symptom.
2. `## By eval outcome` — split tasks on `eval_passed` (authoritative external evaluator). List passed task_ids first, then failed task_ids. For each failed task, give a *why* line sourced from the trajectory body and `judge_cause` (when present): the dataset's expected answer and the evaluator's textual reason are both intentionally withheld from the frontmatter, so the "why" must come from behaviour signals in the body, not from the evaluator. Do not bucket primarily by `judge_verdict` — it is opinion, `eval_passed` is the authoritative correctness signal.
3. `## Tool health` — for each tool that appears in `tool_error_counts` with error_rate >= 0.3 across the batch (compute rate from tool_call_counts / tool_error_counts), list the tool name, affected task_ids, and a short failure signature quote from one body.
4. `## Named capability gaps` — aggregate `judge_missing` strings across tasks with non-empty values. Group similar gaps using the agent's own phrasing (two strings naming the same underlying shape become one bullet). Each bullet lists affected task_ids.
5. `## Candidate artifact hypotheses` — short bullets, one per distinct cluster you found. Each hypothesis is a phrase pointing at a lever the reflect agent might take (e.g. "tool_web_search_unreliable — consider replacement"). Do NOT write artifact code; the reflect agent will decide.

Do not invent task_ids. Do not invent tool names. Every claim must be traceable to a trajectory you actually read; when in doubt, omit.

**Adaptive exit**: once you have classified distinct failure clusters and read two additional trajectories without adding a new cluster, stop reading and emit the summary. Staying past diminishing returns wastes budget that the reflect agent needs for authoring.

Return only the markdown block. No preamble, no farewell.
"""


_WORKER_SPECS: dict[str, WorkerSpec] = {
    "trajectory-digester": WorkerSpec(
        kind="trajectory-digester",
        allowed_tools=frozenset({"Read", "Glob", "Grep", "Bash"}),
        system_prompt=_TRAJECTORY_DIGESTER_PROMPT,
        max_steps=20,
        max_cost_usd=0.5,
        description="Digest a batch of trajectory .md files and return a structured symptom summary.",
    ),
}

# Reserved for later PRs — listed so the tool's description and error
# messages stay in sync once they land.
_RESERVED_KINDS = frozenset({"artifact-auditor", "processor-probe-dryrun"})


class _StaticSystemPromptBuilder:
    """Returns a fixed string; used for pre-declared worker system prompts."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def build(self, workspace: Any = None) -> str:
        return self._text


def _make_worker_child_config_fn(
    spec: WorkerSpec,
) -> Callable[["HarnessConfig", dict, int, int, Any, str], "HarnessConfig"]:
    """Build a child_config_fn for ``build_spawn_fn`` that applies ``spec``.

    Semantics:
    - Tool registry = parent tools ∩ spec.allowed_tools (spawn_subagent is
      intentionally **not** added — worker is a leaf).
    - System prompt = spec.system_prompt (static builder, not inherited).
    - Everything else inherits via :func:`_default_child_config`.
    """
    from ...processors.context.system_prompt import SystemPromptProcessor
    from ...tools.spawn_subagent import _default_child_config

    def _child_config_fn(
        parent_harness_config: "HarnessConfig",
        overrides: dict,
        child_depth: int,
        max_depth: int,
        runtime_tracer: Any = None,
        parent_run_id: str = "",
    ) -> "HarnessConfig":
        # Force the allowed-tools list; strip any upstream free-form override.
        effective_overrides = dict(overrides or {})
        effective_overrides["tools"] = []  # we build the registry manually
        effective_overrides["system_prompt"] = ""  # ditto for prompt

        child_cfg = _default_child_config(
            parent_harness_config,
            effective_overrides,
            child_depth,
            max_depth,
            runtime_tracer,
            parent_run_id,
        )

        # Rebuild the tool registry from parent, keeping only allowed names.
        new_reg = InMemoryToolRegistry()
        parent_reg: Any = parent_harness_config.tool_registry
        parent_tools = getattr(parent_reg, "_tools", {}) if parent_reg is not None else {}
        for name, tool_obj in parent_tools.items():
            if name in spec.allowed_tools:
                new_reg.register(tool_obj)
        child_cfg = child_cfg.copy(tool_registry=new_reg)

        # Overwrite SystemPromptProcessor's builder with a static one so the
        # worker can't see the reflect agent's guide — it uses only its
        # narrower prompt.
        static_builder = _StaticSystemPromptBuilder(spec.system_prompt)
        replaced = False
        new_rt: list = []
        for proc in getattr(child_cfg, "_rt_procs", []):
            if isinstance(proc, SystemPromptProcessor):
                new_rt.append(SystemPromptProcessor(static_builder))
                replaced = True
            else:
                new_rt.append(proc)
        if not replaced:
            new_rt.append(SystemPromptProcessor(static_builder))
        child_cfg = child_cfg.copy()
        child_cfg._rt_procs = new_rt

        return child_cfg

    return _child_config_fn


def make_spawn_reflect_worker_tool(
    *,
    inner_model: "ModelConfig",
    parent_harness_config: "HarnessConfig",
    max_depth: int = 2,
) -> Tool:
    """Return the ``spawn_reflect_worker`` tool bound to ``parent_harness_config``.

    The returned tool accepts three arguments: ``kind`` (enum, currently
    only ``"trajectory-digester"``), ``task`` (natural-language framing for
    the worker), and ``files`` (optional list of absolute paths that will
    be rendered into the worker's first user message as a markdown list).
    """
    from ...tools.spawn_subagent import build_spawn_fn

    kinds_desc = ", ".join(sorted(_WORKER_SPECS)) or "(none)"
    reserved_desc = (
        f" Reserved but not yet implemented: {', '.join(sorted(_RESERVED_KINDS))}." if _RESERVED_KINDS else ""
    )

    async def _dispatch(
        kind: str = "trajectory-digester",
        task: str = "",
        files: list[str] | None = None,
    ) -> str:
        files = list(files or [])
        spec = _WORKER_SPECS.get(kind)
        if spec is None:
            return f"unknown worker kind {kind!r}; available: {kinds_desc}." + reserved_desc
        if not task.strip():
            return "task description is empty; describe what the worker should do."

        child_cfg_fn = _make_worker_child_config_fn(spec)
        spawn_fn = build_spawn_fn(
            inner_model,
            parent_harness_config,
            child_config_fn=child_cfg_fn,
            max_depth=max_depth,
        )

        # Scale step budget with *actual input size*, not file count:
        # 50 × 1KB trajectories and 50 × 500KB trajectories need very
        # different step budgets. We measure total bytes of the passed
        # files and allocate ~1 step per 50KB of input plus orientation
        # overhead. File-count fallback preserves the old heuristic when
        # file sizes aren't measurable (e.g. remote paths). Cap at 80 so
        # a pathological call can't exhaust the parent budget; cost
        # budget is the second guard.
        from pathlib import Path as _Path

        total_bytes = 0
        for raw in files:
            try:
                fp = _Path(raw)
                if fp.is_file():
                    total_bytes += fp.stat().st_size
            except Exception:  # noqa: BLE001 — best-effort sizing
                continue
        if total_bytes > 0:
            steps_from_bytes = total_bytes // 50_000 + 8
            effective_max_steps = min(80, max(spec.max_steps, steps_from_bytes))
        else:
            effective_max_steps = min(60, max(spec.max_steps, 2 * len(files) + 6))

        rendered_task = _render_worker_user_message(task, files)
        return await spawn_fn(
            task=rendered_task,
            model="",
            system_prompt="",
            tools=[],
            max_steps=effective_max_steps,
            max_cost_usd=spec.max_cost_usd,
            wait=True,
            label=f"{spec.kind}",
        )

    tool = Tool(
        name=SPAWN_REFLECT_WORKER_TOOL_NAME,
        description=(
            "Spawn a pre-declared reflect worker. Workers are sync (block for "
            "result), leaf-only (cannot spawn further), and CANNOT write "
            "artifacts. Use when the current-round trajectory set exceeds "
            "what you can comfortably read yourself; the worker returns a "
            "structured summary you then act on. "
            f"Available kinds: {kinds_desc}." + reserved_desc
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": sorted(_WORKER_SPECS),
                    "description": "Which worker type to spawn.",
                },
                "task": {
                    "type": "string",
                    "description": (
                        "Short natural-language framing for the worker, e.g. "
                        "'digest these 20 trajectories, focus on tool errors'."
                    ),
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of absolute file paths the worker "
                        "should read. Rendered as a markdown bullet list "
                        "into the worker's first user message."
                    ),
                },
            },
            "required": ["kind", "task"],
        },
        fn=_dispatch,
        tags=[],
        execution_target="local",
    )
    # Mark the factory's importable path so the serializer skips the
    # "non-importable module" warning.  YAML round-trip won't reconstruct
    # a fully-bound tool (the closure captures runtime deps), but the
    # meta-harness config is always built programmatically anyway.
    tool.__hx_target__ = "harnessx.meta_harness.workers.trajectory_digester.make_spawn_reflect_worker_tool"
    return tool


def _render_worker_user_message(task: str, files: list[str]) -> str:
    parts = [task.strip()]
    if files:
        parts.append("")
        parts.append("Files:")
        for p in files:
            parts.append(f"- `{p}`")
    return "\n".join(parts)


__all__ = [
    "SPAWN_REFLECT_WORKER_TOOL_NAME",
    "WorkerSpec",
    "make_spawn_reflect_worker_tool",
]
