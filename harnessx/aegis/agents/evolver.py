# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Evolver agent builder — Stage 2 Evolver that writes a candidate manifest.

Public API:
    EvolverInputs           — dataclass describing a single evolver invocation
    parse_candidate_manifest(md) -> tuple[dict, str]
    build_evolver_harness(inputs) -> HarnessConfig
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from harnessx import HarnessConfig

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_EVOLVER_TEMPLATE = _TEMPLATES_DIR / "evolver.md"

# Harness source root — blocked from Evolver read scope.
from harnessx.aegis._paths import HARNESSX_SRC_ROOT as _HARNESSX_SRC_ROOT

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass
class EvolverInputs:
    """Inputs for the single Evolver session.

    The Evolver produces K candidates in one session (K chosen by the
    agent from evidence). Each candidate writes its manifest at
    ``candidates_dir / C-R{round}-NN.md`` and its applied YAML +
    assets under ``applied_root / C-R{round}-NN/``. The orchestrator
    enumerates whatever the Evolver actually wrote.

    ``ask_more_brief_path``: when supplied (ask-more subcall from the
    Critic), the Evolver runs in ask-more mode with no scratch dir.
    """

    round: int
    current_config_path: Path
    landscape_path: Path
    digests_dir: Path
    trajectories_dir: Path
    candidates_dir: Path
    applied_root: Path
    # Ask-more mode: Critic calls us with a specific candidate_id to
    # clarify. When set, we run in single-manifest (no-scratch) mode and
    # the resulting text is appended to the candidate file by the Critic.
    ask_more_brief_path: Path | None = None
    ask_more_candidate_id: str | None = None
    ask_more_candidate_path: Path | None = None
    # Meta-agent trajectory dir — each Evolver session writes its
    # event stream + messages here for post-hoc inspection.
    sessions_dir: Path | None = None
    # Benchmark context string passed to the evolver template so it can
    # tailor guidance (e.g. "tau2" prevents PlainMarkdownSystemPromptBuilder).
    benchmark_context: str = ""


# ---------------------------------------------------------------------------
# Manifest parser
# ---------------------------------------------------------------------------


def parse_candidate_manifest(md: str) -> tuple[dict, str]:
    """Split a candidate manifest string into (yaml_dict, body_markdown).

    The manifest must start with a YAML frontmatter block delimited by ``---``.
    Raises ``ValueError`` if the frontmatter is missing or cannot be parsed.
    """
    m = _FRONTMATTER_RE.match(md.strip() + "\n")
    if not m:
        raise ValueError(
            "Candidate manifest must begin with a YAML frontmatter block "
            "delimited by '---'. No valid frontmatter found."
        )
    frontmatter_str, body = m.group(1), m.group(2)
    try:
        frontmatter = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse frontmatter YAML: {exc}") from exc
    if not isinstance(frontmatter, dict):
        raise ValueError("Frontmatter YAML must be a mapping (dict).")
    return frontmatter, body


# ---------------------------------------------------------------------------
# Harness builder
# ---------------------------------------------------------------------------


def build_evolver_harness(inputs: EvolverInputs) -> "HarnessConfig":
    """Return a HarnessConfig for an Evolver agent.

    The Evolver reads its brief, the current HarnessConfig, and trajectory /
    digest files within the allowed roots, then writes exactly one candidate
    manifest file.  harnessx source is blocked from reads.

    Caller wraps with ``ModelConfig(main=...).agentic(cfg)`` to get a
    runnable Harness.  This function is pure — no file I/O at call time.
    """
    from harnessx.core.builder import HarnessBuilder
    from harnessx.meta_harness.processors.read_scope_gate import ReadScopeGateProcessor
    from harnessx.meta_harness.processors.write_scope_gate import WriteScopeGateProcessor
    from harnessx.processors.context.system_prompt import SystemPromptProcessor
    from harnessx.aegis._prompt import StaticSystemPromptBuilder, render_template
    from harnessx.processors.control.compaction import CompactionProcessor
    from harnessx.processors.control.cost_guard import CostGuardProcessor
    from harnessx.processors.control.loop_detection import LoopDetectionProcessor
    from harnessx.tools.builtin import (
        bash_tool,
        glob_tool,
        grep_tool,
        read_tool,
        web_fetch_tool,
        web_search_tool,
        write_tool,
    )
    from harnessx.tools.inmemory import InMemoryToolRegistry
    from harnessx.tracing.journal import HarnessJournal

    ask_more_mode = inputs.ask_more_brief_path is not None

    rendered = render_template(
        _EVOLVER_TEMPLATE,
        ask_more_mode=ask_more_mode,
        round=inputs.round,
        landscape_path=str(inputs.landscape_path),
        current_config_path=str(inputs.current_config_path),
        digests_dir=str(inputs.digests_dir),
        trajectories_dir=str(inputs.trajectories_dir),
        candidates_dir=str(inputs.candidates_dir),
        applied_root=str(inputs.applied_root),
        ask_more_brief_path=str(inputs.ask_more_brief_path) if inputs.ask_more_brief_path else "",
        ask_more_candidate_id=inputs.ask_more_candidate_id or "",
        ask_more_candidate_path=str(inputs.ask_more_candidate_path) if inputs.ask_more_candidate_path else "",
        benchmark_context=inputs.benchmark_context,
    )
    system_builder = StaticSystemPromptBuilder(text=rendered)

    tool_reg = InMemoryToolRegistry()
    # Filesystem + shell + web — benchmark-agnostic capabilities so the Evolver
    # can verify API endpoints before coding them (web_fetch / web_search) and
    # probe the target harness's workspace / runtime environment (bash).
    # Tools are surfaced to the LLM via native function schemas, NOT via any
    # system-prompt announcement — registering here is sufficient.
    for t in (
        read_tool,
        write_tool,
        glob_tool,
        grep_tool,
        bash_tool,
        web_search_tool,
        web_fetch_tool,
    ):
        tool_reg.register(t)

    # Write scope:
    # - Primary mode: Evolver writes K candidate manifests + K applied
    #   scratch dirs. Allow whole candidates_dir + whole applied_root.
    # - Ask-more mode: single scratch file (the candidate path the Critic
    #   pointed us at).
    if ask_more_mode:
        _allowed_files = [str(inputs.ask_more_candidate_path)] if inputs.ask_more_candidate_path else []
        _allowed_roots: list[str] = []
    else:
        _allowed_files = []
        _allowed_roots = [str(inputs.candidates_dir), str(inputs.applied_root)]
    write_gate = WriteScopeGateProcessor(
        allowed_roots=tuple(_allowed_roots),
        allowed_files=tuple(_allowed_files),
    )

    from harnessx.aegis._paths import api_reference_files, archive_roots

    harnessx_src_root = str(_HARNESSX_SRC_ROOT.resolve())
    allowed_read_files = (
        str(inputs.landscape_path.resolve()),
        str(inputs.current_config_path.resolve()),
        *([str(inputs.ask_more_brief_path.resolve())] if inputs.ask_more_brief_path else []),
        *api_reference_files(),
    )
    # Block harnessx source + archived-runs directory so the Evolver cannot
    # read prior experiments' data into this round's decision-making. Bash
    # commands referencing absolute paths under these roots are also rejected.
    read_gate = ReadScopeGateProcessor(
        blocked_roots=(harnessx_src_root, *archive_roots()),
        allowed_files=allowed_read_files,
        hint_message=(
            "harnessx/ source is gated except for the living documentation "
            "(base classes + built-in processors + built-in tools). Read "
            "real reference implementations before writing new processor / "
            "tool code — see `harnessx/processors/control/cost_guard.py`, "
            "`harnessx/core/processor.py`, `harnessx/core/events.py`, "
            "`harnessx/core/builder.py`.\n\n"
            "Run state lives outside harnessx/: INDEX.md at the run root "
            "catalogs the landscape synthesis, per-task digests, raw "
            "trajectories, cross-round ledgers, and prior rounds. Read "
            "whatever supports your candidate decisions."
        ),
    )

    # -- Assemble -------------------------------------------------------
    builder = HarnessBuilder().slot(tool_registry=tool_reg)
    if inputs.sessions_dir is not None:
        builder = builder.slot(
            tracer=HarnessJournal(base_dir=str(inputs.sessions_dir), export_jsonl=True),
        )
    cfg = (
        builder.add(SystemPromptProcessor(system_builder))
        .add(write_gate)
        .add(read_gate)
        .add(LoopDetectionProcessor())
        # Per-agent hard cap of $100 — BaseTask's max_cost_usd is the primary
        # budget knob; this gate is just a runtime backstop. Single Evolver
        # now produces K candidates in one session, so the cap must be
        # generous (previous single-candidate $10 gate was the bottleneck).
        .add(CostGuardProcessor(max_usd=100.0))
        .add(
            CompactionProcessor(
                # E5: Evolver BaseTask budget is 600k; keep compaction well
                # below that so it fires at most ~2x per run.
                token_threshold=300000,
                retention_window=4,
                eviction_fraction=0.90,
                summarize_prompt_template=(
                    "You are compacting context for an Evolver agent that reads\n"
                    "failure analysis briefs and produces candidate harness patches.\n"
                    "Return concise Markdown with exactly these sections:\n"
                    "1) Decisions\n"
                    "2) Facts and Constraints\n"
                    "3) Errors and Unresolved Risks\n"
                    "4) Pending Actions\n\n"
                    "Preserve: failure evidence cited, root cause analysis, targeted fix\n"
                    "description, predicted impact, file changes planned, open questions.\n"
                    "Discard: repeated status lines and filler.\n\n"
                    "Conversation to summarize:\n{conversation}"
                ),
            )
        )
        .build()
    )
    return cfg
