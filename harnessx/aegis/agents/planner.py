# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Planner agent builder — writes a single ``landscape.md`` synthesising
the round's evidence. It is no longer a brief dispatcher; the downstream
Evolver reads the landscape + raw digests + trajectories and decides
itself how many candidates to produce.

Public API:
    PlannerInputs    — dataclass describing a single planner invocation
    build_planner_harness(inputs) -> HarnessConfig
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harnessx import HarnessConfig

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_PLANNER_TEMPLATE = _TEMPLATES_DIR / "planner.md"

# Harness source root — blocked from Planner read scope to prevent
# prompt injection and unintended exfiltration of harness internals.
from harnessx.aegis._paths import HARNESSX_SRC_ROOT as _HARNESSX_SRC_ROOT


@dataclass
class PlannerInputs:
    round: int
    overview_path: Path           # per-round overview (was summary_path)
    journal_path: Path
    archive_dir: Path
    current_config_path: Path
    landscape_path: Path          # Planner's sole output
    digests_dir: Path             # Planner Reads individual digests
    reputation_summary: dict
    recent_window: int = 5
    max_cost_usd: float | None = None
    run_root: Path | None = None
    # Meta-agent trajectory dir. Each Planner session writes its event
    # stream + messages to ``sessions_dir/<run_id>.jsonl`` so post-hoc
    # inspection can show what the Planner actually read / thought.
    sessions_dir: Path | None = None


# ---------------------------------------------------------------------------
# Harness builder
# ---------------------------------------------------------------------------

def build_planner_harness(inputs: PlannerInputs) -> "HarnessConfig":
    """Return a HarnessConfig for the Planner agent.

    The Planner reads overview.md + per-task digests + cross-round ledgers
    and writes a single ``landscape.md`` that synthesises the round's
    evidence for the downstream Evolver. harnessx source is blocked from
    reads to prevent prompt injection / source exfiltration.

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

    round_minus_1 = max(0, inputs.round - 1)
    rendered = render_template(
        _PLANNER_TEMPLATE,
        round=inputs.round,
        round_minus_1=round_minus_1,
        recent_window=inputs.recent_window,
        reputation_summary=inputs.reputation_summary,
    )
    system_builder = StaticSystemPromptBuilder(text=rendered)

    tool_reg = InMemoryToolRegistry()
    # Parity with Evolver / Critic so the Planner can look up live
    # documentation (web_search / web_fetch) when cross-round ledger
    # signals point toward an unexplored direction that requires
    # verifying external facts (e.g. "does arXiv expose an OAI-PMH
    # feed for listings?" before naming it as an unattempted direction).
    for t in (
        read_tool, write_tool, glob_tool, grep_tool,
        bash_tool, web_search_tool, web_fetch_tool,
    ):
        tool_reg.register(t)

    # Planner writes only the single landscape.md file.
    write_gate = WriteScopeGateProcessor(
        allowed_files=(str(inputs.landscape_path),),
    )

    from harnessx.aegis._paths import archive_roots
    harnessx_src_root = str(_HARNESSX_SRC_ROOT.resolve())
    allowed_read_files = (
        str(inputs.overview_path.resolve()),
        str(inputs.journal_path.resolve()),
        str(inputs.current_config_path.resolve()),
    )
    read_gate = ReadScopeGateProcessor(
        blocked_roots=(harnessx_src_root, *archive_roots()),
        allowed_files=allowed_read_files,
        hint_message=(
            "harnessx/ source is gated for the Planner agent. Run root "
            "has INDEX.md cataloguing cross-round ledgers + prior rounds. "
            "Consult whatever supports your landscape synthesis."
        ),
    )

    # -- Assemble -------------------------------------------------------
    builder = (
        HarnessBuilder()
        .slot(tool_registry=tool_reg)
    )
    if inputs.sessions_dir is not None:
        builder = builder.slot(
            tracer=HarnessJournal(base_dir=str(inputs.sessions_dir), export_jsonl=True),
        )
    cfg = (
        builder
        .add(SystemPromptProcessor(system_builder))
        .add(write_gate)
        .add(read_gate)
        .add(LoopDetectionProcessor())
        # Cost guard. Per-agent hard cap of $100 — BaseTask's max_cost_usd is the
        # primary budget knob; this gate is just a runtime backstop.
        .add(CostGuardProcessor(max_usd=inputs.max_cost_usd or 100.0))
        .add(
            CompactionProcessor(
                # E5: Planner BaseTask budget is 500k; keep compaction well
                # below that so it fires at most ~2x per run.
                token_threshold=240000,
                retention_window=4,
                eviction_fraction=0.90,
                summarize_prompt_template=(
                    "You are compacting context for a Planner agent that reads\n"
                    "task analysis summaries and issues Evolver briefs.\n"
                    "Return concise Markdown with exactly these sections:\n"
                    "1) Decisions\n"
                    "2) Facts and Constraints\n"
                    "3) Errors and Unresolved Risks\n"
                    "4) Pending Actions\n\n"
                    "Preserve: bucket reputation, archive candidates referenced, "
                    "brief assignments made or planned, hard quota rules, open blockers.\n"
                    "Discard: repeated status lines and filler.\n\n"
                    "Conversation to summarize:\n{conversation}"
                ),
            )
        )
        .build()
    )
    return cfg
