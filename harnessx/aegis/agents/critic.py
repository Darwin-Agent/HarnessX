# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Critic agent builder — Stage 3 Critic that ranks candidates independently.

Public API:
    CriticInputs             — dataclass describing a single critic invocation
    parse_decision(md) -> tuple[dict, str]
    make_ask_evolver_tool(candidates_dir, evolver_runner, max_turns_per_candidate) -> Tool
    build_critic_harness(inputs, evolver_runner, *, ablation_allow_briefs) -> HarnessConfig
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import yaml

if TYPE_CHECKING:
    from harnessx import HarnessConfig

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_CRITIC_TEMPLATE = _TEMPLATES_DIR / "critic.md"

# Harness source root — blocked from Critic read scope.
from harnessx.aegis._paths import HARNESSX_SRC_ROOT as _HARNESSX_SRC_ROOT

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass
class CriticInputs:
    round: int
    candidates_dir: Path
    verdicts_dir: Path
    decision_path: Path
    digests_dir: Path
    trajectories_dir: Path
    sessions_dir: Path   # rollout sessions dir the Critic may read
    journal_path: Path
    current_config_path: Path
    max_ask_more: int = 2
    # Meta-agent trajectory dir — each Critic session writes its event
    # stream + messages here for post-hoc inspection.
    meta_sessions_dir: Path | None = None


# ---------------------------------------------------------------------------
# Decision parser
# ---------------------------------------------------------------------------

def parse_decision(md: str) -> tuple[dict, str]:
    """Split a decision.md string into (yaml_dict, body_markdown).

    The decision must start with a YAML frontmatter block delimited by ``---``.
    Raises ``ValueError`` if the frontmatter is missing or cannot be parsed.
    """
    m = _FRONTMATTER_RE.match(md.strip() + "\n")
    if not m:
        raise ValueError(
            "Decision must begin with a YAML frontmatter block "
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
# ask_evolver tool factory
# ---------------------------------------------------------------------------

def make_ask_evolver_tool(
    candidates_dir: Path,
    evolver_runner: Callable,
    max_turns_per_candidate: int = 2,
):
    """Return a Tool that lets the Critic ask a fresh mini-Evolver a question.

    The returned Tool appends each Q/A exchange to the candidate's .md file
    so there is a persistent trace of the Critic's clarification requests.

    Parameters
    ----------
    candidates_dir:
        Directory containing C-*.md candidate files.
    evolver_runner:
        Async callable ``(candidate_id: str, question: str) -> str`` that
        runs a fresh mini-Evolver and returns its answer.
    max_turns_per_candidate:
        Hard cap on how many ask-more turns the Critic may use per candidate.
        Exceeding this raises ``RuntimeError``.
    """
    from harnessx.tools.base import tool

    # Closure state — per-candidate turn counter.
    _turn_counts: dict[str, int] = {}

    @tool(
        name="ask_evolver",
        description=(
            "Ask the Evolver a clarifying question about a specific candidate. "
            "The Evolver runs a fresh mini-session and returns a cited answer. "
            "Use when a candidate's evidence is ambiguous or incomplete. "
            f"Maximum {max_turns_per_candidate} turns per candidate."
        ),
    )
    async def ask_evolver(candidate_id: str, question: str) -> str:
        count = _turn_counts.get(candidate_id, 0)
        if count >= max_turns_per_candidate:
            raise RuntimeError(
                f"max_turns ({max_turns_per_candidate}) exhausted for {candidate_id}"
            )

        answer = await evolver_runner(candidate_id, question)

        # Increment only after successful call.
        _turn_counts[candidate_id] = count + 1
        turn_number = _turn_counts[candidate_id]

        # Append Q/A to the candidate file for an auditable trace.
        cand_file = candidates_dir / f"{candidate_id}.md"
        append_block = (
            f"\n\n## Ask-more Response (turn {turn_number})\n\n"
            f"**Q:** {question}\n\n"
            f"**A:** {answer}\n"
        )
        with cand_file.open("a", encoding="utf-8") as fh:
            fh.write(append_block)

        return answer

    return ask_evolver


# ---------------------------------------------------------------------------
# Harness builder
# ---------------------------------------------------------------------------

def build_critic_harness(
    inputs: CriticInputs,
    evolver_runner: Callable,
    *,
    ablation_allow_briefs: bool = False,
) -> "HarnessConfig":
    """Return a HarnessConfig for the Critic agent.

    The Critic reads all candidates, digests, trajectories, journal, and the
    current HarnessConfig.  It MUST NOT read briefs/ (independence guarantee).
    harnessx source is also blocked.

    The ``ablation_allow_briefs`` flag is reserved for T28 ablation runs.
    When True the briefs directory is removed from ``blocked_roots`` so the
    Critic can read briefs (poisoning its independence — for ablation study
    only).

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

    # -- ask_evolver tool ---------------------------------------------------
    ask_tool = make_ask_evolver_tool(
        candidates_dir=inputs.candidates_dir,
        evolver_runner=evolver_runner,
        max_turns_per_candidate=inputs.max_ask_more,
    )

    # -- Tool registry --------------------------------------------------
    # Filesystem + shell + web — parity with Evolver so the Critic can
    # independently verify proposed endpoints (web_fetch), cross-check live
    # source docs (web_search), and probe the target harness's environment
    # (bash). Without these the Critic's verdicts on tool candidates
    # collapse to trusting the Evolver's claims.
    tool_reg = InMemoryToolRegistry()
    for t in (
        read_tool, write_tool, glob_tool, grep_tool,
        bash_tool, web_search_tool, web_fetch_tool,
    ):
        tool_reg.register(t)
    tool_reg.register(ask_tool)

    # -- System prompt --------------------------------------------------
    rendered = render_template(
        _CRITIC_TEMPLATE,
        round=inputs.round,
        max_ask_more=inputs.max_ask_more,
    )
    system_builder = StaticSystemPromptBuilder(text=rendered)

    # -- Write scope gate -----------------------------------------------
    # Critic may ONLY write inside verdicts_dir and the decision.md file.
    write_gate = WriteScopeGateProcessor(
        allowed_roots=(str(inputs.verdicts_dir),),
        allowed_files=(str(inputs.decision_path),),
    )

    # -- Read scope gate ------------------------------------------------
    # Block harnessx source tree AND briefs dir (independence guarantee)
    # AND archived prior runs (cross-experiment leakage guarantee).
    from harnessx.aegis._paths import api_reference_files, archive_roots
    harnessx_src_root = str(_HARNESSX_SRC_ROOT.resolve())
    briefs_dir = inputs.candidates_dir.parent / "briefs"

    if ablation_allow_briefs:
        blocked_roots = (harnessx_src_root, *archive_roots())
    else:
        blocked_roots = (
            harnessx_src_root, str(briefs_dir.resolve()), *archive_roots(),
        )

    # Allowlist specific input files the Critic legitimately needs.
    allowed_read_files = (
        str(inputs.journal_path.resolve()),
        str(inputs.current_config_path.resolve()),
        # Living documentation — so the Critic can verify whether the Evolver's
        # processor/tool code uses real HarnessX APIs or is hallucinated.
        # Excludes aegis/ (prevents reading briefs via indirection).
        *api_reference_files(),
    )
    read_gate = ReadScopeGateProcessor(
        blocked_roots=blocked_roots,
        allowed_files=allowed_read_files,
        hint_message=(
            "harnessx/ source is gated except for living documentation (base "
            "classes + built-in processors + built-in tools). When verifying a "
            "candidate that ships new processor/tool code, Read the reference "
            "implementations (e.g. harnessx/processors/control/cost_guard.py, "
            "harnessx/tools/builtin/web_search.py) to confirm the candidate's "
            "hooks / decorators / imports are real API calls. briefs/ is "
            "blocked independently — do NOT try to read briefs; your job is "
            "to verify evidence, not command.\n\n"
            "Cross-round evidence: the run root has an INDEX.md cataloging "
            "data/ ledgers (task_history.jsonl, ship_outcomes.json, "
            "rejected_candidates.jsonl) + reputation.json at run root, plus "
            "prior rounds' artifacts including each round's decision.md. "
            "Use them both for per-candidate verification AND for the "
            "portfolio audit — e.g. check ship_outcomes.json to see if an "
            "earlier ship targeting the same cluster missed, reputation.json "
            "to see which buckets the Evolver has been avoiding, prior "
            "decision.md files to see what strategy_concerns were raised "
            "before and whether they were addressed."
        ),
    )

    # -- Assemble -------------------------------------------------------
    from harnessx.tracing.journal import HarnessJournal
    builder = (
        HarnessBuilder()
        .slot(tool_registry=tool_reg)
    )
    if inputs.meta_sessions_dir is not None:
        builder = builder.slot(
            tracer=HarnessJournal(base_dir=str(inputs.meta_sessions_dir), export_jsonl=True),
        )
    cfg = (
        builder
        .add(SystemPromptProcessor(system_builder))
        .add(write_gate)
        .add(read_gate)
        .add(LoopDetectionProcessor())
        # Per-agent hard cap of $100 — BaseTask's max_cost_usd is the primary
        # budget knob; this gate is just a runtime backstop.
        .add(CostGuardProcessor(max_usd=100.0))
        .add(
            CompactionProcessor(
                # E5: Critic BaseTask budget is 800k; keep compaction well
                # below that so it fires at most ~2x per run rather than 4-7x.
                token_threshold=400000,
                retention_window=4,
                eviction_fraction=0.90,
                summarize_prompt_template=(
                    "You are compacting context for a Critic agent that independently\n"
                    "verifies candidate harness patches against raw trajectory evidence.\n"
                    "Return concise Markdown with exactly these sections:\n"
                    "1) Decisions\n"
                    "2) Facts and Constraints\n"
                    "3) Errors and Unresolved Risks\n"
                    "4) Pending Actions\n\n"
                    "Preserve: candidate verdicts issued (accept/reject/ask-more), "
                    "evidence anchors verified, ask-more Q/A exchanges, ship_ranking "
                    "and rationale, refuted signatures from journal.\n"
                    "Discard: repeated status lines and filler.\n\n"
                    "Conversation to summarize:\n{conversation}"
                ),
            )
        )
        .build()
    )
    return cfg
