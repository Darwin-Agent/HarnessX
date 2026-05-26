"""
Parse HarnessResult into strongly-typed outputs.
Includes fallback path: when DigestAgent fails, build a conservative
DigestReport from pre-computed trajectory signals only.
"""
from __future__ import annotations
import json
import logging
import re

from pydantic import ValidationError
from harnessx.core.harness import HarnessResult

from ..signals.schema import TaskSignals
from ..signals.solvability import SolvabilityJournal
from .schema import DigestReport, PatternImprovability, SevereRegression

logger = logging.getLogger(__name__)

# gap_type normalization aliases (LLM may output variant spellings)
_GAP_TYPE_ALIASES: dict[str, str] = {
    "behavioral": "behavior",
    "behavioral_gap": "behavior",
    "knowledge_gap": "knowledge",
    "reasoning_gap": "reasoning",
    "capability": "model_gap",
    "model_capability": "model_gap",
    "capability_gap": "model_gap",
    "unclear": "unknown",
    "insufficient": "unknown",
}
_VALID_GAP_TYPES = frozenset(
    ["stability", "behavior", "knowledge", "reasoning", "model_gap", "unknown"]
)


def parse_digest_result(
    result: HarnessResult,
    signals: dict[str, TaskSignals],
    round_idx: int = 0,
) -> DigestReport:
    """
    Extract submit_digest_report parameters from the single DigestAgent HarnessResult.

    level_counts, harness_fixable_ratio, and has_search_targets are derived here
    from patterns — not trusted from LLM output.
    """
    tool_input = _extract_stop_tool_input(result, "submit_digest_report")
    if tool_input is None:
        logger.warning("DigestAgent: submit_digest_report not called, using fallback")
        return fallback_digest(signals, round_idx=round_idx)

    try:
        patterns_raw = tool_input.get("patterns") or {}
        if not isinstance(patterns_raw, dict):
            logger.warning("DigestAgent: patterns field is %s, expected dict — ignoring", type(patterns_raw).__name__)
            patterns_raw = {}
        # Fallback: model sometimes embeds patterns as XML <parameter name="patterns">...</parameter>
        # inside the rationale string instead of passing it as a separate key.
        if not patterns_raw:
            patterns_raw = _extract_patterns_from_rationale(tool_input.get("rationale", ""))
        patterns = {k: PatternImprovability(**v) for k, v in patterns_raw.items()}
    except (ValidationError, TypeError, AttributeError) as e:
        logger.warning("DigestAgent: parse error (%s), using fallback", e)
        return fallback_digest(signals, round_idx=round_idx)

    # Parse severe_regressions list (new field added to tool schema).
    severe_regressions: list[SevereRegression] = []
    raw_regressions = tool_input.get("severe_regressions") or []
    if isinstance(raw_regressions, list):
        for item in raw_regressions:
            if not isinstance(item, dict):
                continue
            try:
                severe_regressions.append(SevereRegression(**item))
            except (TypeError, ValidationError) as e:
                logger.warning("DigestAgent: skipping malformed severe_regression entry (%s): %s", e, item)

    needs_revert = bool(tool_input.get("needs_revert", False))

    report = DigestReport(
        round=0,  # backfilled by orchestrator
        pass_rate=sum(s.outcome.rollout_pass_rate for s in signals.values()) / max(len(signals), 1),
        total_tasks=len(signals),
        failed_tasks=sum(1 for s in signals.values() if not s.outcome.eval_passed),
        patterns=patterns,
        severe_regressions=severe_regressions,
        has_severe_regression=bool(severe_regressions),
        needs_revert=needs_revert,
        priority_pattern=_validated_priority_pattern(
            tool_input.get("priority_pattern"), patterns
        ),
        rationale=tool_input.get("rationale", ""),
    )

    _fill_derived_counts(report)
    _derive_routing_flags(report)
    return report


def fallback_digest(
    signals: dict[str, TaskSignals],
    round_idx: int,
) -> DigestReport:
    """
    Fallback when DigestAgent is entirely unavailable.
    Uses only pre-computed trajectory signals, no LLM calls.
    level1_fixable -> stability/Level1, unclear -> unknown/Level3.
    """
    level1_tasks = [
        tid for tid, s in signals.items()
        if s.outcome.mechanical_fixability == "level1_fixable" and not s.outcome.eval_passed
    ]
    unclear_tasks = [
        tid for tid, s in signals.items()
        if s.outcome.mechanical_fixability == "unclear" and not s.outcome.eval_passed
    ]
    failed_count = len(level1_tasks) + len(unclear_tasks)
    pass_rate = sum(s.outcome.rollout_pass_rate for s in signals.values()) / max(len(signals), 1)

    patterns: dict[str, PatternImprovability] = {}
    if level1_tasks:
        patterns["mechanical_level1"] = PatternImprovability(
            gap_type="stability",
            improvability_level=1,
            tasks=level1_tasks,
            count=len(level1_tasks),
            signal="Pre-computed mechanical signals (fallback mode)",
        )
    if unclear_tasks:
        patterns["unclear_level3"] = PatternImprovability(
            gap_type="unknown",
            improvability_level=3,
            tasks=unclear_tasks,
            count=len(unclear_tasks),
            signal="No LLM analysis available (fallback mode)",
        )

    report = DigestReport(
        round=round_idx,
        pass_rate=pass_rate,
        total_tasks=len(signals),
        failed_tasks=failed_count,
        patterns=patterns,
        rationale="Fallback mode: DigestAgent unavailable, using pre-computed signals only.",
    )
    _fill_derived_counts(report)
    _derive_routing_flags(report)
    return report


# ── internal helpers ──────────────────────────────────────────────────────────

def _extract_stop_tool_input(
    result: HarnessResult,
    tool_name: str,
) -> dict | None:
    """
    Extract the input parameters of a stop tool call.

    interrupt_on fires before the tool executes, so the call lands in
    result.interrupted_at. For robustness also scan trajectory steps
    in case the tool was executed normally.
    """
    # Primary: interrupt path — tool call in result.interrupted_at
    if result.is_interrupted and result.interrupted_at is not None:
        if result.interrupted_at.name == tool_name:
            return result.interrupted_at.input

    # Fallback: scan trajectory (tool may have been executed normally)
    for step in result.trajectory.steps:
        if step.action is None:
            continue
        for tc in step.action.tool_calls:
            if tc.name == tool_name:
                return tc.input
    return None


def _validated_priority_pattern(
    raw: str | None,
    patterns: dict,
) -> str | None:
    """Return priority_pattern only if it exists in patterns; otherwise pick highest-count L1/2."""
    if raw and raw in patterns:
        return raw
    if raw:
        logger.debug("parse: priority_pattern %r not in patterns, falling back to highest-count", raw)
    # fallback: highest-count pattern at the lowest improvability level
    best = min(
        (p for p in patterns.values()),
        key=lambda p: (p.improvability_level, -p.count),
        default=None,
    )
    return next((k for k, v in patterns.items() if v is best), None) if best else None


def _normalize_gap_type(raw: str) -> str:
    normalized = _GAP_TYPE_ALIASES.get(raw.lower().strip(), raw.lower().strip())
    return normalized if normalized in _VALID_GAP_TYPES else "unknown"


def _fill_derived_counts(report: DigestReport) -> None:
    """Derive level_counts and harness_fixable_ratio from patterns in-place."""
    counts: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    for p in report.patterns.values():
        lvl = p.improvability_level
        if lvl in counts:
            counts[lvl] += p.count
    report.level_counts = counts
    fixable = counts[1] + counts[2]
    report.harness_fixable_ratio = fixable / max(report.failed_tasks, 1)


def _extract_patterns_from_rationale(rationale: str) -> dict:
    """Extract patterns dict when model mistakenly embedded it inside rationale as XML.

    Handles the case where model outputs:
      rationale="...text...</rationale>\\n<parameter name=\\"patterns\\">{ json }</parameter>"
    """
    m = re.search(r'<parameter\s+name=["\']patterns["\']>\s*(\{.*)', rationale, re.DOTALL)
    if not m:
        return {}
    fragment = m.group(1)
    # Strip closing </parameter> tag if present
    fragment = re.sub(r'\s*</parameter>.*$', '', fragment, flags=re.DOTALL).strip()
    try:
        result, _ = json.JSONDecoder().raw_decode(fragment)
        if isinstance(result, dict):
            logger.warning("DigestAgent: recovered patterns from rationale XML embedding (%d keys)", len(result))
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


def _derive_routing_flags(report: DigestReport) -> None:
    """
    Mechanically derive has_search_targets from report data.

    has_search_targets: L1/2 fixable patterns exist → EvolveAgent searches existing
                        processors and decides internally whether to tune params
                        or implement a new processor.
    """
    report.has_search_targets = (
        report.level_counts.get(1, 0) + report.level_counts.get(2, 0)
    ) > 0
