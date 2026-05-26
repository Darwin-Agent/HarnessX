"""
Layer 1 signal extraction runner.

Entry point::

    from harnessx.experimental.harness_evol.signals.runner import extract_signals

    # Option A: glob pattern — auto-discover + auto-group by task_id
    extract_signals(
        source="/data/.benchmarks/*/runs/*/",
        output_dir=Path("out/"),
        score_fn=my_score_fn,
    )

    # Option B: pre-aggregated rollouts
    extract_signals(
        source=[
            {"task_name": "task-a", "rollout_dirs": [Path("runs/task-a/run1"), Path("runs/task-a/run2")]},
            {"task_name": "task-b", "rollout_dirs": [Path("runs/task-b/run1")]},
        ],
        output_dir=Path("out/"),
        score_fn=my_score_fn,
    )

Outputs written to *output_dir*:
  ``{task_id}.json``         — one per task, full Layer 1 signals
  ``all_tasks_summary.json`` — cross-task aggregates + failure-pattern clusters
  ``signals_report.md``      — compact human/LLM-readable Markdown report

ScoreFn interface
-----------------
Implement and pass a ``ScoreFn`` to override eval results from the JSONL trace::

    def my_score_fn(session_dir: Path) -> tuple[bool, float, dict | None]:
        # Return (eval_passed, eval_score, feedback_or_None)
        ...

``feedback`` is stored opaquely per rollout; Layer 2 (DigestAgent) reads it.

Session discovery (glob mode)
------------------------------
Each matched directory must contain at least one ``{run_id}_trace.jsonl``.
Multiple sessions with the same task_id are treated as separate rollouts (k > 1).

task_id resolution (priority order):
  1. Walk up parents; if any parent name contains ``__``, use the part before it.
     Covers benchmarks that use ``{task_name}__{trial_id}`` directory naming.
  2. Read first non-trace JSONL; if first line is ``session_start`` with ``task_id``, use it.
  3. Fall back to the session directory's parent's parent name.

CLI
---
::

    python -m harnessx.experimental.harness_evol.signals.runner \\
        "/data/.benchmarks/*/runs/*/" out/ \\
        --score-fn recipe.my_bench.score:my_score_fn
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Callable

from .extractor import TrajectorySignalExtractor
from .parser import parse_session_rollout
from .schema import RolloutData, TaskSignals
from .solvability import SolvabilityJournal

# Benchmark-specific scoring function.
# Takes a session_dir Path, returns (eval_passed, eval_score, feedback).
# feedback is a free-form dict (or None) containing benchmark-specific evaluation
# artifacts. Layer 1 stores feedback opaquely; Layer 2 (DigestAgent) reads it.
ScoreFn = Callable[[Path], tuple[bool, float, dict | None]]

logger = logging.getLogger(__name__)


# ── task_id resolution ────────────────────────────────────────────────────────

def _task_id_from_path(session_dir: Path) -> str | None:
    """Walk parent dirs; return prefix before '__' if found (e.g. task_name__trial_id)."""
    for parent in session_dir.parents:
        if "__" in parent.name:
            return parent.name.split("__")[0]
    return None


def _task_id_from_session_start(session_dir: Path) -> str | None:
    """Read first non-trace JSONL; extract task_id from session_start event."""
    for jsonl in sorted(session_dir.glob("*.jsonl")):
        if "_trace" in jsonl.name or "_state" in jsonl.name:
            continue
        try:
            with jsonl.open(encoding="utf-8") as f:
                line = f.readline().strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "session_start" and rec.get("task_id"):
                return str(rec["task_id"])
        except Exception:
            pass
        break
    return None


def _resolve_task_id(session_dir: Path) -> str:
    tid = _task_id_from_path(session_dir)
    if tid:
        return tid
    tid = _task_id_from_session_start(session_dir)
    if tid:
        return tid
    return session_dir.parent.parent.name


# ── discovery + grouping ──────────────────────────────────────────────────────

def _discover_task_rollouts(session_dir_pattern: str) -> dict[str, list[Path]]:
    matched = sorted(
        Path(p) for p in Path("/").glob(session_dir_pattern.lstrip("/"))
        if Path(p).is_dir()
    )
    if not matched:
        matched = sorted(p for p in Path(".").glob(session_dir_pattern) if p.is_dir())
    if not matched:
        logger.warning("No directories matched pattern: %s", session_dir_pattern)
        return {}

    groups: dict[str, list[Path]] = {}
    for session_dir in matched:
        task_id = _resolve_task_id(session_dir)
        groups.setdefault(task_id, []).append(session_dir)

    logger.info(
        "Discovered %d session(s) across %d task(s) from pattern: %s",
        len(matched), len(groups), session_dir_pattern,
    )
    return groups


# ── main entry point ──────────────────────────────────────────────────────────

def extract_signals(
    source: str | list[dict],
    output_dir: Path | str,
    *,
    score_fn: ScoreFn | None = None,
    solvability_journal: SolvabilityJournal | None = None,
    round_idx: int = 0,
    indent: int = 2,
) -> dict[str, TaskSignals]:
    """
    Extract Layer 1 signals and write outputs to *output_dir*.

    Parameters
    ----------
    source:
        Either a glob pattern string for auto-discovery, or a pre-aggregated list of dicts::

            [{"task_name": "task-a", "rollout_dirs": [Path("run1"), Path("run2")]}, ...]

    output_dir:
        Directory where outputs are written.  Created if it does not exist.
    score_fn:
        Optional benchmark-specific scoring function.
        Signature: ``(session_dir: Path) -> (eval_passed: bool, eval_score: float, feedback: dict | None)``
        Overrides eval_passed/eval_score/eval_feedback from the JSONL parser.
    solvability_journal:
        Pre-existing journal for cross-round historical signals.
        If None, a fresh journal is created internally (single-round use).
        Pass a loaded journal for multi-round evolution (orchestrator manages lifecycle).
    round_idx:
        Round index passed to ``solvability_journal.update()``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(source, str):
        groups = _discover_task_rollouts(source)
    else:
        groups = {
            d["task_name"]: [Path(p) for p in d["rollout_dirs"]]
            for d in source
        }
        logger.info("Using %d pre-aggregated task(s)", len(groups))

    if not groups:
        return {}

    task_runs: dict[str, list[RolloutData]] = {}
    for task_id, session_dirs in groups.items():
        rollouts = []
        for sd in session_dirs:
            r = parse_session_rollout(sd)
            if r is None:
                logger.warning("Failed to parse session: %s", sd)
                continue
            if score_fn is not None:
                try:
                    passed, score, feedback = score_fn(sd)
                    r = dataclasses.replace(r, eval_passed=passed, eval_score=score, eval_feedback=feedback)
                except Exception as exc:
                    logger.warning("score_fn failed for %s: %s", sd, exc)
            rollouts.append(r)
        task_runs[task_id] = rollouts
        logger.info("  %s: %d rollout(s) (%d session dirs)", task_id, len(rollouts), len(session_dirs))

    if solvability_journal is None:
        solvability_journal = SolvabilityJournal()
    solvability_journal.update(round_idx, task_runs)

    extractor = TrajectorySignalExtractor()
    signals = extractor.extract_batch(task_runs, solvability_journal)

    source_label = source if isinstance(source, str) else ""

    for task_id, sig in signals.items():
        out_file = output_dir / f"{task_id}.json"
        out_file.write_text(
            json.dumps(dataclasses.asdict(sig), default=str, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )
    logger.info("Wrote %d task JSON file(s) to %s", len(signals), output_dir)

    summary_path = output_dir / "all_tasks_summary.json"
    summary_path.write_text(
        json.dumps(_build_summary(signals, source_label), default=str, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote all_tasks_summary.json to %s", summary_path)

    report_path = output_dir / "signals_report.md"
    report_path.write_text(
        _build_signals_report(signals, source_label),
        encoding="utf-8",
    )
    logger.info("Wrote signals_report.md to %s", report_path)

    return signals


# ── cross-task summary ────────────────────────────────────────────────────────

def _level1_category(s: TaskSignals) -> str:
    """One of: all_pass | partial_current | partial_regression | never_solved"""
    if s.outcome.all_rollouts_passed:
        return "all_pass"
    if s.outcome.any_rollout_passed:
        return "partial_current"
    if s.history.ever_passed:
        return "partial_regression"
    return "never_solved"


def _build_summary(signals: dict[str, TaskSignals], source_pattern: str) -> dict:
    total = len(signals)
    cats = {tid: _level1_category(s) for tid, s in signals.items()}
    all_pass      = sum(1 for c in cats.values() if c == "all_pass")
    k_div         = sum(1 for c in cats.values() if c == "partial_current")
    regression    = sum(1 for c in cats.values() if c == "partial_regression")
    never_solved  = sum(1 for c in cats.values() if c == "never_solved")

    exit_counts: Counter = Counter(s.outcome.exit_reason for s in signals.values())
    fix_counts: Counter = Counter(s.outcome.mechanical_fixability for s in signals.values())

    clusters: dict[str, list[str]] = {}
    for task_id, s in sorted(signals.items()):
        for tag in s.outcome.failure_pattern_tags:
            clusters.setdefault(tag, []).append(task_id)

    all_err_tools: Counter = Counter()
    for s in signals.values():
        all_err_tools.update(s.rep_rollout.tool_error_histogram)

    tasks = []
    for task_id in sorted(signals):
        s = signals[task_id]
        pvf = s.pass_vs_fail
        rep = s.rep_rollout
        tasks.append({
            "task_id": task_id,
            "level1_category": cats[task_id],
            "pass_rate": s.outcome.rollout_pass_rate,
            "rollout_count": s.meta.rollout_count,
            "exit_reason": s.outcome.exit_reason,
            "failure_pattern_tags": s.outcome.failure_pattern_tags,
            "mechanical_fixability": s.outcome.mechanical_fixability,
            "rep_total_steps": rep.total_steps,
            "rep_total_tokens": rep.total_tokens,
            "rep_tool_error_rate": rep.tool_error_rate,
            "failing_steps_min": pvf.failing_steps_min,
            "failing_steps_max": pvf.failing_steps_max,
            "failing_steps_mean": pvf.failing_steps_mean,
            "failing_tokens_mean": pvf.failing_tokens_mean,
            "passing_tokens_mean": pvf.passing_tokens_mean,
            "is_partial_trace": s.meta.is_partial_trace,
        })

    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_pattern": source_pattern,
        "stats": {
            "total_tasks": total,
            "all_pass": all_pass,
            "partial_current": k_div,       # any_rollout_passed AND NOT all — task solvable, needs stabilisation
            "partial_regression": regression,  # ever_passed AND current all_fail — dangerous, was solvable
            "never_solved": never_solved,    # ever_passed=False AND all_fail — capability gap
        },
        "exit_reason_distribution": dict(exit_counts.most_common()),
        "fixability_distribution": dict(fix_counts.most_common()),
        "failure_pattern_clusters": {k: v for k, v in sorted(clusters.items())},
        "top_error_tools": dict(all_err_tools.most_common(10)),
        "tasks": tasks,
    }


# ── Markdown report ───────────────────────────────────────────────────────────

_CAT_ORDER = {"all_pass": 0, "partial_current": 1, "partial_regression": 2, "never_solved": 3}
_CAT_ICON  = {"all_pass": "🟢", "partial_current": "🟡", "partial_regression": "🔴", "never_solved": "⚫"}


def _build_signals_report(signals: dict[str, TaskSignals], source_pattern: str = "") -> str:
    from .schema import SLOW_TOOL_THRESHOLD_MS

    lines: list[str] = []

    total = len(signals)
    cats = {tid: _level1_category(s) for tid, s in signals.items()}
    n_all_pass   = sum(1 for c in cats.values() if c == "all_pass")
    n_k_div      = sum(1 for c in cats.values() if c == "partial_current")
    n_regression = sum(1 for c in cats.values() if c == "partial_regression")
    n_never      = sum(1 for c in cats.values() if c == "never_solved")
    n_unstable   = sum(1 for s in signals.values() if "unstable_pass" in s.outcome.failure_pattern_tags)

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("# Layer 1 Signals Report")
    lines.append("")
    if source_pattern:
        lines.append(f"**Source:** `{source_pattern}`")
        lines.append("")
    lines.append(
        f"**Tasks:** {total}"
        f"  |  🟢 **All-pass:** {n_all_pass}"
        f" ({n_unstable} unstable)"
        f"  |  🟡 **Partial (current):** {n_k_div}"
        f"  |  🔴 **Partial (regression):** {n_regression}"
        f"  |  ⚫ **Never solved:** {n_never}"
    )
    lines.append("")
    lines.append(
        "> **Level-1 taxonomy:** "
        "🟢 all-pass = stable/needs stabilisation  "
        "🟡 partial-current = k-divergence this round  "
        "🔴 partial-regression = was solvable, now all fail  "
        "⚫ never-solved = no evidence of solvability"
    )
    lines.append("")

    # ── Quick Reference table (sorted: cat order → pass_rate desc → task_id) ─
    sorted_tasks = sorted(
        signals.keys(),
        key=lambda tid: (_CAT_ORDER.get(cats[tid], 9), -signals[tid].outcome.rollout_pass_rate, tid),
    )

    lines.append("## Quick Reference")
    lines.append("")
    lines.append("| Cat | Task | k | Pass | Steps(rep) | Fail steps range | Tokens(rep) | Err% | Tags | Exit | Fix |")
    lines.append("|-----|------|---|------|------------|------------------|-------------|------|------|------|-----|")

    for task_id in sorted_tasks:
        s = signals[task_id]
        pvf = s.pass_vs_fail
        rep = s.rep_rollout

        pass_str = f"{pvf.passing_rollout_count}/{s.meta.rollout_count}"
        if pvf.failing_rollout_count > 0 and pvf.failing_steps_min != pvf.failing_steps_max:
            fail_range = f"{pvf.failing_steps_min}–{pvf.failing_steps_max}"
        elif pvf.failing_rollout_count > 0:
            fail_range = str(pvf.failing_steps_min)
        else:
            fail_range = "—"
        err_pct   = f"{rep.tool_error_rate * 100:.0f}%"
        tags_str  = " ".join(f"`{t}`" for t in s.outcome.failure_pattern_tags[:3])
        fix       = s.outcome.mechanical_fixability.replace("level1_fixable", "fixable").replace("unclear", "?")
        partial_note = "⚠" if s.meta.is_partial_trace else ""
        cat_icon  = _CAT_ICON.get(cats[task_id], "?")

        lines.append(
            f"| {cat_icon} | {task_id} | {s.meta.rollout_count} | {pass_str} "
            f"| {rep.total_steps}{partial_note} | {fail_range} "
            f"| {rep.total_tokens:,} | {err_pct} | {tags_str} | {s.outcome.exit_reason} | {fix} |"
        )

    lines.append("")

    # ── Inner helper: render one per-task H3 block ────────────────────────────
    def _render_task(task_id: str, s: TaskSignals) -> list[str]:
        tl: list[str] = []
        pvf = s.pass_vs_fail
        rep = s.rep_rollout

        tl.append(f"### {task_id}")
        tl.append("")

        if s.meta.task_description:
            tl.append(f"> {s.meta.task_description[:200].replace(chr(10), ' ')}")
            tl.append("")

        if s.meta.rep_rollout_path:
            tl.append(f"**Rep rollout:** `{s.meta.rep_rollout_path}`")
            tl.append("")

        tl.append(
            f"**Result:** {pvf.passing_rollout_count}/{s.meta.rollout_count} passed"
            f"  |  pass_rate={s.outcome.rollout_pass_rate:.0%}"
            f"  |  exit={s.outcome.exit_reason}"
        )
        if len(s.outcome.exit_reason_counts) > 1:
            dist = ", ".join(f"{k}×{v}" for k, v in sorted(s.outcome.exit_reason_counts.items()))
            tl.append(f"  exit distribution: {dist}")
        if s.outcome.failure_pattern_tags:
            tl.append(f"  tags: {', '.join(s.outcome.failure_pattern_tags)}")
        tl.append("")

        tl.append("**Rep rollout (single-rollout signals)**")
        tl.append("")
        wall_s = rep.total_wall_clock_ms / 1000 if rep.total_wall_clock_ms else None
        wall_str = f"  wall_clock={wall_s:.0f}s" if wall_s is not None else ""
        tl.append(
            f"- steps={rep.total_steps}  tokens={rep.total_tokens:,}  "
            f"input={rep.total_input_tokens:,}  output={rep.total_output_tokens:,}"
            f"  cost=${rep.total_cost_usd:.4f}{wall_str}"
        )
        if rep.avg_model_inference_ms:
            slow_inf = f"  ⚠ long_inference={rep.long_model_inference_count}" if rep.long_model_inference_count else ""
            tl.append(
                f"- model_inference: avg={rep.avg_model_inference_ms:.0f}ms"
                f"  max={rep.max_model_inference_ms:.0f}ms{slow_inf}"
            )
        if rep.task_end_error:
            tl.append(f"- ⚠ **task_end_error:** `{rep.task_end_error[:120]}`")
        if s.meta.is_partial_trace:
            tl.append("- ⚠ **partial trace** — signals reflect only the captured tail")
        tl.append("")

        # cross-rollout section
        if pvf.failing_rollout_count > 0 or pvf.passing_rollout_count > 0:
            all_pass_s   = s.outcome.all_rollouts_passed
            k_div_s      = s.outcome.any_rollout_passed and not s.outcome.all_rollouts_passed
            regression_s = not s.outcome.any_rollout_passed and s.history.ever_passed and pvf.failing_rollout_count > 0
            chronic_s    = not s.outcome.any_rollout_passed and not s.history.ever_passed and pvf.failing_rollout_count > 0

            if all_pass_s:
                tl.append(f"**Stability — all {s.meta.rollout_count} rollouts passed**")
                tl.append("")
                tl.append(
                    f"- steps {pvf.passing_steps_min}–{pvf.passing_steps_max}"
                    f"  mean={pvf.passing_steps_mean:.1f}  cv={pvf.passing_steps_cv:.2f}"
                    f"  |  tokens mean={pvf.passing_tokens_mean:,.0f}  cv={pvf.passing_tokens_cv:.2f}"
                    f"  |  err_rate={pvf.passing_error_rate_mean:.1%}"
                )
                if "unstable_pass" in s.outcome.failure_pattern_tags:
                    tl.append(
                        f"> ⚠ **UNSTABLE PASS** — steps_cv={pvf.passing_steps_cv:.2f} > 0.30"
                        f"  tokens_cv={pvf.passing_tokens_cv:.2f}"
                        f"  |  evolver target: reduce path variance while keeping pass_rate=100%"
                    )

            elif k_div_s:
                tl.append(
                    f"**k-divergence — {pvf.passing_rollout_count}/{s.meta.rollout_count} passed**"
                    f"  *(task is solvable; harness must stabilise)*"
                )
                tl.append("")
                tl.append(
                    f"- Failing  ({pvf.failing_rollout_count}): "
                    f"steps {pvf.failing_steps_min}–{pvf.failing_steps_max}"
                    f"  mean={pvf.failing_steps_mean:.1f}  cv={pvf.failing_steps_cv:.2f}"
                    f"  |  tokens mean={pvf.failing_tokens_mean:,.0f}"
                    f"  min={pvf.failing_tokens_min:,}  max={pvf.failing_tokens_max:,}  cv={pvf.failing_tokens_cv:.2f}"
                    f"  |  err_rate={pvf.failing_error_rate_mean:.1%}"
                )
                tl.append(
                    f"- Passing  ({pvf.passing_rollout_count}): "
                    f"steps {pvf.passing_steps_min}–{pvf.passing_steps_max}"
                    f"  mean={pvf.passing_steps_mean:.1f}  cv={pvf.passing_steps_cv:.2f}"
                    f"  |  tokens mean={pvf.passing_tokens_mean:,.0f}  cv={pvf.passing_tokens_cv:.2f}"
                    f"  |  err_rate={pvf.passing_error_rate_mean:.1%}"
                )

            elif regression_s:
                h = s.history
                danger = "dangerous_regression" in s.outcome.failure_pattern_tags
                stable_note = " (WAS STABLE)" if h.was_stable else ""
                if danger:
                    heading = (
                        f"**🔴 DANGEROUS REGRESSION{stable_note} —"
                        f" task previously all-passed, now all {pvf.failing_rollout_count} fail**"
                        f"  *(solvable — regression)*"
                    )
                else:
                    heading = (
                        f"**🔴 Historical regression —"
                        f" task previously passed (round {h.last_passed_round}),"
                        f" now all {pvf.failing_rollout_count} fail**"
                        f"  *(solvable — regression)*"
                    )
                tl.append(heading)
                tl.append("")
                hist_note = ""
                if h.current_vs_hist_token_delta is not None:
                    hist_note = (
                        f"  |  vs hist best: token_delta={h.current_vs_hist_token_delta:+,}"
                        f"  step_delta={h.current_vs_hist_step_delta:+.0f}"
                    )
                tl.append(
                    f"- Failing  ({pvf.failing_rollout_count}): "
                    f"steps {pvf.failing_steps_min}–{pvf.failing_steps_max}"
                    f"  mean={pvf.failing_steps_mean:.1f}  cv={pvf.failing_steps_cv:.2f}"
                    f"  |  tokens mean={pvf.failing_tokens_mean:,.0f}"
                    f"  min={pvf.failing_tokens_min:,}  max={pvf.failing_tokens_max:,}  cv={pvf.failing_tokens_cv:.2f}"
                    f"  |  err_rate={pvf.failing_error_rate_mean:.1%}"
                    + hist_note
                )

            elif chronic_s:
                tl.append(
                    f"**⚫ Never solved — all {pvf.failing_rollout_count} rollouts failed**"
                    f"  *(no evidence of solvability across any round)*"
                )
                tl.append("")
                tl.append(
                    f"- Failing  ({pvf.failing_rollout_count}): "
                    f"steps {pvf.failing_steps_min}–{pvf.failing_steps_max}"
                    f"  mean={pvf.failing_steps_mean:.1f}  cv={pvf.failing_steps_cv:.2f}"
                    f"  |  tokens mean={pvf.failing_tokens_mean:,.0f}"
                    f"  min={pvf.failing_tokens_min:,}  max={pvf.failing_tokens_max:,}  cv={pvf.failing_tokens_cv:.2f}"
                    f"  |  err_rate={pvf.failing_error_rate_mean:.1%}"
                )

            tl.append("")

            # Tool diff (non-all-pass scenarios)
            if not all_pass_s and (pvf.failing_tool_histogram or pvf.passing_tool_histogram):
                all_tools = set(pvf.failing_tool_histogram) | set(pvf.passing_tool_histogram)
                diffs = [
                    (t, pvf.failing_tool_histogram.get(t, 0), pvf.passing_tool_histogram.get(t, 0))
                    for t in all_tools
                    if abs(pvf.failing_tool_histogram.get(t, 0) - pvf.passing_tool_histogram.get(t, 0)) > 0
                ]
                diffs.sort(key=lambda x: -abs(x[1] - x[2]))
                if diffs:
                    tl.append("- Tool diff (fail vs pass): " +
                               "  ".join(f"{t}={f}↔{p}" for t, f, p in diffs[:6]))
                tl.append("")

        # Test failure summary (cross-rollout, from CTRF feedback)
        if pvf.test_failure_summary:
            tl.append("**Test failures (cross-rollout):**")
            tl.append("")
            tl.append("| Test | Pass | Fail | Rollouts | Sample trace (300 chars) |")
            tl.append("|------|------|------|----------|--------------------------|")
            for tf in pvf.test_failure_summary[:15]:
                trace = (tf.sample_trace or "")[:100].replace("|", "\\|").replace("\n", " ")
                tname = tf.test_name[:60].replace("|", "¦")
                tl.append(
                    f"| {tname} | {tf.passed_count} | {tf.failed_count}"
                    f" | {tf.rollouts_tested} | {trace} |"
                )
            if len(pvf.test_failure_summary) > 15:
                tl.append(f"  *(+{len(pvf.test_failure_summary) - 15} more tests)*")
            tl.append("")

        if rep.tool_error_rate > 0 or rep.failed_tool_calls:
            tl.append(
                f"**Tool errors (rep):** rate={rep.tool_error_rate:.1%}  "
                f"unrecovered={sum(1 for f in rep.failed_tool_calls if not f.recovered)}"
            )
            if rep.error_category_counts:
                err_cats_str = ", ".join(f"{k}={v}" for k, v in sorted(rep.error_category_counts.items()))
                tl.append(f"  categories: {err_cats_str}")
            if rep.failed_tool_calls:
                tl.append("")
                tl.append("| step | tool | rec | error (120 chars) | rollout |")
                tl.append("|------|------|-----|-------------------|---------|")
                for fc in rep.failed_tool_calls[:10]:
                    err = fc.error_summary[:120].replace("|", "\\|").replace("\n", " ")
                    rec_mark = "✓" if fc.recovered else "✗"
                    rp = fc.rollout_path.split("/")[-3] if fc.rollout_path else ""
                    tl.append(f"| {fc.step_id} | {fc.tool_name} | {rec_mark} | {err} | {rp} |")
                if len(rep.failed_tool_calls) > 10:
                    tl.append(f"  *(+{len(rep.failed_tool_calls) - 10} more)*")
            tl.append("")

        if rep.slow_tool_calls:
            tl.append(f"**Slow tool calls (rep, >{SLOW_TOOL_THRESHOLD_MS // 1000}s):** {len(rep.slow_tool_calls)}")
            tl.append("")
            tl.append("| step | tool | input | duration_ms | err_after | rollout |")
            tl.append("|------|------|-------|-------------|-----------|---------|")
            for sc in rep.slow_tool_calls[:5]:
                err_after = "✗" if sc.followed_by_error else ""
                rp = sc.rollout_path.split("/")[-3] if sc.rollout_path else ""
                inp = sc.tool_input_summary[:80].replace("|", "\\|") if sc.tool_input_summary else ""
                tl.append(f"| {sc.step_id} | {sc.tool_name} | {inp} | {sc.duration_ms:,} | {err_after} | {rp} |")
            if len(rep.slow_tool_calls) > 5:
                tl.append(f"  *(+{len(rep.slow_tool_calls) - 5} more)*")
            tl.append("")

        if rep.repeated_sequences:
            tl.append(f"**Repeated sequences (rep):** {len(rep.repeated_sequences)}")
            for rs in rep.repeated_sequences[:3]:
                tl.append(f"  - `{rs.tool_name}` ×{rs.count} starting step {rs.first_step}")
            tl.append("")

        if rep.compaction_events:
            tl.append(
                f"**Compaction (rep):** {len(rep.compaction_events)} event(s)  "
                f"steps_after_last={rep.steps_after_last_compaction}  "
                f"pre_err={rep.pre_compaction_error_rate:.1%}  "
                f"post_err={rep.post_compaction_error_rate:.1%}"
            )
            tl.append("")

        if rep.tool_call_histogram:
            top = sorted(rep.tool_call_histogram.items(), key=lambda x: -x[1])[:8]
            tl.append("**Tools (rep):** " + "  ".join(f"{t}={c}" for t, c in top))
            tl.append("")

        h = s.history
        if h.ever_passed or h.ever_all_passed or h.rate_regressed:
            hist_note = (
                f"ever_passed={h.ever_passed}  ever_all_passed={h.ever_all_passed}  "
                f"last_round={h.last_passed_round}  stable={h.was_stable}"
                f"  rounds_without_flip={h.rounds_without_flip}"
            )
            if h.hist_best_pass_rate is not None:
                cur_rate = s.outcome.rollout_pass_rate
                rate_note = f"  hist_best_pass_rate={h.hist_best_pass_rate:.0%}  current={cur_rate:.0%}"
                if h.rate_regressed:
                    rate_note += f"  ⚠ RATE_REGRESSED (drop={h.hist_best_pass_rate - cur_rate:.0%})"
                hist_note += rate_note
            tl.append(f"**History:** {hist_note}")
            if h.current_vs_hist_token_delta is not None:
                tl.append(
                    f"  hist_best: tokens={h.hist_best_passing_tokens:,}"
                    f"  steps={h.hist_best_passing_steps}"
                    f"  |  current delta: token={h.current_vs_hist_token_delta:+,}"
                    f"  step={h.current_vs_hist_step_delta:+.0f}"
                )
            tl.append("")

        tl.append(
            f"**Fixability:** `{s.outcome.mechanical_fixability}` — {s.outcome.mechanical_fixability_signal}"
        )
        tl.append("")
        tl.append("---")
        tl.append("")
        return tl

    # ── 🟢 All-pass section ────────────────────────────────────────────────────
    ap_tasks = [tid for tid in sorted_tasks if cats[tid] == "all_pass"]
    lines.append(f"## 🟢 All-pass — {len(ap_tasks)} tasks")
    lines.append("")
    if ap_tasks:
        unstable = [tid for tid in ap_tasks if "unstable_pass" in signals[tid].outcome.failure_pattern_tags]
        stable_count = len(ap_tasks) - len(unstable)
        lines.append(
            f"- Stable (steps_cv ≤ 0.30): **{stable_count}**  "
            f"  Unstable (steps_cv > 0.30): **{len(unstable)}**"
        )
        if unstable:
            lines.append("")
            lines.append("**Unstable passes** (evolver target — reduce path variance while keeping pass_rate=100%):")
            for tid in unstable:
                s = signals[tid]
                lines.append(
                    f"  - `{tid}`: steps_cv={s.pass_vs_fail.passing_steps_cv:.2f}"
                    f"  tokens_cv={s.pass_vs_fail.passing_tokens_cv:.2f}"
                )
        ap_tags: Counter = Counter()
        for tid in ap_tasks:
            for tag in signals[tid].outcome.failure_pattern_tags:
                if tag != "all_pass":
                    ap_tags[tag] += 1
        if ap_tags:
            lines.append("")
            lines.append("**Behavior tags:** " + "  ".join(f"`{t}`×{c}" for t, c in ap_tags.most_common(8)))
        lines.append("")
        # Per-task H3: unstable first (steps_cv desc), then stable
        ap_sorted = sorted(ap_tasks, key=lambda tid: -signals[tid].pass_vs_fail.passing_steps_cv)
        for tid in ap_sorted:
            lines.extend(_render_task(tid, signals[tid]))
    else:
        lines.append("*(no tasks in this category)*")
        lines.append("")

    # ── 🟡 Partial (current k-divergence) section ─────────────────────────────
    kd_tasks = [tid for tid in sorted_tasks if cats[tid] == "partial_current"]
    lines.append(f"## 🟡 Partial (current k-divergence) — {len(kd_tasks)} tasks")
    lines.append("")
    if kd_tasks:
        # Highlight rate_regression tasks (pass rate dropped from historical best).
        rate_reg_tasks = [
            tid for tid in kd_tasks
            if "rate_regression" in signals[tid].outcome.failure_pattern_tags
        ]
        if rate_reg_tasks:
            lines.append(
                f"⚠ **Rate regression** ({len(rate_reg_tasks)} tasks — pass rate dropped below historical best):"
            )
            for tid in rate_reg_tasks:
                h = signals[tid].history
                cur = signals[tid].outcome.rollout_pass_rate
                best = h.hist_best_pass_rate
                lines.append(
                    f"  - `{tid}`: current={cur:.0%}  hist_best={best:.0%}"
                    f"  drop={best - cur:.0%}"
                    f"  last_passed_round={h.last_passed_round}"
                )
            lines.append(
                "\n  → These tasks were previously performing better. "
                "Investigate whether a recent harness change caused the regression "
                "and consider reverting the responsible change."
            )
            lines.append("")
        # L2: priority table by pass_rate asc (lowest first = highest need)
        kd_by_priority = sorted(kd_tasks, key=lambda tid: signals[tid].outcome.rollout_pass_rate)
        lines.append("**Priority** (lowest pass rate = highest stabilisation need):")
        lines.append("")
        lines.append("| Task | pass_rate | hist_best | rate_regressed | k | err_rate(fail) | err_rate(pass) | step_diff |")
        lines.append("|------|-----------|-----------|----------------|---|----------------|----------------|-----------|")
        for tid in kd_by_priority:
            s = signals[tid]
            pvf = s.pass_vs_fail
            step_diff = pvf.failing_steps_mean - pvf.passing_steps_mean
            hist_best_str = f"{s.history.hist_best_pass_rate:.0%}" if s.history.hist_best_pass_rate is not None else "—"
            reg_flag = "⚠ YES" if "rate_regression" in s.outcome.failure_pattern_tags else "no"
            lines.append(
                f"| {tid} | {s.outcome.rollout_pass_rate:.0%} | {hist_best_str} | {reg_flag}"
                f" | {s.meta.rollout_count}"
                f" | {pvf.failing_error_rate_mean:.1%} | {pvf.passing_error_rate_mean:.1%}"
                f" | {step_diff:+.0f} |"
            )
        lines.append("")
        kd_tags: Counter = Counter()
        for tid in kd_tasks:
            for tag in signals[tid].outcome.failure_pattern_tags:
                if tag != "k_divergence":
                    kd_tags[tag] += 1
        if kd_tags:
            lines.append("**Behavior tags:** " + "  ".join(f"`{t}`×{c}" for t, c in kd_tags.most_common(8)))
            lines.append("")
        for tid in kd_by_priority:
            lines.extend(_render_task(tid, signals[tid]))
    else:
        lines.append("*(no tasks in this category)*")
        lines.append("")

    # ── 🔴 Partial (regression) section ───────────────────────────────────────
    reg_tasks = [tid for tid in sorted_tasks if cats[tid] == "partial_regression"]
    lines.append(f"## 🔴 Partial (regression) — {len(reg_tasks)} tasks")
    lines.append("")
    if reg_tasks:
        dangerous  = [tid for tid in reg_tasks if "dangerous_regression" in signals[tid].outcome.failure_pattern_tags]
        historical = [tid for tid in reg_tasks if tid not in dangerous]
        lines.append(
            f"- **Dangerous regression** (was stable → all fail): **{len(dangerous)}**"
            f"   **Historical regression** (ever passed → all fail): **{len(historical)}**"
        )
        if dangerous:
            lines.append("")
            lines.append("**Dangerous regressions** (highest evolver priority — re-stabilise immediately):")
            for tid in dangerous:
                h = signals[tid].history
                delta_note = (
                    f"  token_delta={h.current_vs_hist_token_delta:+,}"
                    if h.current_vs_hist_token_delta is not None else ""
                )
                lines.append(
                    f"  - `{tid}`: last_passed_round={h.last_passed_round}"
                    f"  hist_best_tokens={h.hist_best_passing_tokens}" + delta_note
                )
        lines.append("")
        reg_sorted = sorted(reg_tasks, key=lambda tid: (
            0 if "dangerous_regression" in signals[tid].outcome.failure_pattern_tags else 1,
            -signals[tid].history.rounds_without_flip,
            tid,
        ))
        for tid in reg_sorted:
            lines.extend(_render_task(tid, signals[tid]))
    else:
        lines.append("*(no tasks in this category)*")
        lines.append("")

    # ── ⚫ Never solved section ────────────────────────────────────────────────
    ns_tasks = [tid for tid in sorted_tasks if cats[tid] == "never_solved"]
    lines.append(f"## ⚫ Never solved — {len(ns_tasks)} tasks")
    lines.append("")
    if ns_tasks:
        ns_exit: Counter = Counter(signals[tid].outcome.exit_reason for tid in ns_tasks)
        lines.append(
            "**Exit reason distribution:** "
            + "  ".join(f"`{k}`×{v}" for k, v in ns_exit.most_common())
        )
        lines.append("")
        ns_tags: Counter = Counter()
        for tid in ns_tasks:
            for tag in signals[tid].outcome.failure_pattern_tags:
                if tag not in ("k_divergence", "all_pass"):
                    ns_tags[tag] += 1
        if ns_tags:
            lines.append("**Behavior tags:** " + "  ".join(f"`{t}`×{c}" for t, c in ns_tags.most_common(8)))
            lines.append("")
        ns_sorted = sorted(ns_tasks, key=lambda tid: (
            0 if "chronic_failure" in signals[tid].outcome.failure_pattern_tags else 1,
            -signals[tid].pass_vs_fail.failing_tokens_mean,
            tid,
        ))
        for tid in ns_sorted:
            lines.extend(_render_task(tid, signals[tid]))
    else:
        lines.append("*(no tasks in this category)*")
        lines.append("")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(default_score_fn: str | None = None) -> None:
    import argparse
    import importlib
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Extract Layer 1 signals from session directories.",
    )
    parser.add_argument(
        "pattern",
        help='Glob pattern for session dirs, e.g. "/data/.benchmarks/*/runs/*/"',
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write outputs (per-task JSON + summary + report)",
    )
    parser.add_argument(
        "--round", type=int, default=0, metavar="N",
        help="Round index (used if --journal is given)",
    )
    parser.add_argument(
        "--journal", type=Path, default=None, metavar="PATH",
        help="Path to solvability_journal.json for cross-round historical signals",
    )
    score_fn_default_help = (
        f"Benchmark-specific scoring function as 'module:func' "
        f"(default: {default_score_fn}). "
        "The function must accept a Path (session_dir) and return (bool, float, dict|None)."
    ) if default_score_fn else (
        "Benchmark-specific scoring function as 'module:func', e.g. "
        "'recipe.my_bench.score:my_score_fn'. "
        "The function must accept a Path (session_dir) and return (bool, float, dict|None)."
    )
    parser.add_argument("--score-fn", default=default_score_fn, metavar="MODULE:FUNC", help=score_fn_default_help)
    args = parser.parse_args()

    score_fn = None
    if args.score_fn:
        module_name, func_name = args.score_fn.rsplit(":", 1)
        score_fn = getattr(importlib.import_module(module_name), func_name)

    journal = SolvabilityJournal.load(args.journal) if args.journal and Path(args.journal).exists() else None
    results = extract_signals(
        args.pattern,
        Path(args.output_dir),
        score_fn=score_fn,
        solvability_journal=journal,
        round_idx=args.round,
    )

    if not results:
        sys.exit(1)

    all_pass = sum(1 for s in results.values() if s.outcome.all_rollouts_passed)
    partial = sum(1 for s in results.values() if s.outcome.any_rollout_passed and not s.outcome.all_rollouts_passed)
    print(f"\nTasks: {len(results)}  All-pass: {all_pass}  Partial: {partial}  All-fail: {len(results) - all_pass - partial}\n")
    print(f"{'Task':<40} {'k':>2} {'Pass':>5} {'Steps':>6} {'Tokens':>8} {'Err%':>5} {'Fix':<14} Tags")
    print("-" * 110)
    for task_id in sorted(results):
        s = results[task_id]
        pvf = s.pass_vs_fail
        rep = s.rep_rollout
        pass_str = f"{pvf.passing_rollout_count}/{s.meta.rollout_count}"
        err_pct = f"{rep.tool_error_rate * 100:.0f}%"
        fix = s.outcome.mechanical_fixability.replace("level1_fixable", "fixable")
        tags = ",".join(s.outcome.failure_pattern_tags[:3])
        print(
            f"{task_id:<40} {s.meta.rollout_count:>2} {pass_str:>5} "
            f"{rep.total_steps:>6} {rep.total_tokens:>8,} {err_pct:>5} {fix:<14} {tags}"
        )


if __name__ == "__main__":
    main()
