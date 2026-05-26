"""
Task builder for DigestAgent.

Single agent, single task.  The agent:
  1. Reads pre-computed trajectory signals from the task description.
  2. Explores trajectory files freely (Read/Grep/Glob) for evidence.
  3. Writes gap classifications and running notes to the evolution notebook.
  4. When analysis is complete, calls submit_digest_report once to terminate.

max_steps=200 covers full trajectory exploration + synthesis in one coherent
reasoning chain.  The notebook is the only writable file (WriteScopeGateProcessor).
"""
from __future__ import annotations

from pathlib import Path

from harnessx.core.harness import BaseTask

from ..signals.schema import TaskSignals
from ..signals.solvability import SolvabilityJournal
from ..signals.runner import _level1_category


def build_digest_task(
    signals: dict[str, TaskSignals],
    solvability_journal: SolvabilityJournal,
    round_idx: int,
    trajectories_dir: Path,
    *,
    max_steps: int = 200,
) -> BaseTask:
    """
    Build the single DigestAgent task for one round.

    The agent explores trajectories and synthesises the routing report in one
    continuous reasoning chain, writing intermediate notes to the notebook.

    Parameters
    ----------
    max_steps:
        Step budget for the agent.  Default 200.  Override from the evolution
        config to tune exploration depth vs. cost.
    """
    failed = {tid: s for tid, s in signals.items() if not s.outcome.eval_passed}
    if not failed:
        return BaseTask(
            description="No failed tasks to analyse. Call submit_digest_report with empty patterns={}, empty priority_pattern='', empty rationale=''.",
            interrupt_on=["submit_digest_report"],
            max_steps=2,
        )

    description = _render_digest_data(
        signals, solvability_journal, round_idx, trajectories_dir,
    )
    return BaseTask(
        description=description,
        interrupt_on=["submit_digest_report"],
        max_steps=max_steps,
    )


# ── task description renderer ─────────────────────────────────────────────────

def _render_digest_data(
    signals: dict[str, TaskSignals],
    solvability_journal: SolvabilityJournal,
    round_idx: int,
    trajectories_dir: Path,
) -> str:
    total = len(signals)
    failed = sum(1 for s in signals.values() if not s.outcome.eval_passed)
    passed = total - failed
    pass_rate = passed / total if total else 0.0

    l1_counts: dict[str, list[str]] = {
        "all_pass": [], "partial_current": [], "partial_regression": [], "never_solved": [],
    }
    for tid, s in signals.items():
        l1_counts[_level1_category(s)].append(tid)
    n_unstable = sum(1 for s in signals.values() if "unstable_pass" in s.outcome.failure_pattern_tags)

    lines: list[str] = []

    # ── workflow instructions ──────────────────────────────────────────────────
    lines.append(
        "Analyse harness performance patterns across ALL tasks and produce a routing report.\n"
        "\n"
        "Workflow:\n"
        "1. Read `signals_report.md` (path below) — automated Python analysis of all trajectories,\n"
        "   grouped by outcome: all_pass, partial, never_solved.\n"
        "2. Read `{task_id}.json` for any task you want to investigate in depth — contains structured\n"
        "   signals, all rollout paths, and cross-rollout stats. Failed tasks are priority but\n"
        "   all_pass unstable and never_solved tasks are equally valid targets.\n"
        "3. Read raw trajectory JSONL files (paths in each JSON) for step-by-step evidence.\n"
        "4. Write per-task classifications and running notes to the evolution notebook.\n"
        "5. Synthesise all failure patterns and routing decision.\n"
        "6. Call submit_digest_report once with the complete report,\n"
        "   ensuring every pattern cites evidence from signals or trajectory steps you read.\n"
    )

    # ── round summary ─────────────────────────────────────────────────────────
    lines.append(f"## Round {round_idx} Summary")
    lines.append(f"- total: {total}  |  passed: {passed}  |  failed: {failed}  |  pass_rate: {pass_rate:.1%}")
    # Identify rate_regression tasks (pass rate dropped from historical best, some still pass)
    rate_reg_tasks = [
        tid for tid in l1_counts["partial_current"]
        if "rate_regression" in signals[tid].outcome.failure_pattern_tags
    ]
    lines.append(
        f"- 🟢 all_pass={len(l1_counts['all_pass'])} ({n_unstable} unstable)"
        f"  🟡 partial_current={len(l1_counts['partial_current'])}"
        f"  🔴 partial_regression={len(l1_counts['partial_regression'])}"
        f"  ⚫ never_solved={len(l1_counts['never_solved'])}"
    )
    if l1_counts["partial_regression"]:
        lines.append(
            "  → partial_regression tasks (was solvable, now all fail — highest priority): "
            + ", ".join(l1_counts["partial_regression"][:10])
        )
    if rate_reg_tasks:
        rate_details = []
        for tid in rate_reg_tasks[:10]:
            h = signals[tid].history
            cur = signals[tid].outcome.rollout_pass_rate
            best = h.hist_best_pass_rate or 0.0
            rate_details.append(f"{tid} ({cur:.0%}↓{best:.0%})")
        lines.append(
            "  → ⚠ rate_regression tasks (pass rate dropped below historical best"
            " — check if a recent change caused this regression): "
            + ", ".join(rate_details)
        )
    lines.append("")

    # ── file locations ─────────────────────────────────────────────────────────
    lines.append("## Pre-computed Trajectory Analysis\n")
    lines.append(
        f"- `signals_report.md`: `{trajectories_dir / 'signals_report.md'}`\n"
        f"- `{{task_id}}.json`: `{trajectories_dir}/{{task_id}}.json`\n"
        f"  (task IDs listed in signals_report.md)\n"
    )

    # ── historical solvability (cross-round, not in signals_report.md) ────────
    lines.append("## Historical Solvability\n")
    all_records = solvability_journal.get_all()
    if all_records:
        stable = [tid for tid, r in all_records.items() if r.consecutive_pass_rounds >= 2]
        chronic = [tid for tid, r in all_records.items() if r.rounds_without_flip >= 3]
        dangerous_regressions = [
            (tid, signals[tid])
            for tid, r in all_records.items()
            if r.ever_all_passed and tid in signals and not signals[tid].outcome.eval_passed
        ]

        if stable:
            lines.append(f"- stable pass (≥2 consecutive rounds): {', '.join(stable[:8])}")
        if dangerous_regressions:
            lines.append(
                f"\n- ⚠ dangerous regression (previously all-pass, now all-fail,"
                f" {len(dangerous_regressions)} tasks):"
            )
            for tid, s in dangerous_regressions[:8]:
                h = s.history
                stable_note = (
                    f" regressed after {h.consecutive_pass_rounds_before} stable rounds"
                    if h.was_stable
                    else (f" last_passed_round={h.last_passed_round}" if h.last_passed_round is not None else "")
                )
                delta_note = ""
                if h.current_vs_hist_token_delta is not None:
                    delta_note = (
                        f"  token_delta={h.current_vs_hist_token_delta:+}"
                        f"  step_delta={h.current_vs_hist_step_delta:+.1f}"
                    )
                lines.append(f"  - **{tid}**:{stable_note}{delta_note}")
            lines.append(
                "\n  → If token_delta significantly positive and task was stable,"
                " consider recommended_mode=revert"
            )
        if chronic:
            lines.append(f"- chronically failing (≥3 rounds without flip): {', '.join(chronic[:8])}")
        lines.append("")
    else:
        lines.append("(no history — this is round 1)\n")

    return "\n".join(lines)
