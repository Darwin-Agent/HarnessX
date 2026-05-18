# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Per-round regression watchlist.

When a round's ship makes a previously passing (or partially passing)
task worse, the system used to bury the signal: ship_outcomes.json's
``hit_rate`` only counts improvements among the ship's own predicted
tasks, so collateral damage on un-predicted tasks stayed invisible.
Planner / Evolver / Critic then ran the next round looking for "what
should we attack" without seeing "what did we just break".

This module computes a deterministic regression list from
``data/task_history.jsonl`` (k-aware, so PARTIAL → ALL_FAIL counts even
without any AF→PASS dimension). The orchestrator writes the result to
``R{N}/regressions.md`` at the start of Stage P; Planner/Evolver/Critic
prompts include it in their explicit read list so the LLMs cannot miss
a regression by accidental synthesis omission.

A regression is the inverse of the improvement grades in
:mod:`harnessx.aegis.data.ledger`:

    regressed_hard      ALL_PASS → ALL_FAIL
    regressed_soft      ALL_PASS → PARTIAL
    regressed_partial   PARTIAL  → lower PARTIAL or ALL_FAIL

We also surface which ships at round=N are joint suspects (those ships
built round N's config and therefore caused round N's regressions),
with each ship's bucket — Critic uses this to decide whether to
iterate-from a specific bucket or pivot.
"""

from __future__ import annotations

import json
from pathlib import Path


def _classify(flags: list[bool] | None) -> str | None:
    if not flags:
        return None
    if all(flags):
        return "ALL_PASS"
    if not any(flags):
        return "ALL_FAIL"
    return "PARTIAL"


def _pass_rate(flags: list[bool] | None) -> float | None:
    if not flags:
        return None
    return sum(flags) / len(flags)


def _grade(prev_flags: list[bool] | None, curr_flags: list[bool] | None) -> str | None:
    """Return the regression grade if there is one, else None."""
    p = _pass_rate(prev_flags)
    c = _pass_rate(curr_flags)
    if p is None or c is None:
        return None
    if p == 1.0 and c == 0.0:
        return "regressed_hard"
    if p == 1.0 and 0.0 < c < 1.0:
        return "regressed_soft"
    if 0.0 < p < 1.0 and c < p:
        return "regressed_partial"
    return None


def detect_regressions(
    run_root: Path,
    round_n: int,
) -> list[dict]:
    """Compare round_n-1 vs round_n task states; one row per regressed task.

    Returns ``[]`` when round_n is 0 or there is no prior data.
    """
    if round_n <= 0:
        return []
    hist_path = run_root / "data" / "task_history.jsonl"
    if not hist_path.exists():
        return []

    per_task: dict[str, dict[int, list[bool]]] = {}
    for line in hist_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid = row.get("task_id")
        if not tid:
            continue
        flags = row.get("passed_flags")
        if not isinstance(flags, list) or not flags:
            flags = [bool(row.get("passed", False))]
        per_task.setdefault(tid, {})[int(row.get("round", 0))] = [bool(b) for b in flags]

    # Joint-suspect ships are those tagged round=N: in aegis the commit
    # event for round N produces the config that round N's task batch
    # actually runs against, so a regression visible by comparing
    # round N-1 vs round N is caused by ships at round=N. (Common bug:
    # filtering on round-1 here points at the prior config, which is
    # exactly the one that did NOT regress.)
    suspects: list[dict] = []
    so_path = run_root / "data" / "ship_outcomes.json"
    if so_path.exists():
        try:
            outcomes = json.loads(so_path.read_text(encoding="utf-8"))
            if isinstance(outcomes, list):
                for o in outcomes:
                    if int(o.get("round", -1)) == round_n:
                        suspects.append(
                            {
                                "ship_id": o.get("ship_id"),
                                "bucket": o.get("bucket"),
                            }
                        )
        except json.JSONDecodeError:
            pass

    out: list[dict] = []
    for tid, by_round in per_task.items():
        prev = by_round.get(round_n - 1)
        curr = by_round.get(round_n)
        grade = _grade(prev, curr)
        if grade is None:
            continue
        out.append(
            {
                "task_id": tid,
                "prev_state": _classify(prev),
                "curr_state": _classify(curr),
                "prev_flags": prev,
                "curr_flags": curr,
                "grade": grade,
                "joint_suspect_ships": list(suspects),
            }
        )
    out.sort(
        key=lambda r: ({"regressed_hard": 0, "regressed_soft": 1, "regressed_partial": 2}[r["grade"]], r["task_id"])
    )
    return out


def render_regressions_md(round_n: int, regressions: list[dict]) -> str:
    """Markdown rendering written to ``R{N}/regressions.md``.

    Always emits a file (even when empty) so downstream prompts can
    reliably refer to ``R{N}/regressions.md`` without conditional
    presence checks.
    """
    lines: list[str] = []
    lines.append(f"# Regressions detected in R{round_n}")
    lines.append("")
    lines.append(
        "Tasks whose pass-state worsened versus the previous round, "
        "computed mechanically from `data/task_history.jsonl`. "
        "Each entry is a *joint* attribution: any ship at round={n} is "
        "a candidate cause (those ships built this round's config and "
        "thus produced this round's regressions) and is listed below. "
        "Use `data/ship_outcomes.json[].evidence_per_task` to separate "
        "ships with mechanical evidence (tools/processor) from ships "
        "without (prompt/config — joint by definition).".format(n=round_n)
    )
    lines.append("")

    if not regressions:
        lines.append("_No regressions detected this round._")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Summary")
    lines.append("")
    by_grade: dict[str, list[dict]] = {}
    for r in regressions:
        by_grade.setdefault(r["grade"], []).append(r)
    for grade in ("regressed_hard", "regressed_soft", "regressed_partial"):
        rows = by_grade.get(grade, [])
        if rows:
            ids = ", ".join(f"`{r['task_id'][:8]}`" for r in rows)
            lines.append(f"- **{grade}**: {len(rows)} — {ids}")
    lines.append("")

    if regressions and regressions[0].get("joint_suspect_ships"):
        ships = regressions[0]["joint_suspect_ships"]
        lines.append(f"## Joint-suspect ships from R{round_n}")
        lines.append("")
        for s in ships:
            lines.append(f"- `{s.get('ship_id')}` (bucket=`{s.get('bucket')}`)")
        lines.append("")

    lines.append("## Per-task detail")
    lines.append("")
    lines.append("| task_id | prev | curr | grade | prev_flags → curr_flags |")
    lines.append("|---|---|---|---|---|")
    for r in regressions:
        lines.append(
            f"| `{r['task_id']}` | {r['prev_state']} | {r['curr_state']} "
            f"| {r['grade']} | {r['prev_flags']} → {r['curr_flags']} |"
        )
    lines.append("")
    lines.append(
        "**Required action**: Evolver MUST either include at least one "
        "candidate that addresses these regressions (typically by iterating "
        "from the suspect ship with bucket=`prompt`/`config`) OR explicitly "
        "state in the manifest body why the regression is acceptable / "
        "transient / out of scope. Critic verifies this in portfolio_audit."
    )
    return "\n".join(lines)


def write_regressions_md(
    run_root: Path,
    round_n: int,
    out_path: Path | None = None,
) -> Path:
    """Compute + write ``R{N}/regressions.md``. Returns the output path."""
    regressions = detect_regressions(run_root, round_n)
    if out_path is None:
        out_path = run_root / f"R{round_n}" / "regressions.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_regressions_md(round_n, regressions),
        encoding="utf-8",
    )
    return out_path
