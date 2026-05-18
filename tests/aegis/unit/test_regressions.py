"""Regression watchlist: detect tasks that worsened vs the prior round.

The classic miss this catches: R{N-1} ships a prompt change that flips
some predicted-pass tasks (good) but also breaks an unrelated previously-
ALL_PASS task (bad). ship_outcomes.json's hit_rate counts only the
predicted-task improvements, so the collateral damage stays invisible
unless something explicitly walks task_history transitions. That's this
file.
"""

from __future__ import annotations

import json
from pathlib import Path


from harnessx.aegis.data.regressions import (
    detect_regressions,
    render_regressions_md,
    write_regressions_md,
)


def _seed_history(run_root: Path, rows: list[dict]) -> None:
    p = run_root / "data" / "task_history.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _seed_outcomes(run_root: Path, outcomes: list[dict]) -> None:
    p = run_root / "data" / "ship_outcomes.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(outcomes), encoding="utf-8")


def test_no_regressions_at_round_zero(tmp_path: Path) -> None:
    """Round 0 has no prior round → empty list, no error."""
    _seed_history(
        tmp_path,
        [
            {"round": 0, "task_id": "t1", "passed_flags": [True, True]},
        ],
    )
    assert detect_regressions(tmp_path, 0) == []


def test_regressed_hard(tmp_path: Path) -> None:
    """ALL_PASS → ALL_FAIL is the worst grade; must be at top of summary."""
    _seed_history(
        tmp_path,
        [
            {"round": 0, "task_id": "t1", "passed_flags": [True, True]},
            {"round": 1, "task_id": "t1", "passed_flags": [False, False]},
        ],
    )
    rs = detect_regressions(tmp_path, 1)
    assert len(rs) == 1
    assert rs[0]["task_id"] == "t1"
    assert rs[0]["grade"] == "regressed_hard"
    assert rs[0]["prev_state"] == "ALL_PASS"
    assert rs[0]["curr_state"] == "ALL_FAIL"


def test_regressed_soft(tmp_path: Path) -> None:
    _seed_history(
        tmp_path,
        [
            {"round": 0, "task_id": "t1", "passed_flags": [True, True]},
            {"round": 1, "task_id": "t1", "passed_flags": [True, False]},
        ],
    )
    rs = detect_regressions(tmp_path, 1)
    assert rs[0]["grade"] == "regressed_soft"


def test_regressed_partial(tmp_path: Path) -> None:
    """PARTIAL with lower pass-rate is regressed_partial. PARTIAL→ALL_FAIL
    qualifies as the steepest fall within this grade (rate dropped to 0)."""
    _seed_history(
        tmp_path,
        [
            {"round": 0, "task_id": "t1", "passed_flags": [True, False]},
            {"round": 1, "task_id": "t1", "passed_flags": [False, False]},
        ],
    )
    rs = detect_regressions(tmp_path, 1)
    assert rs[0]["grade"] == "regressed_partial"


def test_no_regression_if_unchanged(tmp_path: Path) -> None:
    _seed_history(
        tmp_path,
        [
            {"round": 0, "task_id": "t1", "passed_flags": [True, True]},
            {"round": 1, "task_id": "t1", "passed_flags": [True, True]},
            {"round": 0, "task_id": "t2", "passed_flags": [False, False]},
            {"round": 1, "task_id": "t2", "passed_flags": [False, False]},
        ],
    )
    assert detect_regressions(tmp_path, 1) == []


def test_no_regression_when_partial_flags_reorder(tmp_path: Path) -> None:
    """[True, False] → [False, True] has the same pass_rate (0.5) and is
    pure ordering noise, not a regression."""
    _seed_history(
        tmp_path,
        [
            {"round": 0, "task_id": "t1", "passed_flags": [True, False]},
            {"round": 1, "task_id": "t1", "passed_flags": [False, True]},
        ],
    )
    assert detect_regressions(tmp_path, 1) == []


def test_improvements_are_not_listed(tmp_path: Path) -> None:
    _seed_history(
        tmp_path,
        [
            {"round": 0, "task_id": "t1", "passed_flags": [False, False]},
            {"round": 1, "task_id": "t1", "passed_flags": [True, True]},
        ],
    )
    assert detect_regressions(tmp_path, 1) == []


def test_joint_suspects_are_round_n_ships(tmp_path: Path) -> None:
    """Ships at round=N built round N's config, so they own any
    regression that surfaces by comparing R{N-1} → R{N}. Filtering by
    the wrong round (an off-by-one to round_n-1) would point at the
    last-known-good config and let the actual culprit slip through."""
    _seed_history(
        tmp_path,
        [
            {"round": 0, "task_id": "t1", "passed_flags": [True, True]},
            {"round": 1, "task_id": "t1", "passed_flags": [False, False]},
        ],
    )
    _seed_outcomes(
        tmp_path,
        [
            # Ships at round=1 — these built R1's config, so they're suspects.
            {"ship_id": "C-R1-01", "round": 1, "bucket": "prompt"},
            {"ship_id": "C-R1-02", "round": 1, "bucket": "config"},
            # Ship at round=0 — would only matter if our filter is off-by-one.
            {"ship_id": "C-R0-99", "round": 0, "bucket": "tools"},
        ],
    )
    rs = detect_regressions(tmp_path, 1)
    assert len(rs) == 1
    suspects = rs[0]["joint_suspect_ships"]
    suspect_ids = {s["ship_id"] for s in suspects}
    assert suspect_ids == {"C-R1-01", "C-R1-02"}, suspect_ids


def test_render_md_empty_state(tmp_path: Path) -> None:
    md = render_regressions_md(round_n=1, regressions=[])
    assert "No regressions detected this round" in md
    assert "# Regressions detected in R1" in md


def test_render_md_includes_required_action(tmp_path: Path) -> None:
    md = render_regressions_md(
        round_n=2,
        regressions=[
            {
                "task_id": "abc12345-deadbeef",
                "prev_state": "ALL_PASS",
                "curr_state": "ALL_FAIL",
                "prev_flags": [True, True],
                "curr_flags": [False, False],
                "grade": "regressed_hard",
                "joint_suspect_ships": [{"ship_id": "C-R1-X", "bucket": "prompt"}],
            },
        ],
    )
    assert "regressed_hard" in md
    assert "abc12345" in md
    assert "Required action" in md
    assert "C-R1-X" in md


def test_write_regressions_md_creates_file(tmp_path: Path) -> None:
    _seed_history(
        tmp_path,
        [
            {"round": 0, "task_id": "t1", "passed_flags": [True, True]},
            {"round": 1, "task_id": "t1", "passed_flags": [False, False]},
        ],
    )
    out = write_regressions_md(tmp_path, 1)
    assert out.exists()
    body = out.read_text()
    assert "regressed_hard" in body
    assert out.parent.name == "R1"
