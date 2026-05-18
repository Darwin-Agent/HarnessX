"""Regression: ``backfill_ship_outcomes`` must be k-agnostic.

The previous implementation read ``passed = any(passed_flags)`` and only
credited ALL_FAIL → any-pass transitions. For k>=2 this scored every
PARTIAL_PASS → ALL_PASS stabilization as 0 (the canonical evolve win for
variance-reduction was invisible). Tests here lock in:

  * full_unlock     ALL_FAIL → ALL_PASS         credited
  * partial_unlock  ALL_FAIL → PARTIAL          credited
  * stabilized      PARTIAL  → ALL_PASS         credited (the key new case)
  * improved        PARTIAL  → higher PARTIAL   credited (k>=3)
  * regressed_*     ALL_PASS → worse            tracked but not credited
  * legacy field    flipped_to_pass_in_ship_round preserves AF→any-pass
  * k=1 fallback    rows without passed_flags still backfill correctly
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harnessx.aegis.data import ledger
from harnessx.aegis.data.ledger import (
    _classify_state,
    _grade_transition,
    backfill_ship_outcomes,
    record_ship_outcome,
)


def _seed_history(run_root: Path, rows: list[dict]) -> None:
    """Append rows to data/task_history.jsonl."""
    p = run_root / "data" / "task_history.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_outcomes(run_root: Path) -> list[dict]:
    return json.loads((run_root / "data" / "ship_outcomes.json").read_text())


# ─── classifier + grader unit checks ─────────────────────────────────────


def test_classify_state_k_agnostic() -> None:
    assert _classify_state(None) is None
    assert _classify_state([]) is None
    assert _classify_state([True]) == "ALL_PASS"
    assert _classify_state([False]) == "ALL_FAIL"
    assert _classify_state([True, True]) == "ALL_PASS"
    assert _classify_state([False, False]) == "ALL_FAIL"
    assert _classify_state([True, False]) == "PARTIAL"
    # k=3 cases
    assert _classify_state([True, True, True]) == "ALL_PASS"
    assert _classify_state([False, False, False]) == "ALL_FAIL"
    assert _classify_state([True, False, False]) == "PARTIAL"
    assert _classify_state([True, True, False]) == "PARTIAL"


def test_grade_transition_named_categories() -> None:
    # k=1
    assert _grade_transition([False], [True]) == "full_unlock"
    assert _grade_transition([True], [False]) == "regressed_hard"
    assert _grade_transition([True], [True]) == "noop_pass"
    assert _grade_transition([False], [False]) == "noop_fail"

    # k=2
    assert _grade_transition([False, False], [True, True]) == "full_unlock"
    assert _grade_transition([False, False], [True, False]) == "partial_unlock"
    assert _grade_transition([True, False], [True, True]) == "stabilized"  # ★ THE FIX
    assert _grade_transition([True, True], [False, False]) == "regressed_hard"
    assert _grade_transition([True, True], [True, False]) == "regressed_soft"
    assert _grade_transition([True, False], [False, False]) == "regressed_partial"

    # k=3 — "improved" only meaningful for k>=3
    assert _grade_transition([True, False, False], [True, True, False]) == "improved"
    assert _grade_transition([True, True, False], [True, True, True]) == "stabilized"

    # missing data
    assert _grade_transition(None, [True]) == "unknown"
    assert _grade_transition([True], None) == "unknown"


# ─── backfill end-to-end ─────────────────────────────────────────────────


def test_backfill_credits_partial_to_all_pass(tmp_path: Path) -> None:
    """The headline regression: PARTIAL → ALL_PASS used to score 0."""
    run = tmp_path
    # Seed task_history: 2 predicted tasks. Both PARTIAL_PASS in R0,
    # both ALL_PASS in R1 → ship was a stabilization win.
    _seed_history(run, [
        {"round": 0, "task_id": "t1", "passed": True, "passed_flags": [True, False]},
        {"round": 0, "task_id": "t2", "passed": True, "passed_flags": [False, True]},
        {"round": 1, "task_id": "t1", "passed": True, "passed_flags": [True, True]},
        {"round": 1, "task_id": "t2", "passed": True, "passed_flags": [True, True]},
    ])
    record_ship_outcome(
        run, round_n=1, shipped_cid="C-R1-X", bucket="prompt",
        predicted_tasks=["t1", "t2"], rejected_sibling_cids=[], signature="sig",
    )
    backfill_ship_outcomes(run)
    [entry] = _read_outcomes(run)

    assert entry["hit_rate"] == "2/2", \
        "PARTIAL→ALL_PASS stabilization must count as a hit"
    assert entry["hit_rate_strict"] == "0/2", \
        "Strict (full_unlock only) — no AF→AP here"
    assert entry["flipped_to_pass_in_ship_round"] == [], \
        "Legacy AF→any-pass field must stay empty for stabilization"
    assert entry["flipped_by_category"]["stabilized"] == ["t1", "t2"]


def test_backfill_credits_full_unlock(tmp_path: Path) -> None:
    """ALL_FAIL → ALL_PASS — the strictest progress class."""
    run = tmp_path
    _seed_history(run, [
        {"round": 0, "task_id": "t1", "passed": False, "passed_flags": [False, False]},
        {"round": 1, "task_id": "t1", "passed": True, "passed_flags": [True, True]},
    ])
    record_ship_outcome(
        run, round_n=1, shipped_cid="C", bucket="tools",
        predicted_tasks=["t1"], rejected_sibling_cids=[], signature="s",
    )
    backfill_ship_outcomes(run)
    [entry] = _read_outcomes(run)
    assert entry["hit_rate"] == "1/1"
    assert entry["hit_rate_strict"] == "1/1"
    assert entry["flipped_to_pass_in_ship_round"] == ["t1"]
    assert entry["flipped_by_category"]["full_unlock"] == ["t1"]


def test_backfill_partial_unlock_in_legacy_flipped(tmp_path: Path) -> None:
    """ALL_FAIL → PARTIAL is partial unlock — still escapes ALL_FAIL,
    so it should appear in the legacy ``flipped_to_pass_in_ship_round``."""
    run = tmp_path
    _seed_history(run, [
        {"round": 0, "task_id": "t1", "passed": False, "passed_flags": [False, False]},
        {"round": 1, "task_id": "t1", "passed": True, "passed_flags": [True, False]},
    ])
    record_ship_outcome(
        run, round_n=1, shipped_cid="C", bucket="prompt",
        predicted_tasks=["t1"], rejected_sibling_cids=[], signature="s",
    )
    backfill_ship_outcomes(run)
    [entry] = _read_outcomes(run)
    assert entry["flipped_to_pass_in_ship_round"] == ["t1"]
    assert entry["flipped_by_category"]["partial_unlock"] == ["t1"]
    assert entry["hit_rate_strict"] == "0/1"  # not full unlock
    assert entry["hit_rate"] == "1/1"


def test_backfill_regression_tracked_not_credited(tmp_path: Path) -> None:
    run = tmp_path
    _seed_history(run, [
        {"round": 0, "task_id": "t1", "passed": True, "passed_flags": [True, True]},
        {"round": 1, "task_id": "t1", "passed": False, "passed_flags": [False, False]},
    ])
    record_ship_outcome(
        run, round_n=1, shipped_cid="C", bucket="prompt",
        predicted_tasks=["t1"], rejected_sibling_cids=[], signature="s",
    )
    backfill_ship_outcomes(run)
    [entry] = _read_outcomes(run)
    assert entry["hit_rate"] == "0/1"
    assert entry["flipped_by_category"]["regressed_hard"] == ["t1"]


def test_backfill_k1_legacy_rows(tmp_path: Path) -> None:
    """Old rows without ``passed_flags`` must still backfill via ``passed``."""
    run = tmp_path
    _seed_history(run, [
        {"round": 0, "task_id": "t1", "passed": False},  # no passed_flags
        {"round": 1, "task_id": "t1", "passed": True},
    ])
    record_ship_outcome(
        run, round_n=1, shipped_cid="C", bucket="prompt",
        predicted_tasks=["t1"], rejected_sibling_cids=[], signature="s",
    )
    backfill_ship_outcomes(run)
    [entry] = _read_outcomes(run)
    # k=1: only full_unlock or noop possible.
    assert entry["flipped_by_category"].get("full_unlock") == ["t1"]
    assert entry["hit_rate"] == "1/1"


def test_backfill_k3_improved_grade(tmp_path: Path) -> None:
    """k=3: PARTIAL with higher pass-rate gets ``improved`` (not stabilized)."""
    run = tmp_path
    _seed_history(run, [
        {"round": 0, "task_id": "t1", "passed_flags": [True, False, False]},   # 1/3
        {"round": 1, "task_id": "t1", "passed_flags": [True, True, False]},    # 2/3
    ])
    record_ship_outcome(
        run, round_n=1, shipped_cid="C", bucket="prompt",
        predicted_tasks=["t1"], rejected_sibling_cids=[], signature="s",
    )
    backfill_ship_outcomes(run)
    [entry] = _read_outcomes(run)
    assert entry["flipped_by_category"]["improved"] == ["t1"]
    assert entry["hit_rate"] == "1/1"
    assert entry["hit_rate_strict"] == "0/1"
    assert entry["predicted_tasks_status_latest"]["t1"] == "partial"


def test_backfill_status_latest_three_way(tmp_path: Path) -> None:
    run = tmp_path
    _seed_history(run, [
        {"round": 0, "task_id": "t_pass", "passed_flags": [True, True]},
        {"round": 0, "task_id": "t_part", "passed_flags": [True, False]},
        {"round": 0, "task_id": "t_fail", "passed_flags": [False, False]},
        {"round": 1, "task_id": "t_pass", "passed_flags": [True, True]},
        {"round": 1, "task_id": "t_part", "passed_flags": [True, False]},
        {"round": 1, "task_id": "t_fail", "passed_flags": [False, False]},
    ])
    record_ship_outcome(
        run, round_n=1, shipped_cid="C", bucket="prompt",
        predicted_tasks=["t_pass", "t_part", "t_fail"],
        rejected_sibling_cids=[], signature="s",
    )
    backfill_ship_outcomes(run)
    [entry] = _read_outcomes(run)
    s = entry["predicted_tasks_status_latest"]
    assert s["t_pass"] == "passing"
    assert s["t_part"] == "partial"
    assert s["t_fail"] == "still_failing"
