# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Adaptive k=2 focus-set computation.

The focus set picks which tasks this round should run twice to unlock
PARTIAL_PASS pattern signal (the Digester's highest-information template).
Priority: last-ship predicted tasks still failing > bouncer tasks.
"""
from pathlib import Path

from harnessx.aegis.data import ledger


def _run_recipe_focus(tmp_path: Path, round_idx: int, max_focus: int) -> set[str]:
    # Import the recipe lazily so test discovery doesn't pull benchmark deps.
    import sys, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from recipe.gaia_evolver.run_meta_aegis import _compute_focus_set
    return _compute_focus_set(tmp_path, round_idx, max_focus)


def test_focus_set_is_empty_at_r0(tmp_path):
    # Pre-populate some history just to show it's still empty.
    ledger.append_task_history(tmp_path, [
        {"round": 0, "task_id": "t1", "passed": True},
    ])
    assert _run_recipe_focus(tmp_path, round_idx=0, max_focus=10) == set()


def test_focus_set_empty_when_max_focus_zero(tmp_path):
    ledger.append_task_history(tmp_path, [
        {"round": 0, "task_id": "t1", "passed": True},
        {"round": 1, "task_id": "t1", "passed": False},  # bouncer
    ])
    assert _run_recipe_focus(tmp_path, round_idx=2, max_focus=0) == set()


def test_focus_set_picks_bouncer_tasks(tmp_path):
    # Always-pass, always-fail, and a bouncer.
    for r in range(3):
        ledger.append_task_history(tmp_path, [
            {"round": r, "task_id": "always_pass", "passed": True},
            {"round": r, "task_id": "always_fail", "passed": False},
            {"round": r, "task_id": "bouncer", "passed": r == 1},
        ])
    focus = _run_recipe_focus(tmp_path, round_idx=3, max_focus=10)
    assert "bouncer" in focus
    assert "always_pass" not in focus
    assert "always_fail" not in focus


def test_focus_set_prioritizes_still_failing_predicted_tasks(tmp_path):
    # Set up: ship@R1 predicted A and B. A stays fail, B passes.
    # Also a bouncer task that should show up but lower priority.
    ledger.record_ship_outcome(
        tmp_path, round_n=1, shipped_cid="C-R1-01",
        bucket="prompt", predicted_tasks=["target_A", "target_B"],
        rejected_sibling_cids=[],
    )
    ledger.append_task_history(tmp_path, [
        {"round": 0, "task_id": "target_A", "passed": False},
        {"round": 0, "task_id": "target_B", "passed": False},
        {"round": 0, "task_id": "bouncer_X", "passed": True},
        {"round": 1, "task_id": "target_A", "passed": False},  # still fail
        {"round": 1, "task_id": "target_B", "passed": True},   # passed (ship took)
        {"round": 1, "task_id": "bouncer_X", "passed": False},
    ])
    ledger.backfill_ship_outcomes(tmp_path)

    focus = _run_recipe_focus(tmp_path, round_idx=2, max_focus=2)
    # max_focus=2 — priority 1 (predicted) should eat both slots before bouncer
    # because target_A is still_failing and target_B is passing but still
    # predicted (we want to re-verify it didn't regress).
    assert "target_A" in focus


def test_focus_set_respects_max_focus_cap(tmp_path):
    # Ten bouncer tasks but cap is 3.
    for r in range(2):
        for i in range(10):
            ledger.append_task_history(tmp_path, [{
                "round": r, "task_id": f"b{i}", "passed": (r + i) % 2 == 0,
            }])
    focus = _run_recipe_focus(tmp_path, round_idx=2, max_focus=3)
    assert len(focus) <= 3
