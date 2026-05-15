# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Unit tests for the ship-aware rollback helpers in run_meta_aegis.

Covers the three small pure helpers added alongside the rollback feature:
``_read_latest_commit_shipments``, ``_append_rollback_reputation``,
``_append_rollback_audit``. Tests also exercise the last-ship-info dedup
invariant (only the MOST RECENT commit event for the named round is used,
not earlier partial writes).
"""
from __future__ import annotations

import json
from pathlib import Path

from recipe.gaia_evolver.run_meta_aegis import (
    _append_rollback_audit,
    _append_rollback_reputation,
    _read_latest_commit_shipments,
)


def _write_audit(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def test_read_commit_returns_empty_when_audit_missing(tmp_path):
    assert _read_latest_commit_shipments(tmp_path, round_n=1) == []


def test_read_commit_returns_empty_when_round_had_no_ship(tmp_path):
    _write_audit(tmp_path / "audit.jsonl", [
        {"round": 1, "stage": "4", "kind": "revert",
         "payload": {"shipped_cids": [], "shipped_by_bucket": {}}},
    ])
    assert _read_latest_commit_shipments(tmp_path, round_n=1) == []


def test_read_commit_extracts_shipments_for_round(tmp_path):
    _write_audit(tmp_path / "audit.jsonl", [
        {"round": 2, "stage": "4", "kind": "commit",
         "payload": {
             "shipped_cids": ["C-R2-01"],
             "shipped_by_bucket": {"prompt": "C-R2-01"},
         }},
    ])
    result = _read_latest_commit_shipments(tmp_path, round_n=2)
    assert result == [("C-R2-01", "prompt")]


def test_read_commit_handles_multi_ship(tmp_path):
    _write_audit(tmp_path / "audit.jsonl", [
        {"round": 6, "stage": "4", "kind": "commit",
         "payload": {
             "shipped_cids": ["C-R6-01", "C-R6-02"],
             "shipped_by_bucket": {
                 "processor": "C-R6-01", "tools": "C-R6-02",
             },
         }},
    ])
    result = _read_latest_commit_shipments(tmp_path, round_n=6)
    assert set(result) == {("C-R6-01", "processor"), ("C-R6-02", "tools")}


def test_read_commit_picks_latest_when_multiple_writes_same_round(tmp_path):
    """If the orchestrator writes more than one commit event for a round
    (shouldn't happen in practice but is not prevented by the schema), the
    helper must use the last one so the caller sees the final state."""
    _write_audit(tmp_path / "audit.jsonl", [
        {"round": 3, "stage": "4", "kind": "commit",
         "payload": {
             "shipped_cids": ["C-R3-STALE"],
             "shipped_by_bucket": {"prompt": "C-R3-STALE"},
         }},
        {"round": 3, "stage": "4", "kind": "commit",
         "payload": {
             "shipped_cids": ["C-R3-LATEST"],
             "shipped_by_bucket": {"prompt": "C-R3-LATEST"},
         }},
    ])
    assert _read_latest_commit_shipments(tmp_path, round_n=3) == [
        ("C-R3-LATEST", "prompt"),
    ]


def test_read_commit_ignores_other_rounds(tmp_path):
    _write_audit(tmp_path / "audit.jsonl", [
        {"round": 2, "stage": "4", "kind": "commit",
         "payload": {"shipped_cids": ["C-R2-01"],
                     "shipped_by_bucket": {"prompt": "C-R2-01"}}},
        {"round": 3, "stage": "4", "kind": "commit",
         "payload": {"shipped_cids": ["C-R3-01"],
                     "shipped_by_bucket": {"processor": "C-R3-01"}}},
    ])
    assert _read_latest_commit_shipments(tmp_path, round_n=2) == [
        ("C-R2-01", "prompt"),
    ]


def test_append_reputation_creates_file_if_missing(tmp_path):
    _append_rollback_reputation(tmp_path, ["prompt", "processor"])
    data = json.loads((tmp_path / "reputation.json").read_text())
    assert data["prompt"] == [False]
    assert data["processor"] == [False]


def test_append_reputation_appends_to_existing_history(tmp_path):
    rep_path = tmp_path / "reputation.json"
    rep_path.write_text(json.dumps({
        "prompt": [True, True],
        "tools": [False, True],
    }))
    _append_rollback_reputation(tmp_path, ["prompt"])
    data = json.loads(rep_path.read_text())
    assert data["prompt"] == [True, True, False]
    # Other buckets untouched.
    assert data["tools"] == [False, True]


def test_append_reputation_handles_duplicate_buckets(tmp_path):
    """If the same bucket appears twice in shipped list (shouldn't happen
    but defensive), each should append a False."""
    _append_rollback_reputation(tmp_path, ["prompt", "prompt"])
    data = json.loads((tmp_path / "reputation.json").read_text())
    assert data["prompt"] == [False, False]


def test_append_audit_event_wellformed(tmp_path):
    # Pre-existing audit with prior entries.
    (tmp_path / "audit.jsonl").write_text(
        '{"round": 1, "stage": "P", "kind": "preprocess"}\n'
    )
    _append_rollback_audit(
        tmp_path,
        round_idx=3,
        rolled_back_cids=["C-R3-01"],
        pre_ship_rate=0.75,
        post_ship_rate=0.688,
        delta_count=-4,
        reason="Δrate=-0.062 ≤ -0.05 AND Δcount=-4 ≤ -3",
    )
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 2
    last = json.loads(lines[-1])
    assert last["round"] == 3
    assert last["stage"] == "R"
    assert last["kind"] == "rollback"
    p = last["payload"]
    assert p["rolled_back_cids"] == ["C-R3-01"]
    assert p["pre_ship_rate"] == 0.75
    assert p["post_ship_rate"] == 0.688
    assert p["delta_count"] == -4
    assert "Δrate" in p["reason"]
    assert "ts" in last


def test_rollback_threshold_semantics_fires_on_r2_to_r3_drop():
    """Sanity check: the -5pp / -3 task threshold catches the real-world
    regression we just observed in aegis_64_v3 (R2=48, R3=44, Δ=-4
    tasks / -6.2pp), but does NOT fire on single-task noise (R5→R6 was
    -1 task / -1.5pp)."""
    def would_rollback(pre_passed: int, post_passed: int, n: int = 64) -> bool:
        pre_rate = pre_passed / n
        post_rate = post_passed / n
        delta_rate = post_rate - pre_rate
        delta_count = post_passed - pre_passed
        return delta_rate <= -0.05 and delta_count <= -3

    # The R2→R3 regression in the pilot — should rollback.
    assert would_rollback(48, 44)
    # R5→R6 was -1 task — noise, keep.
    assert not would_rollback(47, 46)
    # R8→R9 was -1 task — noise, keep.
    assert not would_rollback(49, 48)
    # R0 baseline variance: 47→43 is also catastrophic (-4 tasks / -6.2pp).
    # This would trip the threshold, but the ship-aware guard skips rollback
    # when last_ship_info is None (i.e., no recent ship to blame); that
    # guard is tested via the integration path, not here.
    assert would_rollback(47, 43)
    # Improvement should never rollback.
    assert not would_rollback(44, 48)
    # At N=64, -3 tasks is 4.69pp — below the 5pp threshold.
    # BOTH thresholds must be met (conservative AND), so -3 tasks alone
    # does NOT roll back. Requires -4 tasks at N=64.
    assert not would_rollback(50, 47)
    assert would_rollback(50, 46)  # -4 tasks = -6.25pp, both conditions met
