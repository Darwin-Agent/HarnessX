# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Ledger persistence — cross-round structured state the agents can Read.

These tests exercise the four append/rewrite helpers plus INDEX.md
refresh. The goal of the ledger is to turn prose journal entries into
greppable structured rows so Planner/Evolver/Critic can synthesize
cross-round patterns on their own.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harnessx.aegis.data import ledger


# ---------------------------------------------------------------------------
# task_history.jsonl
# ---------------------------------------------------------------------------

def test_append_task_history_writes_one_row_per_task(tmp_path: Path):
    ledger.append_task_history(tmp_path, [
        {"round": 0, "task_id": "t1", "level": "L1", "passed": True, "steps": 5},
        {"round": 0, "task_id": "t2", "level": "L2", "passed": False, "exit": "budget", "steps": 20},
    ])
    rows = ledger.read_task_history(tmp_path)
    assert len(rows) == 2
    assert rows[0]["task_id"] == "t1"
    assert rows[0]["passed"] is True
    assert rows[1]["exit"] == "budget"
    assert rows[1]["steps"] == 20


def test_append_task_history_accepts_dataclass(tmp_path: Path):
    rec = ledger.TaskRecord(round=3, task_id="t9", level="L3",
                            passed=True, steps=7, cost_usd=0.42)
    ledger.append_task_history(tmp_path, [rec])
    rows = ledger.read_task_history(tmp_path)
    assert rows[0]["task_id"] == "t9"
    assert rows[0]["cost_usd"] == 0.42


def test_append_task_history_skips_rows_without_task_id(tmp_path: Path):
    ledger.append_task_history(tmp_path, [
        {"round": 0, "task_id": "", "passed": True},
        {"round": 0, "task_id": "real", "passed": False},
    ])
    rows = ledger.read_task_history(tmp_path)
    assert len(rows) == 1
    assert rows[0]["task_id"] == "real"


# ---------------------------------------------------------------------------
# ship_outcomes.json  (list, rewritten)
# ---------------------------------------------------------------------------

def test_record_ship_outcome_writes_entry(tmp_path: Path):
    ledger.record_ship_outcome(
        tmp_path, round_n=1, shipped_cid="C-R1-02",
        bucket="processor",
        predicted_tasks=["t1", "t2"],
        rejected_sibling_cids=["C-R1-01", "C-R1-03"],
        signature="abc123",
    )
    outs = ledger.read_ship_outcomes(tmp_path)
    assert len(outs) == 1
    assert outs[0]["ship_id"] == "C-R1-02"
    assert outs[0]["bucket"] == "processor"
    assert outs[0]["predicted_tasks"] == ["t1", "t2"]
    assert outs[0]["hit_rate"] is None  # not yet backfilled


def test_backfill_ship_outcomes_computes_hit_rate_from_task_history(tmp_path: Path):
    # R0: t1 fails, t2 fails (baseline).
    # R1: ship C-R1-02 predicting t1 and t2 will pass.
    # R1 rollouts: t1 passes (credit), t2 still fails.
    ledger.append_task_history(tmp_path, [
        {"round": 0, "task_id": "t1", "passed": False},
        {"round": 0, "task_id": "t2", "passed": False},
    ])
    ledger.record_ship_outcome(
        tmp_path, round_n=1, shipped_cid="C-R1-02",
        bucket="processor", predicted_tasks=["t1", "t2"],
        rejected_sibling_cids=[],
    )
    ledger.append_task_history(tmp_path, [
        {"round": 1, "task_id": "t1", "passed": True},
        {"round": 1, "task_id": "t2", "passed": False},
    ])
    ledger.backfill_ship_outcomes(tmp_path)

    outs = ledger.read_ship_outcomes(tmp_path)
    entry = outs[0]
    assert entry["flipped_to_pass_in_ship_round"] == ["t1"]
    assert entry["hit_rate"] == "1/2"
    assert entry["predicted_tasks_status_latest"]["t1"] == "passing"
    assert entry["predicted_tasks_status_latest"]["t2"] == "still_failing"


def test_backfill_ship_outcomes_tracks_regression_across_future_rounds(tmp_path: Path):
    # R0: t1 fails. R1 ship claims t1. R1: t1 passes (hit).
    # R2 rollouts: t1 regresses back to fail. predicted_tasks_status_latest
    # should now say 'still_failing' even though the ship itself credited it.
    ledger.append_task_history(tmp_path, [
        {"round": 0, "task_id": "t1", "passed": False},
    ])
    ledger.record_ship_outcome(
        tmp_path, round_n=1, shipped_cid="C-R1-01",
        bucket="prompt", predicted_tasks=["t1"], rejected_sibling_cids=[],
    )
    ledger.append_task_history(tmp_path, [
        {"round": 1, "task_id": "t1", "passed": True},
        {"round": 2, "task_id": "t1", "passed": False},
    ])
    ledger.backfill_ship_outcomes(tmp_path)

    entry = ledger.read_ship_outcomes(tmp_path)[0]
    # Ship round credit: t1 flipped fail->pass at R1 so it's counted.
    assert entry["flipped_to_pass_in_ship_round"] == ["t1"]
    # But latest status at R2 shows the regression.
    assert entry["predicted_tasks_status_latest"]["t1"] == "still_failing"


# ---------------------------------------------------------------------------
# rejected_candidates.jsonl
# ---------------------------------------------------------------------------

def test_append_rejected_candidates_truncates_long_excerpts(tmp_path: Path):
    ledger.append_rejected_candidates(tmp_path, 3, [
        {
            "candidate_id": "C-R3-02",
            "bucket": "tools",
            "predicted_tasks": ["t5"],
            "rejection_text_excerpt": "x" * 1000,
        },
    ])
    rows = ledger.read_rejected_candidates(tmp_path)
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "C-R3-02"
    assert rows[0]["bucket"] == "tools"
    assert len(rows[0]["rejection_text_excerpt"]) == 400
    assert rows[0]["revived_as"] == []  # no revival yet
    # author_confidence / novelty_dimension removed from schema — agents
    # decide without self-rated scalars.
    assert "author_confidence" not in rows[0]
    assert "novelty_dimension" not in rows[0]


def test_backfill_rejected_revivals_detects_archive_pointers(tmp_path: Path):
    # Rejected at R3.
    ledger.append_rejected_candidates(tmp_path, 3, [
        {"candidate_id": "C-R3-02", "bucket": "tools",
         "predicted_tasks": ["t5"], "rejection_text_excerpt": "low confidence"},
    ])
    # R4 brief revives it.
    r4_briefs = tmp_path / "R4" / "briefs"
    r4_briefs.mkdir(parents=True)
    (r4_briefs / "B-R4-02.md").write_text(
        "---\nbrief_id: B-R4-02\nbucket: tools\nslot_type: regular\n"
        "lead_pointer: archive:C-R3-02\n---\n\nBody."
    )

    ledger.backfill_rejected_revivals(tmp_path, [r4_briefs])
    rows = ledger.read_rejected_candidates(tmp_path)
    assert rows[0]["revived_as"] == [{"round": 4, "brief_id": "B-R4-02"}]


def test_backfill_rejected_revivals_is_idempotent(tmp_path: Path):
    ledger.append_rejected_candidates(tmp_path, 3, [
        {"candidate_id": "C-R3-02", "rejection_text_excerpt": ""},
    ])
    # No briefs yet.
    ledger.backfill_rejected_revivals(tmp_path, [])
    rows_before = ledger.read_rejected_candidates(tmp_path)
    # Call again — file should not grow.
    ledger.backfill_rejected_revivals(tmp_path, [])
    rows_after = ledger.read_rejected_candidates(tmp_path)
    assert rows_before == rows_after


# ---------------------------------------------------------------------------
# INDEX.md
# ---------------------------------------------------------------------------

def test_refresh_index_md_creates_catalog(tmp_path: Path):
    path = ledger.refresh_index_md(tmp_path, current_round=2)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    # Catalog must mention each ledger so agents can discover them.
    assert "task_history.jsonl" in text
    assert "ship_outcomes.json" in text
    assert "rejected_candidates.jsonl" in text
    # No forced reading list — the "use your judgment" phrasing matters.
    assert "your judgment" in text.lower() or "your discretion" in text.lower()


def test_refresh_index_md_lists_existing_round_dirs(tmp_path: Path):
    (tmp_path / "R0").mkdir()
    (tmp_path / "R1").mkdir()
    (tmp_path / "R2").mkdir()
    ledger.refresh_index_md(tmp_path, current_round=3)
    text = (tmp_path / "INDEX.md").read_text(encoding="utf-8")
    assert "R0" in text and "R1" in text and "R2" in text


def test_refresh_index_md_flags_present_vs_missing_ledgers(tmp_path: Path):
    # Before any ledger writes, INDEX should mark data files "not yet populated".
    ledger.refresh_index_md(tmp_path, current_round=0)
    text = (tmp_path / "INDEX.md").read_text(encoding="utf-8")
    assert "not yet populated" in text

    # After we append task_history, INDEX regen should flag it present.
    ledger.append_task_history(tmp_path, [
        {"round": 0, "task_id": "t1", "passed": True},
    ])
    ledger.refresh_index_md(tmp_path, current_round=1)
    text2 = (tmp_path / "INDEX.md").read_text(encoding="utf-8")
    # The task_history line should now say (present). Backtick closes before
    # the status tag: `data/task_history.jsonl` (present)
    assert "task_history.jsonl` (present)" in text2
