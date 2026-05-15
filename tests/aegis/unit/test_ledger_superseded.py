# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""v0.9.5 — ledger.mark_ship_superseded + ship_ledger_for_gate."""
from __future__ import annotations

import json
from pathlib import Path

from harnessx.aegis.data import ledger


def _write_outcomes(run_root: Path, entries: list[dict]) -> Path:
    p = ledger.data_dir(run_root) / "ship_outcomes.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return p


def test_mark_ship_superseded_sets_field(tmp_path):
    _write_outcomes(tmp_path, [
        {"ship_id": "C-R5-01", "round": 5, "bucket": "config"},
        {"ship_id": "C-R5-02", "round": 5, "bucket": "prompt"},
    ])
    ledger.mark_ship_superseded(tmp_path, "C-R5-01", "C-R6-03")
    data = json.loads((ledger.data_dir(tmp_path) / "ship_outcomes.json").read_text())
    target = next(e for e in data if e["ship_id"] == "C-R5-01")
    other = next(e for e in data if e["ship_id"] == "C-R5-02")
    assert target["superseded_by"] == "C-R6-03"
    assert other.get("superseded_by") in (None, "")


def test_mark_ship_superseded_is_idempotent(tmp_path):
    _write_outcomes(tmp_path, [
        {"ship_id": "C-R5-01", "round": 5, "superseded_by": "C-R6-03"},
    ])
    # Second claim must NOT overwrite the first.
    ledger.mark_ship_superseded(tmp_path, "C-R5-01", "C-R7-01")
    data = json.loads((ledger.data_dir(tmp_path) / "ship_outcomes.json").read_text())
    assert data[0]["superseded_by"] == "C-R6-03"


def test_mark_ship_superseded_no_op_when_missing(tmp_path):
    _write_outcomes(tmp_path, [])
    # Should not raise when target doesn't exist
    ledger.mark_ship_superseded(tmp_path, "C-R5-99", "C-R6-01")


def test_ship_ledger_for_gate_keyed_by_ship_id(tmp_path):
    _write_outcomes(tmp_path, [
        {"ship_id": "C-R1-01", "round": 1, "bucket": "prompt", "hit_rate": "5/9", "superseded_by": None},
        {"ship_id": "C-R5-01", "round": 5, "bucket": "config", "hit_rate": "0/5", "superseded_by": "C-R6-03"},
    ])
    snap = ledger.ship_ledger_for_gate(tmp_path)
    assert set(snap.keys()) == {"C-R1-01", "C-R5-01"}
    assert snap["C-R5-01"]["superseded_by"] == "C-R6-03"
    assert snap["C-R1-01"]["bucket"] == "prompt"


def test_ship_ledger_for_gate_empty_when_no_ledger(tmp_path):
    assert ledger.ship_ledger_for_gate(tmp_path) == {}


def test_record_ship_outcome_initializes_superseded_by_none(tmp_path):
    ledger.record_ship_outcome(
        tmp_path,
        round_n=5,
        shipped_cid="C-R5-01",
        bucket="config",
        predicted_tasks=["t1"],
        rejected_sibling_cids=[],
    )
    data = json.loads((ledger.data_dir(tmp_path) / "ship_outcomes.json").read_text())
    assert data[0]["superseded_by"] is None
