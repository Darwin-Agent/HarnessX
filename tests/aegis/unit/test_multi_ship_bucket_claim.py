# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Stage 4 bucket-claim logic: multiple candidates can ship in one round as
long as their `bucket` fields are disjoint."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from harnessx.aegis.stages.commit import run_stage_4
from harnessx.aegis.gates.structure import GateResult


def _write_manifest(path: Path, cid: str, bucket: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    md = (
        f"---\n"
        f"candidate_id: {cid}\n"
        f"bucket: {bucket}\n"
        f"file_changes:\n"
        f"  - path: /scratch/{cid}.py\n"
        f"    action: create\n"
        f"    diff_summary: x\n"
        f"predicted_impact:\n"
        f"  tasks_will_pass: [t1]\n"
        f"---\n\n"
        f"## Failure Evidence\n"
        f"- `trajectories/t1_r0.jsonl#step_1` — x\n"
    )
    path.write_text(md)


def _write_applied(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"processors": []}))


@pytest.mark.asyncio
async def test_two_candidates_different_buckets_both_ship(tmp_path):
    candidates_info = {}
    for cid, bucket in [("C-R1-01", "prompt"), ("C-R1-02", "tools")]:
        manifest = tmp_path / "candidates" / f"{cid}.md"
        applied = tmp_path / "applied" / cid / "config.yaml"
        _write_manifest(manifest, cid, bucket)
        _write_applied(applied)
        candidates_info[cid] = (manifest, applied)

    decision = {
        "decision_type": "ship",
        "ship_ranking": [
            {"candidate_id": "C-R1-01"},
            {"candidate_id": "C-R1-02"},
        ],
    }

    all_ok = {
        "structure": GateResult(True, ""),
        "novelty": GateResult(True, ""),
        "canonicalize": GateResult(True, ""),
        "replay": GateResult(True, ""),
    }
    archived = []
    with patch(
        "harnessx.aegis.stages.commit._run_all_gates",
        new=AsyncMock(return_value=all_ok),
    ):
        result = await run_stage_4(
            round_n=1,
            decision=decision,
            candidates_info=candidates_info,
            refuted_signatures=set(),
            commit_fn=None,
            archive_fn=lambda cid, ctx: archived.append((cid, ctx)),
            replay_model=None,
        )

    assert result["shipped_cids"] == ["C-R1-01", "C-R1-02"]
    assert result["shipped_by_bucket"] == {
        "prompt": "C-R1-01", "tools": "C-R1-02",
    }
    # Nothing archived — both shipped, no unsfhipped siblings.
    assert archived == []


@pytest.mark.asyncio
async def test_v093_same_bucket_multi_ship_is_legal(tmp_path):
    """v0.9.3: bucket-disjoint constraint removed. Two candidates in the
    same bucket can both ship as long as both pass their gates. This
    enables legitimate same-bucket multi-ship (e.g., two independent
    prompt micro-rules targeting different failure clusters). Conflict
    resolution is compose-layer (last-writer-wins per key)."""
    candidates_info = {}
    for cid, bucket in [("C-R1-01", "prompt"), ("C-R1-02", "prompt")]:
        manifest = tmp_path / "candidates" / f"{cid}.md"
        applied = tmp_path / "applied" / cid / "config.yaml"
        _write_manifest(manifest, cid, bucket)
        _write_applied(applied)
        candidates_info[cid] = (manifest, applied)

    decision = {
        "decision_type": "ship",
        "ship_ranking": [
            {"candidate_id": "C-R1-01"},
            {"candidate_id": "C-R1-02"},
        ],
    }
    all_ok = {
        "structure": GateResult(True, ""),
        "novelty": GateResult(True, ""),
        "canonicalize": GateResult(True, ""),
        "replay": GateResult(True, ""),
    }
    gate_calls = []

    async def tracking_gate(cid, *a, **kw):
        gate_calls.append(cid)
        return all_ok

    archived = []
    with patch(
        "harnessx.aegis.stages.commit._run_all_gates",
        side_effect=tracking_gate,
    ):
        result = await run_stage_4(
            round_n=1,
            decision=decision,
            candidates_info=candidates_info,
            refuted_signatures=set(),
            commit_fn=None,
            archive_fn=lambda cid, ctx: archived.append((cid, ctx)),
            replay_model=None,
        )

    # Both ship — no bucket-disjoint claim anymore.
    assert set(result["shipped_cids"]) == {"C-R1-01", "C-R1-02"}
    # Both gated.
    assert set(gate_calls) == {"C-R1-01", "C-R1-02"}


@pytest.mark.asyncio
async def test_second_candidate_ships_when_first_fails_gate(tmp_path):
    """If C1 fails replay gate, C2 (same or different bucket) should still
    get a chance."""
    candidates_info = {}
    for cid, bucket in [("C-R1-01", "prompt"), ("C-R1-02", "tools")]:
        manifest = tmp_path / "candidates" / f"{cid}.md"
        applied = tmp_path / "applied" / cid / "config.yaml"
        _write_manifest(manifest, cid, bucket)
        _write_applied(applied)
        candidates_info[cid] = (manifest, applied)

    decision = {
        "decision_type": "ship",
        "ship_ranking": [
            {"candidate_id": "C-R1-01"},
            {"candidate_id": "C-R1-02"},
        ],
    }
    first_fails = {
        "structure": GateResult(True, ""),
        "novelty": GateResult(True, ""),
        "canonicalize": GateResult(True, ""),
        "replay": GateResult(False, "broke"),
    }
    all_ok = {
        "structure": GateResult(True, ""),
        "novelty": GateResult(True, ""),
        "canonicalize": GateResult(True, ""),
        "replay": GateResult(True, ""),
    }
    call_num = {"i": 0}

    async def varying_gate(cid, *a, **kw):
        call_num["i"] += 1
        return first_fails if call_num["i"] == 1 else all_ok

    archived = []
    with patch(
        "harnessx.aegis.stages.commit._run_all_gates",
        side_effect=varying_gate,
    ):
        result = await run_stage_4(
            round_n=1,
            decision=decision,
            candidates_info=candidates_info,
            refuted_signatures=set(),
            commit_fn=None,
            archive_fn=lambda cid, ctx: archived.append((cid, ctx)),
            replay_model=None,
        )
    # C-R1-01 failed, C-R1-02 shipped.
    assert result["shipped_cids"] == ["C-R1-02"]
