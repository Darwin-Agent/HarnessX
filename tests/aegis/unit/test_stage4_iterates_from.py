# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""v0.9.5 — Stage 4 end-to-end for iterates_from candidates."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from harnessx.aegis.stages.commit import run_stage_4
from harnessx.aegis.gates.structure import GateResult


def _write_manifest(path: Path, cid: str, *, iterates_from: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fm_lines = [
        "---",
        f"candidate_id: {cid}",
        "bucket: config",
        "file_changes:",
        f"  - path: /scratch/{cid}.yaml",
        "    action: modify",
        "    diff_summary: revert",
        "predicted_impact:",
        "  tasks_will_pass: [t1]",
        "capability_evidence: []",
    ]
    if iterates_from:
        fm_lines.insert(3, f"iterates_from: {iterates_from}")
    md = "\n".join(fm_lines) + "\n---\n\n"
    md += (
        "## Failure Evidence\n"
        f"Target {iterates_from or 'none'} hit_rate=0/5; regressed: "
        "`trajectories/t1_r4.jsonl#step_1` — passed before; "
        "`trajectories/t1_r5.jsonl#step_3` — fails after.\n"
    )
    path.write_text(md)


def _write_applied(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"processors": []}))


@pytest.mark.asyncio
async def test_stage4_iterate_ships_and_returns_supersedes(tmp_path):
    """Iterate candidate passes all gates → stage4 returns supersedes pair."""
    manifest = tmp_path / "candidates" / "C-R6-03.md"
    applied = tmp_path / "applied" / "C-R6-03" / "config.yaml"
    _write_manifest(manifest, "C-R6-03", iterates_from="C-R5-01")
    _write_applied(applied)

    decision = {
        "decision_type": "ship",
        "ship_ranking": [{"candidate_id": "C-R6-03"}],
    }
    all_ok = {
        "structure": GateResult(True, ""),
        "novelty": GateResult(True, ""),
        "canonicalize": GateResult(True, ""),
        "replay": GateResult(True, ""),
    }
    with patch(
        "harnessx.aegis.stages.commit._run_all_gates",
        new=AsyncMock(return_value=all_ok),
    ):
        result = await run_stage_4(
            round_n=6,
            decision=decision,
            candidates_info={"C-R6-03": (manifest, applied)},
            refuted_signatures=set(),
            commit_fn=None,
            archive_fn=lambda cid, ctx: None,
            replay_model=None,
            prior_ships={"C-R5-01": {"round": 5, "bucket": "config"}},
        )

    assert result["shipped_cids"] == ["C-R6-03"]
    assert result.get("supersedes") == [{"new": "C-R6-03", "target": "C-R5-01"}]


@pytest.mark.asyncio
async def test_stage4_regular_candidate_has_empty_supersedes(tmp_path):
    manifest = tmp_path / "candidates" / "C-R6-01.md"
    applied = tmp_path / "applied" / "C-R6-01" / "config.yaml"
    _write_manifest(manifest, "C-R6-01", iterates_from=None)
    _write_applied(applied)

    decision = {
        "decision_type": "ship",
        "ship_ranking": [{"candidate_id": "C-R6-01"}],
    }
    all_ok = {
        "structure": GateResult(True, ""),
        "novelty": GateResult(True, ""),
        "canonicalize": GateResult(True, ""),
        "replay": GateResult(True, ""),
    }
    with patch(
        "harnessx.aegis.stages.commit._run_all_gates",
        new=AsyncMock(return_value=all_ok),
    ):
        result = await run_stage_4(
            round_n=6,
            decision=decision,
            candidates_info={"C-R6-01": (manifest, applied)},
            refuted_signatures=set(),
            commit_fn=None,
            archive_fn=lambda cid, ctx: None,
            replay_model=None,
        )

    assert result["shipped_cids"] == ["C-R6-01"]
    assert result.get("supersedes") == []
