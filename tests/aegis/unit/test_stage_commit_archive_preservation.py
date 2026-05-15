# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""I1 — when an earlier candidate fails a gate and a later candidate ships,
the earlier candidate's archive context must retain the gate-failure dict
rather than being overwritten by ``{"reason": "not_selected"}``.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from harnessx.aegis.stages.commit import run_stage_4


@pytest.mark.asyncio
async def test_earlier_gate_failure_context_preserved_on_later_ship(tmp_path):
    """If C1 fails structure gate, then C2 ships, C1's archive ctx must
    retain the structure-failure details — not be overwritten with
    'not_selected'."""
    manifest_1 = tmp_path / "C-R1-01.md"
    manifest_1.write_text(
        "---\ncandidate_id: C-R1-01\nbucket: prompt\nslot_type: regular\n"
        "file_changes: [{path: a.py, action: modify, diff_summary: x}]\n"
        "predicted_impact: {tasks_will_pass: [], tasks_at_risk: [], confidence: 0.5}\n---\n\n"
        "## Failure Evidence\n[trajectories/t1.jsonl#step_0]\n"
    )
    manifest_2 = tmp_path / "C-R1-02.md"
    manifest_2.write_text(
        "---\ncandidate_id: C-R1-02\nbucket: tools\nslot_type: regular\n"
        "file_changes: [{path: b.py, action: modify, diff_summary: y}]\n"
        "predicted_impact: {tasks_will_pass: [], tasks_at_risk: [], confidence: 0.5}\n"
        "capability_evidence: []\n---\n\n"
        "## Failure Evidence\n[trajectories/t1.jsonl#step_0]\n"
    )

    cfg_yaml = tmp_path / "cfg.yaml"
    from harnessx.core.builder import HarnessBuilder
    HarnessBuilder().build().to_yaml_file(cfg_yaml)

    decision = {
        "decision_type": "ship",
        "ship_ranking": [
            {"candidate_id": "C-R1-01"},  # tried first
            {"candidate_id": "C-R1-02"},  # shipped after C1 fails
        ],
    }
    candidates_info = {
        "C-R1-01": (manifest_1, cfg_yaml),
        "C-R1-02": (manifest_2, cfg_yaml),
    }

    calls: list[tuple[str, dict]] = []

    def fake_archive(cid, ctx):
        calls.append((cid, ctx))

    from harnessx.aegis.stages import commit as commit_mod
    real_run = commit_mod._run_all_gates

    async def patched(cid, manifest_path, candidate_config_path, refuted_signatures, *, replay_model=None, **kwargs):
        if cid == "C-R1-01":
            from harnessx.aegis.stages.commit import GateVerdict
            return {
                "structure": GateVerdict(ok=False, reason="fake structural fail"),
                "novelty": GateVerdict(ok=False, reason="skipped"),
                "canonicalize": GateVerdict(ok=False, reason="skipped"),
                "replay": GateVerdict(ok=False, reason="skipped"),
            }
        return await real_run(cid, manifest_path, candidate_config_path, refuted_signatures)

    with patch.object(commit_mod, "_run_all_gates", side_effect=patched):
        result = await run_stage_4(
            round_n=1, decision=decision,
            candidates_info=candidates_info,
            refuted_signatures=set(),
            commit_fn=None,
            archive_fn=fake_archive,
        )

    assert result["shipped_cid"] == "C-R1-02"
    # C-R1-01 should have been archived with gate-failure context, NOT
    # overwritten with 'not_selected'.
    c1_archivings = [ctx for cid, ctx in calls if cid == "C-R1-01"]
    assert len(c1_archivings) == 1, (
        f"C-R1-01 archived {len(c1_archivings)} times (should be 1)"
    )
    c1_ctx = c1_archivings[0]
    # The gate-failure ctx is the dict of GateVerdicts — it should NOT be
    # the 'not_selected' reason string.
    assert c1_ctx != {"reason": "not_selected"}, (
        f"C-R1-01 failure context was overwritten: {c1_ctx}"
    )
