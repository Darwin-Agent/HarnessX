import pytest
from pathlib import Path
from harnessx.aegis.stages.commit import run_stage_4


@pytest.mark.asyncio
async def test_stage_4_noops_on_broken_decision_chain(tmp_path):
    """If decision cites a candidate_id that isn't in candidates_info,
    Stage 4 returns no-op with reason 'decision_chain_broken:...'
    rather than silently continuing to the next candidate."""
    decision = {
        "decision_type": "ship",
        "ship_ranking": [{"candidate_id": "C-R1-99"}],  # doesn't exist
    }
    # Only C-R1-01 exists in candidates_info.
    manifest = tmp_path / "C-R1-01.md"
    manifest.write_text(
        "---\ncandidate_id: C-R1-01\nbucket: tools\nslot_type: regular\n"
        "file_changes:\n  - path: a.py\n    action: modify\n    diff_summary: x\n"
        "predicted_impact:\n  tasks_will_pass: [t1]\n  tasks_at_risk: []\n"
        "  confidence: 0.5\n---\n\n"
        "## Failure Evidence\n[trajectories/t1.jsonl#step_0]\n"
    )
    candidates_info = {"C-R1-01": (manifest, tmp_path / "applied.yaml")}
    archived: list = []
    result = await run_stage_4(
        round_n=1, decision=decision, candidates_info=candidates_info,
        refuted_signatures=set(),
        commit_fn=None,
        archive_fn=lambda cid, ctx: archived.append((cid, ctx)),
    )
    assert result["shipped_cid"] is None
    assert "decision_chain_broken" in (result["reason"] or "")
    assert any("decision_chain" in str(a[1]) for a in archived)
