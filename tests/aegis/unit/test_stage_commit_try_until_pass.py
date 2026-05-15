import pytest
from unittest.mock import patch, AsyncMock
from pathlib import Path
from harnessx.aegis.stages.commit import run_stage_4, GateVerdict


@pytest.mark.asyncio
async def test_first_candidate_passes_all_gates_ships(tmp_path):
    # setup candidate manifest files that parse_candidate_manifest can read
    cand1 = tmp_path / "C-R5-01.md"
    cand1.write_text(
        "---\ncandidate_id: C-R5-01\nslot_type: regular\nbucket: tools\n"
        "file_changes:\n  - path: a.py\n    diff_sha_after: abc1\n"
        "predicted_impact:\n  tasks_will_pass: [t1]\n  tasks_at_risk: []\n"
        "  confidence: 0.5\n---\n\n## Failure Evidence\n[trajectories/t1.jsonl#step_0]\n"
    )
    cfg1 = tmp_path / "config_a.yaml"
    cfg1.write_text("processors: []\n")

    cand2 = tmp_path / "C-R5-02.md"
    cand2.write_text(cand1.read_text().replace("C-R5-01", "C-R5-02").replace("abc1", "abc2"))
    cfg2 = tmp_path / "config_b.yaml"
    cfg2.write_text("processors: []\n")

    decision = {
        "decision_type": "ship",
        "ship_ranking": [
            {"candidate_id": "C-R5-01"},
            {"candidate_id": "C-R5-02"},
        ],
    }
    candidates = {
        "C-R5-01": (cand1, cfg1),
        "C-R5-02": (cand2, cfg2),
    }

    async def all_pass(*a, **kw):
        return {"structure": GateVerdict(True, ""), "novelty": GateVerdict(True, ""),
                "canonicalize": GateVerdict(True, ""), "replay": GateVerdict(True, "")}

    with patch("harnessx.aegis.stages.commit._run_all_gates", new=all_pass):
        result = await run_stage_4(
            round_n=5, decision=decision,
            candidates_info=candidates,
            refuted_signatures=set(),
            commit_fn=None,
            archive_fn=lambda cid, ctx: None,
        )
    assert result["shipped_cid"] == "C-R5-01"


@pytest.mark.asyncio
async def test_first_candidate_fails_fallback_to_second(tmp_path):
    cand1 = tmp_path / "C-R5-01.md"
    cand1.write_text(
        "---\ncandidate_id: C-R5-01\nslot_type: regular\nbucket: tools\n"
        "file_changes:\n  - path: a.py\n    diff_sha_after: abc1\n"
        "predicted_impact:\n  tasks_will_pass: [t1]\n  tasks_at_risk: []\n"
        "  confidence: 0.5\n---\n\n## Failure Evidence\n[trajectories/t1.jsonl#step_0]\n"
    )
    cfg1 = tmp_path / "config_a.yaml"; cfg1.write_text("processors: []\n")
    cand2 = tmp_path / "C-R5-02.md"
    cand2.write_text(cand1.read_text().replace("C-R5-01", "C-R5-02").replace("abc1", "abc2"))
    cfg2 = tmp_path / "config_b.yaml"; cfg2.write_text("processors: []\n")

    decision = {
        "decision_type": "ship",
        "ship_ranking": [
            {"candidate_id": "C-R5-01"},
            {"candidate_id": "C-R5-02"},
        ],
    }
    candidates = {"C-R5-01": (cand1, cfg1), "C-R5-02": (cand2, cfg2)}

    async def mixed_gate(cid, *a, **kw):
        if cid == "C-R5-01":
            return {"structure": GateVerdict(True, ""),
                    "novelty": GateVerdict(False, "refuted"),
                    "canonicalize": GateVerdict(True, ""),
                    "replay": GateVerdict(True, "")}
        return {"structure": GateVerdict(True, ""),
                "novelty": GateVerdict(True, ""),
                "canonicalize": GateVerdict(True, ""),
                "replay": GateVerdict(True, "")}

    archived = []
    with patch("harnessx.aegis.stages.commit._run_all_gates", new=mixed_gate):
        result = await run_stage_4(
            round_n=5, decision=decision,
            candidates_info=candidates,
            refuted_signatures=set(),
            commit_fn=None,
            archive_fn=lambda cid, ctx: archived.append(cid),
        )
    assert result["shipped_cid"] == "C-R5-02"
    assert "C-R5-01" in archived


@pytest.mark.asyncio
async def test_stage4_short_circuits_after_structure_failure(tmp_path):
    """E6: when a cheap gate fails, canonicalize+replay must be marked
    'skipped: earlier gate failed' rather than actually executed."""
    from harnessx.aegis.stages.commit import _run_all_gates

    # Candidate manifest that will fail the structure gate (no Failure
    # Evidence section on a regular slot, zero anchors in body).
    cand = tmp_path / "C-R1-01.md"
    cand.write_text(
        "---\ncandidate_id: C-R1-01\nslot_type: regular\nbucket: tools\n"
        "file_changes:\n  - path: a.py\n    diff_sha_after: abc\n"
        "predicted_impact:\n  tasks_will_pass: [t1]\n  tasks_at_risk: []\n"
        "  confidence: 0.5\n---\n\n## Nothing here\n"
    )
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("processors: []\n")

    replay_called = {"n": 0}

    async def fail_if_called(*a, **kw):
        replay_called["n"] += 1
        return True

    from unittest.mock import patch
    with patch(
        "harnessx.aegis.stages.commit.check_replay_smoke",
        new=fail_if_called,
    ):
        results = await _run_all_gates(
            "C-R1-01", cand, cfg, refuted_signatures=set(),
            replay_model=object(),  # non-None so replay would run if reached
        )

    assert not results["structure"].ok
    assert results["canonicalize"].reason == "skipped: earlier gate failed"
    assert results["replay"].reason == "skipped: earlier gate failed"
    assert replay_called["n"] == 0, (
        "replay gate should not have been invoked after structure failed"
    )


@pytest.mark.asyncio
async def test_all_candidates_fail_noop(tmp_path):
    cand = tmp_path / "C-R5-01.md"
    cand.write_text(
        "---\ncandidate_id: C-R5-01\nslot_type: regular\nbucket: tools\n"
        "file_changes:\n  - path: a.py\n    diff_sha_after: abc1\n"
        "predicted_impact:\n  tasks_will_pass: [t1]\n  tasks_at_risk: []\n"
        "  confidence: 0.5\n---\n\n## Failure Evidence\n[trajectories/t1.jsonl#step_0]\n"
    )
    cfg = tmp_path / "cfg.yaml"; cfg.write_text("processors: []\n")

    decision = {"decision_type": "ship", "ship_ranking": [{"candidate_id": "C-R5-01"}]}
    candidates = {"C-R5-01": (cand, cfg)}

    async def fail_gate(cid, *a, **kw):
        return {"structure": GateVerdict(False, "bad anchor"),
                "novelty": GateVerdict(True, ""),
                "canonicalize": GateVerdict(True, ""),
                "replay": GateVerdict(True, "")}

    archived = []
    with patch("harnessx.aegis.stages.commit._run_all_gates", new=fail_gate):
        result = await run_stage_4(
            round_n=5, decision=decision,
            candidates_info=candidates,
            refuted_signatures=set(),
            commit_fn=None,
            archive_fn=lambda cid, ctx: archived.append(cid),
        )
    assert result["shipped_cid"] is None
    assert archived == ["C-R5-01"]
