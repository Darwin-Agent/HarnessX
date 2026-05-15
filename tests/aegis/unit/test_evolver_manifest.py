import pytest
from harnessx.aegis.agents.evolver import (
    EvolverInputs,
    build_evolver_harness,
    parse_candidate_manifest,
)


def test_parse_full_manifest():
    md = """---
candidate_id: C-R5-02
brief_id: B-R5-01
slot_type: regular
bucket: tools
fact_or_strategy: fact
novelty_dimension: lateral
file_changes:
  - path: harnessx/tools/ToolB.py
    action: modify
    diff_sha_after: abc123
predicted_impact:
  tasks_will_pass: [gaia_01]
  tasks_at_risk: []
  confidence: 0.7
---

## Failure Evidence
[trajectories/gaia_01_r1.jsonl#step_3]

## Root Cause
something
"""
    mf, body = parse_candidate_manifest(md)
    assert mf["candidate_id"] == "C-R5-02"
    assert mf["bucket"] == "tools"
    assert "Failure Evidence" in body


def test_parse_rejects_missing_frontmatter():
    md = "## Failure Evidence\nnope"
    with pytest.raises(ValueError):
        parse_candidate_manifest(md)


def test_evolver_write_scope_spans_candidates_and_applied(tmp_path):
    """The Evolver writes K candidate manifests + K applied scratch dirs in
    one session. Its write scope must cover both directories as roots."""
    from pathlib import Path

    candidates_dir = tmp_path / "candidates"
    applied_root = tmp_path / "applied"

    inputs = EvolverInputs(
        round=1,
        current_config_path=tmp_path / "config.yaml",
        landscape_path=tmp_path / "landscape.md",
        digests_dir=tmp_path / "digests",
        trajectories_dir=tmp_path / "trajectories",
        candidates_dir=candidates_dir,
        applied_root=applied_root,
    )
    cfg = build_evolver_harness(inputs)

    write_gate_dicts = [
        p for p in (cfg.processors or [])
        if isinstance(p, dict)
        and p.get("_target_", "").endswith("WriteScopeGateProcessor")
    ]
    assert write_gate_dicts, "WriteScopeGateProcessor not found"
    allowed_roots = {Path(p).resolve() for p in write_gate_dicts[0].get("allowed_roots") or []}
    assert candidates_dir.resolve() in allowed_roots
    assert applied_root.resolve() in allowed_roots
