# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Verify Evolver's read gate permits Reading the curated API reference set.

Failure modes this catches:
- Someone tightens the read gate and accidentally blocks living docs
- api_reference_files() returns an empty set (path resolution bug)
- A Reader processor mis-resolves the path
"""
from pathlib import Path

from harnessx.aegis._paths import api_reference_files, HARNESSX_SRC_ROOT


def test_api_reference_set_is_nonempty_and_in_harnessx():
    paths = api_reference_files()
    assert len(paths) >= 50, f"expected >=50 reference files, got {len(paths)}"
    src_root = str(HARNESSX_SRC_ROOT.resolve())
    for p in paths:
        assert p.startswith(src_root), f"{p} is outside {src_root}"
    # Spot-check canonical references are present.
    joined = "\n".join(paths)
    assert "core/processor.py" in joined
    assert "core/events.py" in joined
    assert "processors/control/cost_guard.py" in joined
    assert "tools/builtin/web_search.py" in joined


def test_api_reference_does_NOT_include_aegis_or_templates():
    """Prompt injection / introspection guard: agents must NOT be able to
    Read their own system prompts or the meta-agent's source."""
    paths = api_reference_files()
    for p in paths:
        assert "/aegis/" not in p, f"aegis source leaked: {p}"
        assert "/meta_harness/" not in p, f"legacy meta leaked: {p}"
        assert p.endswith(".py"), f"non-.py file leaked: {p}"


def test_evolver_read_gate_allows_api_reference(tmp_path: Path) -> None:
    """End-to-end: Evolver built with build_evolver_harness has a read gate
    whose allowed_files includes the API reference set."""
    from harnessx.aegis.agents.evolver import build_evolver_harness, EvolverInputs

    cfg = (tmp_path / "cfg.yaml")
    cfg.write_text("processors: []\n")
    (tmp_path / "landscape.md").write_text("# landscape\n")
    (tmp_path / "digests").mkdir()
    (tmp_path / "trajectories").mkdir()
    (tmp_path / "candidates").mkdir()
    (tmp_path / "applied").mkdir()
    inputs = EvolverInputs(
        round=1,
        current_config_path=cfg,
        landscape_path=tmp_path / "landscape.md",
        digests_dir=tmp_path / "digests",
        trajectories_dir=tmp_path / "trajectories",
        candidates_dir=tmp_path / "candidates",
        applied_root=tmp_path / "applied",
    )
    hc = build_evolver_harness(inputs)
    # Find the ReadScopeGate processor.
    read_gate_entry = None
    for p in (hc.processors or []):
        if isinstance(p, dict) and "ReadScopeGate" in p.get("_target_", ""):
            read_gate_entry = p
            break
    assert read_gate_entry is not None, "ReadScopeGate processor not found"
    allowed = set(str(x) for x in read_gate_entry.get("allowed_files", []))
    refs = set(api_reference_files())
    # Every API reference file should be in allowed_files.
    missing = refs - allowed
    assert not missing, f"{len(missing)} API reference files not in gate: {list(missing)[:3]}"
