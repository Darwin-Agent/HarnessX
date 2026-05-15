"""Evolver mode switching.

Two modes:
- Normal: Evolver writes K candidates + their scratch dirs (K chosen by agent).
- Ask-more: Critic calls the Evolver with a specific candidate_id to
  clarify. No new scratch dir, single-file write at ask_more_candidate_path.
"""
from harnessx.aegis.agents.evolver import build_evolver_harness, EvolverInputs


def _make_normal_inputs(tmp_path):
    return EvolverInputs(
        round=1,
        current_config_path=tmp_path / "config.yaml",
        landscape_path=tmp_path / "landscape.md",
        digests_dir=tmp_path / "digests",
        trajectories_dir=tmp_path / "trajectories",
        candidates_dir=tmp_path / "candidates",
        applied_root=tmp_path / "applied",
    )


def _make_askmore_inputs(tmp_path, scratch_path):
    inp = _make_normal_inputs(tmp_path)
    return EvolverInputs(
        round=inp.round,
        current_config_path=inp.current_config_path,
        landscape_path=inp.landscape_path,
        digests_dir=inp.digests_dir,
        trajectories_dir=inp.trajectories_dir,
        candidates_dir=inp.candidates_dir,
        applied_root=inp.applied_root,
        ask_more_brief_path=tmp_path / "real_manifest.md",
        ask_more_candidate_id="C-R1-01",
        ask_more_candidate_path=scratch_path,
    )


def test_askmore_evolver_write_scope_does_not_include_manifest(tmp_path):
    scratch = tmp_path / "askmore_xyz.md"
    real_manifest = tmp_path / "real_manifest.md"
    real_manifest.write_text("preserved")

    inputs = _make_askmore_inputs(tmp_path, scratch)
    cfg = build_evolver_harness(inputs)
    for p in (cfg.processors or []):
        if "WriteScopeGate" in p.get("_target_", ""):
            allowed_files = set(str(x) for x in p.get("allowed_files", []))
            # Scratch path is the only allowed write target.
            assert str(scratch.resolve()) in allowed_files
            assert str(real_manifest.resolve()) not in allowed_files
            return
    raise AssertionError("WriteScopeGate not found")


def test_askmore_template_omits_normal_branch(tmp_path):
    scratch = tmp_path / "askmore_xyz.md"
    inputs = _make_askmore_inputs(tmp_path, scratch)
    cfg = build_evolver_harness(inputs)
    for p in (cfg.processors or []):
        if "SystemPromptProcessor" in p.get("_target_", ""):
            text = p.get("system_builder", {}).get("text", "")
            # Must render the ask-more branch rather than leak the
            # normal Evolver's K-candidate instructions.
            assert "Evolver (ask-more)" in text
            # Must NOT advertise the candidates/applied scratch output
            # (those are the normal-branch semantics).
            assert "K candidate manifests" not in text
            return
    raise AssertionError("system prompt not found")


def test_normal_evolver_template_keeps_k_candidate_instructions(tmp_path):
    inputs = _make_normal_inputs(tmp_path)
    cfg = build_evolver_harness(inputs)
    for p in (cfg.processors or []):
        if "SystemPromptProcessor" in p.get("_target_", ""):
            text = p.get("system_builder", {}).get("text", "")
            assert "Evolver (ask-more)" not in text
            # Normal branch instructs K-candidate production.
            assert "K candidate manifests" in text or "K candidates" in text or "K ≥" in text
            return
    raise AssertionError("system prompt not found")
