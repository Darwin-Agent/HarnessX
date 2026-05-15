from pathlib import Path
from harnessx.aegis.agents.critic import build_critic_harness, CriticInputs


async def _null_runner(cid, q):
    return ""


def _find_read_gate_blocked_roots(cfg) -> list[Path]:
    """Locate the ReadScopeGateProcessor declaration in a HarnessConfig and
    return its blocked_roots list (as Path objects).

    build_critic_harness uses HarnessBuilder composition which produces a
    declarative ``cfg.processors`` list of ``{"_target_": ..., ...}`` dicts;
    ``_rt_procs`` is empty at build time. Locate the ReadScopeGate dict by its
    ``_target_`` suffix.
    """
    for p in cfg.processors if cfg.processors else []:
        if isinstance(p, dict) and p.get("_target_", "").endswith("ReadScopeGateProcessor"):
            return [Path(x) for x in p.get("blocked_roots", [])]
    raise AssertionError("ReadScopeGateProcessor not found in critic HarnessConfig")


def _make_inputs(tmp_path: Path) -> CriticInputs:
    for sub in ("briefs", "candidates", "digests", "trajectories", "sessions", "verdicts"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "journal.md").write_text("")
    (tmp_path / "config.yaml").write_text("")
    return CriticInputs(
        round=1,
        candidates_dir=tmp_path / "candidates",
        verdicts_dir=tmp_path / "verdicts",
        decision_path=tmp_path / "decision.md",
        digests_dir=tmp_path / "digests",
        trajectories_dir=tmp_path / "trajectories",
        sessions_dir=tmp_path / "sessions",
        journal_path=tmp_path / "journal.md",
        current_config_path=tmp_path / "config.yaml",
    )


def test_briefs_blocked_by_default(tmp_path):
    inputs = _make_inputs(tmp_path)
    cfg = build_critic_harness(inputs, _null_runner)
    blocked = _find_read_gate_blocked_roots(cfg)
    briefs_path = (tmp_path / "briefs").resolve()
    assert briefs_path in blocked


def test_briefs_unblocked_with_ablation_flag(tmp_path):
    inputs = _make_inputs(tmp_path)
    cfg = build_critic_harness(inputs, _null_runner, ablation_allow_briefs=True)
    blocked = _find_read_gate_blocked_roots(cfg)
    briefs_path = (tmp_path / "briefs").resolve()
    assert briefs_path not in blocked
