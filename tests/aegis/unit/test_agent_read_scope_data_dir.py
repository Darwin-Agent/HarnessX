# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Planner/Evolver/Critic read-scope tests — cross-round ledgers accessible.

The Read-scope gate's contract is "block under blocked_roots, allow
otherwise (with allowed_files exceptions)". These tests assert that:
  * harnessx/ source is STILL blocked (no regression)
  * A candidate cross-round ledger path under <run_root>/data/ is NOT blocked
  * INDEX.md at <run_root>/INDEX.md is NOT blocked
  * peer briefs (this round's briefs_dir for Evolver; briefs_dir for Critic)
    remain blocked — independence guarantee stays
"""
from __future__ import annotations

from pathlib import Path

from harnessx.aegis.agents.planner import PlannerInputs, build_planner_harness
from harnessx.aegis.agents.evolver import EvolverInputs, build_evolver_harness
from harnessx.aegis.agents.critic import CriticInputs, build_critic_harness
from harnessx.aegis._paths import HARNESSX_SRC_ROOT


def _read_gate_config(cfg) -> dict:
    """Pull the read-scope gate entry out of a HarnessConfig's processor list.

    HarnessBuilder serialises processors as dicts with `_target_` pointing
    at the concrete class. We don't need a live instance — the blocked_roots
    + allowed_files lists are the whole contract.
    """
    for p in cfg.processors:
        if isinstance(p, dict) and "ReadScopeGateProcessor" in p.get("_target_", ""):
            return p
    raise AssertionError("no ReadScopeGateProcessor entry found in config")


def _is_blocked(gate_dict: dict, path) -> bool:
    from pathlib import Path
    resolved = Path(path).resolve()
    allowed = [Path(x).resolve() for x in gate_dict.get("allowed_files", [])]
    blocked = [Path(x).resolve() for x in gate_dict.get("blocked_roots", [])]
    if any(resolved == a for a in allowed):
        return False
    for root in blocked:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def test_planner_can_read_data_dir_and_index(tmp_path: Path):
    run_root = tmp_path / "run"
    (run_root / "R2" / "briefs").mkdir(parents=True)
    inputs = PlannerInputs(
        round=2,
        overview_path=run_root / "R2" / "summary.md",
        journal_path=run_root / "journal.md",
        archive_dir=run_root / "archive",
        current_config_path=run_root / "R2" / "config.yaml",
        landscape_path=run_root / "R2" / "landscape.md",
        digests_dir=run_root / "R2" / "digests",
        reputation_summary={},
        run_root=run_root,
    )
    for p in (inputs.overview_path, inputs.journal_path, inputs.current_config_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    cfg = build_planner_harness(inputs)
    gate = _read_gate_config(cfg)

    # data/*.jsonl under run root — should pass (not under blocked_roots)
    assert not _is_blocked(gate, run_root / "data" / "task_history.jsonl")
    assert not _is_blocked(gate, run_root / "data" / "ship_outcomes.json")
    assert not _is_blocked(gate, run_root / "INDEX.md")
    # Prior-round artifacts pass
    assert not _is_blocked(gate, run_root / "R1" / "summary.md")
    assert not _is_blocked(gate, run_root / "R0" / "digests" / "t1.md")
    # harnessx/ source still blocked
    assert _is_blocked(gate, HARNESSX_SRC_ROOT / "core" / "processor.py")


def test_evolver_can_read_data_dir_and_index(tmp_path: Path):
    run_root = tmp_path / "run"
    round_dir = run_root / "R3"
    (round_dir / "candidates").mkdir(parents=True)
    (round_dir / "applied").mkdir(parents=True)
    current_config = round_dir / "config.yaml"
    current_config.touch()
    landscape = round_dir / "landscape.md"
    landscape.touch()

    inputs = EvolverInputs(
        round=3,
        current_config_path=current_config,
        landscape_path=landscape,
        digests_dir=round_dir / "digests",
        trajectories_dir=round_dir / "trajectories",
        candidates_dir=round_dir / "candidates",
        applied_root=round_dir / "applied",
    )
    cfg = build_evolver_harness(inputs)
    gate = _read_gate_config(cfg)

    # Cross-round ledgers + INDEX — not blocked
    assert not _is_blocked(gate, run_root / "data" / "task_history.jsonl")
    assert not _is_blocked(gate, run_root / "data" / "rejected_candidates.jsonl")
    assert not _is_blocked(gate, run_root / "INDEX.md")
    # harnessx/aegis source still blocked
    assert _is_blocked(gate, HARNESSX_SRC_ROOT / "aegis" / "orchestrator.py")


def test_critic_can_read_data_dir_and_index(tmp_path: Path):
    run_root = tmp_path / "run"
    round_dir = run_root / "R4"
    candidates_dir = round_dir / "candidates"
    candidates_dir.mkdir(parents=True)
    (round_dir / "briefs").mkdir()
    current_config = round_dir / "config.yaml"
    current_config.touch()
    journal_path = run_root / "journal.md"
    journal_path.touch()

    inputs = CriticInputs(
        round=4,
        candidates_dir=candidates_dir,
        verdicts_dir=round_dir / "verdicts",
        decision_path=round_dir / "decision.md",
        digests_dir=round_dir / "digests",
        trajectories_dir=round_dir / "trajectories",
        sessions_dir=run_root / "sessions",
        journal_path=journal_path,
        current_config_path=current_config,
    )

    async def noop_runner(cid, q):
        return ""
    cfg = build_critic_harness(inputs, evolver_runner=noop_runner)
    gate = _read_gate_config(cfg)

    # Data + INDEX accessible
    assert not _is_blocked(gate,run_root / "data" / "ship_outcomes.json")
    assert not _is_blocked(gate,run_root / "data" / "rejected_candidates.jsonl")
    assert not _is_blocked(gate,run_root / "INDEX.md")

    # briefs/ still blocked (critic independence)
    assert _is_blocked(gate,round_dir / "briefs" / "B-R4-01.md")

    # harnessx source still blocked (except living-docs allow-list)
    assert _is_blocked(gate,HARNESSX_SRC_ROOT / "aegis" / "agents" / "critic.py")
