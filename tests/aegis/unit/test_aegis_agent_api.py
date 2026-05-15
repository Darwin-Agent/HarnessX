from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harnessx.aegis import AegisAgent, compute_changeset


def test_aegis_agent_exposes_evolve(tmp_path):
    agent = AegisAgent(num_evolvers=4, budget_per_round_usd=5.0)
    assert hasattr(agent, "evolve")
    import inspect
    sig = inspect.signature(agent.evolve)
    assert {"current_config", "trajectories_dir", "output_dir"}.issubset(sig.parameters)


def test_compute_changeset_reexport():
    # re-exported from meta_harness for recipe compat
    assert callable(compute_changeset)


@pytest.mark.asyncio
async def test_evolve_returns_applied_yaml(tmp_path):
    """When orchestrator ships a candidate, evolve returns the applied
    HarnessConfig YAML path — not the manifest. The recipe then loads
    it via ``HarnessConfig.from_yaml_file``."""
    from harnessx.core.builder import HarnessBuilder

    run_dir = tmp_path / "run"
    output_dir = run_dir / "R1" / "evolve"
    output_dir.mkdir(parents=True, exist_ok=True)

    current_config_path = tmp_path / "config.yaml"
    current_config_path.write_text("version: 1\n")

    # Pre-create the shipped candidate's applied YAML where the Evolver
    # would normally write it during Stage 2.
    applied_dir = run_dir / "R1" / "applied" / "C-R1-01"
    applied_dir.mkdir(parents=True, exist_ok=True)
    shipped_yaml = applied_dir / "config.yaml"
    # Minimal valid HarnessConfig YAML so any downstream load/canonicalize
    # sanity checks still succeed.
    HarnessBuilder().build().to_yaml_file(shipped_yaml)

    agent = AegisAgent(num_evolvers=2, model_config=MagicMock())

    fake_run_round = AsyncMock(return_value={"shipped_cid": "C-R1-01", "reason": None})

    with patch("harnessx.aegis.AegisOrchestrator") as MockOrch:
        instance = MockOrch.return_value
        instance.run_round = fake_run_round
        result_path = await agent.evolve(
            current_config_path,
            trajectories_dir=tmp_path / "trajs",
            output_dir=output_dir,
            round_n=1,
        )

    assert result_path == shipped_yaml


@pytest.mark.asyncio
async def test_evolve_returns_config_when_no_ship(tmp_path):
    """If no candidate ships, evolve returns the original config path."""
    run_dir = tmp_path / "run"
    output_dir = run_dir / "R1" / "evolve"
    output_dir.mkdir(parents=True, exist_ok=True)

    current_config_path = tmp_path / "config.yaml"
    current_config_path.write_text("version: 1\n")

    agent = AegisAgent(num_evolvers=2, model_config=MagicMock())

    fake_run_round = AsyncMock(return_value={"shipped_cid": None, "reason": "critic_failed"})

    with patch("harnessx.aegis.AegisOrchestrator") as MockOrch:
        instance = MockOrch.return_value
        instance.run_round = fake_run_round
        result_path = await agent.evolve(
            current_config_path,
            trajectories_dir=tmp_path / "trajs",
            output_dir=output_dir,
            round_n=1,
        )

    assert result_path == current_config_path
