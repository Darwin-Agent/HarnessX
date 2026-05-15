# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""I2 — ``AegisAgent.evolve`` validates and coerces pass_flags_by_task
shape: scalar bool → [bool]; other types raise TypeError.
"""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from harnessx.aegis import AegisAgent


@pytest.mark.asyncio
async def test_pass_flags_scalar_bool_coerced(tmp_path):
    """Calling evolve with {tid: True} instead of {tid: [True]} is coerced,
    not an error. Makes the API less footgun-prone for external callers."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("processors: []\n")

    captured = {}

    async def capture_run_round(**kwargs):
        captured.update(kwargs)
        return {"shipped_cid": None, "reason": "test"}

    with patch("harnessx.aegis.AegisOrchestrator") as MockOrch:
        instance = MagicMock()
        instance.run_round = AsyncMock(side_effect=capture_run_round)
        MockOrch.return_value = instance
        agent = AegisAgent(num_evolvers=1, model_config=MagicMock())
        await agent.evolve(
            current_config=cfg_path,
            trajectories_dir=tmp_path,
            output_dir=tmp_path / "R1" / "evolve",
            pass_flags_by_task={"t1": True, "t2": [False, True]},  # mixed
            round_n=1,
        )

    assert captured["pass_flags_by_task"] == {"t1": [True], "t2": [False, True]}


@pytest.mark.asyncio
async def test_pass_flags_invalid_type_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("processors: []\n")
    with patch("harnessx.aegis.AegisOrchestrator"):
        agent = AegisAgent(num_evolvers=1, model_config=MagicMock())
        with pytest.raises(TypeError, match="must be bool or list"):
            await agent.evolve(
                current_config=cfg_path,
                trajectories_dir=tmp_path,
                output_dir=tmp_path / "R1" / "evolve",
                pass_flags_by_task={"t1": "yes"},  # wrong
                round_n=1,
            )
