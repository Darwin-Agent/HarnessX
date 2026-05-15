from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
from harnessx.aegis.gates.replay import check_replay_smoke


@pytest.mark.asyncio
async def test_replay_skipped_when_no_model_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("processors: []\ntool_registry:\n  builtin: []\n  custom: []\n")
    result = await check_replay_smoke(cfg_path)
    assert result.ok
    assert "skipped" in result.reason.lower()


@pytest.mark.asyncio
async def test_replay_delegates_to_meta_harness(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("processors: []\ntool_registry:\n  builtin: []\n  custom: []\n")

    with patch("harnessx.aegis.gates.replay._run_meta_replay", new=AsyncMock(return_value=True)):
        result = await check_replay_smoke(cfg_path, model_config=MagicMock())
    assert result.ok


@pytest.mark.asyncio
async def test_replay_failure_reported(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("processors: []\ntool_registry:\n  builtin: []\n  custom: []\n")

    with patch("harnessx.aegis.gates.replay._run_meta_replay",
               new=AsyncMock(side_effect=RuntimeError("boom"))):
        result = await check_replay_smoke(cfg_path, model_config=MagicMock())
    assert not result.ok
    assert "boom" in result.reason
