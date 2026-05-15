import json
from pathlib import Path
from unittest.mock import MagicMock

from harnessx.aegis.orchestrator import AegisOrchestrator
from harnessx.aegis.data.scoreboard import Scoreboard, ShipRecord


def test_orchestrator_initialises_empty_scoreboard(tmp_path: Path):
    o = AegisOrchestrator(
        run_dir=tmp_path, num_evolvers=1, model_config=MagicMock(),
    )
    assert (tmp_path / "scoreboard.json").exists() is False
    assert o.scoreboard.ships == []
    assert o.scoreboard.to_dict()["by_bucket"] == {}


def test_orchestrator_loads_existing_scoreboard(tmp_path: Path):
    sb = Scoreboard()
    sb.add_ship(ShipRecord(
        cid="C-R1-01", round=1, bucket="processor",
        predicted_tasks=("t1",),
        flipped_in_ship_round=("t1",),
    ))
    sb.save(tmp_path / "scoreboard.json")
    o = AegisOrchestrator(
        run_dir=tmp_path, num_evolvers=1, model_config=MagicMock(),
    )
    assert len(o.scoreboard.ships) == 1
    assert o.scoreboard.ships[0].cid == "C-R1-01"


def test_orchestrator_persists_scoreboard_on_finalize(tmp_path: Path):
    o = AegisOrchestrator(
        run_dir=tmp_path, num_evolvers=1, model_config=MagicMock(),
    )
    o.scoreboard.add_ship(ShipRecord(
        cid="C-R2-01", round=2, bucket="tools",
        predicted_tasks=("t2",),
        flipped_in_ship_round=(),
    ))
    o._persist_state()
    assert (tmp_path / "scoreboard.json").exists()
    loaded = json.loads((tmp_path / "scoreboard.json").read_text())
    assert loaded["by_bucket"]["tools"]["ships"] == 1
