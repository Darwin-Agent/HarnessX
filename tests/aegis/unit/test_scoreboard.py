import json
from pathlib import Path
from harnessx.aegis.data.scoreboard import Scoreboard, ShipRecord


def _rec(cid, round_n, bucket, predicted, flipped):
    return ShipRecord(
        cid=cid, round=round_n, bucket=bucket,
        predicted_tasks=tuple(predicted),
        flipped_in_ship_round=tuple(flipped),
    )


def test_empty_scoreboard_has_version_and_zero_rollups():
    sb = Scoreboard()
    d = sb.to_dict()
    assert d["version"] == 1
    assert d["by_bucket"] == {}
    assert d["ships"] == []


def test_add_ship_updates_by_bucket_rollup():
    sb = Scoreboard()
    sb.add_ship(_rec("C-R1-01", 1, "processor",
                     predicted=["t1", "t2"], flipped=["t1", "t2"]))
    d = sb.to_dict()
    assert d["by_bucket"]["processor"]["ships"] == 1
    assert d["by_bucket"]["processor"]["predicted"] == 2
    assert d["by_bucket"]["processor"]["flipped"] == 2
    assert d["by_bucket"]["processor"]["hit_rate"] == 1.0


def test_multiple_ships_accumulate():
    sb = Scoreboard()
    sb.add_ship(_rec("C-R1-01", 1, "processor",
                     predicted=["t1", "t2"], flipped=["t1"]))
    sb.add_ship(_rec("C-R2-01", 2, "processor",
                     predicted=["t3"], flipped=["t3"]))
    d = sb.to_dict()
    assert d["by_bucket"]["processor"]["ships"] == 2
    assert d["by_bucket"]["processor"]["predicted"] == 3
    assert d["by_bucket"]["processor"]["flipped"] == 2
    assert d["by_bucket"]["processor"]["hit_rate"] == round(2 / 3, 4)


def test_add_ship_idempotent_on_cid():
    sb = Scoreboard()
    sb.add_ship(_rec("C-R1-01", 1, "processor",
                     predicted=["t1"], flipped=[]))
    sb.add_ship(_rec("C-R1-01", 1, "processor",
                     predicted=["t1"], flipped=["t1"]))  # retry with filled flipped
    d = sb.to_dict()
    assert len(d["ships"]) == 1
    assert d["by_bucket"]["processor"]["flipped"] == 1


def test_persist_and_reload_round_trips(tmp_path: Path):
    path = tmp_path / "scoreboard.json"
    sb = Scoreboard()
    sb.add_ship(_rec("C-R1-01", 1, "processor",
                     predicted=["t1"], flipped=["t1"]))
    sb.last_updated_round = 1
    sb.save(path)
    assert path.exists()

    loaded = Scoreboard.load(path)
    assert loaded.to_dict() == sb.to_dict()


def test_load_missing_returns_empty(tmp_path: Path):
    sb = Scoreboard.load(tmp_path / "nope.json")
    assert sb.to_dict()["version"] == 1
    assert sb.ships == []


def test_load_corrupt_returns_empty(tmp_path: Path):
    path = tmp_path / "scoreboard.json"
    path.write_text("{not-valid-json")
    sb = Scoreboard.load(path)
    assert sb.ships == []
