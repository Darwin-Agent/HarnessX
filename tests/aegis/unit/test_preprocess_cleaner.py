import json
from pathlib import Path
from harnessx.aegis.stages.preprocess import clean_trajectory


def _write_raw(path: Path, events: list[dict]) -> None:
    with path.open("w") as fp:
        for e in events:
            fp.write(json.dumps(e) + "\n")


def test_cleaner_dedups_repeated_tool_output(tmp_path):
    raw = tmp_path / "task_01_r1.jsonl.raw"
    _write_raw(raw, [
        {"type": "tool_result", "tool": "Grep", "output": "hit1"},
        {"type": "tool_result", "tool": "Grep", "output": "hit1"},
        {"type": "tool_result", "tool": "Grep", "output": "hit2"},
    ])
    out = tmp_path / "clean.jsonl"
    clean_trajectory(raw, out)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2


def test_cleaner_externalizes_large_content(tmp_path):
    raw = tmp_path / "task_01_r1.jsonl.raw"
    big = "x" * 5000
    _write_raw(raw, [{"type": "message", "content": big}])
    out = tmp_path / "clean.jsonl"
    clean_trajectory(raw, out, externalize_threshold=2048,
                     media_dir=tmp_path / "media")
    clean = json.loads(out.read_text().strip())
    assert "content_ref" in clean
    assert "content" not in clean or len(clean.get("content", "")) < 2048


def test_cleaner_preserves_structure(tmp_path):
    raw = tmp_path / "task_01_r1.jsonl.raw"
    _write_raw(raw, [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "tool_call", "tool": "Bash", "input": {"cmd": "ls"}},
        {"type": "tool_result", "tool": "Bash", "output": "file"},
    ])
    out = tmp_path / "clean.jsonl"
    clean_trajectory(raw, out)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 3
    assert json.loads(lines[1])["tool"] == "Bash"
