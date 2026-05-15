import asyncio
import json
from pathlib import Path
from unittest.mock import patch
import pytest


@pytest.mark.asyncio
async def test_stage_p_end_to_end(tmp_path):
    raw_dir = tmp_path / "sessions"
    raw_dir.mkdir()
    (raw_dir / "task_01_r1.jsonl.raw").write_text(
        '{"type":"message","content":"hi"}\n'
        '{"type":"tool_result","tool":"Bash","output":"ok"}\n'
    )

    pass_flags = {"task_01": [False]}

    async def fake_run_digester(inputs, harness):
        inputs.digest_out_path.parent.mkdir(parents=True, exist_ok=True)
        inputs.digest_out_path.write_text(
            "pattern: FAIL\n"
            "failure_mode: tool_error\n"
            "observation [trajectories/task_01_r1.jsonl.raw#msg_0]\n"
        )

    from harnessx.aegis.stages.preprocess import run_stage_p
    with patch("harnessx.aegis.stages.preprocess._run_digester", new=fake_run_digester):
        result = await run_stage_p(
            raw_dir=raw_dir,
            trajectories_dir=tmp_path / "trajectories",
            digests_dir=tmp_path / "digests",
            summary_path=tmp_path / "summary.md",
            pass_flags_by_task=pass_flags,
            harness_factory=None,
            concurrency=2,
        )
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "digests" / "task_01.md").exists()
    summary = (tmp_path / "summary.md").read_text()
    # The per-task index should list the tool_error failure_mode tag.
    assert "tool_error" in summary
