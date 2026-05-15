import json
from pathlib import Path
from harnessx.aegis.stages.preprocess import aggregate_digests, classify_pattern


def test_classify_pattern_all_fail():
    assert classify_pattern([False, False]) == "ALL_FAIL"


def test_classify_pattern_all_pass():
    assert classify_pattern([True, True]) == "ALL_PASS"


def test_classify_pattern_partial():
    assert classify_pattern([True, False]) == "PARTIAL_PASS"
    assert classify_pattern([False, True]) == "PARTIAL_PASS"


def test_classify_single_rollout():
    assert classify_pattern([True]) == "PASS"
    assert classify_pattern([False]) == "FAIL"


def test_aggregate_produces_flat_overview_no_forced_clustering(tmp_path):
    """Forced-by-string clustering was removed — each digest gets its own
    row. The Planner synthesises real cross-trace themes downstream."""
    digests = tmp_path / "digests"
    digests.mkdir()
    (digests / "task_01.md").write_text(
        "pattern: ALL_FAIL\n"
        "failure_mode: tool_order\n"
        "observation [trajectories/task_01_r1.jsonl#step_5]\n"
    )
    (digests / "task_02.md").write_text(
        "pattern: ALL_FAIL\n"
        "failure_mode: tool_order\n"
        "observation [trajectories/task_02_r1.jsonl#step_3]\n"
    )
    (digests / "task_03.md").write_text(
        "pattern: ALL_PASS\n"
        "strategy: read_before_write\n"
        "observation [trajectories/task_03_r1.jsonl#step_2]\n"
    )

    summary_path = tmp_path / "summary.md"
    aggregate_digests(digests_dir=digests, summary_path=summary_path)

    assert summary_path.exists()
    summary = summary_path.read_text()
    # All three task_ids appear in the flat overview.
    assert "task_01" in summary and "task_02" in summary and "task_03" in summary
    assert "tool_order" in summary
    # Each digest gets its own line in the per-task index — no forced
    # clustering is produced.
    assert summary.count("pattern=ALL_FAIL") == 2
    assert summary.count("pattern=ALL_PASS") == 1
