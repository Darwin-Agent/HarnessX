# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""C2: cross-round ship-follow-up check.

After a ship in round N, round N+1's Stage P compares the shipped
candidate's predicted_tasks_pass against the actual pass set. Any
mismatch is appended to summary.md as "Previous-round ship follow-up"
so the Planner sees "ship didn't take" evidence and doesn't reinforce
a no-op change.
"""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from harnessx.aegis.orchestrator import AegisOrchestrator
from harnessx.aegis.data.journal import Journal, RoundEntry


@pytest.mark.asyncio
async def test_summary_gets_followup_block_when_previous_ship_predictions_miss(tmp_path):
    # Pre-populate journal with a R1 ship entry that claimed 3 tasks would pass.
    orch = AegisOrchestrator(
        run_dir=tmp_path,
        num_evolvers=2, model_config=MagicMock(),
    )
    orch.journal.append(RoundEntry(
        round=1, action="ship", shipped_cid="C-R1-02",
        hypothesis_signatures=[], refuted_signatures=[],
        hit_rate=None, narrative="shipped C-R1-02",
        predicted_tasks_pass=["t1", "t2", "t3"],
    ))

    async def mock_p(**kw):
        # Simulate Stage P writing a summary.md
        kw["summary_path"].write_text("# Summary\n\nBaseline content.\n")
        return {
            "task_count": 3, "cluster_count": 1,
            "actionability": 1.0, "actionability_reason": "mock",
        }

    async def mock_s1(**kw):
        # Abort after Stage P so we don't need more plumbing — return a
        # "landscape not written" result that causes the orchestrator to
        # finalise the round early.
        return {"landscape_written": False, "landscape_path": "",
                "frontmatter": {}}

    with patch("harnessx.aegis.orchestrator.run_stage_p",
               new=AsyncMock(side_effect=mock_p)), \
         patch("harnessx.aegis.orchestrator.run_stage_1",
               new=AsyncMock(side_effect=mock_s1)):
        # Provide pass_flags showing t1 passed but t2 and t3 still fail
        await orch.run_round(
            round_n=2,
            raw_sessions_dir=tmp_path / "sessions",
            pass_flags_by_task={
                "t1": [True],
                "t2": [False],
                "t3": [False],
            },
            current_config_path=tmp_path / "cfg.yaml",
        )

    summary_text = (tmp_path / "R2" / "summary.md").read_text()
    assert "Previous-round ship follow-up" in summary_text
    assert "t2" in summary_text
    assert "t3" in summary_text
    # t1 passed so should NOT be listed as still-failing
    # (use anchor check to avoid substring false positive)
    assert "- t1\n" not in summary_text
    # Should explicitly tell Planner what to do
    assert "DIFFERENT bucket" in summary_text


@pytest.mark.asyncio
async def test_no_followup_block_when_previous_was_noop(tmp_path):
    orch = AegisOrchestrator(
        run_dir=tmp_path,
        num_evolvers=2, model_config=MagicMock(),
    )
    orch.journal.append(RoundEntry(
        round=1, action="no_op", shipped_cid=None,
        hypothesis_signatures=[], refuted_signatures=[],
        hit_rate=None, narrative="no-op",
        predicted_tasks_pass=[],
    ))

    async def mock_p(**kw):
        kw["summary_path"].write_text("# Summary\n\nBaseline content.\n")
        return {
            "task_count": 1, "cluster_count": 0,
            "actionability": 1.0, "actionability_reason": "mock",
        }

    async def mock_s1(**kw):
        return {"landscape_written": False, "landscape_path": "",
                "frontmatter": {}}

    with patch("harnessx.aegis.orchestrator.run_stage_p",
               new=AsyncMock(side_effect=mock_p)), \
         patch("harnessx.aegis.orchestrator.run_stage_1",
               new=AsyncMock(side_effect=mock_s1)):
        await orch.run_round(
            round_n=2,
            raw_sessions_dir=tmp_path / "sessions",
            pass_flags_by_task={"t1": [False]},
            current_config_path=tmp_path / "cfg.yaml",
        )

    summary_text = (tmp_path / "R2" / "summary.md").read_text()
    assert "Previous-round ship follow-up" not in summary_text
