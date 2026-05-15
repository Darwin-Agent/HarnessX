import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from harnessx.aegis.orchestrator import AegisOrchestrator


@pytest.mark.asyncio
async def test_orchestrator_happy_path(tmp_path):
    orch = AegisOrchestrator(
        run_dir=tmp_path,
        num_evolvers=2,
        model_config=MagicMock(),
    )

    with patch("harnessx.aegis.orchestrator.run_stage_p", new=AsyncMock(
        return_value={"task_count": 2, "cluster_count": 1,
                      "actionability": 1.0, "actionability_reason": "mock"})), \
         patch("harnessx.aegis.orchestrator.run_stage_1", new=AsyncMock(
        return_value={"landscape_written": True,
                      "landscape_path": str(tmp_path / "R1" / "landscape.md"),
                      "frontmatter": {"round": 1, "top_themes": ["x"]}})), \
         patch("harnessx.aegis.orchestrator.run_stage_2", new=AsyncMock(
        return_value={"ok_count": 1,
                      "candidate_paths": [tmp_path / "C-R1-01.md"],
                      "results": [("C-R1-01", True, "")]})), \
         patch("harnessx.aegis.orchestrator.run_stage_3", new=AsyncMock(
        return_value={"decision": {"decision_type": "ship",
                                    "ship_ranking": [{"candidate_id": "C-R1-01"}]},
                      "decision_body": "ok", "critic_failed": False})), \
         patch("harnessx.aegis.orchestrator.run_stage_4", new=AsyncMock(
        return_value={"shipped_cid": "C-R1-01", "gate_results": {}, "reason": None})):

        result = await orch.run_round(
            round_n=1,
            raw_sessions_dir=tmp_path / "sessions",
            pass_flags_by_task={"t1": [False, False], "t2": [True, True]},
            current_config_path=tmp_path / "config.yaml",
        )
    assert result["shipped_cid"] == "C-R1-01"
    audit_path = tmp_path / "audit.jsonl"
    assert audit_path.exists()
    assert audit_path.stat().st_size > 0
