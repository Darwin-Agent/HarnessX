# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Tests for StatefulTrajectory.to_markdown()."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from harnessx.core.events import (
    EvalResult,
    ModelResponseEvent,
    ToolCall,
    ToolResultEvent,
    Usage,
)
from harnessx.core.trajectory import (
    FullStateSnapshot,
    StatefulTrajectory,
    StateDelta,
    TrajectoryStep,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_step(
    step_id: int,
    *,
    thinking: str = "",
    content: str = "",
    tool_calls: tuple = (),
    observations: list | None = None,
) -> TrajectoryStep:
    snapshot = FullStateSnapshot(
        step_id=step_id,
        messages=(),
        slots={},
        cumulative_tokens=0,
        cumulative_cost_usd=0.0,
    )
    delta = StateDelta(step_id=step_id, operations=())
    action = ModelResponseEvent(
        run_id="r",
        step_id=step_id,
        content=content,
        thinking=thinking,
        tool_calls=tool_calls,
        usage=Usage(),
    )
    return TrajectoryStep(
        step_id=step_id,
        state_snapshot=snapshot,
        state_delta=delta,
        action=action,
        observation=observations or [],
        event=None,
    )


def _traj(*steps: TrajectoryStep) -> StatefulTrajectory:
    return StatefulTrajectory(run_id="run-1", steps=list(steps))


# ── level validation ─────────────────────────────────────────────────────────


class TestLevelValidation:
    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="level"):
            _traj().to_markdown(level="verbose")


# ── summary level ─────────────────────────────────────────────────────────────


class TestSummaryLevel:
    def test_summary_with_result(self):
        result = SimpleNamespace(
            exit_reason="done",
            total_steps=3,
            final_output="the answer",
            eval_result=EvalResult(passed=True, score=1.0, reason="ok"),
        )
        md = _traj(_make_step(1)).to_markdown(level="summary", result=result)
        assert "# Trajectory" in md
        assert "exit_reason: done" in md
        assert "total_steps: 3" in md
        assert "final_output: the answer" in md
        assert "eval_passed: True" in md
        assert "eval_score: 1.0" in md
        assert "eval_reason: ok" in md

    def test_summary_truncates_long_output(self):
        long_out = "x" * 1000
        result = SimpleNamespace(
            exit_reason="done",
            total_steps=1,
            final_output=long_out,
            eval_result=None,
        )
        md = _traj().to_markdown(level="summary", result=result)
        # Summary truncates to 500 chars.
        assert "x" * 500 in md
        assert "x" * 501 not in md

    def test_summary_omits_diagnostics_and_steps(self):
        """Summary level must not emit Diagnostics / Execution Steps."""
        traj = _traj(
            _make_step(
                1,
                content="hello",
                tool_calls=(ToolCall(id="1", name="Bash", input={"cmd": "ls"}),),
            )
        )
        md = traj.to_markdown(level="summary")
        assert "Diagnostics" not in md
        assert "Execution Steps" not in md
        assert "Bash" not in md

    def test_summary_no_result_falls_back_to_steps_count(self):
        traj = _traj(_make_step(1), _make_step(2))
        md = traj.to_markdown(level="summary")
        assert "total_steps: 2" in md


# ── full level ────────────────────────────────────────────────────────────────


class TestFullLevel:
    def test_full_emits_all_sections(self):
        task = SimpleNamespace(
            description="solve puzzle",
            task_id="T1",
            max_steps=10,
            max_cost_usd=1.0,
        )
        result = SimpleNamespace(
            exit_reason="done",
            total_steps=1,
            final_output="42",
            eval_result=EvalResult(passed=True, score=1.0, reason="correct"),
            total_tokens=123,
            total_cost_usd=0.05,
        )
        traj = _traj(
            _make_step(
                1,
                thinking="think",
                content="answer",
                tool_calls=(ToolCall(id="c1", name="Bash", input={"cmd": "ls"}),),
                observations=[
                    ToolResultEvent(
                        run_id="r",
                        step_id=1,
                        tool_name="Bash",
                        result="file.txt",
                        error=None,
                    )
                ],
            )
        )
        md = traj.to_markdown(level="full", task=task, result=result)
        assert "# Trajectory: T1" in md
        assert "## Task" in md
        assert "solve puzzle" in md
        assert "## Result" in md
        assert "## Diagnostics" in md
        assert "1/10 (10% budget)" in md
        assert "tokens: 123" in md
        assert "cost: $0.050/$1.00 (5% budget)" in md
        assert "## Execution Steps" in md
        assert "### Step 1" in md
        assert "#### Thinking" in md
        assert "#### Response" in md
        assert "**Bash**" in md
        assert "-> Bash: file.txt" in md

    def test_full_renders_tool_error_in_step(self):
        traj = _traj(
            _make_step(
                1,
                content="",
                observations=[
                    ToolResultEvent(
                        run_id="r",
                        step_id=1,
                        tool_name="Bash",
                        result="",
                        error="boom",
                    )
                ],
            )
        )
        md = traj.to_markdown(level="full")
        assert "-> Bash: ERROR: boom" in md

    def test_full_without_config_omits_harness_config(self):
        md = _traj(_make_step(1)).to_markdown(level="full")
        assert "## Harness Config" not in md

    def test_full_with_config_lists_processors_and_tools(self):
        class _FakeRegistry:
            def list_names(self):
                return ["Bash", "Read"]

        class _P:
            _singleton_group = "mygroup"
            _order = 50

        config = SimpleNamespace(
            processors={"*": [_P()]},
            tool_registry=_FakeRegistry(),
        )
        md = _traj(_make_step(1)).to_markdown(level="full", config=config)
        assert "## Harness Config" in md
        assert "mygroup(50)" in md
        assert "Tools: [Bash, Read]" in md

    def test_full_diagnostics_top_tools_sorted(self):
        traj = _traj(
            _make_step(
                1,
                observations=[
                    ToolResultEvent(run_id="r", step_id=1, tool_name="Bash", result="a"),
                    ToolResultEvent(run_id="r", step_id=1, tool_name="Bash", result="b"),
                    ToolResultEvent(run_id="r", step_id=1, tool_name="Read", result="c"),
                ],
            )
        )
        md = traj.to_markdown(level="full")
        assert "tool_calls: 3" in md
        assert "Bash(2)" in md
        assert "Read(1)" in md

    def test_full_diagnostics_error_rate(self):
        traj = _traj(
            _make_step(
                1,
                observations=[
                    ToolResultEvent(run_id="r", step_id=1, tool_name="Bash", result="", error="fail"),
                    ToolResultEvent(run_id="r", step_id=1, tool_name="Read", result="x"),
                ],
            )
        )
        md = traj.to_markdown(level="full")
        assert "errors: 1 (error_rate=50%)" in md

    def test_empty_trajectory_still_renders(self):
        md = _traj().to_markdown(level="full")
        assert "# Trajectory" in md
        # No execution steps section when there are no steps
        assert "Execution Steps" not in md
