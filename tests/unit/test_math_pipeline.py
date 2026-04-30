# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from harnessx.core.trajectory import (
    StatefulTrajectory,
    TrajectoryStep,
    FullStateSnapshot,
    StateDelta,
)
from harnessx.core.events import ToolResultEvent
from harnessx.rl.task import RLTask, NullPRM, EnhancedToolSuccessPRM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    step_id: int,
    reward: float = 0.0,
    tool_results: list[tuple[bool, str]] | None = None,
) -> TrajectoryStep:
    """Build a TrajectoryStep.

    tool_results: list of (is_error, result_str) tuples.
    is_error=True sets obs.error; "Error:..." result string also counts as error.
    """
    snap = FullStateSnapshot(
        step_id=step_id,
        messages=(),
        slots={},
        cumulative_tokens=0,
        cumulative_cost_usd=0.0,
    )
    delta = StateDelta(step_id=step_id, operations=())
    obs = []
    for is_error, result_str in tool_results or []:
        obs.append(
            ToolResultEvent(
                run_id="r1",
                step_id=step_id,
                tool_name="code_interpreter",
                tool_call_id="tc1",
                result=result_str,
                error="execution failed" if is_error else None,
            )
        )
    return TrajectoryStep(
        step_id=step_id,
        state_snapshot=snap,
        state_delta=delta,
        action=None,
        observation=obs,
        event=None,
        reward=reward,
    )


def _make_traj(rewards: list[float], tool_results_per_step: list | None = None) -> StatefulTrajectory:
    traj = StatefulTrajectory(run_id="test-traj")
    n = len(rewards)
    for i, r in enumerate(rewards):
        per_step = (tool_results_per_step or [None] * n)[i] or []
        traj.add_step(_make_step(i, reward=r, tool_results=per_step))
    return traj


class _FakeSample:
    """Minimal stand-in for slime.utils.types.Sample."""

    def __init__(self, prompt="", label="", metadata=None, response=""):
        self.prompt = prompt
        self.label = label
        self.metadata = metadata or {}
        self.response = response


# ===========================================================================
# MathTaskBuilder
# ===========================================================================


class TestMathTaskBuilder:
    def setup_method(self):
        from recipe.slime.math.builder import MathTaskBuilder

        self.builder = MathTaskBuilder()

    def test_build_from_str_prompt(self):
        sample = _FakeSample(prompt="What is 2+2?", label="4")
        task = self.builder.build(sample)
        assert isinstance(task, RLTask)
        assert task.description == "What is 2+2?"
        assert task.label == "4"
        assert task.task_type == "math"

    def test_build_from_list_prompt_user_role(self):
        sample = _FakeSample(
            prompt=[{"role": "user", "content": "Solve x^2=4"}],
            label="2",
        )
        task = self.builder.build(sample)
        assert task.description == "Solve x^2=4"
        assert task.label == "2"

    def test_build_from_list_prompt_with_system(self):
        """AIME format: system + user messages — system stored in metadata."""
        sample = _FakeSample(
            prompt=[
                {"role": "system", "content": "You are a math expert."},
                {"role": "user", "content": "Compute AIME 2024 Problem 1."},
            ],
            label="42",
        )
        task = self.builder.build(sample)
        assert task.description == "Compute AIME 2024 Problem 1."
        assert task.metadata.get("system_prompt") == "You are a math expert."

    def test_build_no_system_prompt_not_in_metadata(self):
        """str prompt — metadata should NOT contain system_prompt key."""
        sample = _FakeSample(prompt="2+2=?", label="4")
        task = self.builder.build(sample)
        assert "system_prompt" not in task.metadata

    def test_build_empty_label(self):
        sample = _FakeSample(prompt="x=?", label=None)
        task = self.builder.build(sample)
        assert task.label == ""


# ===========================================================================
# RetoolCompatPRM
# ===========================================================================


class TestRetoolCompatPRM:
    def setup_method(self):
        from recipe.slime.math.rewards import RetoolCompatPRM

        self.prm = RetoolCompatPRM()

    def test_is_terminal_only(self):
        assert self.prm.is_terminal_only is True

    @pytest.mark.asyncio
    async def test_positive_terminal_no_shaping(self):
        """Correct answer: terminal >= 0, no tool bonus applied."""
        # 4 tool-call turns — bonus would be large if shaping happened
        traj = _make_traj(
            rewards=[1.0, 1.0, 1.0, 1.0],
            tool_results_per_step=[
                [(False, "42")],
                [(False, "ok")],
                [(False, "ok")],
                [(False, "ok")],
            ],
        )
        scores = await self.prm.score_steps(traj)
        assert all(s == 1.0 for s in scores)

    @pytest.mark.asyncio
    async def test_negative_terminal_zero_tool_turns(self):
        """No tool calls → no bonus → raw terminal propagated."""
        traj = _make_traj(rewards=[-1.0, -1.0])
        scores = await self.prm.score_steps(traj)
        # (0 - 2) / 2 * 0.1 = -0.1 → terminal + (-0.1) = -1.1 → capped at min(-0.6, -1.1) = -1.1
        assert all(s <= 0 for s in scores)
        assert all(s == scores[0] for s in scores)

    @pytest.mark.asyncio
    async def test_negative_terminal_with_tool_turns_bonus(self):
        """4 tool-call turns with terminal=-1.0.

        bonus = (4-2)/2 * 0.1 = 0.1
        adjusted = min(-0.6, -1.0 + 0.1) = min(-0.6, -0.9) = -0.9
        """
        traj = _make_traj(
            rewards=[-1.0] * 4,
            tool_results_per_step=[
                [(False, "x=1")],
                [(False, "x=2")],
                [(False, "x=3")],
                [(False, "x=4")],
            ],
        )
        scores = await self.prm.score_steps(traj)
        expected = pytest.approx(-0.9, abs=1e-9)
        assert all(s == expected for s in scores)

    @pytest.mark.asyncio
    async def test_cap_at_minus_0_6(self):
        """Many tool turns: bonus cannot push score above -0.6."""
        traj = _make_traj(
            rewards=[-1.0] * 20,
            tool_results_per_step=[[(False, "ok")] for _ in range(20)],
        )
        scores = await self.prm.score_steps(traj)
        # min(-0.6, large_bonus) = -0.6
        assert all(s == pytest.approx(-0.6, abs=1e-9) for s in scores)

    @pytest.mark.asyncio
    async def test_empty_trajectory(self):
        traj = StatefulTrajectory(run_id="empty")
        scores = await self.prm.score_steps(traj)
        assert scores == []

    @pytest.mark.asyncio
    async def test_none_trajectory(self):
        scores = await self.prm.score_steps(None)
        assert scores == []

    @pytest.mark.asyncio
    async def test_all_steps_get_same_scalar(self):
        """is_terminal_only: all steps share the same adjusted scalar."""
        traj = _make_traj(
            rewards=[-1.0] * 3,
            tool_results_per_step=[[(False, "1")], [(False, "2")], [(False, "3")]],
        )
        scores = await self.prm.score_steps(traj)
        assert len(set(scores)) == 1  # all identical


# ===========================================================================
# math_format_reward
# ===========================================================================


class TestMathFormatReward:
    def setup_method(self):
        from recipe.slime.math.rewards import math_format_reward

        self.fn = math_format_reward

    def _eval_result(self, reward: float, pred: str = ""):
        from harnessx.core.events import EvalResult

        return EvalResult(passed=reward >= 0, score=reward, reason=pred, reward=reward)

    def test_boxed_wrong_answer_gets_bonus(self):
        sample = _FakeSample(response=r"The answer is \boxed{42}.")
        result = self.fn(sample, self._eval_result(-1.0, pred="42"), None)
        assert result["score_delta"] == pytest.approx(0.1)
        assert result["has_boxed_answer"] == 1

    def test_boxed_correct_answer_no_bonus(self):
        """Correct answer: no format bonus (terminal >= 0)."""
        sample = _FakeSample(response=r"Therefore \boxed{4}.")
        result = self.fn(sample, self._eval_result(1.0, pred="4"), None)
        assert result["score_delta"] == pytest.approx(0.0)
        assert result["has_boxed_answer"] == 1

    def test_no_boxed_no_bonus(self):
        sample = _FakeSample(response="The answer is 42.")
        result = self.fn(sample, self._eval_result(-1.0, pred="42"), None)
        assert result["score_delta"] == pytest.approx(0.0)
        assert result["has_boxed_answer"] == 0

    def test_pred_extracted_from_eval_result(self):
        sample = _FakeSample(response=r"\boxed{7}")
        result = self.fn(sample, self._eval_result(-1.0, pred="7"), None)
        assert result["pred"] == "7"

    def test_none_eval_result_treated_as_negative_terminal(self):
        """eval_result=None → terminal=-1.0 (default)."""
        sample = _FakeSample(response=r"\boxed{x}")
        result = self.fn(sample, None, None)
        assert result["score_delta"] == pytest.approx(0.1)

    def test_format_score_equals_score_delta(self):
        sample = _FakeSample(response=r"\boxed{3}")
        result = self.fn(sample, self._eval_result(-1.0), None)
        assert result["format_score"] == result["score_delta"]

    def test_returns_all_required_keys(self):
        sample = _FakeSample(response="")
        result = self.fn(sample, self._eval_result(1.0), None)
        assert {
            "score_delta",
            "format_score",
            "pred",
            "has_boxed_answer",
        } <= result.keys()


# ===========================================================================
# NullPRM (harnessx.rl.task)
# ===========================================================================


class TestNullPRM:
    def setup_method(self):
        self.prm = NullPRM()

    def test_is_terminal_only(self):
        assert self.prm.is_terminal_only is True

    @pytest.mark.asyncio
    async def test_returns_terminal_per_step(self):
        traj = _make_traj(rewards=[1.0, 1.0, 1.0])
        traj.steps[-1].reward = 0.5  # backfilled terminal
        scores = await self.prm.score_steps(traj)
        # NullPRM returns each step's .reward as-is
        assert scores[-1] == pytest.approx(0.5)
        assert len(scores) == 3

    @pytest.mark.asyncio
    async def test_empty(self):
        assert await self.prm.score_steps(None) == []
        assert await self.prm.score_steps(StatefulTrajectory(run_id="e")) == []


# ===========================================================================
# EnhancedToolSuccessPRM (harnessx.rl.task)
# ===========================================================================


class TestEnhancedToolSuccessPRM:
    def setup_method(self):
        self.prm = EnhancedToolSuccessPRM(
            success_bonus=0.05,
            error_penalty=0.10,
            loop_penalty=0.20,
        )

    def test_is_NOT_terminal_only(self):
        assert self.prm.is_terminal_only is False

    @pytest.mark.asyncio
    async def test_success_bonus_added(self):
        """Successful tool call: step.reward + success_bonus."""
        traj = _make_traj(
            rewards=[0.0],
            tool_results_per_step=[[(False, "output")]],
        )
        scores = await self.prm.score_steps(traj)
        assert scores[0] == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_error_via_obs_error_field(self):
        """obs.error set → error_penalty deducted."""
        traj = _make_traj(
            rewards=[0.0],
            tool_results_per_step=[[(True, "")]],  # is_error=True
        )
        scores = await self.prm.score_steps(traj)
        assert scores[0] == pytest.approx(-0.10)

    @pytest.mark.asyncio
    async def test_error_via_result_string(self):
        """obs.result starting with 'Error:' → error_penalty (obs.error may be None)."""
        # is_error=False (no obs.error) but result starts with "Error:"
        step = _make_step(0, reward=0.0, tool_results=[(False, "Error: ZeroDivisionError")])
        traj = StatefulTrajectory(run_id="t")
        traj.add_step(step)
        scores = await self.prm.score_steps(traj)
        assert scores[0] == pytest.approx(-0.10)

    @pytest.mark.asyncio
    async def test_loop_penalty_on_last_step(self):
        """loop_detected exit_reason adds loop_penalty on the final step."""
        traj = _make_traj(rewards=[0.0, 0.0])
        scores = await self.prm.score_steps(traj, exit_reason="loop_detected")
        assert scores[-1] == pytest.approx(-0.20)
        assert scores[0] == pytest.approx(0.0)  # no loop penalty on non-last

    @pytest.mark.asyncio
    async def test_no_loop_penalty_on_normal_exit(self):
        traj = _make_traj(rewards=[0.0])
        scores = await self.prm.score_steps(traj, exit_reason="done")
        assert scores[0] == pytest.approx(0.0)

    def test_aggregate_positive_terminal(self):
        step_rewards = [1.05, 1.10, 1.00]  # deltas above terminal=1.0
        score = self.prm.aggregate(terminal=1.0, step_rewards=step_rewards)
        # bonuses = 0.05 + 0.10 = 0.15 (cap 0.30) → 1.0 + 0.15 = 1.15
        assert score == pytest.approx(1.15, abs=1e-9)

    def test_aggregate_negative_terminal_clamped(self):
        """neg_floor=-1.10, neg_ceil=-0.60: result clamped to that range."""
        step_rewards = [-1.05, -1.15]  # penalty deltas
        score = self.prm.aggregate(terminal=-1.0, step_rewards=step_rewards)
        assert -1.10 <= score <= -0.60

    def test_aggregate_empty(self):
        assert self.prm.aggregate(terminal=0.5, step_rewards=[]) == pytest.approx(0.5)
