# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
import pytest
from harnessx.processors.evaluation.strategies.evaluators.prm import (
    TerminalPRM,
    DiscountedPRM,
    ToolSuccessPRM,
    ProcessRewardModel,
)
from harnessx.core.trajectory import (
    StatefulTrajectory,
    TrajectoryStep,
    FullStateSnapshot,
    StateDelta,
)
from harnessx.core.events import ToolResultEvent


def _make_snapshot(step_id: int) -> FullStateSnapshot:
    return FullStateSnapshot(
        step_id=step_id,
        messages=(),
        slots={},
        cumulative_tokens=0,
        cumulative_cost_usd=0.0,
    )


def _make_delta(step_id: int) -> StateDelta:
    return StateDelta(step_id=step_id, operations=())


def _make_step(step_id: int, reward: float = 0.0, tool_errors: list[bool] | None = None) -> TrajectoryStep:
    obs = []
    for has_error in tool_errors or []:
        obs.append(
            ToolResultEvent(
                run_id="r1",
                step_id=step_id,
                tool_name="Bash",
                tool_call_id="tc1",
                result="",
                error="error!" if has_error else None,
            )
        )
    step = TrajectoryStep(
        step_id=step_id,
        state_snapshot=_make_snapshot(step_id),
        state_delta=_make_delta(step_id),
        action=None,
        observation=obs,
        event=None,
        reward=reward,
    )
    return step


def _make_trajectory(rewards: list[float], tool_errors_per_step: list[list[bool]] | None = None) -> StatefulTrajectory:
    traj = StatefulTrajectory(run_id="traj-test")
    for i, r in enumerate(rewards):
        errs = (tool_errors_per_step or [[]] * len(rewards))[i]
        traj.add_step(_make_step(i, reward=r, tool_errors=errs))
    return traj


class _DummyTask:
    description = "test"


# ─── TerminalPRM ──────────────────────────────────────────────────────────────


class TestPrm:
    @pytest.mark.asyncio
    async def test_terminal_prm_uniform(self):
        traj = _make_trajectory([1.0, 1.0, 1.0])
        traj.steps[-1].reward = 0.8
        prm = TerminalPRM()
        scores = await prm.score_steps(traj, _DummyTask())
        assert len(scores) == 3
        assert all(s == 0.8 for s in scores)

    @pytest.mark.asyncio
    async def test_terminal_prm_empty(self):
        traj = StatefulTrajectory(run_id="r1")
        prm = TerminalPRM()
        scores = await prm.score_steps(traj, _DummyTask())
        assert scores == []

    # ─── DiscountedPRM ────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_discounted_prm_length(self):
        traj = _make_trajectory([0.0, 0.0, 1.0])
        prm = DiscountedPRM(gamma=0.9)
        scores = await prm.score_steps(traj, _DummyTask())
        assert len(scores) == 3

    @pytest.mark.asyncio
    async def test_discounted_prm_last_step_is_terminal(self):
        """Last step score should equal terminal reward (γ^0 = 1.0)."""
        traj = _make_trajectory([0.0, 0.0, 1.0])
        traj.steps[-1].reward = 1.0
        prm = DiscountedPRM(gamma=0.9)
        scores = await prm.score_steps(traj, _DummyTask())
        assert abs(scores[-1] - 1.0) < 1e-9

    @pytest.mark.asyncio
    async def test_discounted_prm_decreasing(self):
        """Earlier steps have smaller scores."""
        traj = _make_trajectory([0.0, 0.0, 0.0, 1.0])
        traj.steps[-1].reward = 1.0
        prm = DiscountedPRM(gamma=0.9)
        scores = await prm.score_steps(traj, _DummyTask())
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    @pytest.mark.asyncio
    async def test_discounted_prm_invalid_gamma(self):
        with pytest.raises(ValueError):
            DiscountedPRM(gamma=0.0)

    # ─── ToolSuccessPRM ───────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_tool_success_prm_bonus(self):
        traj = _make_trajectory(
            rewards=[0.5, 0.5],
            tool_errors_per_step=[[False], [True]],
        )
        prm = ToolSuccessPRM(success_bonus=0.1, error_penalty=0.2)
        scores = await prm.score_steps(traj, _DummyTask())
        assert len(scores) == 2
        # step 0: success → 0.5 + 0.1 = 0.6
        assert abs(scores[0] - 0.6) < 1e-9
        # step 1: error → 0.5 - 0.2 = 0.3
        assert abs(scores[1] - 0.3) < 1e-9

    @pytest.mark.asyncio
    async def test_tool_success_prm_no_tools(self):
        traj = _make_trajectory(rewards=[0.5, 0.5], tool_errors_per_step=[[], []])
        prm = ToolSuccessPRM()
        scores = await prm.score_steps(traj, _DummyTask())
        assert all(s == 0.5 for s in scores)

    # ─── Protocol conformance ──────────────────────────────────────────────────────

    def test_prm_protocol(self):
        assert isinstance(TerminalPRM(), ProcessRewardModel)
        assert isinstance(DiscountedPRM(), ProcessRewardModel)
        assert isinstance(ToolSuccessPRM(), ProcessRewardModel)
