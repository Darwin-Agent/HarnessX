# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .....core.harness import BaseTask
    from .....core.trajectory import StatefulTrajectory


@runtime_checkable
class ProcessRewardModel(Protocol):
    """Assigns per-step rewards to a trajectory.

    All implementations must return len(result) == len(trajectory.steps).
    """

    async def score_steps(
        self,
        trajectory: "StatefulTrajectory",
        task: "BaseTask",
    ) -> list[float]: ...


class TerminalPRM:
    """All steps share the terminal reward. Use when no step-level signal is available."""

    async def score_steps(self, trajectory: "StatefulTrajectory", task: "BaseTask") -> list[float]:
        terminal = trajectory.steps[-1].reward if trajectory.steps else 0.0
        return [terminal] * len(trajectory.steps)


class DiscountedPRM:
    """Reverse-discounted reward: r_t = γ^(T-t) * terminal_reward.

    Args:
        gamma: Discount factor in (0, 1]. Near 1.0 = near-uniform; near 0.0 = last step only.
    """

    def __init__(self, gamma: float = 0.95) -> None:
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.gamma = gamma

    async def score_steps(self, trajectory: "StatefulTrajectory", task: "BaseTask") -> list[float]:
        if not trajectory.steps:
            return []
        terminal = trajectory.steps[-1].reward
        T = len(trajectory.steps)
        return [(self.gamma ** (T - 1 - i)) * terminal for i in range(T)]


class ToolSuccessPRM:
    """Per-step bonus/penalty based on tool call outcomes plus terminal reward.

    reward_t = terminal_reward + Σ(success_bonus or -error_penalty per tool call)
    """

    def __init__(self, success_bonus: float = 0.05, error_penalty: float = 0.10) -> None:
        self.success_bonus = success_bonus
        self.error_penalty = error_penalty

    async def score_steps(self, trajectory: "StatefulTrajectory", task: "BaseTask") -> list[float]:
        rewards = []
        for step in trajectory.steps:
            delta = sum(-self.error_penalty if obs.error else self.success_bonus for obs in step.observation)
            rewards.append(step.reward + delta)
        return rewards


class LLMJudgePRM:
    """Calls an LLM to score each step's action quality. Expensive — for offline annotation.

    The scorer model is resolved from the harness's providers registry via
    provider_key (default "evaluator").

    Args:
        provider_key: Providers registry key for the scorer sub-harness.
        scale:        Multiply raw 0–1 score by this factor (default 1.0).
    """

    def __init__(self, provider_key: str = "evaluator", scale: float = 1.0) -> None:
        self._provider_key = provider_key
        self.scale = scale
        self._sub_harnesses: dict = {}

    def _bind_sub_harnesses(self, sub_harnesses: dict) -> None:
        self._sub_harnesses = dict(sub_harnesses)

    async def score_steps(self, trajectory: "StatefulTrajectory", task: "BaseTask") -> list[float]:
        from .....core.harness import BaseTask as _BT

        sub = self._sub_harnesses.get(self._provider_key)
        if sub is None:
            logger.warning(
                "LLMJudgePRM: sub-harness '%s' not registered, returning 0.0",
                self._provider_key,
            )
            return [0.0] * len(trajectory.steps)

        rewards = []
        for step in trajectory.steps:
            action_summary = step.action.content or "" if step.action is not None else ""
            if step.action is not None and step.action.tool_calls:
                action_summary += f"\nTool calls: {', '.join(tc.name for tc in step.action.tool_calls)}"

            _desc_text = (
                task.description
                if isinstance(task.description, str)
                else " ".join(
                    b.get("text", "") for b in task.description if isinstance(b, dict) and b.get("type") == "text"
                )
            )
            prompt = (
                f"Task: {_desc_text}\n\nStep {step.step_id} action:\n{action_summary}\n\n"
                "On a scale of 0.0 to 1.0, how much did this action contribute "
                "to completing the task? Reply with only a number."
            )
            try:
                result = await sub.run(
                    _BT(description=prompt, max_steps=1),
                    parent_run_id=trajectory.run_id,
                )
                score = max(
                    0.0,
                    min(self.scale, float(result.final_output.strip()) * self.scale),
                )
            except Exception as exc:
                logger.warning(
                    "LLMJudgePRM: scoring step %d failed (%s), using 0.0",
                    step.step_id,
                    exc,
                )
                score = 0.0
            rewards.append(score)
        return rewards
