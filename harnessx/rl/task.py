# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from harnessx.core.harness import BaseTask

if TYPE_CHECKING:
    from harnessx.core.events import EvalResult, TaskEndEvent
    from harnessx.core.trajectory import StatefulTrajectory


@dataclass
class RLTask(BaseTask):
    """
    RL-specific BaseTask subclass.

    Carries per-sample metadata needed by evaluators and loggers.
    The ``label`` field is the ground truth answer for the evaluator.
    The ``task_type`` field routes to the correct HarnessConfigSpec.

    Inherits from BaseTask:
        description: str          — task text shown to the model (user message)
        success_criteria: str     — optional hint for evaluator
        max_steps: int            — from HarnessConfigSpec.max_steps
        token_budget: int | None
        metadata: dict
        is_done(state) -> bool    — default False; subclass to check slots
    """

    label: str = ""  # ground truth (compared by evaluator)
    task_type: str = ""  # "math" | "code" | ... (for logging/routing)


class TaskBuilder(Protocol):
    """Converts an arbitrary sample to an RLTask.

    This protocol is intentionally framework-agnostic: ``sample`` is typed
    as ``Any`` so the same protocol works for different training-framework
    sample representations.

    Framework-specific builders (e.g. MathTaskBuilder) live in recipe/ and
    implement this protocol.
    """

    def build(self, sample: Any) -> RLTask: ...


class RLEvaluator(Protocol):
    """Protocol for RL task evaluators.

    Evaluators are instantiated once per run (closing over ``task.label``)
    by ``build_harness_config()`` and passed to ``EvaluationProcessor``.

    ``EvaluationProcessor`` fires ``on_task_end()`` which calls this method.
    ``TaskEndEvent.final_output`` is the model's last non-empty text content.
    ``TaskEndEvent`` has NO task reference — close over ``task.label`` in ``__init__``.

    Framework-specific implementations live in recipe/:
        recipe/slime/evaluators/math_evaluator.py → MathBoxedEvaluator
    """

    async def evaluate(self, event: "TaskEndEvent") -> "EvalResult": ...


class ProcessRewardModel:
    """Base class for per-step reward models.

    Concrete implementations live in recipe/slime/rewards/prm.py:
        NullPRM              — pure terminal reward, no shaping
        RetoolCompatPRM      — exact retool reward shaping
        EnhancedToolSuccessPRM — tool success/failure per-step deltas

    is_terminal_only:
        True  — ``score_steps()`` already returns the final scalar propagated
                to all steps (e.g. RetoolCompatPRM, NullPRM).  ``reward_func()``
                uses ``step_rewards[-1]`` directly; ``aggregate()`` is never called.
        False — ``score_steps()`` returns per-step deltas; ``reward_func()`` calls
                ``aggregate(terminal, step_rewards)`` to combine them.
    """

    is_terminal_only: bool = False

    async def score_steps(
        self,
        traj: "StatefulTrajectory | None",
        exit_reason: str = "done",
    ) -> list[float]:
        """Return a per-step reward list (same length as traj.steps)."""
        raise NotImplementedError

    def aggregate(self, terminal: float, step_rewards: list[float]) -> float:
        """Aggregate per-step rewards with terminal into a scalar score.

        Default: sum tool deltas above terminal (capped +0.3) minus deltas
        below terminal (capped -0.5), constrained to [-1.1, -0.6] for
        negative terminal.  Override in subclasses to embed PRM-specific caps.
        """
        if not step_rewards:
            return terminal
        tool_bonuses = sum(r - terminal for r in step_rewards if r > terminal)
        tool_penalties = sum(terminal - r for r in step_rewards if r < terminal)
        score = terminal + min(tool_bonuses, 0.3) - min(tool_penalties, 0.5)
        if terminal < 0:
            score = max(-1.1, min(-0.6, score))
        return score


# ---------------------------------------------------------------------------
# NullPRM — pure terminal reward, no shaping
# ---------------------------------------------------------------------------


class NullPRM(ProcessRewardModel):
    """Pure terminal reward — no per-step shaping.

    Use as a sanity-check baseline before enabling any reward shaping.
    ``is_terminal_only=True``: score_steps() returns the terminal scalar
    propagated to all steps; reward_func() must NOT call aggregate().
    """

    is_terminal_only: bool = True

    async def score_steps(
        self,
        traj: "StatefulTrajectory | None",
        exit_reason: str = "done",
    ) -> list[float]:
        if not traj or not traj.steps:
            return []
        return [step.reward for step in traj.steps]


# ---------------------------------------------------------------------------
# EnhancedToolSuccessPRM — generic per-step tool success/failure shaping
# ---------------------------------------------------------------------------


class EnhancedToolSuccessPRM(ProcessRewardModel):
    """
    Per-step reward shaping based on tool success/failure signals.

    Generic — works for any tool-using task (math, code, search, …).
    Reads directly from StatefulTrajectory.steps[t].observation
    (ToolResultEvent objects with .error attribute).

    Two error channels must both be checked:
        obs.error  — set when the tool registry raises a Python exception
        obs.result — code_interpreter never raises; all execution errors
                     are returned as "Error: ..." strings (obs.error stays None)

    Per-step delta:
        +success_bonus  per successful tool call
        -error_penalty  per failed tool call
        -loop_penalty   on the last step if exit_reason == "loop_detected"

    step.reward is the backfilled terminal value; the PRM adds deltas on top.

    is_terminal_only=False: score_steps() returns per-step deltas;
    reward_func() calls aggregate(terminal, step_rewards) to combine them.
    """

    def __init__(
        self,
        success_bonus: float = 0.05,
        error_penalty: float = 0.10,
        loop_penalty: float = 0.20,
        max_bonus_cap: float = 0.30,  # cap on total tool bonus above terminal
        max_penalty_cap: float = 0.50,  # cap on total tool penalty below terminal
        neg_floor: float = -1.10,  # lower bound when terminal < 0
        neg_ceil: float = -0.60,  # upper bound when terminal < 0
    ) -> None:
        self.success_bonus = success_bonus
        self.error_penalty = error_penalty
        self.loop_penalty = loop_penalty
        self.max_bonus_cap = max_bonus_cap
        self.max_penalty_cap = max_penalty_cap
        self.neg_floor = neg_floor
        self.neg_ceil = neg_ceil

    async def score_steps(
        self,
        traj: "StatefulTrajectory | None",
        exit_reason: str = "done",
    ) -> list[float]:
        if not traj or not traj.steps:
            return []

        rewards: list[float] = []
        last_idx = len(traj.steps) - 1

        for i, step in enumerate(traj.steps):
            delta: float = 0.0
            for obs in step.observation:
                is_error = bool(obs.error) or (isinstance(obs.result, str) and obs.result.startswith("Error:"))
                if is_error:
                    delta -= self.error_penalty
                else:
                    delta += self.success_bonus
            if i == last_idx and exit_reason == "loop_detected":
                delta -= self.loop_penalty
            rewards.append(step.reward + delta)

        return rewards

    def aggregate(self, terminal: float, step_rewards: list[float]) -> float:
        """Combine per-step deltas with terminal, applying constructor-param caps."""
        if not step_rewards:
            return terminal
        tool_bonuses = sum(r - terminal for r in step_rewards if r > terminal)
        tool_penalties = sum(terminal - r for r in step_rewards if r < terminal)
        score = terminal + min(tool_bonuses, self.max_bonus_cap) - min(tool_penalties, self.max_penalty_cap)
        if terminal < 0:
            score = max(self.neg_floor, min(self.neg_ceil, score))
        return score
