# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from harnessx.rl.task import ProcessRewardModel

if TYPE_CHECKING:
    from harnessx.core.events import EvalResult
    from harnessx.core.trajectory import StatefulTrajectory


# ---------------------------------------------------------------------------
# RetoolCompatPRM — exact retool math reward shaping
# ---------------------------------------------------------------------------


class RetoolCompatPRM(ProcessRewardModel):
    """Exact retool reward shaping: tool-use bonus on negative-terminal episodes.

    Replicates retool reward_func() shaping logic exactly:
        if result["score"] < 0:
            tool_call_reward = (num_turns - 2) / 2 * 0.1
            result["score"] = min(-0.6, result["score"] + tool_call_reward)

    Logic:
    - Positive terminal (correct answer): no shaping
    - Negative terminal (wrong/no answer):
        - Count tool-call turns (steps with at least one tool execution)
        - Give +0.05 bonus per turn above 2 tool-call turns
        - Penalise 0-1 tool turns (bonus is negative): pushes score below terminal
          (e.g. 0 turns → -0.1, score = -1.1; 1 turn → -0.05, score = -1.05)
        - Cap result at -0.6 (tool bonus cannot flip the sign to positive)

    Example: terminal=-1.0, 0 tool-call turns → bonus=-0.1  → score=min(-0.6,-1.1)=-1.1
    Example: terminal=-1.0, 2 tool-call turns → bonus=0.0   → score=min(-0.6,-1.0)=-1.0
    Example: terminal=-1.0, 4 tool-call turns → bonus=+0.1  → score=min(-0.6,-0.9)=-0.9
    Example: terminal=-1.0, 8 tool-call turns → bonus=+0.3  → score=min(-0.6,-0.7)=-0.7

    is_terminal_only=True: score_steps() already returns the final adjusted scalar
    propagated to all steps.  reward_func() uses step_rewards[-1] directly and
    MUST NOT call aggregate() (which would double-count the tool bonus).
    """

    is_terminal_only: bool = True

    async def score_steps(
        self,
        traj: "StatefulTrajectory | None",
        exit_reason: str = "done",
    ) -> list[float]:
        if not traj or not traj.steps:
            return []

        terminal = traj.steps[-1].reward  # backfilled terminal reward

        if terminal >= 0:
            # Positive terminal: no shaping needed
            return [terminal] * len(traj.steps)

        # Count steps that had tool executions.
        # tool_call_reward can be negative (< 2 tool turns are penalised),
        # zero (exactly 2 turns), or positive (> 2 turns, +0.05 per extra pair).
        # Retool intentionally pushes 0–1 tool-turn episodes below terminal to
        # discourage not using tools at all.
        # min(-0.6, ...) caps the result from above: the tool bonus cannot push
        # a wrong-answer episode above -0.6 (i.e., cannot flip the sign).
        num_tool_turns = sum(1 for s in traj.steps if s.observation)
        tool_call_reward = (num_tool_turns - 2) / 2 * 0.1
        adjusted = min(-0.6, terminal + tool_call_reward)
        return [adjusted] * len(traj.steps)


# ---------------------------------------------------------------------------
# math_format_reward — \\boxed{} format bonus
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\s*\{")

# Small bonus when the model uses \boxed{} on a *wrong* answer.
# Helps maintain the SFT-trained format convention early in RL when
# the model might start forgetting the \boxed{} requirement.
# Applied only for negative-terminal episodes so it never inflates correct rewards.
_FORMAT_BONUS: float = 0.1


def math_format_reward(
    sample: Any,
    eval_result: "EvalResult | None",
    traj: "StatefulTrajectory | None",
) -> dict:
    """
    Extra reward function for math tasks.

    Contract (extra_reward_fn protocol):
        fn(sample, eval_result, traj) -> dict

    Returns dict with:
        "score_delta"     : float  — added to prm_adjusted to form final score
        "format_score"    : same as score_delta (for decomposition logging)
        "pred"            : predicted answer string (from eval_result.reason)
        "has_boxed_answer": 1 or 0 (whether model used \\boxed{} notation)

    Adds a small format bonus (+0.1) when the model:
    1. Used \\boxed{} notation in the response, AND
    2. Got the answer wrong (terminal < 0)

    Args:
        sample:      Slime Sample with .response field
        eval_result: EvalResult from EvaluationProcessor (or None if missing)
        traj:        StatefulTrajectory (unused here, available for richer PRMs)
    """
    terminal: float = float(eval_result.reward) if eval_result is not None else -1.0
    pred: str = str(eval_result.reason) if eval_result is not None else ""

    # Check for \boxed{} in model's response.
    # sample.response includes tool tokens, but math tool output never contains
    # LaTeX \boxed{} so any match here is from the model's text.
    has_boxed: bool = bool(_BOXED_RE.search(getattr(sample, "response", "") or ""))

    format_score: float = _FORMAT_BONUS if (has_boxed and terminal < 0) else 0.0

    return {
        "score_delta": format_score,
        "format_score": format_score,
        "pred": pred,
        "has_boxed_answer": int(has_boxed),
    }
