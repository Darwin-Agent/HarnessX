# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING

from harnessx.core.events import EvalResult

if TYPE_CHECKING:
    from harnessx.core.events import TaskEndEvent
    from harnessx.rl.task import RLTask

try:
    from slime.rollout.rm_hub.math_dapo_utils import (
        compute_score as _math_dapo_compute_score,
    )

    _SLIME_AVAILABLE = True
except ImportError:
    _math_dapo_compute_score = None  # type: ignore[assignment]
    _SLIME_AVAILABLE = False


class MathBoxedEvaluator:
    """
    Evaluator for math problems using \\boxed{} answer extraction.

    Must be per-run instantiated (not a singleton) because it closes over
    task.label at construction time.  EvaluationProcessor calls
        evaluator.evaluate(event: TaskEndEvent)
    and TaskEndEvent has no task reference.

    Usage:
        evaluator = MathBoxedEvaluator(task)  # per-run, inside build_rl_harness_config()
    """

    def __init__(self, task: "RLTask") -> None:
        self._label = task.label

    async def evaluate(self, event: "TaskEndEvent") -> EvalResult:
        """Score the final model output against the ground-truth label.

        Mirrors retool's ``solution_str = sample.prompt + sample.response``
        approach: search for \\boxed{} across ALL assistant turns rather than
        only the last model generation (event.final_output).  This ensures that
        if the episode is cut short by budget_exceeded (max_steps), an answer
        produced in an earlier turn is still found.
        """
        if not _SLIME_AVAILABLE or _math_dapo_compute_score is None:
            raise RuntimeError(
                "MathBoxedEvaluator requires the 'slime' package. Install it or use a different evaluator."
            )
        # Concatenate all assistant turns to replicate retool's full-response search.
        # final_messages is the complete message list at episode end.
        full_response = (
            " ".join(
                m.content
                for m in (event.final_messages or ())
                if m.role == "assistant" and isinstance(m.content, str) and m.content
            )
            or event.final_output
        )

        result = _math_dapo_compute_score(
            full_response,
            self._label,
            strict_box_verify=True,
        )
        # math_dapo_compute_score returns {"score": float, "acc": bool, "pred": str}
        return EvalResult(
            passed=bool(result.get("acc", False)),
            score=float(result.get("score", -1.0)),
            reason=str(result.get("pred", "")),
            reward=float(result.get("score", -1.0)),
        )
