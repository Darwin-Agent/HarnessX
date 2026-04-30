# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
from .....core.events import EvalResult, TaskEndEvent


class SelfVerifyEvaluator:
    """Structural heuristic evaluator — no LLM call.

    Checks that the agent produced a response and that key terms from the
    success criteria appear in the output. Score = coverage ratio of criteria tokens.
    Use LLMJudgeEvaluator for semantic evaluation.
    """

    _STOPWORDS = frozenset(
        {
            "should",
            "must",
            "needs",
            "create",
            "produce",
            "write",
            "using",
            "given",
            "ensure",
            "check",
            "verify",
            "contains",
            "output",
            "result",
            "provide",
            "include",
            "return",
            "following",
        }
    )

    async def evaluate(self, event: TaskEndEvent) -> EvalResult:
        final_output = event.final_output.strip() if event.final_output else ""

        if not final_output or len(final_output) <= 20:
            return EvalResult(passed=False, score=0.0, reason="Agent produced no response", reward=0.0)

        if not event.success_criteria:
            return EvalResult(
                passed=True,
                score=1.0,
                reason="No success criteria; agent responded",
                reward=1.0,
            )

        content_lower = final_output.lower()
        criteria_tokens = [
            w
            for w in event.success_criteria.lower().split()
            if len(w) >= 5 and w.isalpha() and w not in self._STOPWORDS
        ]

        if not criteria_tokens:
            return EvalResult(
                passed=True,
                score=0.5,
                reason="Criteria too generic for structural check; use LLMJudgeEvaluator",
                reward=0.5,
            )

        matched = sum(1 for t in criteria_tokens if t in content_lower)
        ratio = matched / len(criteria_tokens)
        return EvalResult(
            passed=ratio >= 0.5,
            score=round(ratio, 2),
            reason=f"Structural check: {matched}/{len(criteria_tokens)} criteria tokens found.",
            reward=round(ratio, 2),
        )
