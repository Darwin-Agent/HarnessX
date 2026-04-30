# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, AsyncIterator

from ...core.events import TaskEndEvent
from ...core.processor import MultiHookProcessor

if TYPE_CHECKING:
    pass


class EvaluationProcessor(MultiHookProcessor):
    """Call an evaluator on task end and inject the result into TaskEndEvent.

    The evaluator receives the full ``TaskEndEvent`` (including
    ``final_output``, ``success_criteria``, and ``final_messages``) so it
    has all the information it needs without requiring a live ``State``
    reference.
    """

    required_providers: frozenset = frozenset()

    _singleton_group = "evaluation"

    def __init__(self, evaluator: object):
        self.evaluator = evaluator
        # Bind sub-harnesses so LLMJudgeEvaluator can call sub_harness.run()
        self._sub_harnesses: dict = {}

    def _bind_sub_harnesses(self, sub_harnesses: dict) -> None:
        self._sub_harnesses = dict(sub_harnesses)
        if hasattr(self.evaluator, "_bind_sub_harnesses"):
            self.evaluator._bind_sub_harnesses(sub_harnesses)

    async def on_task_end(self, event: TaskEndEvent) -> AsyncIterator[TaskEndEvent]:
        try:
            eval_result = await self.evaluator.evaluate(event)
        except Exception as exc:
            from ...core.events import EvalResult

            eval_result = EvalResult(
                passed=False,
                score=0.0,
                reason=f"Evaluator raised {type(exc).__name__}: {exc}",
                reward=0.0,
            )
        yield dataclasses.replace(event, eval_result=eval_result)
