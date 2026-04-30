# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
import json
import re
from .....core.events import EvalResult, TaskEndEvent


_JUDGE_PROMPT = """\
You are an objective evaluator. Score whether the agent completed the task.

Task description:
{description}

Success criteria:
{criteria}

Agent conversation (last {n_msgs} messages):
{conversation}

Respond with a JSON object only — no markdown, no explanation outside JSON:
{{"score": <float 0.0-1.0>, "passed": <true|false>, "reason": "<one sentence>"}}

Scoring guide:
- 1.0: criteria fully met
- 0.7–0.9: mostly met, minor gaps
- 0.4–0.6: partially met
- 0.1–0.3: attempted but failed
- 0.0: no meaningful attempt
"""


class LLMJudgeEvaluator:
    """Uses a language model to evaluate task completion against success_criteria.

    The judge model is resolved from the harness's sub-harnesses registry via
    ``provider_key`` (default ``"evaluator"``).
    """

    def __init__(self, provider_key: str = "evaluator", max_conv_messages: int = 15):
        self._provider_key = provider_key
        self.max_conv_messages = max_conv_messages
        self._sub_harnesses: dict = {}

    def _bind_sub_harnesses(self, sub_harnesses: dict) -> None:
        self._sub_harnesses = dict(sub_harnesses)

    async def evaluate(self, event: TaskEndEvent) -> EvalResult:
        if not event.success_criteria:
            has_response = bool(event.final_output)
            return EvalResult(
                passed=has_response,
                score=1.0 if has_response else 0.0,
                reason="No success criteria; checked agent responded",
                reward=1.0 if has_response else 0.0,
            )

        sub = self._sub_harnesses.get(self._provider_key)
        if sub is None:
            return EvalResult(
                passed=False,
                score=0.0,
                reason=f"Judge sub-harness '{self._provider_key}' not registered in providers",
                reward=0.0,
            )

        recent = list(event.final_messages)[-self.max_conv_messages :] if event.final_messages else []
        conversation = (
            "\n".join(f"[{m.role.upper()}]: {(m.content or '')[:400]}" for m in recent)
            or f"[ASSISTANT]: {event.final_output[:400]}"
        )

        prompt = _JUDGE_PROMPT.format(
            description=(getattr(event, "task_description", "") or event.final_output[:200]),
            criteria=event.success_criteria[:300],
            n_msgs=len(recent),
            conversation=conversation,
        )

        from .....core.harness import BaseTask as _BT

        try:
            result = await sub.run(_BT(description=prompt, max_steps=1), parent_run_id=event.run_id)
            text = result.final_output.strip()
        except Exception as exc:
            return EvalResult(passed=False, score=0.0, reason=f"Judge run failed: {exc}", reward=0.0)

        try:
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
            data = json.loads(text)
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            passed = bool(data.get("passed", score >= 0.6))
            return EvalResult(
                passed=passed,
                score=score,
                reason=str(data.get("reason", "")),
                reward=score,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return EvalResult(
                passed=False,
                score=0.0,
                reason=f"Judge response parse error: {exc} — raw: {text[:200]}",
                reward=0.0,
            )
