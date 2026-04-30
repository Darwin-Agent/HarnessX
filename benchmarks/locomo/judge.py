# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""EverMemOS-aligned LLM Judge for LoCoMo evaluation.

Ports the v7f judge logic from Light-MemoryStack ``eval/metrics.ts``:
gpt-4o-mini, 3 runs, temp 0.3, generous grading ("touches on the same
topic" = CORRECT).  Aligned report excludes adversarial category.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Awaitable, Callable

LLMCallFn = Callable[[str], Awaitable[str | None]]

_ACCURACY_JUDGE_SYSTEM = (
    "You are an expert grader that determines if answers to questions match a gold standard answer."
)

_ACCURACY_JUDGE_PROMPT = """\
You are grading answers to questions posed by one user to another user. \
These questions are about something one user should know about the other \
user based on their prior conversations.

Here is an example:
Question: What did the user bring back from Hawaii?
Gold Answer: A shell necklace
Generated Answer: The user brought back a beautiful shell necklace from \
their trip to Hawaii. They mentioned it was a gift for their partner.

In this case, the generated answer is longer but it touches on the same \
topic as the gold answer. So it should be graded as CORRECT. Be generous \
in your grading \u2014 as long as the generated answer touches on the same \
topic as the gold answer, it should be counted as CORRECT.

For time-related questions, gold answers will be specific dates while \
generated answers may use relative references. Accept relative time \
references (e.g., "last Tuesday", "a few weeks ago") as correct if they \
refer to the same time period. Also accept different date formats \
(e.g., "May 7th" vs "7 May" vs "May 7, 2023") as correct.

Now grade the following:

Question: {question}
Gold Answer: {reference}
Generated Answer: {prediction}

First, provide a one-sentence explanation of your reasoning. \
Then provide your label.

Do NOT include both CORRECT and WRONG in your response, \
or it will break the evaluation script.

Respond in JSON format:
{{"reasoning": "your one-sentence explanation", "label": "CORRECT" or "WRONG"}}"""

_ADVERSARIAL_CATEGORIES = frozenset({"adversarial_qa", "adversarial"})


async def _single_judge_call(
    call_llm_fn: LLMCallFn,
    question: str,
    reference: str,
    prediction: str,
) -> int:
    """Single judge invocation.  Returns 1 = CORRECT, 0 = WRONG."""
    import json
    import re

    prompt = (
        _ACCURACY_JUDGE_SYSTEM
        + "\n\n"
        + _ACCURACY_JUDGE_PROMPT.replace("{question}", question)
        .replace("{reference}", reference)
        .replace("{prediction}", prediction)
    )
    text = await call_llm_fn(prompt)
    if not text:
        return 0

    text = text.strip()

    # Try JSON parse first (aligned with v7f metrics.ts:274-285)
    try:
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            parsed = json.loads(json_match.group(0))
            label = parsed.get("label", "")
            if isinstance(label, str):
                return 1 if label.strip().upper() == "CORRECT" else 0
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: text matching (aligned with v7f metrics.ts:287-291)
    lower = text.lower()
    if "correct" in lower and "wrong" not in lower and "incorrect" not in lower:
        return 1
    if "wrong" in lower or "incorrect" in lower:
        return 0

    return 0


def _make_judge_call_llm(
    model: str,
    temperature: float,
    extra_headers: dict[str, str] | None = None,
) -> LLMCallFn:
    """Create an async LLM call function for judge invocations.

    Routes through the Anthropic proxy when ANTHROPIC_BASE_URL is set,
    otherwise falls back to LiteLLM.
    """
    import os

    if os.environ.get("ANTHROPIC_BASE_URL"):
        from anthropic import AsyncAnthropic

        _client = AsyncAnthropic()

        async def _call(prompt: str) -> str | None:
            try:
                response = await _client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=temperature,
                )
                return response.content[0].text.strip()
            except Exception:  # noqa: BLE001
                return None

        return _call

    async def _call(prompt: str) -> str | None:
        import litellm

        kwargs: dict = {}
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=temperature,
                **kwargs,
            )
            return response.choices[0].message.content.strip()
        except Exception:  # noqa: BLE001
            return None

    return _call


async def judge_accuracy(
    question: str,
    reference: str,
    prediction: str,
    *,
    model: str = "azure_openai/gpt-4o-mini",
    num_runs: int = 3,
    temperature: float | None = None,
    extra_headers: dict[str, str] | None = None,
) -> float:
    """EverMemOS-aligned accuracy: *num_runs* independent judgments, return mean.

    Returns float in [0, 1].  E.g. 0.667 means 2/3 runs said CORRECT.
    Temperature defaults to 0.3 for multi-run, 0 for single-run (v7f convention).
    """
    if temperature is None:
        temperature = 0.3 if num_runs > 1 else 0.0

    call_llm = _make_judge_call_llm(model, temperature, extra_headers)

    correct = 0
    for _ in range(num_runs):
        try:
            correct += await _single_judge_call(call_llm, question, reference, prediction)
        except Exception:  # noqa: BLE001
            pass
    return correct / num_runs if num_runs > 0 else 0.0


def compute_aligned_accuracy(
    results: list[dict],
    *,
    exclude_adversarial: bool = True,
) -> dict[str, float]:
    """Per-category and overall accuracy from judge results.

    Returns dict like ``{"overall": 82.5, "single_hop_qa": 90.0, ...}``.
    Values are **percentages**.  Excludes adversarial by default (EverMemOS convention).
    """
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in results:
        cat = r.get("category", "")
        if exclude_adversarial and cat in _ADVERSARIAL_CATEGORIES:
            continue
        judge_val = r.get("llm_judge")
        if judge_val is not None:
            by_cat[cat].append(judge_val)

    agg: dict[str, float] = {}
    all_scores: list[float] = []
    for cat, scores in sorted(by_cat.items()):
        agg[cat] = sum(scores) / len(scores) * 100 if scores else 0.0
        all_scores.extend(scores)
    agg["overall"] = sum(all_scores) / len(all_scores) * 100 if all_scores else 0.0
    return agg
