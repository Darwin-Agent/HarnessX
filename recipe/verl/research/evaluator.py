# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Evaluator for GAIA research QA tasks."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from harnessx.core.events import EvalResult

if TYPE_CHECKING:
    from harnessx.core.events import TaskEndEvent
    from harnessx.rl.task import RLTask

_ACCURACY_WEIGHT = 0.7
_SPECIAL_TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")
_MAX_ANSWER_WORDS = 100


def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip()).lower()


def _extract_answer(solution_str: str) -> tuple[str, bool]:
    cleaned = _SPECIAL_TOKEN_RE.sub("", solution_str)
    last_open = cleaned.rfind("<answer>")
    last_close = cleaned.rfind("</answer>")
    if last_open != -1 and last_close > last_open:
        return cleaned[last_open + len("<answer>"):last_close].strip(), True

    if "</think>" in cleaned:
        after_think = cleaned.split("</think>")[-1].strip()
        after_think = re.sub(r"<tool_call>.*?</tool_call>", "", after_think, flags=re.DOTALL)
        after_think = re.sub(r"<tool_response>.*?</tool_response>", "", after_think, flags=re.DOTALL)
        after_think = re.sub(r"\bassistant\b", "", after_think).strip()
        if after_think:
            return after_think, False

    return cleaned.strip(), False


def _try_numeric_match(answer: str, gt: str, tolerance: float = 0.05) -> bool:
    num_re = re.compile(r"[-+]?\d[\d,]*\.?\d*")
    a_nums = num_re.findall(answer.replace(",", ""))
    g_nums = num_re.findall(gt.replace(",", ""))
    if not a_nums or not g_nums:
        return False
    try:
        a_val = float(a_nums[-1])
        g_val = float(g_nums[-1])
        if g_val == 0:
            return a_val == 0
        return abs(a_val - g_val) / abs(g_val) <= tolerance
    except (ValueError, ZeroDivisionError):
        return False


def compute_accuracy_score(full_response: str, ground_truth: str) -> float:
    """Compute accuracy: 1.0 (exact), 0.8 (contains/numeric), 0.6 (partial), 0.0 (miss).

    If the answer is not extracted from an <answer> tag, the score is discounted by 0.5.
    """
    answer_text, has_answer_tag = _extract_answer(full_response)
    if not answer_text:
        return 0.0
    if len(answer_text.split()) > _MAX_ANSWER_WORDS:
        return 0.0

    norm_answer = _normalize(answer_text)
    norm_gt = _normalize(ground_truth)

    if norm_gt and norm_answer == norm_gt:
        score = 1.0
    elif norm_gt and norm_gt in norm_answer:
        score = 0.8
    elif _try_numeric_match(norm_answer, norm_gt, tolerance=0.05):
        score = 0.8
    elif norm_answer and norm_answer in norm_gt and len(norm_answer) > 0:
        score = 0.6
    else:
        score = 0.0

    if not has_answer_tag:
        score *= 0.8

    return score


class ResearchAnswerEvaluator:
    """Evaluator for GAIA-style research QA tasks.

    Scoring: extract <answer>...</answer>, substring/numeric match.
    Returns EvalResult with reward = 0.8 * accuracy_score.
    """

    def __init__(self, task: "RLTask") -> None:
        self._label = task.label

    async def evaluate(self, event: "TaskEndEvent") -> EvalResult:
        full_response = (
            " ".join(
                m.content
                for m in (event.final_messages or ())
                if m.role == "assistant" and isinstance(m.content, str) and m.content
            )
            or event.final_output
        )

        accuracy = compute_accuracy_score(full_response, self._label)
        reward = _ACCURACY_WEIGHT * accuracy
        pred, _ = _extract_answer(full_response)

        return EvalResult(
            passed=accuracy >= 1.0,
            score=accuracy,
            reason=pred[:200],
            reward=reward,
        )
