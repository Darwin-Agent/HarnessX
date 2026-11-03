# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Extra reward function for research tasks — format + tool_call bonuses."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harnessx.core.events import EvalResult
    from harnessx.core.trajectory import StatefulTrajectory

_FORMAT_WEIGHT = 0.1
_TOOL_CALL_WEIGHT = 0.2

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_QWEN_FUNCTION_RE = re.compile(r"<function=\w+>.*?</function>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)
_TERMINAL_SPECIAL_RE = re.compile(r"(?:<\|(?:im_end|endoftext|eot_id|end)\|>\s*)+$")


def _check_format_from_traj(traj: "StatefulTrajectory | None") -> float:
    """Strict format check per assistant message in trajectory.

    Each assistant message must be exactly:
        <think>...</think>\n\n<tool_call>...</tool_call>
        OR
        <think>...</think>\n\n<answer>...</answer>

    Only \\n\\n between </think> and the next block. Nothing else allowed.
    Last message must have <answer>.
    """
    if traj is None or not traj.steps:
        return 0.0
    has_final_answer = False
    for step in traj.steps:
        if step.action is None or not step.action.content:
            continue
        content = _TERMINAL_SPECIAL_RE.sub("", step.action.content).rstrip()
        if "</think>" not in content:
            return 0.0
        after_think = content.split("</think>", 1)[-1]
        # Must be exactly \n\n followed by <tool_call> or <answer>
        if after_think.startswith("\n\n<tool_call>") and after_think.rstrip().endswith("</tool_call>"):
            pass
        elif after_think.startswith("\n\n<answer>") and after_think.rstrip().endswith("</answer>"):
            has_final_answer = True
        else:
            return 0.0
    return 1.0 if has_final_answer else 0.0


def _check_tool_call_from_traj(traj: "StatefulTrajectory | None") -> float:
    """Returns diversity_ratio (unique calls / total calls) if all blocks parse.
    Returns 0.0 if no tool calls or any fails to parse.
    """
    if traj is None or not traj.steps:
        return 0.0
    call_signatures: list[str] = []
    for step in traj.steps:
        if step.action is None or not step.action.content:
            continue
        for match in _TOOL_CALL_BLOCK_RE.finditer(step.action.content):
            block = match.group(1).strip()
            if not block or not _QWEN_FUNCTION_RE.search(block):
                return 0.0
            call_signatures.append(block)
    if not call_signatures:
        return 0.0
    unique_ratio = len(set(call_signatures)) / len(call_signatures)
    return unique_ratio


def research_extra_reward_fn(
    sample: Any,
    eval_result: "EvalResult | None",
    traj: "StatefulTrajectory | None",
) -> dict:
    """Extra reward: 0.1*format + 0.1*tool_call added to prm_adjusted."""
    pred: str = str(eval_result.reason) if eval_result is not None else ""
    accuracy: float = float(eval_result.score) if eval_result is not None else 0.0

    format_score = _check_format_from_traj(traj)
    tool_call_score = _check_tool_call_from_traj(traj)

    score_delta = _FORMAT_WEIGHT * format_score + _TOOL_CALL_WEIGHT * tool_call_score

    return {
        "score_delta": score_delta,
        "format_reward": _FORMAT_WEIGHT * format_score,
        "tool_call_reward": _TOOL_CALL_WEIGHT * tool_call_score,
        "accuracy": accuracy,
        "pred": pred,
    }
