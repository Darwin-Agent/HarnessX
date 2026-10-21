# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""verl-compatible reward function.

Called by verl's reward loop via custom_reward_function config:
    compute_score(data_source, solution_str, ground_truth, extra_info) -> dict

Format and tool_call scores are computed in the agent loop (rewards.py) using
structured trajectory messages, then passed via extra_info["rollout_reward_scores"].
This function reads those pre-computed values and combines with accuracy.
"""
from __future__ import annotations

import re

from recipe.verl.research.evaluator import compute_accuracy_score

_FORMAT_WEIGHT = 0.1
_TOOL_CALL_WEIGHT = 0.2
_ACCURACY_WEIGHT = 0.7


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> dict:
    """verl reward function: 0.7*accuracy + 0.1*format + 0.2*tool_call(diversity).

    Format and tool_call scores come from extra_info (pre-computed in agent loop).
    Falls back to 0.0 if not available.
    """
    extra_info = extra_info or {}

    accuracy = compute_accuracy_score(solution_str, ground_truth)

    # Read pre-computed scores from agent loop (via rollout_reward_scores)
    rollout_scores = extra_info.get("rollout_reward_scores", {}) or {}
    format_score = float(rollout_scores.get("format_score", 0.0))
    tool_call_score = float(rollout_scores.get("tool_call_score", 0.0))

    score = (
        _ACCURACY_WEIGHT * accuracy
        + _FORMAT_WEIGHT * format_score
        + _TOOL_CALL_WEIGHT * tool_call_score
    )

    return {
        "score": score,
        "accuracy": accuracy,
        "format_reward": _FORMAT_WEIGHT * format_score,
        "tool_call_reward": _TOOL_CALL_WEIGHT * tool_call_score,
    }
