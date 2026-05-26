# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Stage 5 — adjudicate previous round's ship. Auto-revert if hit_rate < 0.5."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class AdjudicationResult:
    hit_rate: float | None
    shipped_cid: str | None
    action: str  # "keep" | "revert" | "skip"
    regressed_tasks: list[str]


def compute_hit_rate(
    predicted: list[str], actually_passed: list[str]
) -> float | None:
    if not predicted:
        return None
    passed_set = set(actually_passed)
    hits = sum(1 for t in predicted if t in passed_set)
    return hits / len(predicted)


def should_revert(
    *, hit_rate: float | None, threshold: float = 0.5, enabled: bool = True
) -> bool:
    if not enabled:
        return False
    if hit_rate is None:
        return False
    return hit_rate < threshold


def adjudicate_previous_round(
    *,
    prev_shipped_cid: str | None,
    prev_predicted_tasks_pass: list[str],
    prev_predicted_tasks_at_risk: list[str],
    current_round_pass_set: set[str],
    prev_round_pass_set: set[str],
    threshold: float = 0.5,
    auto_revert_enabled: bool = True,
    revert_fn: Callable[[str], None] | None = None,
) -> AdjudicationResult:
    if prev_shipped_cid is None:
        return AdjudicationResult(None, None, "skip", [])

    hit_rate = compute_hit_rate(
        predicted=prev_predicted_tasks_pass,
        actually_passed=list(current_round_pass_set),
    )
    regressed = [t for t in prev_round_pass_set if t not in current_round_pass_set]

    if should_revert(hit_rate=hit_rate, threshold=threshold, enabled=auto_revert_enabled):
        if revert_fn is not None:
            revert_fn(prev_shipped_cid)
        return AdjudicationResult(hit_rate, prev_shipped_cid, "revert", regressed)
    return AdjudicationResult(hit_rate, prev_shipped_cid, "keep", regressed)
