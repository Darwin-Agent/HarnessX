# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Unit tests for the best-so-far gating kernel in recipe/gaia_evolver/run.py.

The kernel is a pure function; we drive it with synthetic rounds and check
decisions, baseline updates, and the drift-proof invariant.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from recipe.gaia_evolver.run import _score_and_gate  # noqa: E402


# Sentinels: _score_and_gate keeps the config object opaque; strings are fine.
CFG_R0 = "cfg-r0"
CFG_R1 = "cfg-r1"
CFG_R2 = "cfg-r2"
CFG_R3 = "cfg-r3"


def _gate(rate: float, cost: float, idx: int, cfg, best, *, tol=0.0, cw=0.0, passed=None):
    # Default passed to rate * 10 (simulate 10 tasks) so the absolute count
    # delta is large enough to exceed the noise threshold when rates differ.
    if passed is None:
        passed = int(rate * 10)
    return _score_and_gate(
        round_pass_rate=rate,
        round_cost=cost,
        round_idx=idx,
        round_config=cfg,
        round_passed=passed,
        best=best,
        tolerance=tol,
        cost_weight=cw,
    )


def test_r0_accepts_and_becomes_baseline() -> None:
    decision, _reason, new_best, reverted = _gate(0.6, 5.0, 0, CFG_R0, None)
    assert decision == "ACCEPTED"
    assert reverted is None
    assert new_best == (0.6, 5.0, CFG_R0, 0, 6)


def test_strict_improvement_dethrones_baseline() -> None:
    best = (0.6, 5.0, CFG_R0, 0, 6)
    decision, _reason, new_best, reverted = _gate(0.8, 5.0, 1, CFG_R1, best)
    assert decision == "ACCEPTED"
    assert reverted is None
    assert new_best == (0.8, 5.0, CFG_R1, 1, 8)


def test_equal_score_does_not_dethrone() -> None:
    best = (0.6, 5.0, CFG_R0, 0, 6)
    decision, _reason, new_best, reverted = _gate(0.6, 5.0, 1, CFG_R1, best)
    assert decision == "ACCEPTED"
    assert reverted is None
    # R0 remains the baseline even though R1 tied.
    assert new_best == best


def test_regression_within_tolerance_accepts_without_promoting() -> None:
    best = (0.6, 5.0, CFG_R0, 0, 6)
    decision, _reason, new_best, reverted = _gate(0.55, 5.0, 1, CFG_R1, best, tol=0.1)
    assert decision == "ACCEPTED"
    assert reverted is None
    # Critical: baseline does NOT drift down to .55 under tolerance.
    assert new_best == best


def test_regression_past_tolerance_reverts() -> None:
    best = (0.6, 5.0, CFG_R0, 0, 6)
    decision, _reason, new_best, reverted = _gate(0.3, 5.0, 1, CFG_R1, best, tol=0.1)
    assert decision == "REVERTED"
    assert reverted == CFG_R0
    # Baseline preserved.
    assert new_best == best


def test_cost_weight_penalises_cost_regression() -> None:
    # Use passed=60 (large enough that count_delta=0 still triggers revert
    # because the pass_count_noise_threshold only guards score regressions,
    # and here the score regression is driven by cost, not pass count).
    # Actually: same passed count → count_delta=0 < threshold=3 → noise guard
    # kicks in. So we need the passed counts to differ by >= 3 to bypass it.
    # Use passed=6 for best but passed=2 for round to get count_delta=4.
    best = (0.6, 5.0, CFG_R0, 0, 6)
    # Same pass_rate but 2x cost → cost_delta_ratio = 1.0
    # score = 0.6 - 0.2 * 1.0 = 0.4, best_score = 0.6
    # Without tolerance, 0.4 < 0.6 → REVERTED (count_delta=4 >= 3).
    decision, _reason, _new_best, reverted = _gate(0.6, 10.0, 1, CFG_R1, best, tol=0.0, cw=0.2, passed=2)
    assert decision == "REVERTED"
    assert reverted == CFG_R0


def test_cost_weight_does_not_penalise_cost_drop() -> None:
    best = (0.6, 5.0, CFG_R0, 0, 6)
    # Same pass_rate but cost halved. cost_delta_ratio < 0, but we max()
    # against 0 so penalty=0 — round ties on score, does not dethrone.
    decision, _reason, new_best, reverted = _gate(0.6, 2.5, 1, CFG_R1, best, tol=0.0, cw=1.0)
    assert decision == "ACCEPTED"
    assert reverted is None
    assert new_best == best  # tie → keep earliest


def test_baseline_cannot_drift_over_many_tolerated_regressions() -> None:
    """Drift-proof invariant: N consecutive rounds each slightly worse
    than the current baseline must not move the baseline."""
    best = None
    rates = [0.60, 0.58, 0.58, 0.57, 0.56]
    costs = [5.0, 5.0, 5.0, 5.0, 5.0]
    cfgs = [CFG_R0, CFG_R1, CFG_R2, CFG_R3, "cfg-r4"]
    for i, (r, c, cfg) in enumerate(zip(rates, costs, cfgs)):
        _decision, _reason, best, _reverted = _gate(r, c, i, cfg, best, tol=0.05)
    # Best is whatever R0 set — R1..R4 all landed within tolerance but never
    # strictly beat R0, so R0 remains.
    assert best == (0.60, 5.0, CFG_R0, 0, 6)


def test_reason_text_contains_baseline_round_identifier() -> None:
    best = (0.6, 5.0, CFG_R0, 0, 6)
    _decision, reason_accept, _, _ = _gate(0.6, 5.0, 1, CFG_R1, best)
    _decision, reason_revert, _, _ = _gate(0.1, 5.0, 1, CFG_R1, best, tol=0.1)
    # Both reasons should cite which baseline round was used, so memo readers
    # can look up the right config.yaml for diff comparison.
    assert "R0" in reason_accept
    assert "R0" in reason_revert
