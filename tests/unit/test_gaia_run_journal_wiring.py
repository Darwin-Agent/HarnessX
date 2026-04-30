# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Tests the gaia_evolver → journal attribution wiring.

The recipe layer computes attribution between rounds and back-fills
the journal entry for the round that just evaluated. This test
exercises the compute + fill path with realistic round records
(without running the full evolver — which needs a model provider).
"""

from __future__ import annotations

from pathlib import Path

from harnessx.meta_harness import journal


def _make_entry(tmp_path: Path, round_idx: int, predicted: list[str]) -> Path:
    """Write a journal with one entry for ``round_idx`` that predicts
    ``predicted`` task ids will flip.
    """
    jp = tmp_path / "learnings.md"
    journal.append_entry(
        jp,
        journal.JournalEntrySpec(
            round=round_idx,
            label=f"round {round_idx} bet",
            hypothesis_id=f"h_r{round_idx}",
            levers=["action"],
            predicted_affected=predicted,
            prose="### Why\nsynthetic\n",
        ),
    )
    return jp


def test_accepted_round_attribution(tmp_path: Path) -> None:
    """Round predicts 2 flips; 1 actually flipped, 1 still F."""
    jp = _make_entry(tmp_path, 1, ["task_a", "task_b"])

    prev_passed = set()  # R0: nothing passed
    cur_passed = {"task_a"}  # R1: task_a passed, task_b still fails

    attr = journal.compute_attribution(
        predicted=["task_a", "task_b"],
        passed_now=cur_passed,
        passed_before=prev_passed,
    )
    assert attr == {"task_a": "flipped", "task_b": "still_F"}

    ok = journal.fill_gating(jp, 1, "accepted", attr)
    assert ok is True

    entry = journal.read_entries(jp)[0]
    assert entry.gating_outcome == "accepted"
    assert entry.gating_attribution == attr


def test_reverted_round_attribution_records_regression(tmp_path: Path) -> None:
    """A task that was passing but regressed under the new config
    should be recorded as ``regressed`` — strong signal the bet hurt."""
    jp = _make_entry(tmp_path, 1, ["task_a"])

    prev_passed = {"task_a"}  # task_a was passing in R0
    cur_passed: set[str] = set()  # R1: nothing passes — regression

    attr = journal.compute_attribution(
        predicted=["task_a"],
        passed_now=cur_passed,
        passed_before=prev_passed,
    )
    assert attr == {"task_a": "regressed"}

    journal.fill_gating(jp, 1, "reverted", attr)
    entry = journal.read_entries(jp)[0]
    assert entry.gating_outcome == "reverted"


def test_noop_round_outcome(tmp_path: Path) -> None:
    """Byte-identical config = noop. Attribution is still computed
    honestly — the orchestrator uses the ``noop`` outcome signal so
    the next agent knows the round was a deliberate no-change, not a
    failed attempt that got rolled back."""
    jp = _make_entry(tmp_path, 1, ["task_a"])

    attr = journal.compute_attribution(
        predicted=["task_a"],
        passed_now={"task_a"},
        passed_before={"task_a"},
    )
    journal.fill_gating(jp, 1, "noop", attr)

    entry = journal.read_entries(jp)[0]
    assert entry.gating_outcome == "noop"
    assert entry.gating_attribution == {"task_a": "still_T"}


def test_no_journal_entry_is_tolerated(tmp_path: Path) -> None:
    """When the meta-agent forgot to write a journal entry for the
    round (legacy runs, crashes), the recipe's fill_gating call should
    return False — not raise. The recipe logs and continues."""
    jp = tmp_path / "learnings.md"
    # No entry written yet.
    ok = journal.fill_gating(jp, 1, "accepted", {"task_a": "flipped"})
    assert ok is False


def test_multi_round_attribution_builds_context(tmp_path: Path) -> None:
    """After two rounds of attribution, build_context should produce
    a scoreboard row that credits the action lever correctly."""
    jp = _make_entry(tmp_path, 1, ["task_a", "task_b"])
    # Simulate a second round (R2) with a different hypothesis.
    journal.append_entry(
        jp,
        journal.JournalEntrySpec(
            round=2,
            label="round 2 bet",
            hypothesis_id="h_r2",
            levers=["configuration"],
            predicted_affected=["task_c"],
            prose="### Why\nconfig nudge\n",
        ),
    )

    # Fill attribution for R1: task_a flipped, task_b still_F.
    journal.fill_gating(
        jp,
        1,
        "accepted",
        {"task_a": "flipped", "task_b": "still_F"},
    )
    # R2: task_c flipped.
    journal.fill_gating(
        jp,
        2,
        "accepted",
        {"task_c": "flipped"},
    )

    ctx = journal.build_context(journal_path=jp, current_round=3, output_path=tmp_path / "CTX.md")
    assert ctx is not None
    text = ctx.read_text(encoding="utf-8")
    # Action: 1 attempt, 1 accepted, 1/2 predicted hits.
    assert "| action | 1 | 1 | 0 | 1/2 |" in text
    # Configuration: 1 attempt, 1 accepted, 1/1 predicted hits.
    assert "| configuration | 1 | 1 | 0 | 1/1 |" in text
