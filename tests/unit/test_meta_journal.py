# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Tests for harnessx.meta_harness.journal — structured multi-round memory.

Covers append/read round-trip, frontmatter parsing forgiveness, gating
back-fill (idempotent), context-file generation, and the invalid-lever
guard.

Note: ``tests/unit/test_journal.py`` covers the unrelated
``harnessx.tracing.journal`` module — this file is about the
meta-agent's cross-round journal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harnessx.meta_harness.journal import (
    JournalEntrySpec,
    append_entry,
    build_context,
    fill_gating,
    latest_entry,
    read_entries,
)


def _make_spec(round_idx: int, label: str, **kw) -> JournalEntrySpec:
    defaults = dict(
        round=round_idx,
        label=label,
        hypothesis_id=f"h_r{round_idx}",
        levers=["configuration"],
        predicted_affected=[f"task_{round_idx}"],
        prose=f"### Why\nRound {round_idx} reasoning.\n",
    )
    defaults.update(kw)
    return JournalEntrySpec(**defaults)


# ─── Append + round-trip ──────────────────────────────────────────────────


def test_append_and_read_roundtrip(tmp_path: Path) -> None:
    jp = tmp_path / "journal.md"
    append_entry(jp, _make_spec(0, "baseline"))
    append_entry(jp, _make_spec(1, "action lever"))

    entries = read_entries(jp)
    assert len(entries) == 2
    assert entries[0].round == 0
    assert entries[0].label == "baseline"
    assert entries[0].hypothesis_id == "h_r0"
    assert entries[0].levers == ["configuration"]
    assert entries[0].predicted_affected == ["task_0"]
    assert entries[0].gating_outcome == "pending"
    assert entries[0].gating_attribution == {}
    assert entries[1].round == 1


def test_append_preserves_prose(tmp_path: Path) -> None:
    jp = tmp_path / "journal.md"
    prose = (
        "### Why\nbecause X\n\n"
        "### Changes\n- tools/foo.py\n- processors/bar.py\n\n"
        "### Evidence\n- `task_a` step 5: empty result\n"
    )
    append_entry(jp, _make_spec(0, "x", prose=prose))
    entries = read_entries(jp)
    assert "tools/foo.py" in entries[0].prose
    assert "Evidence" in entries[0].prose


def test_duplicate_round_rejected(tmp_path: Path) -> None:
    jp = tmp_path / "journal.md"
    append_entry(jp, _make_spec(0, "a"))
    with pytest.raises(ValueError, match="already has a Round 0"):
        append_entry(jp, _make_spec(0, "b"))


def test_invalid_lever_rejected(tmp_path: Path) -> None:
    jp = tmp_path / "journal.md"
    with pytest.raises(ValueError, match="unknown lever"):
        append_entry(jp, _make_spec(0, "x", levers=["nonsense"]))


def test_read_empty_file_returns_empty(tmp_path: Path) -> None:
    jp = tmp_path / "nope.md"
    assert read_entries(jp) == []


def test_read_handles_malformed_frontmatter(tmp_path: Path) -> None:
    """A round section missing the frontmatter block is skipped, not
    a hard error — the journal is forgiving by design."""
    jp = tmp_path / "journal.md"
    jp.write_text(
        "## Round 0 — broken\n\nno frontmatter here at all\n\n"
        "## Round 1 — good\n\n"
        "<!-- journal:frontmatter\n"
        "round: 1\ntimestamp: '2026-01-01T00:00:00Z'\n"
        "hypothesis_id: h_r1\nlevers: [configuration]\n"
        "predicted_affected: [task_x]\n"
        "gating_outcome: pending\n"
        "gating_attribution: pending\n"
        "-->\n\n"
        "### Why\n\ngood entry\n",
        encoding="utf-8",
    )
    entries = read_entries(jp)
    assert len(entries) == 1
    assert entries[0].round == 1
    assert entries[0].hypothesis_id == "h_r1"


def test_latest_entry(tmp_path: Path) -> None:
    jp = tmp_path / "journal.md"
    assert latest_entry(jp) is None
    append_entry(jp, _make_spec(0, "a"))
    append_entry(jp, _make_spec(1, "b"))
    assert latest_entry(jp).round == 1


# ─── fill_gating ──────────────────────────────────────────────────────────


def test_fill_gating_updates_frontmatter(tmp_path: Path) -> None:
    jp = tmp_path / "journal.md"
    append_entry(jp, _make_spec(0, "baseline"))

    ok = fill_gating(
        jp,
        0,
        "accepted",
        {"task_0": "flipped", "task_1": "still_F"},
    )
    assert ok is True

    entries = read_entries(jp)
    assert entries[0].gating_outcome == "accepted"
    assert entries[0].gating_attribution == {
        "task_0": "flipped",
        "task_1": "still_F",
    }


def test_fill_gating_idempotent(tmp_path: Path) -> None:
    jp = tmp_path / "journal.md"
    append_entry(jp, _make_spec(0, "baseline"))

    attr = {"task_0": "flipped"}
    ok1 = fill_gating(jp, 0, "accepted", attr)
    before = jp.read_text(encoding="utf-8")
    ok2 = fill_gating(jp, 0, "accepted", attr)
    after = jp.read_text(encoding="utf-8")

    assert ok1 is True and ok2 is True
    assert before == after  # idempotent — no rewrite when state matches


def test_fill_gating_missing_round_returns_false(tmp_path: Path) -> None:
    jp = tmp_path / "journal.md"
    append_entry(jp, _make_spec(0, "baseline"))
    ok = fill_gating(jp, 99, "accepted", {"task_0": "flipped"})
    assert ok is False


def test_fill_gating_invalid_outcome(tmp_path: Path) -> None:
    jp = tmp_path / "journal.md"
    append_entry(jp, _make_spec(0, "baseline"))
    with pytest.raises(ValueError, match="outcome must be"):
        fill_gating(jp, 0, "banana", {"task_0": "flipped"})


def test_fill_gating_preserves_prose(tmp_path: Path) -> None:
    """The prose body and the other frontmatter fields must survive a
    fill_gating update unchanged — the update is a surgical edit."""
    jp = tmp_path / "journal.md"
    prose = "### Why\nkey reasoning\n\n### Changes\n- tools/foo.py\n"
    append_entry(jp, _make_spec(0, "baseline", prose=prose))

    fill_gating(jp, 0, "accepted", {"task_0": "flipped"})
    text = jp.read_text(encoding="utf-8")

    assert "key reasoning" in text
    assert "tools/foo.py" in text

    # Frontmatter still has the original hypothesis_id.
    entries = read_entries(jp)
    assert entries[0].hypothesis_id == "h_r0"
    assert entries[0].predicted_affected == ["task_0"]


# ─── build_context ────────────────────────────────────────────────────────


def test_build_context_renders_recent_rounds(tmp_path: Path) -> None:
    jp = tmp_path / "journal.md"
    append_entry(jp, _make_spec(0, "baseline", levers=["configuration"]))
    append_entry(
        jp,
        _make_spec(
            1,
            "pdf tool",
            hypothesis_id="h_pdf",
            levers=["action"],
            predicted_affected=["task_a", "task_b"],
        ),
    )
    fill_gating(jp, 0, "accepted", {"task_0": "still_F"})
    fill_gating(jp, 1, "accepted", {"task_a": "flipped", "task_b": "still_F"})

    ctx_path = build_context(jp, current_round=2, output_path=tmp_path / "CTX.md")
    assert ctx_path is not None
    ctx = ctx_path.read_text(encoding="utf-8")

    # Scoreboard shows lever attempt + raw hits + Beta-posterior + side-effects.
    assert "| action | 1 | 1 | 0 | 1/2 | 0.50 (n_eff=2.0) | — |" in ctx
    assert "| configuration | 1 | 1 | 0 | 0/1 | 0.34 (n_eff=0.9) | — |" in ctx
    # Hypothesis table mentions the labels.
    assert "baseline" in ctx
    assert "pdf tool" in ctx
    assert "1/2 flipped" in ctx


def test_build_context_skips_current_round(tmp_path: Path) -> None:
    """Entries at or after ``current_round`` are excluded — only prior
    rounds inform the next one."""
    jp = tmp_path / "journal.md"
    append_entry(jp, _make_spec(0, "a"))
    append_entry(jp, _make_spec(1, "b"))

    # No prior rounds exist before round 0.
    assert build_context(jp, current_round=0, output_path=tmp_path / "CTX.md") is None

    # Only R0 counts when preparing R1.
    ctx_path = build_context(jp, current_round=1, output_path=tmp_path / "CTX.md")
    assert ctx_path is not None
    ctx = ctx_path.read_text(encoding="utf-8")
    assert "last 1" in ctx  # recent window capped at available prior


def test_build_context_flags_reverted(tmp_path: Path) -> None:
    jp = tmp_path / "journal.md"
    append_entry(jp, _make_spec(0, "flawed idea", hypothesis_id="h_bad"))
    fill_gating(jp, 0, "reverted", {"task_0": "still_F"})

    ctx_path = build_context(jp, current_round=1, output_path=tmp_path / "CTX.md")
    assert ctx_path is not None
    ctx = ctx_path.read_text(encoding="utf-8")
    assert "Reverted hypotheses" in ctx
    assert "h_bad" in ctx


def test_build_context_no_journal_returns_none(tmp_path: Path) -> None:
    assert (
        build_context(
            tmp_path / "nonexistent.md",
            current_round=1,
            output_path=tmp_path / "CTX.md",
        )
        is None
    )
