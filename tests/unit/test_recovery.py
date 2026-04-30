# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Recovery tests — dual-track rebuild invariants (spec §C, 4 cases)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from harnessx.tracing.journal import HarnessJournal
from harnessx.core.events import BeforeModelEvent, Message


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _make_snapshot(run_id: str = "run-abc", step: int = 2) -> dict:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "raw_messages": [],
        "messages": [],
        "step": step,
        "cumulative_tokens": 0,
        "cumulative_input_tokens": 0,
        "cumulative_output_tokens": 0,
        "cumulative_cost_usd": 0.0,
        "max_steps": 50,
        "token_budget": None,
        "max_cost_usd": None,
        "spawn_depth": 0,
        "slots": {},
        "pending_subagents": {},
    }


def _make_journal(tmp_path: Path, session_id: str) -> HarnessJournal:
    return HarnessJournal(
        base_dir=str(tmp_path / "sessions"),
        export_jsonl=True,
        silent=True,
        session_id=session_id,
    )


def _write_session_fixtures(
    tmp_path: Path,
    session_id: str,
    run_id: str,
    snap: dict,
    records: list[dict],
) -> Path:
    """Write state JSON + session index + JSONL, return the JSONL path."""
    session_dir = tmp_path / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    state_path = session_dir / f"{run_id}_state.json"
    state_path.write_text(json.dumps(snap), encoding="utf-8")

    index = {
        "schema_version": 1,
        "session_id": session_id,
        "run_ids": [run_id],
        "latest_run_id": run_id,
        "latest_state_path": f"sessions/{session_id}/{run_id}_state.json",
        "latest_config_path": "harness_config.yaml",
        "updated_at": "2026-04-20T00:00:00.000Z",
    }
    (tmp_path / "sessions" / f"{session_id}.json").write_text(json.dumps(index), encoding="utf-8")

    jsonl_path = session_dir / f"{run_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return jsonl_path


# ══════════════════════════════════════════════════════════════════════════════
# C-1. raw + delta rebuild correctness
#
# For each raw_* anchor, the rebuild picks the matching delta when present
# and falls back to raw otherwise.  raw_messages always uses the raw anchor.
# ══════════════════════════════════════════════════════════════════════════════


class TestRawDeltaRebuild:
    """C-1: delta records override effective track; raw track uses raw anchor."""

    def _jsonl_path(self, tmp_path: Path, session_id: str, run_id: str) -> Path:
        return tmp_path / "sessions" / session_id / f"{run_id}.jsonl"

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_user_delta_overrides_effective_only(self, tmp_path):
        """raw_user + user delta → effective=delta, raw=raw."""
        path = self._jsonl_path(tmp_path, "s1", "r1")
        self._write_jsonl(
            path,
            [
                {"type": "raw_user", "step": 0, "timestamp": 0.0, "message": {"role": "user", "content": "original"}},
                {
                    "type": "user",
                    "step": 0,
                    "timestamp": 0.1,
                    "message": {"role": "user", "content": "wrapped: original"},
                },
            ],
        )
        raw, eff = HarnessJournal._rebuild_message_tracks_from_jsonl(path, up_to_step=1)
        assert len(raw) == 1
        assert len(eff) == 1
        assert raw[0].content == "original"
        assert eff[0].content == "wrapped: original"

    def test_assistant_delta_overrides_effective_only(self, tmp_path):
        """raw_assistant + assistant delta → effective=delta, raw=raw."""
        path = self._jsonl_path(tmp_path, "s2", "r2")
        self._write_jsonl(
            path,
            [
                {
                    "type": "raw_assistant",
                    "step": 0,
                    "timestamp": 0.0,
                    "message": {"role": "assistant", "content": "raw answer"},
                },
                {
                    "type": "assistant",
                    "step": 0,
                    "timestamp": 0.1,
                    "message": {"role": "assistant", "content": "processed answer"},
                    "meta": {"model": "test", "stop_reason": "stop", "usage": {}},
                },
            ],
        )
        raw, eff = HarnessJournal._rebuild_message_tracks_from_jsonl(path, up_to_step=1)
        assert raw[0].content == "raw answer"
        assert eff[0].content == "processed answer"

    def test_no_delta_raw_used_for_both_tracks(self, tmp_path):
        """raw_assistant without delta → raw_messages and messages both use raw."""
        path = self._jsonl_path(tmp_path, "s3", "r3")
        self._write_jsonl(
            path,
            [
                {"type": "raw_user", "step": 0, "timestamp": 0.0, "message": {"role": "user", "content": "question"}},
                {
                    "type": "raw_assistant",
                    "step": 1,
                    "timestamp": 0.1,
                    "message": {"role": "assistant", "content": "answer"},
                },
            ],
        )
        raw, eff = HarnessJournal._rebuild_message_tracks_from_jsonl(path, up_to_step=2)
        assert len(raw) == 2
        assert len(eff) == 2
        assert raw[0].content == eff[0].content == "question"
        assert raw[1].content == eff[1].content == "answer"

    def test_user_delta_only_on_last_raw_user_at_step(self, tmp_path):
        """History raw_user records at same step are NOT overridden by user delta.

        step=0 carries both historical context (ctx=history) and the current turn.
        The user delta must apply only to the LAST raw_user at that step.
        """
        path = self._jsonl_path(tmp_path, "s4", "r4")
        self._write_jsonl(
            path,
            [
                # step=0: historical context user (first raw_user at step 0)
                {
                    "type": "raw_user",
                    "step": 0,
                    "timestamp": 0.0,
                    "message": {"role": "user", "content": "history message"},
                    "meta": {"ctx": "history"},
                },
                # step=0: current turn user (second raw_user at step 0)
                {
                    "type": "raw_user",
                    "step": 0,
                    "timestamp": 0.01,
                    "message": {"role": "user", "content": "current turn"},
                },
                # user delta applies only to last raw_user at step 0
                {
                    "type": "user",
                    "step": 0,
                    "timestamp": 0.02,
                    "message": {"role": "user", "content": "wrapped: current turn"},
                },
            ],
        )
        raw, eff = HarnessJournal._rebuild_message_tracks_from_jsonl(path, up_to_step=1)
        assert len(raw) == 2
        assert len(eff) == 2
        # History message: raw == eff (no delta applied)
        assert raw[0].content == "history message"
        assert eff[0].content == "history message"
        # Current turn: raw = original, eff = delta
        assert raw[1].content == "current turn"
        assert eff[1].content == "wrapped: current turn"

    def test_incomplete_step_excluded(self, tmp_path):
        """Records at step >= up_to_step are excluded (incomplete last step)."""
        path = self._jsonl_path(tmp_path, "s5", "r5")
        self._write_jsonl(
            path,
            [
                {
                    "type": "raw_user",
                    "step": 0,
                    "timestamp": 0.0,
                    "message": {"role": "user", "content": "complete step"},
                },
                {
                    "type": "raw_assistant",
                    "step": 1,
                    "timestamp": 0.1,
                    "message": {"role": "assistant", "content": "incomplete reply"},
                },
            ],
        )
        # up_to_step=1 excludes step=1
        raw, eff = HarnessJournal._rebuild_message_tracks_from_jsonl(path, up_to_step=1)
        assert len(raw) == 1
        assert raw[0].role == "user"


# ══════════════════════════════════════════════════════════════════════════════
# C-2. _validate_rebuilt_tracks rejects invariant violations
# ══════════════════════════════════════════════════════════════════════════════


class TestValidateRebuiltTracks:
    """C-2: _validate_rebuilt_tracks detects dual-track invariant violations."""

    def _msg(self, role: str, content: str = "x", tcid: str | None = None) -> Message:
        return Message(role=role, content=content, tool_call_id=tcid)

    def test_length_mismatch_raises(self):
        """len(raw_messages) != len(messages) → ValueError."""
        raw = [self._msg("user")]
        eff = [self._msg("user"), self._msg("assistant")]
        with pytest.raises(ValueError, match="len"):
            HarnessJournal._validate_rebuilt_tracks(
                session_id="s",
                run_id="r",
                raw_messages=raw,
                messages=eff,
                strict=True,
            )

    def test_role_mismatch_raises(self):
        """Different roles at same index → ValueError."""
        raw = [self._msg("user", "q")]
        eff = [self._msg("assistant", "q")]
        with pytest.raises(ValueError):
            HarnessJournal._validate_rebuilt_tracks(
                session_id="s",
                run_id="r",
                raw_messages=raw,
                messages=eff,
                strict=True,
            )

    def test_tool_call_id_mismatch_raises(self):
        """Different tool_call_ids at same tool index → ValueError."""
        raw = [self._msg("tool", "result", tcid="tc-1")]
        eff = [self._msg("tool", "result", tcid="tc-2")]
        with pytest.raises(ValueError, match="tool_call_id"):
            HarnessJournal._validate_rebuilt_tracks(
                session_id="s",
                run_id="r",
                raw_messages=raw,
                messages=eff,
                strict=True,
            )

    def test_valid_tracks_no_raise(self):
        """Matching tracks → no exception."""
        raw = [self._msg("user", "raw content"), self._msg("assistant", "raw reply")]
        eff = [self._msg("user", "eff content"), self._msg("assistant", "eff reply")]
        HarnessJournal._validate_rebuilt_tracks(
            session_id="s",
            run_id="r",
            raw_messages=raw,
            messages=eff,
            strict=True,
        )

    def test_strict_false_skips_all_checks(self):
        """strict=False: mismatched tracks pass without raising."""
        raw = [self._msg("user")]
        eff = [self._msg("assistant"), self._msg("user")]
        HarnessJournal._validate_rebuilt_tracks(
            session_id="s",
            run_id="r",
            raw_messages=raw,
            messages=eff,
            strict=False,
        )

    def test_empty_tracks_valid(self):
        """Empty tracks pass validation (wake() with no history)."""
        HarnessJournal._validate_rebuilt_tracks(
            session_id="s",
            run_id="r",
            raw_messages=[],
            messages=[],
            strict=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# C-3. Cross-segment recovery via context_snapshot records
# ══════════════════════════════════════════════════════════════════════════════


class TestCrossSegmentRecovery:
    """C-3: After a SegmentBoundaryEvent, context_snapshot is the base for rebuild."""

    def _jsonl_path(self, tmp_path: Path, session_id: str, run_id: str) -> Path:
        return tmp_path / "sessions" / session_id / f"{run_id}.jsonl"

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_context_snapshot_raw_provides_base(self, tmp_path):
        """context_snapshot_raw messages form the base of raw_messages; records after it are appended."""
        path = self._jsonl_path(tmp_path, "sx", "rx")
        self._write_jsonl(
            path,
            [
                # Segment boundary snapshots at step=5
                {
                    "type": "context_snapshot_raw",
                    "step": 5,
                    "timestamp": 5.0,
                    "messages": [
                        {"role": "user", "content": "old turn"},
                        {"role": "assistant", "content": "old reply"},
                    ],
                },
                {
                    "type": "context_snapshot",
                    "step": 5,
                    "timestamp": 5.0,
                    "messages": [
                        {"role": "user", "content": "old turn"},
                        {"role": "assistant", "content": "old reply"},
                    ],
                },
                # New turn after compaction
                {
                    "type": "raw_user",
                    "step": 5,
                    "timestamp": 5.1,
                    "message": {"role": "user", "content": "new question"},
                },
            ],
        )
        raw, eff = HarnessJournal._rebuild_message_tracks_from_jsonl(path, up_to_step=6)
        # Base = 2 messages from context_snapshot_raw + 1 new raw_user = 3 total
        assert len(raw) == 3
        assert len(eff) == 3
        assert raw[0].content == "old turn"
        assert raw[2].content == "new question"

    def test_context_snapshot_provides_effective_base(self, tmp_path):
        """context_snapshot (effective) is the base for messages; raw base from context_snapshot_raw."""
        path = self._jsonl_path(tmp_path, "sy", "ry")
        self._write_jsonl(
            path,
            [
                {
                    "type": "context_snapshot_raw",
                    "step": 3,
                    "timestamp": 3.0,
                    "messages": [{"role": "user", "content": "verbatim turn"}],
                },
                {
                    "type": "context_snapshot",
                    "step": 3,
                    "timestamp": 3.0,
                    "messages": [{"role": "user", "content": "summarized turn"}],
                },
                {
                    "type": "raw_assistant",
                    "step": 3,
                    "timestamp": 3.1,
                    "message": {"role": "assistant", "content": "next reply"},
                },
            ],
        )
        raw, eff = HarnessJournal._rebuild_message_tracks_from_jsonl(path, up_to_step=4)
        assert raw[0].content == "verbatim turn"
        assert eff[0].content == "summarized turn"
        # Both tracks get the next assistant message
        assert raw[1].content == "next reply"
        assert eff[1].content == "next reply"

    def test_full_wake_with_snapshot_and_new_records(self, tmp_path):
        """End-to-end wake() with context_snapshot + new records after boundary."""
        run_id = "run-boundary"
        session_id = "sess-boundary"
        snap = _make_snapshot(run_id=run_id, step=4)
        records = [
            {
                "type": "context_snapshot_raw",
                "step": 3,
                "timestamp": 3.0,
                "messages": [
                    {"role": "user", "content": "prior turn"},
                    {"role": "assistant", "content": "prior reply"},
                ],
            },
            {
                "type": "context_snapshot",
                "step": 3,
                "timestamp": 3.0,
                "messages": [
                    {"role": "user", "content": "prior turn"},
                    {"role": "assistant", "content": "prior reply"},
                ],
            },
            {"type": "raw_user", "step": 3, "timestamp": 3.1, "message": {"role": "user", "content": "new turn"}},
            {
                "type": "raw_assistant",
                "step": 3,
                "timestamp": 3.2,
                "message": {"role": "assistant", "content": "new reply"},
            },
        ]
        _write_session_fixtures(tmp_path, session_id, run_id, snap, records)

        state = HarnessJournal.wake(session_id, str(tmp_path))
        assert len(state.messages) == 4
        assert len(state.raw_messages) == 4
        assert state.messages[0].content == "prior turn"
        assert state.messages[3].content == "new reply"


# ══════════════════════════════════════════════════════════════════════════════
# C-4. Synthetic user: meta.synthetic=true written by journal + dual-track parity
# ══════════════════════════════════════════════════════════════════════════════


class TestSyntheticUserDualTrack:
    """C-4: Synthetic user injection is recorded correctly; dual-track invariants hold."""

    def _jsonl_path(self, tmp_path: Path, session_id: str, run_id: str) -> Path:
        return tmp_path / "sessions" / session_id / f"{run_id}.jsonl"

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_synthetic_raw_user_appears_in_both_tracks(self, tmp_path):
        """Synthetic raw_user (meta.synthetic=true) is appended to both raw_messages and messages."""
        path = self._jsonl_path(tmp_path, "ss1", "rs1")
        self._write_jsonl(
            path,
            [
                # Synthetic user injected by before_model processor
                {
                    "type": "raw_user",
                    "step": 0,
                    "timestamp": 0.0,
                    "message": {"role": "user", "content": "verify step"},
                    "meta": {"synthetic": True, "injected_at_hook": "before_model"},
                },
            ],
        )
        raw, eff = HarnessJournal._rebuild_message_tracks_from_jsonl(path, up_to_step=1)
        assert len(raw) == 1
        assert len(eff) == 1
        assert raw[0].role == "user"
        assert eff[0].role == "user"
        assert raw[0].content == "verify step"
        assert eff[0].content == "verify step"

    def test_synthetic_user_with_delta_override(self, tmp_path):
        """A subsequent processor modifying the synthetic user produces a user delta.

        After rebuild: raw=original synthetic content, effective=modified content.
        """
        path = self._jsonl_path(tmp_path, "ss2", "rs2")
        self._write_jsonl(
            path,
            [
                {
                    "type": "raw_user",
                    "step": 0,
                    "timestamp": 0.0,
                    "message": {"role": "user", "content": "synthetic injection"},
                    "meta": {"synthetic": True, "injected_at_hook": "before_model"},
                },
                # A subsequent processor modified the synthetic user
                {
                    "type": "user",
                    "step": 0,
                    "timestamp": 0.01,
                    "message": {"role": "user", "content": "enhanced injection"},
                },
            ],
        )
        raw, eff = HarnessJournal._rebuild_message_tracks_from_jsonl(path, up_to_step=1)
        assert raw[0].content == "synthetic injection"
        assert eff[0].content == "enhanced injection"

    def test_wake_with_synthetic_user_passes_invariants(self, tmp_path):
        """wake() with a synthetic raw_user record: track invariants hold after rebuild."""
        run_id = "run-syn"
        session_id = "sess-syn"
        snap = _make_snapshot(run_id=run_id, step=1)
        records = [
            # Normal history user at step 0
            {
                "type": "raw_user",
                "step": 0,
                "timestamp": 0.0,
                "message": {"role": "user", "content": "original user turn"},
            },
            # Synthetic user injected at step 0 by before_model processor
            {
                "type": "raw_user",
                "step": 0,
                "timestamp": 0.01,
                "message": {"role": "user", "content": "synthetic verify step"},
                "meta": {"synthetic": True, "injected_at_hook": "before_model"},
            },
        ]
        _write_session_fixtures(tmp_path, session_id, run_id, snap, records)

        # wake() must not raise ValueError from _validate_rebuilt_tracks
        state = HarnessJournal.wake(session_id, str(tmp_path))
        assert len(state.messages) == len(state.raw_messages)
        assert all(r.role == e.role for r, e in zip(state.raw_messages, state.messages))

    def test_dual_track_invariants_after_full_sequence(self, tmp_path):
        """C-4: Complete sequence with user/assistant/tool records: invariants hold."""
        run_id = "run-full"
        session_id = "sess-full"
        snap = _make_snapshot(run_id=run_id, step=3)
        records = [
            {"type": "raw_user", "step": 0, "timestamp": 0.0, "message": {"role": "user", "content": "first question"}},
            {
                "type": "raw_assistant",
                "step": 1,
                "timestamp": 1.0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "tc1", "name": "bash", "input": {"cmd": "ls"}}],
                },
            },
            {
                "type": "raw_tool",
                "step": 2,
                "timestamp": 2.0,
                "message": {"role": "tool", "content": "file1.txt", "tool_call_id": "tc1"},
            },
        ]
        _write_session_fixtures(tmp_path, session_id, run_id, snap, records)

        state = HarnessJournal.wake(session_id, str(tmp_path))
        assert len(state.raw_messages) == 3
        assert len(state.messages) == 3

        # role parity
        for raw_m, eff_m in zip(state.raw_messages, state.messages):
            assert raw_m.role == eff_m.role

        # tool_call_id parity for tool messages
        tool_raw = [m for m in state.raw_messages if m.role == "tool"]
        tool_eff = [m for m in state.messages if m.role == "tool"]
        for r, e in zip(tool_raw, tool_eff):
            assert r.tool_call_id == e.tool_call_id


# ══════════════════════════════════════════════════════════════════════════════
# Fix 1. Journal writes source_processor + reason keys in synthetic user meta
# ══════════════════════════════════════════════════════════════════════════════


class TestSyntheticUserMetaFields:
    """Journal-emitted raw_user for synthetic injection must include source_processor/reason."""

    def test_synthetic_meta_has_source_processor_and_reason_keys(self, tmp_path):
        """before_model: synthetic raw_user meta must carry source_processor and reason."""
        session_id = "smeta-1"
        run_id = "run-smeta"
        journal = _make_journal(tmp_path, session_id)
        journal._open_files(run_id)

        u1 = Message(role="user", content="original question")
        a1 = Message(role="assistant", content="reply")
        synth = Message(role="user", content="verify step")

        # Simulate: on_raw_event captures pre-processor snapshot (no synthetic yet)
        raw_event = BeforeModelEvent(run_id=run_id, step_id=0, messages=(u1, a1))
        asyncio.run(journal.on_raw_event(raw_event))

        # After processor: +1 synthetic user at tail
        after_event = BeforeModelEvent(run_id=run_id, step_id=0, messages=(u1, a1, synth))
        asyncio.run(journal.on_event(after_event))

        journal._close_files()

        jsonl_path = tmp_path / "sessions" / session_id / f"{run_id}.jsonl"
        records = [json.loads(ln) for ln in jsonl_path.read_text().splitlines() if ln.strip()]
        syn_records = [r for r in records if r.get("type") == "raw_user"]

        assert len(syn_records) == 1, "expected exactly one raw_user record for synthetic injection"
        meta = syn_records[0].get("meta", {})
        assert meta.get("synthetic") is True
        assert meta.get("injected_at_hook") == "before_model"
        # Fix 1: these keys must be present (journal sets them to None when unknown)
        assert "source_processor" in meta, "source_processor key missing from synthetic meta"
        assert "reason" in meta, "reason key missing from synthetic meta"
        assert meta["source_processor"] is None  # journal cannot resolve the processor name
        assert meta["reason"] is None  # must be provided by the processor itself

    def test_non_synthetic_injection_not_flagged(self, tmp_path):
        """before_model without length change: no raw_user synthetic record written."""
        session_id = "smeta-2"
        run_id = "run-smeta2"
        journal = _make_journal(tmp_path, session_id)
        journal._open_files(run_id)

        u1 = Message(role="user", content="question")
        # Simulate processor wrapping the user content (no length change)
        raw_event = BeforeModelEvent(run_id=run_id, step_id=0, messages=(u1,))
        asyncio.run(journal.on_raw_event(raw_event))

        wrapped = Message(role="user", content="wrapped: question")
        after_event = BeforeModelEvent(run_id=run_id, step_id=0, messages=(wrapped,))
        asyncio.run(journal.on_event(after_event))

        journal._close_files()

        jsonl_path = tmp_path / "sessions" / session_id / f"{run_id}.jsonl"
        records = [json.loads(ln) for ln in jsonl_path.read_text().splitlines() if ln.strip()]
        syn_records = [r for r in records if r.get("type") == "raw_user"]
        # No synthetic injection — no raw_user record from on_event(BeforeModelEvent)
        assert all(r.get("meta", {}).get("synthetic") is not True for r in syn_records)
