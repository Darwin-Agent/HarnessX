# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from harnessx.tracing.journal import HarnessJournal
from harnessx.core.events import Message
from harnessx.core.state import State, StateSlot


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_journal(tmp_path: Path, session_id: str | None = None) -> HarnessJournal:
    sessions_dir = str(tmp_path / "sessions")
    return HarnessJournal(base_dir=sessions_dir, export_jsonl=True, silent=True, session_id=session_id)


def _make_snapshot(run_id: str = "run-abc123", step: int = 3) -> dict:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "raw_messages": [],
        "messages": [],
        "step": step,
        "cumulative_tokens": 1000,
        "cumulative_input_tokens": 600,
        "cumulative_output_tokens": 400,
        "cumulative_cost_usd": 0.01,
        "max_steps": 50,
        "token_budget": None,
        "max_cost_usd": None,
        "spawn_depth": 0,
        "slots": {},
        "pending_subagents": {},
    }


def _setup_session_dir(journal: HarnessJournal, tmp_path: Path, run_id: str) -> str:
    """Open files for a journal so internal state is ready (mirrors TaskStartEvent side-effects)."""
    journal._open_files(run_id)
    return journal._session_dir


# ── State.snapshot() includes new fields ─────────────────────────────────────


class TestStateSnapshot:
    def test_snapshot_includes_schema_version(self):
        state = State(run_id="r1", max_steps=10)
        snap = state.snapshot()
        assert snap["schema_version"] == 2

    def test_snapshot_includes_token_fields(self):
        state = State(run_id="r1", max_steps=10)
        state.cumulative_input_tokens = 500
        state.cumulative_output_tokens = 300
        snap = state.snapshot()
        assert snap["cumulative_input_tokens"] == 500
        assert snap["cumulative_output_tokens"] == 300

    def test_snapshot_includes_message_tracks(self):
        state = State(run_id="r1", max_steps=10)
        state.raw_messages = [Message(role="user", content="raw")]
        state.messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="ctx"),
        ]
        snap = state.snapshot()
        assert "raw_messages" in snap
        assert "messages" in snap
        restored = State.from_snapshot(snap)
        assert restored.raw_messages[0].content == "raw"
        assert restored.messages[0].role == "system"

    def test_snapshot_includes_slots(self):
        state = State(run_id="r1", max_steps=10)
        state.slots["memory_key"] = StateSlot(slot_type="memory", content="hello", metadata={"k": "v"})
        snap = state.snapshot()
        assert "slots" in snap
        assert "memory_key" in snap["slots"]
        slot_data = snap["slots"]["memory_key"]
        assert slot_data["slot_type"] == "memory"
        assert slot_data["content"] == "hello"
        assert slot_data["metadata"] == {"k": "v"}

    def test_snapshot_includes_max_steps_budget(self):
        state = State(run_id="r1", max_steps=25)
        state.max_cost_usd = 5.0
        snap = state.snapshot()
        assert snap["max_steps"] == 25
        assert snap["max_cost_usd"] == 5.0

    def test_from_snapshot_restores_all_fields(self):
        state = State(run_id="r1", max_steps=10)
        state.cumulative_input_tokens = 111
        state.cumulative_output_tokens = 222
        state.max_cost_usd = 2.5
        state.slots["loop_fingerprints"] = StateSlot(slot_type="control", content=["sha1", "sha2"], metadata={})
        snap = state.snapshot()
        restored = State.from_snapshot(snap)
        assert restored.cumulative_input_tokens == 111
        assert restored.cumulative_output_tokens == 222
        assert restored.max_cost_usd == 2.5
        assert "loop_fingerprints" in restored.slots
        assert restored.slots["loop_fingerprints"].content == ["sha1", "sha2"]


# ── _write_segment_state atomic write ────────────────────────────────────────


class TestAtomicWrite:
    def test_segment_state_written_atomically(self, tmp_path):
        session_id = "s1"
        run_id = "run-001"
        journal = _make_journal(tmp_path, session_id=session_id)
        _setup_session_dir(journal, tmp_path, run_id)

        snapshot = _make_snapshot(run_id=run_id)
        journal._write_segment_state(run_id, snapshot, "2026-04-10T00:00:00.000Z")

        session_dir = tmp_path / "sessions" / session_id
        state_path = session_dir / f"{run_id}_state.json"
        assert state_path.exists(), f"{state_path} must exist"
        assert not Path(str(state_path) + ".tmp").exists(), ".tmp must be cleaned up"

        with open(state_path) as f:
            data = json.load(f)
        assert data["schema_version"] == 2
        assert data["session_id"] == session_id
        assert data["run_id"] == run_id

        journal._close_files()

    def test_session_index_written(self, tmp_path):
        session_id = "sess-42"
        run_id = "run-001"
        journal = _make_journal(tmp_path, session_id=session_id)
        _setup_session_dir(journal, tmp_path, run_id)

        journal._write_session_index(run_id, "2026-04-10T00:00:00.000Z")

        index_path = tmp_path / "sessions" / f"{session_id}.json"
        assert index_path.exists()
        with open(index_path) as f:
            idx = json.load(f)
        assert idx["session_id"] == session_id
        assert idx["latest_run_id"] == run_id
        assert run_id in idx["run_ids"]
        # Default state path uses new layout: runs/{session_id}/{run_id}_state.json
        assert f"{run_id}_state.json" in idx["latest_state_path"]

        journal._close_files()

    def test_session_index_accumulates_runs(self, tmp_path):
        """Multiple segments append to run_ids list without duplication."""
        session_id = "sess-multi"
        journal = _make_journal(tmp_path, session_id=session_id)
        ts = "2026-04-10T00:00:00.000Z"

        run_ids = ["run-001", "run-002", "run-003"]
        _setup_session_dir(journal, tmp_path, run_ids[0])
        for run_id in run_ids:
            journal._current_run_id = run_id
            journal._write_session_index(run_id, ts)

        index_path = tmp_path / "sessions" / f"{session_id}.json"
        with open(index_path) as f:
            idx = json.load(f)
        assert idx["run_ids"] == run_ids
        assert idx["latest_run_id"] == "run-003"

        journal._close_files()


# ── wake() ────────────────────────────────────────────────────────────────────


class TestWake:
    def _write_session_and_state(
        self,
        tmp_path: Path,
        session_id: str,
        run_id: str,
        snap: dict | None = None,
    ) -> None:
        """Set up the directory layout that wake() expects (new format)."""
        if snap is None:
            snap = _make_snapshot(run_id=run_id)

        # Write {run_id}_state.json under sessions/{session_id}/
        session_dir = tmp_path / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        state_path = session_dir / f"{run_id}_state.json"
        with open(state_path, "w") as f:
            json.dump(snap, f)

        # Write session index under sessions/
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(exist_ok=True)
        index = {
            "schema_version": 1,
            "session_id": session_id,
            "run_ids": [run_id],
            "latest_run_id": run_id,
            "latest_state_path": f"sessions/{session_id}/{run_id}_state.json",
            "latest_config_path": "harness_config.yaml",
            "updated_at": "2026-04-10T00:00:00.000Z",
        }
        with open(sessions_dir / f"{session_id}.json", "w") as f:
            json.dump(index, f)

    def test_wake_returns_state(self, tmp_path):
        self._write_session_and_state(tmp_path, "my-session", "run-abc")
        state = HarnessJournal.wake("my-session", str(tmp_path))
        assert isinstance(state, State)
        assert state.run_id == "run-abc"
        assert state.step == 3  # from _make_snapshot default

    def test_wake_session_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Session index not found"):
            HarnessJournal.wake("nonexistent-session", str(tmp_path))

    def test_wake_missing_state_file_raises(self, tmp_path):
        # Write only the index, not the state file
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(exist_ok=True)
        index = {
            "schema_version": 1,
            "session_id": "my-session",
            "run_ids": ["run-abc"],
            "latest_run_id": "run-abc",
            "latest_state_path": "sessions/my-session/run-abc_state.json",
            "latest_config_path": "harness_config.yaml",
            "updated_at": "2026-04-10T00:00:00.000Z",
        }
        with open(sessions_dir / "my-session.json", "w") as f:
            json.dump(index, f)
        # Don't create the state file
        with pytest.raises(FileNotFoundError, match="State file not found"):
            HarnessJournal.wake("my-session", str(tmp_path))

    def test_wake_restores_messages(self, tmp_path):
        # Current schema: messages live in JSONL, not in the state snapshot.
        # Snapshot step=2 → _rebuild_messages_from_jsonl includes records at step<2.
        snap = _make_snapshot(run_id="run-msg", step=2)
        self._write_session_and_state(tmp_path, "sess-msg", "run-msg", snap)

        # Write JSONL with two conversation records at steps 0 and 1.
        session_dir = tmp_path / "sessions" / "sess-msg"
        jsonl_path = session_dir / "run-msg.jsonl"
        records = [
            {
                "type": "user",
                "step": 0,
                "timestamp": 0.0,
                "message": {"role": "user", "content": "Hello"},
            },
            {
                "type": "assistant",
                "step": 1,
                "timestamp": 0.1,
                "message": {"role": "assistant", "content": "Hi there"},
                "meta": {"model": "mock", "stop_reason": "end_turn", "usage": {}},
            },
        ]
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        state = HarnessJournal.wake("sess-msg", str(tmp_path))
        assert len(state.messages) == 2
        assert state.messages[0].role == "user"
        assert state.messages[1].role == "assistant"
        assert len(state.raw_messages) == 2

    def test_wake_jsonl_records_used_model_input_snapshot_ignored(self, tmp_path):
        """JSONL conversation records are the source of truth for messages.

        When both regular records (user/assistant) and a model_input_snapshot exist,
        the regular records win — model_input_snapshot is a deprecated fallback.
        The state file no longer carries raw_messages/messages; they are always
        rebuilt from JSONL.
        """
        snap = _make_snapshot(run_id="run-pri", step=2)
        # State file has no messages — stripped before writing (Phase 6).
        self._write_session_and_state(tmp_path, "sess-pri", "run-pri", snap)

        session_dir = tmp_path / "sessions" / "sess-pri"
        jsonl_path = session_dir / "run-pri.jsonl"
        records = [
            {
                "type": "user",
                "step": 0,
                "timestamp": 0.0,
                "message": {"role": "user", "content": "from_jsonl"},
            },
            {
                "type": "assistant",
                "step": 1,
                "timestamp": 0.1,
                "message": {"role": "assistant", "content": "jsonl_reply"},
            },
            # model_input_snapshot with different content — must be ignored.
            {
                "type": "model_input_snapshot",
                "step": 1,
                "timestamp": 0.2,
                "messages": [{"role": "user", "content": "stale_content"}],
                "tools": [],
            },
        ]
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        state = HarnessJournal.wake("sess-pri", str(tmp_path))
        assert len(state.messages) == 2
        # Regular JSONL records used — model_input_snapshot is ignored.
        assert all("stale" not in str(m.content) for m in state.messages)
        assert state.messages[0].content == "from_jsonl"
        assert state.raw_messages == state.messages

    def test_wake_falls_back_to_model_input_snapshot_when_no_records(self, tmp_path):
        """Last-resort: model_input_snapshot used only when JSONL has no conversation records."""
        snap = _make_snapshot(run_id="run-eff", step=2)
        self._write_session_and_state(tmp_path, "sess-eff", "run-eff", snap)

        session_dir = tmp_path / "sessions" / "sess-eff"
        jsonl_path = session_dir / "run-eff.jsonl"
        # JSONL has only a model_input_snapshot — no user/assistant/raw_* records.
        records = [
            {
                "type": "model_input_snapshot",
                "step": 1,
                "timestamp": 0.2,
                "messages": [
                    {"role": "user", "content": "<wrapped>Hello</wrapped>"},
                ],
                "tools": [],
            },
        ]
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        state = HarnessJournal.wake("sess-eff", str(tmp_path))
        # model_input_snapshot is the only source; used as last resort.
        assert len(state.messages) == 1
        assert "wrapped" in str(state.messages[0].content)
        assert state.raw_messages == state.messages


# ── Crash-safe step_state.json checkpoint ─────────────────────────────────────


class TestStepStateCheckpoint:
    """step_state.json is written per-step so wake() recovers on hard crash."""

    def test_step_state_written_by_journal(self, tmp_path):
        """HarnessJournal writes step_state.json when StepEndEvent carries snapshot."""
        from harnessx.core.events import StepEndEvent

        session_id = "sess-crash"
        run_id = "run-crash"
        journal = _make_journal(tmp_path, session_id=session_id)
        _setup_session_dir(journal, tmp_path, run_id)

        snap = _make_snapshot(run_id=run_id, step=5)
        step_event = StepEndEvent(
            run_id=run_id,
            step_id=5,
            cumulative_tokens=100,
            cumulative_cost_usd=0.01,
            state_snapshot=snap,
        )
        asyncio.run(journal.on_event(step_event))

        # step_state.json lives under sessions/{session_id}/
        step_state_path = tmp_path / "sessions" / session_id / "step_state.json"
        assert step_state_path.exists(), "step_state.json must exist after StepEndEvent"
        with open(step_state_path) as f:
            data = json.load(f)
        assert data["schema_version"] == 2
        assert data["session_id"] == session_id
        assert data["step"] == 5

        journal._close_files()

    def test_session_index_written_at_task_start(self, tmp_path):
        """Session index is written at TaskStartEvent pointing to step_state.json."""
        from harnessx.core.events import TaskStartEvent

        session_id = "sess-start"
        run_id = "run-start"
        journal = _make_journal(tmp_path, session_id=session_id)

        start_event = TaskStartEvent(
            run_id=run_id,
            step_id=0,
            task_description="hello",
            model="gpt-4o",
        )
        asyncio.run(journal.on_event(start_event))

        index_path = tmp_path / "sessions" / f"{session_id}.json"
        assert index_path.exists(), "session index must be written at TaskStartEvent"
        with open(index_path) as f:
            idx = json.load(f)
        # latest_state_path must point to step_state.json in new layout
        assert "step_state.json" in idx["latest_state_path"]
        assert idx["latest_run_id"] == run_id

        journal._close_files()

    def test_crash_recovery_via_step_state(self, tmp_path):
        """wake() can recover from step_state.json when segment state doesn't exist."""
        run_id = "run-crash"
        session_id = "sess-crash-recover"

        # Simulate: run started, 3 steps completed, then crash (no task_end)
        session_dir = tmp_path / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        snap = _make_snapshot(run_id=run_id, step=3)
        step_state_path = session_dir / "step_state.json"
        with open(step_state_path, "w") as f:
            json.dump(snap, f)

        # Session index points to step_state.json (as written at TaskStart)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(exist_ok=True)
        index = {
            "schema_version": 1,
            "session_id": session_id,
            "run_ids": [run_id],
            "latest_run_id": run_id,
            "latest_state_path": f"sessions/{session_id}/step_state.json",
            "latest_config_path": "harness_config.yaml",
            "updated_at": "2026-04-10T00:00:00.000Z",
        }
        with open(sessions_dir / f"{session_id}.json", "w") as f:
            json.dump(index, f)

        # segment state file does NOT exist (crash before task_end)
        assert not (session_dir / f"{run_id}_state.json").exists()

        # wake() must still return a State
        state = HarnessJournal.wake(session_id, str(tmp_path))
        assert isinstance(state, State)
        assert state.step == 3

    def test_step_state_deleted_on_normal_completion(self, tmp_path):
        """step_state.json is cleaned up when task completes normally."""
        from harnessx.core.events import TaskEndEvent

        session_id = "sess-clean"
        run_id = "run-clean"
        journal = _make_journal(tmp_path, session_id=session_id)
        _setup_session_dir(journal, tmp_path, run_id)

        # Pre-create step_state.json (as if steps ran before)
        session_dir = journal._session_dir
        step_state_path = os.path.join(session_dir, "step_state.json")
        with open(step_state_path, "w") as f:
            json.dump({"step": 2}, f)

        snap = _make_snapshot(run_id=run_id, step=2)
        end_event = TaskEndEvent(
            run_id=run_id,
            step_id=2,
            exit_reason="done",
            final_output="Done.",
            total_steps=2,
            total_tokens=500,
            total_input_tokens=300,
            total_output_tokens=200,
            total_cost_usd=0.01,
            state_snapshot=snap,
        )
        asyncio.run(journal.on_event(end_event))

        assert not os.path.exists(step_state_path), "step_state.json must be deleted after normal task completion"

        state_file = os.path.join(session_dir, f"{run_id}_state.json")
        assert os.path.exists(state_file), f"{run_id}_state.json must exist after normal task completion"

        # Index must point to the segment state file
        index_path = tmp_path / "sessions" / f"{session_id}.json"
        with open(index_path) as f:
            idx = json.load(f)
        assert f"{run_id}_state.json" in idx["latest_state_path"]


# ── SegmentBoundaryEvent handling ─────────────────────────────────────────────


class TestSegmentBoundary:
    """HarnessJournal correctly rotates files on SegmentBoundaryEvent."""

    def test_segment_rotation_creates_new_files(self, tmp_path):
        """After SegmentBoundaryEvent, new {run_id}.jsonl and {run_id}_trace.jsonl are opened."""
        from harnessx.core.events import SegmentBoundaryEvent, make_run_id

        session_id = "sess-rotate"
        run_id_1 = "run-seg-1"
        run_id_2 = make_run_id()
        journal = _make_journal(tmp_path, session_id=session_id)
        _setup_session_dir(journal, tmp_path, run_id_1)
        session_dir = Path(journal._session_dir)

        boundary = SegmentBoundaryEvent(
            run_id=run_id_1,
            step_id=5,
            reason="compaction",
            new_run_id=run_id_2,
            state_snapshot=_make_snapshot(run_id=run_id_1, step=5),
        )
        asyncio.run(journal.on_event(boundary))

        # Checkpoint for first segment must be written
        assert (session_dir / f"{run_id_1}_state.json").exists(), (
            f"{run_id_1}_state.json must exist as compaction checkpoint"
        )

        # New segment files must be open
        assert journal._current_run_id == run_id_2
        assert journal._session_file is not None

        journal._close_files()

        # New segment files must exist on disk
        assert (session_dir / f"{run_id_2}.jsonl").exists(), f"{run_id_2}.jsonl must be created after boundary"
        assert (session_dir / f"{run_id_2}_trace.jsonl").exists(), (
            f"{run_id_2}_trace.jsonl must be created after boundary"
        )

    def test_segment_state_contains_reason(self, tmp_path):
        """Checkpoint written at SegmentBoundaryEvent records the boundary reason."""
        from harnessx.core.events import SegmentBoundaryEvent, make_run_id

        session_id = "sess-reason"
        run_id_1 = "run-r1"
        run_id_2 = make_run_id()
        journal = _make_journal(tmp_path, session_id=session_id)
        _setup_session_dir(journal, tmp_path, run_id_1)
        session_dir = Path(journal._session_dir)

        boundary = SegmentBoundaryEvent(
            run_id=run_id_1,
            step_id=3,
            reason="compaction",
            new_run_id=run_id_2,
            state_snapshot=_make_snapshot(run_id=run_id_1, step=3),
        )
        asyncio.run(journal.on_event(boundary))
        journal._close_files()

        state_file = session_dir / f"{run_id_1}_state.json"
        with open(state_file) as f:
            data = json.load(f)
        assert data["segment_end_reason"] == "compaction"
        assert data["segment_run_id"] == run_id_1

    def test_session_index_updated_with_new_run_id(self, tmp_path):
        """Session index lists both run_ids after a segment boundary."""
        from harnessx.core.events import SegmentBoundaryEvent, make_run_id

        session_id = "sess-index"
        run_id_1 = "run-i1"
        run_id_2 = make_run_id()
        journal = _make_journal(tmp_path, session_id=session_id)
        _setup_session_dir(journal, tmp_path, run_id_1)

        # First: write the initial index (as TaskStartEvent would do)
        journal._write_session_index(run_id_1, "2026-04-10T00:00:00.000Z")

        boundary = SegmentBoundaryEvent(
            run_id=run_id_1,
            step_id=5,
            reason="compaction",
            new_run_id=run_id_2,
            state_snapshot=_make_snapshot(run_id=run_id_1, step=5),
        )
        asyncio.run(journal.on_event(boundary))
        journal._close_files()

        index_path = tmp_path / "sessions" / f"{session_id}.json"
        with open(index_path) as f:
            idx = json.load(f)
        assert run_id_1 in idx["run_ids"]
        assert run_id_2 in idx["run_ids"]
        assert idx["latest_run_id"] == run_id_2

    def test_segment_boundary_writes_dual_context_snapshots(self, tmp_path):
        """Compaction boundary writes context_snapshot_raw + context_snapshot."""
        from harnessx.core.events import SegmentBoundaryEvent, Message, make_run_id

        session_id = "sess-dual-snap"
        run_id_1 = "run-dual-1"
        run_id_2 = make_run_id()
        journal = _make_journal(tmp_path, session_id=session_id)
        _setup_session_dir(journal, tmp_path, run_id_1)
        session_dir = Path(journal._session_dir)

        compacted = (
            Message(role="user", content="[Summary] earlier turns"),
            Message(role="assistant", content="Understood."),
        )
        boundary = SegmentBoundaryEvent(
            run_id=run_id_1,
            step_id=2,
            reason="compaction",
            new_run_id=run_id_2,
            state_snapshot=_make_snapshot(run_id=run_id_1, step=2),
            compacted_messages=compacted,
        )
        asyncio.run(journal.on_event(boundary))
        journal._close_files()

        new_segment = session_dir / f"{run_id_2}.jsonl"
        with new_segment.open(encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]

        snap_raw = next((r for r in recs if r.get("type") == "context_snapshot_raw"), None)
        snap_eff = next((r for r in recs if r.get("type") == "context_snapshot"), None)
        assert snap_raw is not None
        assert snap_eff is not None
        assert snap_raw.get("messages") == snap_eff.get("messages")


# ── Phase 4: new-format JSONL writes (raw_* + deltas) ────────────────────────


class TestNewFormatJournalWrites:
    """Verify on_raw_event writes raw_* records and on_event writes deltas only
    when processors actually modified the event content."""

    def _read_session_records(self, journal: HarnessJournal) -> list[dict]:
        path = journal._session_file.name
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _read_trace_records(self, journal: HarnessJournal) -> list[dict]:
        path = journal._trace_file.name
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_on_raw_event_writes_raw_assistant(self, tmp_path):
        """on_raw_event for ModelResponseEvent writes a raw_assistant record."""
        from harnessx.core.events import ModelResponseEvent, Usage

        run_id = "run-raw-a"
        journal = _make_journal(tmp_path, session_id="s-raw-a")
        _setup_session_dir(journal, tmp_path, run_id)
        journal._effective_session = "s-raw-a"

        evt = ModelResponseEvent(
            run_id=run_id,
            step_id=0,
            content="Hello!",
            finish_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
            model="test-model",
        )
        asyncio.run(journal.on_raw_event(evt))

        records = self._read_session_records(journal)
        journal._close_files()

        types = [r.get("type") for r in records]
        assert "raw_assistant" in types
        raw_rec = next(r for r in records if r.get("type") == "raw_assistant")
        assert raw_rec["message"]["content"] == "Hello!"
        assert raw_rec["meta"]["model"] == "test-model"

    def test_on_event_no_delta_when_assistant_content_unchanged(self, tmp_path):
        """When on_raw_event and on_event see identical content, no assistant delta."""
        from harnessx.core.events import ModelResponseEvent, Usage

        run_id = "run-nodelta-a"
        journal = _make_journal(tmp_path, session_id="s-nodelta-a")
        _setup_session_dir(journal, tmp_path, run_id)
        journal._effective_session = "s-nodelta-a"

        evt = ModelResponseEvent(
            run_id=run_id,
            step_id=0,
            content="Same content",
            finish_reason="end_turn",
            usage=Usage(input_tokens=3, output_tokens=2),
            model="m",
        )
        asyncio.run(journal.on_raw_event(evt))  # writes raw_assistant; caches msg
        asyncio.run(journal.on_event(evt))  # same content → no delta

        records = self._read_session_records(journal)
        journal._close_files()

        types = [r.get("type") for r in records]
        assert types.count("raw_assistant") == 1
        assert "assistant" not in types  # no delta

    def test_on_event_writes_assistant_delta_when_content_changed(self, tmp_path):
        """When a processor modifies the assistant message, an assistant delta is written."""
        import dataclasses
        from harnessx.core.events import ModelResponseEvent, ToolCall, Usage

        run_id = "run-delta-a"
        journal = _make_journal(tmp_path, session_id="s-delta-a")
        _setup_session_dir(journal, tmp_path, run_id)
        journal._effective_session = "s-delta-a"

        raw_evt = ModelResponseEvent(
            run_id=run_id,
            step_id=0,
            content="Original",
            finish_reason="end_turn",
            usage=Usage(input_tokens=3, output_tokens=2),
            model="m",
        )
        # Simulate a processor injecting a synthetic tool call.
        modified_evt = dataclasses.replace(
            raw_evt,
            tool_calls=(ToolCall(id="kv-1", name="_synthetic", input={}),),
        )
        asyncio.run(journal.on_raw_event(raw_evt))  # raw: no tool_calls
        asyncio.run(journal.on_event(modified_evt))  # modified: has tool_calls → delta

        records = self._read_session_records(journal)
        journal._close_files()

        types = [r.get("type") for r in records]
        assert "raw_assistant" in types
        assert "assistant" in types  # delta written
        delta = next(r for r in records if r.get("type") == "assistant")
        assert delta["message"].get("tool_calls") is not None

    def test_on_raw_event_writes_raw_tool(self, tmp_path):
        """on_raw_event for ToolResultEvent writes a raw_tool record."""
        from harnessx.core.events import ToolResultEvent

        run_id = "run-raw-t"
        journal = _make_journal(tmp_path, session_id="s-raw-t")
        _setup_session_dir(journal, tmp_path, run_id)
        journal._effective_session = "s-raw-t"

        evt = ToolResultEvent(
            run_id=run_id,
            step_id=0,
            tool_name="Bash",
            tool_call_id="tc-bash-1",
            result="exit 0\n",
            error=None,
        )
        asyncio.run(journal.on_raw_event(evt))

        records = self._read_session_records(journal)
        journal._close_files()

        types = [r.get("type") for r in records]
        assert "raw_tool" in types
        raw_rec = next(r for r in records if r.get("type") == "raw_tool")
        assert raw_rec["message"]["tool_call_id"] == "tc-bash-1"
        assert raw_rec["message"]["content"] == "exit 0\n"

    def test_on_event_no_tool_delta_when_result_unchanged(self, tmp_path):
        """No tool delta record written when processors did not change the tool result."""
        from harnessx.core.events import ToolResultEvent

        run_id = "run-nodelta-t"
        journal = _make_journal(tmp_path, session_id="s-nodelta-t")
        _setup_session_dir(journal, tmp_path, run_id)
        journal._effective_session = "s-nodelta-t"

        evt = ToolResultEvent(
            run_id=run_id,
            step_id=0,
            tool_name="Bash",
            tool_call_id="tc-same",
            result="same output",
            error=None,
        )
        asyncio.run(journal.on_raw_event(evt))  # caches (result, error)
        asyncio.run(journal.on_event(evt))  # same → no delta

        records = self._read_session_records(journal)
        journal._close_files()

        types = [r.get("type") for r in records]
        assert types.count("raw_tool") == 1
        assert "tool" not in types  # no delta

    def test_on_event_writes_tool_delta_when_result_modified(self, tmp_path):
        """A tool delta is written when a processor appended content to the result."""
        import dataclasses
        from harnessx.core.events import ToolResultEvent

        run_id = "run-delta-t"
        journal = _make_journal(tmp_path, session_id="s-delta-t")
        _setup_session_dir(journal, tmp_path, run_id)
        journal._effective_session = "s-delta-t"

        raw_evt = ToolResultEvent(
            run_id=run_id,
            step_id=0,
            tool_name="Bash",
            tool_call_id="tc-mod",
            result="original output",
        )
        modified_evt = dataclasses.replace(
            raw_evt,
            result="original output\n[LoopDetection] repeated call",
        )
        asyncio.run(journal.on_raw_event(raw_evt))
        asyncio.run(journal.on_event(modified_evt))

        records = self._read_session_records(journal)
        journal._close_files()

        types = [r.get("type") for r in records]
        assert "raw_tool" in types
        assert "tool" in types  # delta written
        delta = next(r for r in records if r.get("type") == "tool")
        assert "[LoopDetection]" in delta["message"]["content"]

    def test_on_event_trace_records_input_override(self, tmp_path):
        """Trace record includes input_override when a processor rewrote tool input."""
        import dataclasses
        from harnessx.core.events import ToolCallEvent

        run_id = "run-input-ovrd"
        journal = _make_journal(tmp_path, session_id="s-ovrd")
        _setup_session_dir(journal, tmp_path, run_id)
        journal._effective_session = "s-ovrd"

        raw_call = ToolCallEvent(
            run_id=run_id,
            step_id=0,
            tool_name="Bash",
            tool_call_id="tc-fix",
            tool_input={"command": "original"},
        )
        modified_call = dataclasses.replace(raw_call, tool_input={"command": "corrected"})

        asyncio.run(journal.on_raw_event(raw_call))  # cache raw input
        asyncio.run(journal.on_event(modified_call))  # different → input_override

        trace_path = Path(journal._trace_file.name)
        journal._close_files()

        trace_records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        call_recs = [r for r in trace_records if r.get("event_type") == "tool_call"]
        assert len(call_recs) == 1
        assert call_recs[0].get("input_override") == {"command": "corrected"}

    def test_on_event_no_input_override_when_input_unchanged(self, tmp_path):
        """No input_override in trace when processor did not rewrite the tool input."""
        from harnessx.core.events import ToolCallEvent

        run_id = "run-no-ovrd"
        journal = _make_journal(tmp_path, session_id="s-no-ovrd")
        _setup_session_dir(journal, tmp_path, run_id)
        journal._effective_session = "s-no-ovrd"

        evt = ToolCallEvent(
            run_id=run_id,
            step_id=0,
            tool_name="Bash",
            tool_call_id="tc-same-input",
            tool_input={"command": "ls"},
        )
        asyncio.run(journal.on_raw_event(evt))
        asyncio.run(journal.on_event(evt))  # same input → no override

        trace_path = Path(journal._trace_file.name)
        journal._close_files()

        trace_records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        call_recs = [r for r in trace_records if r.get("event_type") == "tool_call"]
        assert len(call_recs) == 1
        assert "input_override" not in call_recs[0]

    def test_on_event_writes_raw_assistant_when_raw_cache_missing(self, tmp_path):
        """Interrupt-bypass: on_event for ModelResponseEvent writes raw_assistant
        when on_raw_event was never called (raw_msg is None fallback).

        This covers the RunLoop CancelledError path where ModelResponseEvent is
        emitted directly without going through ProcessorChain.
        """
        from harnessx.core.events import ModelResponseEvent, Usage

        run_id = "run-interrupt-assist"
        journal = _make_journal(tmp_path, session_id="s-interrupt-assist")
        _setup_session_dir(journal, tmp_path, run_id)
        journal._effective_session = "s-interrupt-assist"

        # Simulate interrupt: on_raw_event never called, only on_event
        evt = ModelResponseEvent(
            run_id=run_id,
            step_id=2,
            content="user actively interrupted execution",
            finish_reason="stop",
            model="haiku",
            usage=Usage(),
        )
        asyncio.run(journal.on_event(evt))  # no preceding on_raw_event call

        records = self._read_session_records(journal)
        journal._close_files()

        types = [r.get("type") for r in records]
        # Must write raw_assistant (not assistant delta) so rebuild can find it.
        assert "raw_assistant" in types
        assert "assistant" not in types

    def test_on_event_writes_raw_tool_when_raw_cache_missing(self, tmp_path):
        """Interrupt-bypass: on_event for ToolResultEvent writes raw_tool when
        on_raw_event was never called (raw_cached is None fallback).

        Without this fix a standalone 'tool' delta with no raw_tool anchor would
        be silently dropped by _rebuild_messages_from_jsonl Pass B.
        """
        from harnessx.core.events import ToolResultEvent

        run_id = "run-interrupt-tool"
        journal = _make_journal(tmp_path, session_id="s-interrupt-tool")
        _setup_session_dir(journal, tmp_path, run_id)
        journal._effective_session = "s-interrupt-tool"

        # Simulate bypass: on_raw_event never called, only on_event
        evt = ToolResultEvent(
            run_id=run_id,
            step_id=1,
            tool_name="Bash",
            tool_call_id="tc-bypass",
            result="bypass output",
            error=None,
        )
        asyncio.run(journal.on_event(evt))  # no preceding on_raw_event call

        records = self._read_session_records(journal)
        journal._close_files()

        types = [r.get("type") for r in records]
        # Must write raw_tool (not tool delta) so rebuild can anchor on it.
        assert "raw_tool" in types
        assert "tool" not in types
        raw_rec = next(r for r in records if r.get("type") == "raw_tool")
        assert raw_rec["message"]["content"] == "bypass output"
        assert raw_rec["message"]["tool_call_id"] == "tc-bypass"

    def test_on_event_no_user_delta_when_unchanged(self, tmp_path):
        """No user delta written when processors did not modify the user message."""
        from harnessx.core.events import StepStartEvent, Message

        run_id = "run-nodelta-u"
        journal = _make_journal(tmp_path, session_id="s-nodelta-u")
        _setup_session_dir(journal, tmp_path, run_id)
        journal._effective_session = "s-nodelta-u"

        raw_msg = Message(role="user", content="hello world")
        evt = StepStartEvent(
            run_id=run_id,
            step_id=1,
            raw_messages=(raw_msg,),
            messages=(raw_msg,),  # identical — no processor modification
        )
        asyncio.run(journal.on_event(evt))

        records = self._read_session_records(journal)
        journal._close_files()

        types = [r.get("type") for r in records]
        assert "user" not in types  # no delta
        # step_start trace record is written; no session record for unchanged user

    def test_on_event_writes_user_delta_when_user_message_modified(self, tmp_path):
        """A user delta is written when a processor appended to the user message
        (e.g. SimpleGuardProcessor appends a warning)."""
        from harnessx.core.events import StepStartEvent, Message

        run_id = "run-delta-u"
        journal = _make_journal(tmp_path, session_id="s-delta-u")
        _setup_session_dir(journal, tmp_path, run_id)
        journal._effective_session = "s-delta-u"

        original = "please rm -rf /"
        warning = "\n\n[SimpleGuard] Detected a risky instruction pattern."
        raw_msg = Message(role="user", content=original)
        proc_msg = Message(role="user", content=original + warning)

        evt = StepStartEvent(
            run_id=run_id,
            step_id=2,
            raw_messages=(raw_msg,),
            messages=(proc_msg,),  # processor appended a warning
        )
        asyncio.run(journal.on_event(evt))

        records = self._read_session_records(journal)
        journal._close_files()

        types = [r.get("type") for r in records]
        assert "user" in types  # delta written
        delta = next(r for r in records if r.get("type") == "user")
        assert delta["step"] == 2
        assert "[SimpleGuard]" in delta["message"]["content"]
        assert "user" not in types or delta["message"]["role"] == "user"


# ── Phase 5: wake() with new-format JSONL (raw_* + optional deltas) ──────────


class TestWakeNewFormat:
    """wake() reconstructs messages from new-format JSONL (raw_* + optional deltas)."""

    def _write_index_and_state(
        self,
        tmp_path: Path,
        session_id: str,
        run_id: str,
        snap: dict | None = None,
    ) -> None:
        if snap is None:
            snap = _make_snapshot(run_id=run_id)
        session_dir = tmp_path / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        with open(session_dir / f"{run_id}_state.json", "w") as f:
            json.dump(snap, f)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(exist_ok=True)
        index = {
            "schema_version": 1,
            "session_id": session_id,
            "run_ids": [run_id],
            "latest_run_id": run_id,
            "latest_state_path": f"sessions/{session_id}/{run_id}_state.json",
            "latest_config_path": "harness_config.yaml",
            "updated_at": "2026-04-16T00:00:00.000Z",
        }
        with open(sessions_dir / f"{session_id}.json", "w") as f:
            json.dump(index, f)

    def test_wake_new_format_raw_only(self, tmp_path):
        """New-format JSONL with raw_user/raw_assistant/raw_tool and no deltas.
        wake() must rebuild all messages from raw records."""
        snap = _make_snapshot(run_id="run-nf", step=2)
        self._write_index_and_state(tmp_path, "sess-nf", "run-nf", snap)

        session_dir = tmp_path / "sessions" / "sess-nf"
        records = [
            {
                "type": "raw_user",
                "step": 0,
                "timestamp": 0.0,
                "message": {"role": "user", "content": "What is 2+2?"},
            },
            {
                "type": "raw_assistant",
                "step": 0,
                "timestamp": 0.1,
                "message": {"role": "assistant", "content": "4"},
                "meta": {"model": "m", "stop_reason": "end_turn", "usage": {}},
            },
            {
                "type": "raw_tool",
                "step": 1,
                "timestamp": 0.2,
                "message": {
                    "role": "tool",
                    "content": "ok",
                    "tool_call_id": "tc-1",
                    "name": "Bash",
                },
            },
            {
                "type": "raw_assistant",
                "step": 1,
                "timestamp": 0.3,
                "message": {"role": "assistant", "content": "Done"},
                "meta": {"model": "m", "stop_reason": "end_turn", "usage": {}},
            },
        ]
        with (session_dir / "run-nf.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        state = HarnessJournal.wake("sess-nf", str(tmp_path))
        assert len(state.raw_messages) == 4
        assert state.raw_messages[0].role == "user"
        assert state.raw_messages[1].role == "assistant"
        assert state.raw_messages[2].role == "tool"
        assert state.raw_messages[3].role == "assistant"
        assert state.raw_messages[3].content == "Done"

    def test_wake_new_format_uses_assistant_delta(self, tmp_path):
        """New-format: assistant delta (processor-injected tool call) is preferred
        over the raw_assistant record for the same step."""
        snap = _make_snapshot(run_id="run-ad", step=1)
        self._write_index_and_state(tmp_path, "sess-ad", "run-ad", snap)

        session_dir = tmp_path / "sessions" / "sess-ad"
        records = [
            {
                "type": "raw_user",
                "step": 0,
                "timestamp": 0.0,
                "message": {"role": "user", "content": "Hello"},
            },
            {
                "type": "raw_assistant",
                "step": 0,
                "timestamp": 0.1,
                "message": {"role": "assistant", "content": "original"},
                "meta": {},
            },
            # Processor injected a keepalive tool call:
            {
                "type": "assistant",
                "step": 0,
                "timestamp": 0.15,
                "message": {
                    "role": "assistant",
                    "content": "original",
                    "tool_calls": [{"id": "kv-1", "name": "_verify_keepalive", "input": {}}],
                },
                "meta": {},
            },
        ]
        with (session_dir / "run-ad.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        state = HarnessJournal.wake("sess-ad", str(tmp_path))
        assert len(state.raw_messages) == 2
        assert len(state.messages) == 2
        # raw_messages keeps pre-processor content.
        from harnessx.core.events import message_to_dict

        raw_d = message_to_dict(state.raw_messages[1])
        assert "tool_calls" not in raw_d
        # messages uses delta when present.
        eff_d = message_to_dict(state.messages[1])
        assert "tool_calls" in eff_d

    def test_wake_new_format_uses_tool_delta(self, tmp_path):
        """New-format: tool delta (processor-appended warning) replaces raw_tool."""
        snap = _make_snapshot(run_id="run-td", step=1)
        self._write_index_and_state(tmp_path, "sess-td", "run-td", snap)

        session_dir = tmp_path / "sessions" / "sess-td"
        records = [
            {
                "type": "raw_user",
                "step": 0,
                "timestamp": 0.0,
                "message": {"role": "user", "content": "run ls"},
            },
            {
                "type": "raw_assistant",
                "step": 0,
                "timestamp": 0.1,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "tc-loop", "name": "Bash", "input": {"command": "ls"}}],
                },
                "meta": {},
            },
            {
                "type": "raw_tool",
                "step": 0,
                "timestamp": 0.2,
                "message": {
                    "role": "tool",
                    "content": "file1 file2",
                    "tool_call_id": "tc-loop",
                    "name": "Bash",
                },
            },
            # Processor appended a warning:
            {
                "type": "tool",
                "step": 0,
                "timestamp": 0.25,
                "message": {
                    "role": "tool",
                    "content": "file1 file2\n[LoopDetection] repeated call",
                    "tool_call_id": "tc-loop",
                    "name": "Bash",
                },
            },
        ]
        with (session_dir / "run-td.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        state = HarnessJournal.wake("sess-td", str(tmp_path))
        assert len(state.raw_messages) == 3
        assert len(state.messages) == 3
        raw_tool_msg = state.raw_messages[2]
        eff_tool_msg = state.messages[2]
        assert raw_tool_msg.role == "tool"
        assert eff_tool_msg.role == "tool"
        assert "[LoopDetection]" not in str(raw_tool_msg.content)
        assert "[LoopDetection]" in str(eff_tool_msg.content)

    def test_wake_new_format_raw_used_when_no_delta(self, tmp_path):
        """New-format without any delta: raw records are used as-is."""
        snap = _make_snapshot(run_id="run-nodt", step=1)
        self._write_index_and_state(tmp_path, "sess-nodt", "run-nodt", snap)

        session_dir = tmp_path / "sessions" / "sess-nodt"
        records = [
            {
                "type": "raw_user",
                "step": 0,
                "timestamp": 0.0,
                "message": {"role": "user", "content": "hi"},
            },
            {
                "type": "raw_assistant",
                "step": 0,
                "timestamp": 0.1,
                "message": {"role": "assistant", "content": "no modification"},
                "meta": {},
            },
        ]
        with (session_dir / "run-nodt.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        state = HarnessJournal.wake("sess-nodt", str(tmp_path))
        assert len(state.raw_messages) == 2
        assert state.raw_messages[1].content == "no modification"

    def test_wake_recovers_tool_from_interrupt_bypass_path(self, tmp_path):
        """New-format: when on_raw_event was bypassed for a tool result,
        on_event writes raw_tool as fallback.  wake() must still recover it.

        This validates the fix for issue #13 (orphaned tool delta loss):
        a standalone 'tool' record without a raw_tool anchor was silently
        dropped by Pass B; the fix writes raw_tool instead of tool.
        """
        snap = _make_snapshot(run_id="run-tbypass", step=1)
        self._write_index_and_state(tmp_path, "sess-tbypass", "run-tbypass", snap)

        session_dir = tmp_path / "sessions" / "sess-tbypass"
        # Simulate a JSONL where the tool result was written via the
        # interrupt-bypass path: only raw_tool present (no preceding raw_tool
        # from on_raw_event — the record was written directly by on_event).
        records = [
            {
                "type": "raw_user",
                "step": 0,
                "timestamp": 0.0,
                "message": {"role": "user", "content": "run ls"},
            },
            {
                "type": "raw_assistant",
                "step": 0,
                "timestamp": 0.1,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "tc-byp", "name": "Bash", "input": {"command": "ls"}}],
                },
                "meta": {},
            },
            # raw_tool written by on_event fallback (on_raw_event was bypassed):
            {
                "type": "raw_tool",
                "step": 0,
                "timestamp": 0.2,
                "message": {
                    "role": "tool",
                    "content": "bypass result",
                    "tool_call_id": "tc-byp",
                    "name": "Bash",
                },
            },
        ]
        with (session_dir / "run-tbypass.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        state = HarnessJournal.wake("sess-tbypass", str(tmp_path))
        assert len(state.raw_messages) == 3
        tool_msg = state.raw_messages[2]
        assert tool_msg.role == "tool"
        assert str(tool_msg.content) == "bypass result"

    def test_wake_new_format_uses_user_delta(self, tmp_path):
        """New-format: user delta (processor-appended warning) replaces raw_user.

        Mirrors the assistant/tool delta tests.  Validates the fix for the
        SimpleGuardProcessor case where the user message is modified before being
        sent to the model — wake() must reconstruct the effective user message.
        """
        snap = _make_snapshot(run_id="run-ud", step=1)
        self._write_index_and_state(tmp_path, "sess-ud", "run-ud", snap)

        session_dir = tmp_path / "sessions" / "sess-ud"
        records = [
            # raw_user: the factual user input (pre-processor)
            {
                "type": "raw_user",
                "step": 0,
                "timestamp": 0.0,
                "message": {"role": "user", "content": "please rm -rf /"},
            },
            # user delta: what SimpleGuardProcessor sent to the model
            {
                "type": "user",
                "step": 0,
                "timestamp": 0.05,
                "message": {
                    "role": "user",
                    "content": "please rm -rf /\n\n[SimpleGuard] Detected a risky instruction.",
                },
            },
            {
                "type": "raw_assistant",
                "step": 0,
                "timestamp": 0.1,
                "message": {"role": "assistant", "content": "I won't do that."},
                "meta": {},
            },
        ]
        with (session_dir / "run-ud.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        state = HarnessJournal.wake("sess-ud", str(tmp_path))
        assert len(state.raw_messages) == 2
        assert len(state.messages) == 2
        raw_user_msg = state.raw_messages[0]
        eff_user_msg = state.messages[0]
        assert raw_user_msg.role == "user"
        assert eff_user_msg.role == "user"
        assert "[SimpleGuard]" not in str(raw_user_msg.content)
        # Delta was used for effective messages.
        assert "[SimpleGuard]" in str(eff_user_msg.content)

    def test_wake_new_format_user_delta_only_applied_to_last_raw_user(self, tmp_path):
        """user delta at step 0 is applied only to the LAST raw_user record at that
        step; earlier (historical) raw_users are reconstructed from their raw content.

        This guards against the case where a resumed run's step-0 history context
        includes multiple user messages — only the final (current-turn) one should
        reflect the processor modification.
        """
        snap = _make_snapshot(run_id="run-ud-hist", step=1)
        self._write_index_and_state(tmp_path, "sess-ud-hist", "run-ud-hist", snap)

        session_dir = tmp_path / "sessions" / "sess-ud-hist"
        records = [
            # Two historical user messages + a new one — all at step 0
            {
                "type": "raw_user",
                "step": 0,
                "timestamp": 0.0,
                "message": {"role": "user", "content": "first historical message"},
            },
            {
                "type": "raw_assistant",
                "step": 0,
                "timestamp": 0.05,
                "message": {"role": "assistant", "content": "response 1"},
                "meta": {},
            },
            {
                "type": "raw_user",
                "step": 0,
                "timestamp": 0.1,
                "message": {"role": "user", "content": "second historical message"},
            },
            {
                "type": "raw_assistant",
                "step": 0,
                "timestamp": 0.15,
                "message": {"role": "assistant", "content": "response 2"},
                "meta": {},
            },
            # current-turn input (last raw_user at step 0)
            {
                "type": "raw_user",
                "step": 0,
                "timestamp": 0.2,
                "message": {"role": "user", "content": "please rm -rf /"},
            },
            # processor delta for the current-turn input
            {
                "type": "user",
                "step": 0,
                "timestamp": 0.25,
                "message": {
                    "role": "user",
                    "content": "please rm -rf /\n\n[SimpleGuard] risky!",
                },
            },
            {
                "type": "raw_assistant",
                "step": 0,
                "timestamp": 0.3,
                "message": {"role": "assistant", "content": "I won't."},
                "meta": {},
            },
        ]
        with (session_dir / "run-ud-hist.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        state = HarnessJournal.wake("sess-ud-hist", str(tmp_path))
        assert len(state.raw_messages) == 6  # 3 user + 3 asst
        assert len(state.messages) == 6
        raw_user_msgs = [m for m in state.raw_messages if m.role == "user"]
        eff_user_msgs = [m for m in state.messages if m.role == "user"]
        assert len(raw_user_msgs) == 3
        assert len(eff_user_msgs) == 3
        # Historical messages must be unchanged in both tracks.
        assert str(raw_user_msgs[0].content) == "first historical message"
        assert str(raw_user_msgs[1].content) == "second historical message"
        assert str(eff_user_msgs[0].content) == "first historical message"
        assert str(eff_user_msgs[1].content) == "second historical message"
        # Current-turn message: raw stays factual, effective uses delta.
        assert "[SimpleGuard]" not in str(raw_user_msgs[2].content)
        assert "[SimpleGuard]" in str(eff_user_msgs[2].content)

    def test_wake_new_format_raw_user_used_when_no_delta(self, tmp_path):
        """New-format: raw_user is used when no user delta exists for that step."""
        snap = _make_snapshot(run_id="run-ud-nodt", step=1)
        self._write_index_and_state(tmp_path, "sess-ud-nodt", "run-ud-nodt", snap)

        session_dir = tmp_path / "sessions" / "sess-ud-nodt"
        records = [
            {
                "type": "raw_user",
                "step": 0,
                "timestamp": 0.0,
                "message": {"role": "user", "content": "no modifications here"},
            },
            {
                "type": "raw_assistant",
                "step": 0,
                "timestamp": 0.1,
                "message": {"role": "assistant", "content": "ok"},
                "meta": {},
            },
        ]
        with (session_dir / "run-ud-nodt.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        state = HarnessJournal.wake("sess-ud-nodt", str(tmp_path))
        assert len(state.raw_messages) == 2
        assert len(state.messages) == 2
        user_msg = state.raw_messages[0]
        assert user_msg.role == "user"
        assert str(user_msg.content) == "no modifications here"
        assert str(state.messages[0].content) == "no modifications here"

    def test_wake_strict_validation_rejects_mismatched_snapshot_lengths(self, tmp_path):
        """Strict wake validation must fail when raw/effective tracks differ in length."""
        snap = _make_snapshot(run_id="run-strict", step=1)
        self._write_index_and_state(tmp_path, "sess-strict", "run-strict", snap)
        session_dir = tmp_path / "sessions" / "sess-strict"
        records = [
            {
                "type": "context_snapshot_raw",
                "step": 0,
                "timestamp": 0.0,
                "messages": [{"role": "user", "content": "raw-only"}],
            },
            {
                "type": "context_snapshot",
                "step": 0,
                "timestamp": 0.01,
                "messages": [],
            },
        ]
        with (session_dir / "run-strict.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        with pytest.raises(ValueError, match="len\\(raw_messages\\)"):
            HarnessJournal.wake("sess-strict", str(tmp_path))
