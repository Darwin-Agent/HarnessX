# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Any

from ..core.events import (
    BeforeModelEvent,
    Event,
    ModelResponseEvent,
    ProcessorTriggerEvent,
    SegmentBoundaryEvent,
    SpawnSubAgentEvent,
    StepEndEvent,
    StepStartEvent,
    TaskEndEvent,
    TaskStartEvent,
    ToolCallEvent,
    ToolResultEvent,
    _extract_text,
    dict_to_message,
    message_to_dict,
)

# Tool result content larger than this is written to tool_results/{id}.txt
# and referenced by path in the segment jsonl.
INLINE_LIMIT = 2048  # bytes

# Base64 string length above which a media block is externalized to media/.
# 2048 chars ≈ ~1.5 KB raw bytes — small images/audio clips stay inline;
# anything larger goes to an external file and is restored on wake().
MEDIA_INLINE_LIMIT = 2048  # characters


def _iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class HarnessJournal:
    """Streams events to per-segment JSONL files and writes recovery artifacts.

    Each recording *segment* corresponds to one run_id and produces two files:

    * ``{run_id}.jsonl``        — Claude-Code-compatible conversation format
    * ``{run_id}_trace.jsonl``  — execution metadata (timing, cost, approvals)

    A segment ends when either a ``SegmentBoundaryEvent`` is received (e.g. after
    context compaction) or the task completes.  On boundary the journal writes a
    ``{run_id}_state.json`` checkpoint and opens new file handles for the next
    segment.

    Recovery artifacts:

    * ``{run_id}_state.json``     — complete State snapshot (written atomically)
    * ``sessions/{session_id}.json`` — persistent index for wake() recovery

    All writes are immediately flushed so a run is recoverable from disk even if
    the process crashes mid-task.
    """

    _USER_INTERRUPTED_MESSAGE = "user actively interrupted execution"

    def __init__(
        self,
        base_dir: str = "sessions",
        export_jsonl: bool = True,
        silent: bool = False,
        log_level: str = "info",  # kept for API compatibility
        session_id: str | None = None,  # associated session; falls back to run_id
        agent_id: str | None = None,  # when set, routes sessions to AGENT_HOME workspace
        project: str | None = None,  # project name within agent workspace
    ):
        if agent_id is not None and base_dir == "sessions":
            from ..home import agent_workspace_root

            ws_root = agent_workspace_root(agent_id, project)
            base_dir = str(ws_root / "sessions")
        self.base_dir = base_dir
        self.export_jsonl = export_jsonl
        self.silent = silent
        self.session_id = session_id
        self.config_hash: str | None = None  # injected by harness.run() after config write
        self._logger = None
        self._init_logger()

        # Per-run state — reset on TaskStartEvent, cleared on TaskEndEvent
        self._session_dir: str | None = None  # runs/{session_id}/
        self._current_run_id: str | None = None
        self._last_closed_run_id: str | None = None  # run_id of the most recently closed segment
        self._effective_session: str | None = None  # session_id label used in jsonl records
        self._session_file: Any = None
        self._trace_file: Any = None
        self._last_uuid: str | None = None
        self._last_system_prompt: str = ""
        self._last_tools_hash: str = ""
        self._segment_has_context_snapshot: bool = False

        # Buffer for ProcessorTriggerEvents that arrive before the trace file is
        # opened (task_start processors run before tracer.on_event(TaskStartEvent)).
        # Stores raw events (not dicts) so session_id can be resolved at flush time.
        # Flushed immediately after the task_start trace record is written.
        self._pending_triggers: list[ProcessorTriggerEvent] = []

        # Raw-event caches — populated by on_raw_event before processors run,
        # consumed by on_event to decide whether to write a delta record.
        # Cleared at task start and task end to prevent stale data across runs.
        self._raw_assistant: dict[int, dict] = {}  # step_id → raw msg dict
        self._raw_tool_results: dict[str, tuple] = {}  # tool_call_id → (result, error)
        self._raw_tool_inputs: dict[str, dict] = {}  # tool_call_id → raw tool_input

        # Snapshot of assembled messages at step_start time, used to detect
        # synthetic user messages added later by before_model processors.
        self._last_step_messages: tuple = ()
        # Pre-processor BeforeModelEvent messages (set by on_raw_event); consumed
        # by on_event(BeforeModelEvent) to detect synthetic user injections.
        self._before_model_raw_msgs: tuple = ()

    def _init_logger(self) -> None:
        try:
            import structlog

            self._logger = structlog.get_logger()
        except ImportError:
            self._logger = None

    # ── File management ───────────────────────────────────────────────────────

    def _session_dir_path(self) -> str:
        """Return runs/{session_id}/ path, creating it if necessary."""
        assert self._session_dir is not None
        return self._session_dir

    def _open_segment(self, run_id: str) -> None:
        """Open JSONL file handles for a new segment.

        Dedup hashes are reset only when switching to a genuinely new run_id.
        Same run_id means we are resuming or continuing the same JSONL file —
        the prior system/tools records are already there, no need to re-write.
        """
        assert self._session_dir is not None
        # Reset dedup hashes only when this is a different run from the previous one.
        # _last_closed_run_id covers the resume case: after _close_files() clears
        # _current_run_id, we still know which run was last closed.
        prev_run = self._current_run_id or self._last_closed_run_id
        if run_id != prev_run:
            # The run_id differs (or this is a fresh journal instance after process
            # restart).  Before clearing the hashes, check whether the segment file
            # already has tools/system records — if so, restore the hashes from the
            # file so the dedup logic naturally suppresses re-writing identical entries.
            segment_path = os.path.join(self._session_dir, f"{run_id}.jsonl")
            restored = self._restore_hashes_from_file(segment_path)
            if not restored:
                self._last_system_prompt = ""
                self._last_tools_hash = ""
        self._current_run_id = run_id
        if self.export_jsonl:
            self._session_file = open(
                os.path.join(self._session_dir, f"{run_id}.jsonl"),
                "a",
                encoding="utf-8",
            )
            self._trace_file = open(
                os.path.join(self._session_dir, f"{run_id}_trace.jsonl"),
                "a",
                encoding="utf-8",
            )

    def _restore_hashes_from_file(self, path: str) -> bool:
        """Scan an existing segment JSONL file and restore dedup hashes.

        Reads the last ``tools`` and ``system`` records in *path* and stores
        their hashes in ``_last_tools_hash`` / ``_last_system_prompt`` so that
        the dedup logic in ``on_event(StepStartEvent)`` will not re-write
        identical entries after a process restart.

        Returns True if at least one hash was restored, False if the file does
        not exist or contains no tools/system records.
        """
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
        last_tools_hash: str = ""
        last_system_prompt: str = ""
        try:
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except Exception:
                        continue
                    rec_type = rec.get("type")
                    msg = rec.get("message") or {}
                    content = msg.get("content", "")
                    if rec_type == "tools" and isinstance(content, list):
                        last_tools_hash = ",".join(t.get("name", "") for t in content if isinstance(t, dict))
                    elif rec_type == "system" and isinstance(content, str):
                        last_system_prompt = content
        except Exception:
            return False
        if last_tools_hash or last_system_prompt:
            self._last_tools_hash = last_tools_hash
            self._last_system_prompt = last_system_prompt
            return True
        return False

    def _close_segment_files(self) -> None:
        for fh in (self._session_file, self._trace_file):
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
        self._session_file = None
        self._trace_file = None
        # Do NOT reset _last_system_prompt / _last_tools_hash here.
        # Dedup hash ownership belongs to _open_segment: it resets them only
        # when switching to a new run_id (different file).  Resetting here
        # caused double-system records on resume: close cleared the hash, then
        # the resume path (is_resume=True) skipped its own reset, leaving an
        # empty hash that triggered a redundant system write on step 0.
        self._segment_has_context_snapshot = False

    def _open_files(self, run_id: str) -> None:
        """Create session directory and open the first segment."""
        effective_session = self.session_id or run_id
        self._effective_session = effective_session
        session_dir = os.path.join(self.base_dir, effective_session)
        os.makedirs(session_dir, exist_ok=True)
        self._session_dir = session_dir
        self._open_segment(run_id)

    def _close_files(self) -> None:
        self._last_closed_run_id = self._current_run_id  # preserve for resume detection
        self._close_segment_files()
        self._session_dir = None
        self._current_run_id = None
        self._raw_assistant.clear()
        self._raw_tool_results.clear()
        self._raw_tool_inputs.clear()
        self._last_step_messages = ()
        self._before_model_raw_msgs = ()

    # ── Write helpers ─────────────────────────────────────────────────────────

    def _write_session(self, data: dict) -> None:
        if not self._session_file:
            return
        uid = str(uuid.uuid4())
        data["uuid"] = uid
        data["parent_uuid"] = self._last_uuid
        self._last_uuid = uid
        self._session_file.write(json.dumps(data, ensure_ascii=False) + "\n")
        self._session_file.flush()

    def _write_trace(self, data: dict) -> None:
        if not self._trace_file:
            return
        self._trace_file.write(json.dumps(data, ensure_ascii=False) + "\n")
        self._trace_file.flush()

    def _prepare_tool_result(self, event: ToolResultEvent, suffix: str = "") -> tuple[str, dict]:
        """Prepare tool result content for the session JSONL.

        Large results (> INLINE_LIMIT bytes) are written to an external file
        and referenced via ``meta.content_ref`` so the JSONL line stays small.

        Args:
            event:  The ToolResultEvent to extract content from.
            suffix: Optional filename suffix inserted before ``.txt`` (e.g.
                    ``"_raw"`` for the pre-processor snapshot written by
                    ``on_raw_event``).  Default ``""`` produces
                    ``{tool_call_id}.txt``; ``"_raw"`` produces
                    ``{tool_call_id}_raw.txt``.  Using distinct suffixes
                    prevents raw and delta records from overwriting each other.

        Returns:
            (content, meta): ``content`` is the inline string (empty when
            externalized); ``meta`` is a dict with ``content_ref`` /
            ``content_size`` keys when externalized, otherwise empty.
        """
        # TODO: content_blocks (inline image/audio blocks from MCP tools) are not
        # persisted here — only the text result is written to JSONL. Sessions that
        # used multimodal MCP tools cannot be fully reconstructed via wake().
        raw = event.result if not event.error else f"Error: {event.error}"
        if self._session_dir and len(raw.encode()) > INLINE_LIMIT:
            tr_dir = os.path.join(self._session_dir, "tool_results")
            os.makedirs(tr_dir, exist_ok=True)
            fname = f"{event.tool_call_id}{suffix}.txt"
            fpath = os.path.join(tr_dir, fname)
            Path(fpath).write_text(raw, encoding="utf-8")
            return "", {
                "content_ref": f"tool_results/{event.tool_call_id}{suffix}.txt",
                "content_size": len(raw.encode()),
            }
        return raw, {}

    def _externalize_user_content(self, content: "str | list", fname_base: str) -> "tuple[str | list, dict]":
        """Serialize user message content, externalizing large base64 media blocks.

        Handles three block types produced by the gateway's _build_description():
          - Anthropic image:  {"type": "image", "source": {"type": "base64", "data": ...}}
          - Audio:            {"type": "input_audio", "input_audio": {"data": ..., "format": ...}}
          - Video/image_url:  {"type": "image_url", "image_url": {"url": "data:{mime};base64,..."}}

        Large base64 payloads (> MEDIA_INLINE_LIMIT chars) are written to
        ``media/{fname_base}_{idx}.{ext}`` and replaced by a ``media_ref`` key
        so that the JSONL line stays small.  Restored by _restore_media_refs().

        Returns (prepared_content, {}) — meta is reserved for future use.
        """
        if not isinstance(content, list) or not self._session_dir:
            return content, {}

        import base64 as _b64

        media_dir = os.path.join(self._session_dir, "media")
        prepared: list = []
        for idx, block in enumerate(content):
            if not isinstance(block, dict):
                prepared.append(block)
                continue

            block_type = block.get("type", "")

            # Anthropic image: source.data
            if block_type == "image" and isinstance(block.get("source"), dict):
                src = block["source"]
                if src.get("type") == "base64" and len(src.get("data", "")) > MEDIA_INLINE_LIMIT:
                    mime = src.get("media_type", "image/jpeg")
                    ext = mime.split("/")[-1].split("+")[0]
                    fname = f"{fname_base}_{idx}.{ext}"
                    try:
                        os.makedirs(media_dir, exist_ok=True)
                        Path(os.path.join(media_dir, fname)).write_bytes(_b64.standard_b64decode(src["data"]))
                        new_src = {k: v for k, v in src.items() if k != "data"}
                        new_src["media_ref"] = f"media/{fname}"
                        prepared.append({**block, "source": new_src})
                        continue
                    except Exception:
                        pass

            # Audio: input_audio.data
            elif block_type == "input_audio" and isinstance(block.get("input_audio"), dict):
                inner = block["input_audio"]
                if len(inner.get("data", "")) > MEDIA_INLINE_LIMIT:
                    fmt = inner.get("format", "bin")
                    fname = f"{fname_base}_{idx}.{fmt}"
                    try:
                        os.makedirs(media_dir, exist_ok=True)
                        Path(os.path.join(media_dir, fname)).write_bytes(_b64.standard_b64decode(inner["data"]))
                        new_inner = {k: v for k, v in inner.items() if k != "data"}
                        new_inner["media_ref"] = f"media/{fname}"
                        prepared.append({**block, "input_audio": new_inner})
                        continue
                    except Exception:
                        pass

            # Video / image_url with data URI: image_url.url
            elif block_type == "image_url" and isinstance(block.get("image_url"), dict):
                url = block["image_url"].get("url", "")
                if url.startswith("data:") and len(url) > MEDIA_INLINE_LIMIT:
                    try:
                        header, b64_data = url.split(",", 1)
                        mime = header.split(";")[0][5:]  # strip "data:"
                        ext = mime.split("/")[-1].split("+")[0]
                        fname = f"{fname_base}_{idx}.{ext}"
                        os.makedirs(media_dir, exist_ok=True)
                        Path(os.path.join(media_dir, fname)).write_bytes(_b64.standard_b64decode(b64_data))
                        # Keep "data:{mime};base64," prefix (without payload) so restore
                        # can reconstruct the full data URI by appending the re-encoded bytes.
                        new_inner = {**block["image_url"], "url": f"data:{mime};base64,", "media_ref": f"media/{fname}"}
                        prepared.append({**block, "image_url": new_inner})
                        continue
                    except Exception:
                        pass

            prepared.append(block)

        return prepared, {}

    @classmethod
    def _restore_media_refs(cls, content: "str | list", jsonl_path: Path) -> "str | list":
        """Restore media_ref file references back to inline base64 in content blocks.

        Inverse of _externalize_user_content().  Called by _message_from_record()
        when reconstructing messages during wake().
        """
        if not isinstance(content, list):
            return content

        import base64 as _b64

        restored: list = []
        for block in content:
            if not isinstance(block, dict):
                restored.append(block)
                continue

            block_type = block.get("type", "")

            if block_type == "image" and isinstance(block.get("source"), dict):
                src = block["source"]
                ref = src.get("media_ref")
                if ref:
                    try:
                        raw = (jsonl_path.parent / ref).read_bytes()
                        new_src = {k: v for k, v in src.items() if k != "media_ref"}
                        new_src["data"] = _b64.standard_b64encode(raw).decode()
                        restored.append({**block, "source": new_src})
                        continue
                    except Exception:
                        pass

            elif block_type == "input_audio" and isinstance(block.get("input_audio"), dict):
                inner = block["input_audio"]
                ref = inner.get("media_ref")
                if ref:
                    try:
                        raw = (jsonl_path.parent / ref).read_bytes()
                        new_inner = {k: v for k, v in inner.items() if k != "media_ref"}
                        new_inner["data"] = _b64.standard_b64encode(raw).decode()
                        restored.append({**block, "input_audio": new_inner})
                        continue
                    except Exception:
                        pass

            elif block_type == "image_url" and isinstance(block.get("image_url"), dict):
                inner = block["image_url"]
                ref = inner.get("media_ref")
                if ref:
                    try:
                        raw = (jsonl_path.parent / ref).read_bytes()
                        b64 = _b64.standard_b64encode(raw).decode()
                        # url was stored as "data:{mime};base64," — append the payload.
                        new_inner = {k: v for k, v in inner.items() if k != "media_ref"}
                        new_inner["url"] = new_inner.get("url", "") + b64
                        restored.append({**block, "image_url": new_inner})
                        continue
                    except Exception:
                        pass

            restored.append(block)

        return restored

    def _atomic_write_json(self, path: str, data: dict) -> None:
        """Write data as JSON atomically (write .tmp then os.replace)."""
        tmp = path + ".tmp"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
        os.replace(tmp, path)

    def _write_segment_state(
        self,
        run_id: str,
        snapshot: dict,
        ts: str,
        reason: str = "task_end",
    ) -> str:
        """Write {run_id}_state.json and return its relative path from workspace root."""
        if not self._session_dir:
            return ""
        effective_session = self.session_id or run_id
        data = dict(snapshot)
        # Messages are authoritative in the JSONL (raw_* records + deltas).
        # Omit them from the state file so the file stays compact and wake()
        # always rebuilds from JSONL rather than a potentially stale snapshot.
        data.pop("raw_messages", None)
        data.pop("messages", None)
        data["session_id"] = effective_session
        data["segment_run_id"] = run_id
        data["segment_end_reason"] = reason
        data.setdefault("schema_version", 2)
        if self.config_hash:
            data["config_hash"] = self.config_hash
        # Pointer to the full append-only event log for this segment.
        rel_session = os.path.relpath(self._session_dir, str(Path(self.base_dir).resolve().parent))
        data["segment_jsonl_path"] = f"{rel_session}/{run_id}.jsonl"
        fname = f"{run_id}_state.json"
        path = os.path.join(self._session_dir, fname)
        self._atomic_write_json(path, data)
        effective_session_dir = os.path.relpath(self._session_dir, str(Path(self.base_dir).resolve().parent))
        return f"{effective_session_dir}/{fname}"

    def _write_session_index(
        self,
        run_id: str,
        ts: str,
        latest_state_path: str | None = None,
    ) -> None:
        """Update sessions/{session_id}.json with this run.

        Args:
            run_id: The current run identifier.
            ts: ISO timestamp string.
            latest_state_path: Relative path (from workspace root) to the current
                best state snapshot.  Defaults to
                ``runs/{session_id}/{run_id}_state.json``.
                Pass ``runs/{session_id}/step_state.json`` during in-progress runs
                so wake() can recover even if the process crashes before task_end.
        """
        effective_session = self.session_id or run_id
        # base_dir IS the sessions/ directory; index sits at base_dir/{session_id}.json
        sessions_dir = str(Path(self.base_dir).resolve())
        Path(sessions_dir).mkdir(parents=True, exist_ok=True)
        index_path = os.path.join(sessions_dir, f"{effective_session}.json")

        # Load existing index or start fresh
        existing: dict = {}
        if os.path.exists(index_path):
            try:
                with open(index_path, encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        run_ids: list = list(existing.get("run_ids", []))
        if run_id not in run_ids:
            run_ids.append(run_id)

        if latest_state_path is None:
            # Default: point to the definitive state for this segment
            session_rel = os.path.relpath(
                self._session_dir or "",
                str(Path(self.base_dir).resolve().parent),
            )
            latest_state_path = f"{session_rel}/{run_id}_state.json"

        # Point to the content-addressed config when available (precise reproduction).
        # Configs live at agent_home/configs/ (global, not per-workspace), so store
        # an absolute path so wake_config() can find it regardless of which workspace
        # the resume happens from.
        if self.config_hash:
            from ..home import agent_configs_dir as _acd

            config_path = str(_acd() / f"{self.config_hash}.yaml")
        else:
            from ..home import agent_home as _ah

            config_path = str(_ah() / "harness_config.yaml")

        index = {
            "schema_version": 1,
            "session_id": effective_session,
            "run_ids": run_ids,
            "latest_run_id": run_id,
            "latest_state_path": latest_state_path,
            "latest_config_path": config_path,
            "updated_at": ts,
        }
        self._atomic_write_json(index_path, index)

    # ── Public interface ──────────────────────────────────────────────────────

    async def on_event(self, event: Event) -> None:
        if self._logger and not self.silent:
            try:
                self._logger.info(event.type, run_id=event.run_id, step=event.step_id)
                if isinstance(event, ProcessorTriggerEvent) and event.detail:
                    self._logger.debug(
                        event.type,
                        run_id=event.run_id,
                        action=event.action,
                        **event.detail,
                    )
            except Exception:
                pass

        ts = _iso(event.ts)
        run_id = event.run_id
        step = event.step_id
        # Session label used in all jsonl records: the true session_id (not the segment run_id).
        # Resolved once at TaskStartEvent via _open_files and cached in _effective_session.
        sess = self._effective_session or run_id

        if isinstance(event, TaskStartEvent):
            effective_session = self.session_id or run_id
            # Detect resume: check whether a segment file already exists for this run_id
            # (resume passes the original run_id in state.run_id)
            session_dir_candidate = os.path.join(self.base_dir, effective_session)
            segment_path = os.path.join(session_dir_candidate, f"{run_id}.jsonl")
            is_resume = os.path.exists(segment_path) and os.path.getsize(segment_path) > 0

            self._last_uuid = None
            if not is_resume:
                self._last_system_prompt = ""
                self._last_tools_hash = ""
            # Clear raw-event caches so a reused journal instance starts clean.
            self._raw_assistant.clear()
            self._raw_tool_results.clear()
            self._raw_tool_inputs.clear()
            self._last_step_messages = ()
            self._before_model_raw_msgs = ()
            self._open_files(run_id)  # sets self._effective_session
            sess = self._effective_session  # update after _open_files

            if not is_resume:
                self._write_session(
                    {
                        "session_id": sess,
                        "type": "session_start",
                        "step": step,
                        "timestamp": ts,
                        "message": None,
                        "task": event.task_description,
                        "model": event.model,
                    }
                )
            else:
                if event.task_description:
                    # Prefer the full multimodal content from state over the
                    # text-only task_description (which strips non-text blocks
                    # in run_loop).  state.raw_messages[-1] is the user turn
                    # that was just appended by harness.run() before calling run_loop().
                    _resume_content: Any = event.task_description
                    _state = event.state
                    if _state is not None:
                        _raw_msgs = getattr(_state, "raw_messages", [])
                        if _raw_msgs and getattr(_raw_msgs[-1], "role", None) == "user":
                            _resume_content = _raw_msgs[-1].content
                    if isinstance(_resume_content, list):
                        _resume_content, _ = self._externalize_user_content(_resume_content, f"{run_id}_s{step}_cur")
                    self._write_session(
                        {
                            "session_id": sess,
                            "type": "raw_user",
                            "step": step,
                            "timestamp": ts,
                            "message": {
                                "role": "user",
                                "content": _resume_content,
                            },
                        }
                    )
            self._write_trace(
                {
                    "event_type": "task_start" if not is_resume else "turn_start",
                    "session_id": sess,
                    "step": step,
                    "timestamp": ts,
                    **({"task": event.task_description} if is_resume else {}),
                }
            )
            # Flush any ProcessorTriggerEvents that arrived before the file was
            # opened (task_start processors run before tracer.on_event(TaskStartEvent)).
            # Build records now so session_id is resolved to the just-opened session.
            for evt in self._pending_triggers:
                self._write_trace(
                    {
                        "event_type": "processor_trigger",
                        "session_id": sess,
                        "step": evt.step_id,
                        "timestamp": _iso(evt.ts),
                        "processor": evt.processor,
                        "hook": evt.hook,
                        "action": evt.action,
                        "detail": evt.detail,
                    }
                )
            self._pending_triggers.clear()
            # Register this run in the session index immediately (in-progress marker).
            # Points to step_state.json so wake() can recover even on hard crash.
            # Updated to {run_id}_state.json at task_end.
            step_state_path = self._step_state_rel_path()
            self._write_session_index(
                run_id,
                ts,
                latest_state_path=step_state_path,
            )

        elif isinstance(event, StepStartEvent):
            self._write_trace(
                {
                    "event_type": "step_start",
                    "session_id": sess,
                    "step": step,
                    "timestamp": ts,
                    "token_count": event.token_count,
                    "token_budget": getattr(event, "token_budget", None),
                    "context_window": getattr(event, "context_window", None),
                }
            )
            if event.tools:
                tools_hash = ",".join(t.name for t in event.tools)
                if tools_hash != self._last_tools_hash:
                    self._last_tools_hash = tools_hash
                    self._write_session(
                        {
                            "session_id": sess,
                            "type": "tools",
                            "step": step,
                            "timestamp": ts,
                            "message": {
                                "role": "tools",
                                "content": [
                                    {
                                        "name": t.name,
                                        "description": t.description,
                                        "input_schema": t.input_schema,
                                    }
                                    for t in event.tools
                                ],
                            },
                        }
                    )
            if event.system_prompt and event.system_prompt != self._last_system_prompt:
                self._last_system_prompt = event.system_prompt
                self._write_session(
                    {
                        "session_id": sess,
                        "type": "system",
                        "step": step,
                        "timestamp": ts,
                        "message": {"role": "system", "content": event.system_prompt},
                    }
                )
            if step == 0 and not self._segment_has_context_snapshot:
                # Write the genuine factual messages that were in state.raw_messages
                # *before* processor assembly (event.raw_messages). Processor-injected
                # synthetic messages (skills hints, budget nudges, …) live only in
                # event.messages and must NOT be written here — they are re-injected
                # on every step and writing them would cause duplicates on wake().
                # Skipped when a context_snapshot was already written for this segment
                # (post-compaction): the snapshot already contains all prior messages.
                #
                # ctx="history" marks these as historical-context records so that
                # _read_display_messages can skip them in non-first run segments
                # (they are duplicates of messages already shown from earlier runs).
                for _hi, m in enumerate(event.raw_messages):
                    if getattr(m, "role", None) == "user":
                        _hist_content: Any = m.content
                        if isinstance(_hist_content, list):
                            _hist_content, _ = self._externalize_user_content(_hist_content, f"{run_id}_s{step}_h{_hi}")
                        _hist_rec: dict = {
                            "session_id": sess,
                            "type": "raw_user",
                            "step": step,
                            "timestamp": ts,
                            "message": {
                                "role": "user",
                                "content": _hist_content,
                            },
                            "ctx": "history",
                        }
                        if getattr(m, "msg_id", None):
                            _hist_rec["meta"] = {"msg_id": m.msg_id}
                        self._write_session(_hist_rec)
            # Write a "user" delta record if processors modified the current turn's
            # user message.  raw_user (factual, pre-processor) was already written
            # in the TaskStartEvent handler (is_resume path) or in the step-0 history
            # loop above.  Here we detect when processors changed the content and
            # write the effective version so _rebuild_message_tracks_from_jsonl can
            # reconstruct the exact context that was sent to the model.
            #
            # We compare the last user message in event.raw_messages (pre-processor)
            # against the last user message in event.messages (post-processor).
            # Scanning from the end of each list independently handles the common case
            # where processors inject synthetic messages *after* the user message
            # (budget nudges, skill hints) without disturbing the user message index.
            if event.raw_messages and event.messages:
                raw_last_user = next(
                    (m for m in reversed(event.raw_messages) if getattr(m, "role", None) == "user"),
                    None,
                )
                proc_last_user = next(
                    (m for m in reversed(event.messages) if getattr(m, "role", None) == "user"),
                    None,
                )
                if raw_last_user is not None and proc_last_user is not None:
                    raw_text = _extract_text(raw_last_user.content)
                    proc_text = _extract_text(proc_last_user.content)
                    if raw_text != proc_text:
                        _delta_content: Any = proc_last_user.content
                        if isinstance(_delta_content, list):
                            _delta_content, _ = self._externalize_user_content(
                                _delta_content, f"{run_id}_s{step}_udelta"
                            )
                        self._write_session(
                            {
                                "session_id": sess,
                                "type": "user",
                                "step": step,
                                "timestamp": ts,
                                "message": {"role": "user", "content": _delta_content},
                            }
                        )
            # Cache for synthetic-user detection in on_event(BeforeModelEvent).
            self._last_step_messages = event.messages

        elif isinstance(event, BeforeModelEvent):
            # Detect synthetic user messages injected by before_model processors.
            # A synthetic user is one that appears in the post-processor event.messages
            # but was NOT in the pre-processor snapshot (_before_model_raw_msgs).
            # Condition: length increased by exactly 1 and the new tail is role=user.
            _raw_bm = self._before_model_raw_msgs
            if _raw_bm and event.messages and len(event.messages) == len(_raw_bm) + 1:
                synthetic_msg = event.messages[-1]
                if synthetic_msg.role == "user":
                    _syn_meta: dict = {
                        "synthetic": True,
                        "injected_at_hook": "before_model",
                        # source_processor and reason are set by the injecting processor;
                        # the journal only sees the aggregated post-processor event and
                        # cannot resolve the processor name here — callers that need
                        # attribution should annotate the injected Message directly.
                        "source_processor": None,
                        "reason": None,
                    }
                    if getattr(synthetic_msg, "msg_id", None):
                        _syn_meta["msg_id"] = synthetic_msg.msg_id
                    _syn_content: Any = synthetic_msg.content
                    if isinstance(_syn_content, list):
                        _syn_content, _ = self._externalize_user_content(_syn_content, f"{run_id}_s{step}_syn")
                    self._write_session(
                        {
                            "session_id": sess,
                            "type": "raw_user",
                            "step": step,
                            "timestamp": ts,
                            "message": {
                                "role": "user",
                                "content": _syn_content,
                            },
                            "meta": _syn_meta,
                        }
                    )

            self._write_trace(
                {
                    "event_type": "before_model",
                    "session_id": sess,
                    "step": step,
                    "timestamp": ts,
                    "cumulative_cost_usd": event.cumulative_cost_usd,
                }
            )
            # Note: effective model input (messages + tools) is NOT written to the
            # session JSONL here.  The state checkpoint (step_state.json, written at
            # every StepEndEvent) already persists both raw_messages and messages via
            # State.snapshot() — writing the full assembled context at every step
            # would create O(N²) bloat and duplicate the system prompt that is
            # already captured by the "system" record type.  wake() reads the
            # checkpoint to restore state.messages directly.

        elif isinstance(event, ModelResponseEvent):
            # Build message in the same format as message_to_dict() so that
            # dict_to_message() can reconstruct it losslessly during wake().
            msg: dict = {
                "role": "assistant",
                "content": event.content or "",
            }
            if event.tool_calls:
                msg["tool_calls"] = [{"id": tc.id, "name": tc.name, "input": tc.input} for tc in event.tool_calls]
            if event.thinking:
                msg["thinking"] = event.thinking
            if event.thinking_blocks:
                msg["thinking_blocks"] = list(event.thinking_blocks)
            # Write delta only when processors modified the assistant message.
            # on_raw_event already wrote the unmodified version as raw_assistant.
            # When no delta is needed (common case), raw_assistant is the sole record
            # and _rebuild_message_tracks_from_jsonl will use it directly.
            #
            # Special case: if on_raw_event was never called for this step (e.g. the
            # RunLoop emits ModelResponseEvent directly — interrupt path, not via
            # ProcessorChain), raw_msg will be None.  In that case write raw_assistant
            # so the rebuild can find the record via its normal raw_* anchor lookup.
            raw_msg = self._raw_assistant.get(step)
            meta_block: dict = {
                "model": event.model,
                "stop_reason": event.finish_reason,
                "usage": {
                    "input_tokens": event.usage.input_tokens,
                    "output_tokens": event.usage.output_tokens,
                    "cache_read_tokens": event.usage.cache_read_tokens,
                    "cache_write_tokens": event.usage.cache_write_tokens,
                },
            }
            if raw_msg is None:
                # No raw anchor yet — write one now so _rebuild_message_tracks_from_jsonl
                # can locate this message in new-format sessions.
                self._write_session(
                    {
                        "session_id": sess,
                        "type": "raw_assistant",
                        "step": step,
                        "timestamp": ts,
                        "message": msg,
                        "meta": meta_block,
                    }
                )
            elif json.dumps(msg, sort_keys=True) != json.dumps(raw_msg, sort_keys=True):
                # Processors changed the message — write a delta record.
                self._write_session(
                    {
                        "session_id": sess,
                        "type": "assistant",
                        "step": step,
                        "timestamp": ts,
                        "message": msg,
                        # Execution metadata lives in meta, not message — keeps message
                        # format identical to message_to_dict() for zero-transform wake().
                        "meta": meta_block,
                    }
                )

        elif isinstance(event, ToolCallEvent):
            trace_rec: dict = {
                "event_type": "tool_call",
                "session_id": sess,
                "step": step,
                "timestamp": ts,
                "tool_name": event.tool_name,
                "tool_call_id": event.tool_call_id,
                "approved": event.approved,
                "synthetic_result": event.synthetic_result,
            }
            raw_input = self._raw_tool_inputs.get(event.tool_call_id)
            if raw_input is not None and event.tool_input != raw_input:
                trace_rec["input_override"] = event.tool_input
            self._write_trace(trace_rec)

        elif isinstance(event, ToolResultEvent):
            # Write tool delta only when processors modified the result.
            # on_raw_event already wrote the pre-processor version as raw_tool.
            #
            # Special case: if on_raw_event was never called for this tool call
            # (interrupt path or other ProcessorChain bypass), raw_cached is None.
            # Write raw_tool in that case so _rebuild_message_tracks_from_jsonl can
            # locate the message via its raw_* anchor lookup — a standalone "tool"
            # delta with no raw_tool anchor is invisible to Pass B.
            # Mirrors the ModelResponseEvent interrupt-path fallback.
            raw_cached = self._raw_tool_results.get(event.tool_call_id)
            if raw_cached is None:
                # No raw anchor yet — write one now.
                content, result_meta = self._prepare_tool_result(event)
                tool_msg: dict = {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": event.tool_call_id,
                    "name": event.tool_name,
                }
                record: dict = {
                    "session_id": sess,
                    "type": "raw_tool",
                    "step": step,
                    "timestamp": ts,
                    "message": tool_msg,
                }
                if result_meta:
                    record["meta"] = result_meta
                self._write_session(record)
            elif raw_cached != (event.result, event.error):
                # Processors changed the result — write a delta record.
                # message format mirrors message_to_dict() for zero-transform wake():
                #   role="tool", content=<inline or "">, tool_call_id, name
                # Large results: content="" + meta.content_ref → filled on rebuild.
                content, result_meta = self._prepare_tool_result(event)
                tool_msg = {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": event.tool_call_id,
                    "name": event.tool_name,
                }
                record = {
                    "session_id": sess,
                    "type": "tool",
                    "step": step,
                    "timestamp": ts,
                    "message": tool_msg,
                }
                if result_meta:
                    record["meta"] = result_meta
                self._write_session(record)
            # else: raw == effective, no delta needed — raw_tool already covers it.
            self._write_trace(
                {
                    "event_type": "tool_result",
                    "session_id": sess,
                    "step": step,
                    "timestamp": ts,
                    "tool_name": event.tool_name,
                    "tool_call_id": event.tool_call_id,
                    "duration_ms": event.duration_ms,
                    "error": event.error,
                }
            )

        elif isinstance(event, StepEndEvent):
            self._write_trace(
                {
                    "event_type": "step_end",
                    "session_id": sess,
                    "step": step,
                    "timestamp": ts,
                    "cumulative_tokens": event.cumulative_tokens,
                    "cumulative_cost_usd": event.cumulative_cost_usd,
                    "memory_written": event.memory_written,
                }
            )
            # Crash-safe checkpoint: overwrite step_state.json after each step.
            if event.state_snapshot is not None and self._session_dir:
                step_state_path = os.path.join(self._session_dir, "step_state.json")
                data = dict(event.state_snapshot)
                data["session_id"] = sess
                data.setdefault("schema_version", 2)
                if self.config_hash:
                    data["config_hash"] = self.config_hash
                # Messages are rebuilt from JSONL on wake(); strip them here
                # so the crash-recovery checkpoint stays compact.
                data.pop("raw_messages", None)
                data.pop("messages", None)
                self._atomic_write_json(step_state_path, data)

        elif isinstance(event, SegmentBoundaryEvent):
            # 1. Write checkpoint for the completed segment
            old_run_id = self._current_run_id or run_id
            # state_snapshot may be None here (compaction doesn't have State access);
            # in that case we write a minimal marker so the segment is still findable.
            snapshot = event.state_snapshot or {
                "schema_version": 2,
                "run_id": old_run_id,
                "segment_end_reason": event.reason,
            }
            state_rel = self._write_segment_state(old_run_id, snapshot, ts, reason=event.reason)

            # 2. Write segment_boundary marker into the OLD segment's trace (before rotating)
            self._write_trace(
                {
                    "event_type": "segment_boundary",
                    "session_id": sess,
                    "step": step,
                    "timestamp": ts,
                    "reason": event.reason,
                    "previous_run_id": old_run_id,
                    "new_run_id": event.new_run_id,
                    "previous_state_path": state_rel,
                }
            )

            # 3a. When the caller provides a full state snapshot, write step_state.json
            # immediately so the crash-safe pointer in the session index is up-to-date
            # before the process could crash.  Without this, a crash between the boundary
            # and the first StepEndEvent of the new segment would leave step_state.json
            # pointing at stale numeric state (e.g. stale last_sys_prompt_hash).
            if event.state_snapshot is not None and self._session_dir:
                _snap = dict(event.state_snapshot)
                _snap.pop("raw_messages", None)
                _snap.pop("messages", None)
                # Overwrite run_id with new_run_id so crash recovery routes to the
                # new segment, not the old one (snapshot was taken before state.run_id
                # was updated to new_run_id).
                _snap["run_id"] = event.new_run_id
                _snap["session_id"] = sess
                _snap.setdefault("schema_version", 2)
                if self.config_hash:
                    _snap["config_hash"] = self.config_hash
                self._atomic_write_json(os.path.join(self._session_dir, "step_state.json"), _snap)

            # 3b. Update session index to point to the new segment (in-progress)
            new_step_state = self._step_state_rel_path()
            self._write_session_index(event.new_run_id, ts, latest_state_path=new_step_state)

            # 4. Rotate file handles
            self._close_segment_files()
            self._open_segment(event.new_run_id)

            # 5. Write both effective/raw snapshots as the first records of the
            # new JSONL. For compaction boundaries they are intentionally
            # identical (the compacted context), but wake() still tracks them
            # separately to rebuild raw_messages/messages with strict invariants.
            if event.compacted_messages:
                raw_snap = event.compacted_raw_messages or event.compacted_messages
                self._write_session(
                    {
                        "session_id": sess,
                        "type": "context_snapshot_raw",
                        "step": step,
                        "timestamp": ts,
                        "messages": [message_to_dict(m) for m in raw_snap],
                    }
                )
                self._write_session(
                    {
                        "session_id": sess,
                        "type": "context_snapshot",
                        "step": step,
                        "timestamp": ts,
                        "messages": [message_to_dict(m) for m in event.compacted_messages],
                    }
                )
                self._segment_has_context_snapshot = True

        elif isinstance(event, SpawnSubAgentEvent):
            sub_id = getattr(event.sub_task, "run_id", None) or str(uuid.uuid4())
            self._write_session(
                {
                    "session_id": sess,
                    "type": "spawn_sub_agent",
                    "step": step,
                    "timestamp": ts,
                    "message": None,
                    "sub_session_id": sub_id,
                    "sub_session_path": f"sub_agents/{sub_id}.jsonl",
                }
            )

        elif isinstance(event, ProcessorTriggerEvent):
            if self._trace_file is None:
                # task_start processors run before the file is opened; buffer the
                # raw event and flush after TaskStartEvent writes the task_start
                # trace record (so session_id is resolved correctly at flush time).
                self._pending_triggers.append(event)
            else:
                self._write_trace(
                    {
                        "event_type": "processor_trigger",
                        "session_id": sess,
                        "step": step,
                        "timestamp": ts,
                        "processor": event.processor,
                        "hook": event.hook,
                        "action": event.action,
                        "detail": event.detail,
                    }
                )

        elif isinstance(event, TaskEndEvent):
            self._write_session(
                {
                    "session_id": sess,
                    "type": "episode_end",
                    "step": step,
                    "timestamp": ts,
                    "message": None,
                    "exit_reason": event.exit_reason,
                    "total_steps": event.total_steps,
                    "reward": event.eval_result.reward if event.eval_result else None,
                    "passed": event.eval_result.passed if event.eval_result else None,
                }
            )
            self._write_trace(
                {
                    "event_type": "task_end",
                    "session_id": sess,
                    "step": step,
                    "timestamp": ts,
                    "exit_reason": event.exit_reason,
                    "error": event.error or None,
                    "total_steps": event.total_steps,
                    "total_tokens": event.total_tokens,
                    "total_input_tokens": event.total_input_tokens,
                    "total_output_tokens": event.total_output_tokens,
                    "total_cost_usd": event.total_cost_usd,
                }
            )
            # Write final state checkpoint and update session index.
            current_run = self._current_run_id or run_id
            if event.state_snapshot is not None:
                state_rel = self._write_segment_state(current_run, event.state_snapshot, ts, reason="task_end")
                self._write_session_index(current_run, ts, latest_state_path=state_rel)
                # Remove crash-safe step_state.json — segment state supersedes it.
                if self._session_dir:
                    _step_state = os.path.join(self._session_dir, "step_state.json")
                    try:
                        os.remove(_step_state)
                    except FileNotFoundError:
                        pass
            self._close_files()

    async def on_raw_event(self, event: Event) -> None:
        """Called by ProcessorChain before running processors (pre-processor snapshot).

        Writes ``raw_assistant`` / ``raw_tool`` records to the session JSONL so
        that ``on_event`` can compare the post-processor version and emit a
        delta record only when content actually changed.  Also caches raw
        ``tool_input`` so ``on_event`` can record ``input_override`` in the
        trace when a processor rewrites the model's intended tool arguments.

        Record types written here:
        * ``raw_assistant`` — pre-processor model response (always written)
        * ``raw_tool``      — pre-processor tool result (always written)
        * (no ``raw_user`` here; factual user messages are written in on_event —
          ``raw_user`` in TaskStartEvent/step-0 history, ``user`` delta in StepStartEvent)

        The ``_raw_assistant`` / ``_raw_tool_results`` / ``_raw_tool_inputs``
        caches are keyed by ``step_id`` or ``tool_call_id`` and cleared at
        task start/end.
        """
        if not self.export_jsonl or not self._session_file:
            return

        ts = _iso(event.ts)
        step = event.step_id
        sess = self._effective_session or event.run_id

        if isinstance(event, BeforeModelEvent):
            # Cache pre-processor messages so on_event(BeforeModelEvent) can
            # detect synthetic user injections by comparing against the post-
            # processor version.
            self._before_model_raw_msgs = event.messages
            return

        if isinstance(event, ModelResponseEvent):
            msg: dict = {"role": "assistant", "content": event.content or ""}
            if event.tool_calls:
                msg["tool_calls"] = [{"id": tc.id, "name": tc.name, "input": tc.input} for tc in event.tool_calls]
            if event.thinking:
                msg["thinking"] = event.thinking
            if event.thinking_blocks:
                msg["thinking_blocks"] = list(event.thinking_blocks)
            self._raw_assistant[step] = msg
            self._write_session(
                {
                    "session_id": sess,
                    "type": "raw_assistant",
                    "step": step,
                    "timestamp": ts,
                    "message": msg,
                    "meta": {
                        "model": event.model,
                        "stop_reason": event.finish_reason,
                        "usage": {
                            "input_tokens": event.usage.input_tokens,
                            "output_tokens": event.usage.output_tokens,
                            "cache_read_tokens": event.usage.cache_read_tokens,
                            "cache_write_tokens": event.usage.cache_write_tokens,
                        },
                    },
                }
            )

        elif isinstance(event, ToolCallEvent):
            # Cache raw tool_input for input_override detection in on_event.
            # No session record written — the trace record is written in on_event.
            self._raw_tool_inputs[event.tool_call_id] = event.tool_input or {}

        elif isinstance(event, ToolResultEvent):
            # Cache for delta comparison.
            self._raw_tool_results[event.tool_call_id] = (event.result, event.error)
            content, result_meta = self._prepare_tool_result(event, suffix="_raw")
            tool_msg: dict = {
                "role": "tool",
                "content": content,
                "tool_call_id": event.tool_call_id,
                "name": event.tool_name,
            }
            record: dict = {
                "session_id": sess,
                "type": "raw_tool",
                "step": step,
                "timestamp": ts,
                "message": tool_msg,
            }
            if result_meta:
                record["meta"] = result_meta
            self._write_session(record)

    def _step_state_rel_path(self) -> str:
        """Return relative path (from workspace root) to the rolling step_state.json."""
        if not self._session_dir:
            return "step_state.json"
        rel = os.path.relpath(self._session_dir, str(Path(self.base_dir).resolve().parent))
        return f"{rel}/step_state.json"

    async def flush(self) -> None:
        pass  # every write is immediately flushed

    async def export_session_jsonl(self, run_id: str, path: str) -> None:
        pass  # no-op — streaming write handles export

    # ── Recovery API ──────────────────────────────────────────────────────────

    @classmethod
    def _message_from_record(cls, evt: dict, jsonl_path: Path) -> "Any | None":
        """Convert one JSONL record to Message, restoring externalized content."""
        msg_data = dict(evt.get("message") or {})
        if not msg_data:
            return None
        meta = evt.get("meta") or {}
        # Restore large tool result text from external file (content_ref).
        content_ref = meta.get("content_ref")
        if content_ref:
            ref_path = jsonl_path.parent / content_ref
            try:
                msg_data["content"] = ref_path.read_text(encoding="utf-8")
            except Exception:
                pass
        # Restore large media blocks from external files (media_ref in each block).
        if isinstance(msg_data.get("content"), list):
            msg_data["content"] = cls._restore_media_refs(msg_data["content"], jsonl_path)
        return dict_to_message(msg_data)

    @classmethod
    def _rebuild_message_tracks_from_jsonl(cls, jsonl_path: Path, up_to_step: int) -> "tuple[list, list]":
        """Reconstruct both raw and effective message tracks from segment JSONL.

        Supports two JSONL formats:

        **New format** (sessions created after Phase 4):
        * ``raw_user`` / ``raw_assistant`` / ``raw_tool`` — pre-processor content
          (always written by the journal).
        * ``assistant`` / ``tool`` — processor-applied deltas (written only when a
          processor changed the content).

        For each ``raw_assistant`` record the rebuild checks whether a matching
        ``assistant`` delta exists at the same step.  For each ``raw_tool`` record
        it checks whether a matching ``tool`` delta exists for the same
        ``tool_call_id``.  The delta is preferred when present; otherwise the raw
        version is used.  This yields the effective messages (exactly what the model
        saw on the last completed turn).

        **Old format** (legacy sessions before Phase 4, backward compatibility):
        ``user`` / ``assistant`` / ``tool`` records are used directly without any
        delta logic.  A JSONL is considered old-format when it has no ``raw_*``
        type records (after the last ``context_snapshot``).

        **Common to both formats:**
        1. The latest ``context_snapshot_raw`` / ``context_snapshot`` records
           (if present) provide base message lists for raw/effective tracks.
           Only records **after** snapshot line positions are appended.
        2. Records at ``step >= up_to_step`` are excluded (incomplete last step).
        3. External tool results (``meta.content_ref``) are re-read from disk.

        ``up_to_step`` is ``state.step`` from the checkpoint.
        """
        if not jsonl_path.exists():
            return ([], [])

        lines: list[dict] = []
        with jsonl_path.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if raw:
                    try:
                        lines.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass

        # Pass 1: find the latest raw/effective snapshots and their line positions.
        # Using file position (not step number) as the boundary ensures that
        # conversation records written at the same step as the snapshot are
        # correctly included.
        raw_base_messages: list[dict] = []
        effective_base_messages: list[dict] = []
        raw_base_pos: int = -1
        effective_base_pos: int = -1
        for i, evt in enumerate(lines):
            t = evt.get("type")
            if t == "context_snapshot_raw":
                raw_base_messages = list(evt.get("messages") or [])
                raw_base_pos = i
            elif t == "context_snapshot":
                effective_base_messages = list(evt.get("messages") or [])
                effective_base_pos = i

        # Transitional fallback: if only one snapshot type exists, treat it as both.
        if raw_base_pos < 0 and effective_base_pos >= 0:
            raw_base_messages = list(effective_base_messages)
            raw_base_pos = effective_base_pos
        if effective_base_pos < 0 and raw_base_pos >= 0:
            effective_base_messages = list(raw_base_messages)
            effective_base_pos = raw_base_pos

        base_pos = max(raw_base_pos, effective_base_pos)

        # Detect format: new (has raw_* records after base_pos) vs legacy.
        _NEW_RAW_TYPES = {"raw_user", "raw_assistant", "raw_tool"}
        has_raw_format = any(
            lines[i].get("type") in _NEW_RAW_TYPES
            for i in range(base_pos + 1, len(lines))
            if lines[i].get("step", 0) < up_to_step
        )

        raw_messages = [dict_to_message(m) for m in raw_base_messages]
        messages = [dict_to_message(m) for m in effective_base_messages]

        if has_raw_format:
            # ── New format ──────────────────────────────────────────────────────
            # Pass A: collect deltas and count raw_user records per step.
            # Deltas are keyed by step (user/assistant) or tool_call_id (tool).
            user_deltas: dict[int, dict] = {}  # step → evt
            assistant_deltas: dict[int, dict] = {}  # step → evt
            tool_deltas: dict[str, dict] = {}  # tool_call_id → evt
            raw_user_count: dict[int, int] = {}  # step → # of raw_user records
            for i, evt in enumerate(lines):
                if i <= base_pos:
                    continue
                if evt.get("step", 0) >= up_to_step:
                    continue
                t = evt.get("type")
                if t == "user":
                    # Processor-applied delta for the user message at this step.
                    # Keyed by step; multiple deltas at the same step (unlikely)
                    # are resolved by last-write-wins.
                    user_deltas[evt.get("step", 0)] = evt
                elif t == "raw_user":
                    s = evt.get("step", 0)
                    raw_user_count[s] = raw_user_count.get(s, 0) + 1
                elif t == "assistant":
                    assistant_deltas[evt.get("step", 0)] = evt
                elif t == "tool":
                    tcid = (evt.get("message") or {}).get("tool_call_id", "")
                    if tcid:
                        tool_deltas[tcid] = evt

            # Pass B: build effective messages using raw_* + deltas.
            # user delta applies only to the LAST raw_user at each step (the
            # current turn's input), not to historical user messages at the same
            # step that may share a step number (e.g. step-0 history context).
            raw_user_seen: dict[int, int] = {}  # step → count processed so far
            for i, evt in enumerate(lines):
                if i <= base_pos:
                    continue
                if evt.get("step", 0) >= up_to_step:
                    continue
                t = evt.get("type")
                if t == "raw_user":
                    raw_msg = cls._message_from_record(evt, jsonl_path)
                    if raw_msg is not None:
                        raw_messages.append(raw_msg)

                    step_n = evt.get("step", 0)
                    seen = raw_user_seen.get(step_n, 0) + 1
                    raw_user_seen[step_n] = seen
                    is_last_at_step = seen == raw_user_count.get(step_n, 1)
                    # Apply user delta only to the last raw_user at this step.
                    effective_evt = user_deltas.get(step_n, evt) if is_last_at_step else evt
                    eff_msg = cls._message_from_record(effective_evt, jsonl_path)
                    if eff_msg is not None:
                        messages.append(eff_msg)

                elif t == "raw_assistant":
                    raw_msg = cls._message_from_record(evt, jsonl_path)
                    if raw_msg is not None:
                        raw_messages.append(raw_msg)

                    step_n = evt.get("step", 0)
                    # Prefer processor-applied delta; fall back to raw.
                    effective_evt = assistant_deltas.get(step_n, evt)
                    eff_msg = cls._message_from_record(effective_evt, jsonl_path)
                    if eff_msg is not None:
                        messages.append(eff_msg)

                elif t == "raw_tool":
                    raw_msg = cls._message_from_record(evt, jsonl_path)
                    if raw_msg is not None:
                        raw_messages.append(raw_msg)

                    tcid = (evt.get("message") or {}).get("tool_call_id", "")
                    # Prefer processor-applied delta; fall back to raw.
                    effective_evt = tool_deltas.get(tcid, evt) if tcid else evt
                    eff_msg = cls._message_from_record(effective_evt, jsonl_path)
                    if eff_msg is not None:
                        messages.append(eff_msg)

        else:
            # ── Legacy format (backward compatibility) ──────────────────────────
            # Collect records that appear AFTER the last context_snapshot.
            # For the first segment (no context_snapshot, base_pos=-1) the condition
            # ``i <= -1`` is always False, so all records are candidates.
            for i, evt in enumerate(lines):
                if i <= base_pos:
                    continue  # at or before the context_snapshot — already in base
                if evt.get("step", 0) >= up_to_step:
                    continue  # incomplete last step — skip
                t = evt.get("type")
                if t not in ("user", "assistant", "tool"):
                    continue  # system/tools/session_start/episode_end are not messages
                legacy_msg = cls._message_from_record(evt, jsonl_path)
                if legacy_msg is not None:
                    raw_messages.append(legacy_msg)
                    messages.append(legacy_msg)

        return raw_messages, messages

    @classmethod
    def _validate_rebuilt_tracks(
        cls,
        *,
        session_id: str,
        run_id: str,
        raw_messages: "list",
        messages: "list",
        strict: bool = True,
    ) -> None:
        """Validate raw/effective tracks after wake() reconstruction."""
        if not strict:
            return

        if len(raw_messages) != len(messages):
            raise ValueError(
                "wake() rebuilt mismatched tracks for session "
                f"{session_id!r}, run {run_id!r}: "
                f"len(raw_messages)={len(raw_messages)} "
                f"!= len(messages)={len(messages)}"
            )

        for i, (raw_msg, eff_msg) in enumerate(zip(raw_messages, messages)):
            raw_role = getattr(raw_msg, "role", None)
            eff_role = getattr(eff_msg, "role", None)
            if raw_role != eff_role:
                raise ValueError(
                    "wake() rebuilt role-mismatched tracks for session "
                    f"{session_id!r}, run {run_id!r} at index {i}: "
                    f"raw role={raw_role!r}, messages role={eff_role!r}"
                )
            if raw_role == "tool":
                raw_tcid = getattr(raw_msg, "tool_call_id", None)
                eff_tcid = getattr(eff_msg, "tool_call_id", None)
                if raw_tcid != eff_tcid:
                    raise ValueError(
                        "wake() rebuilt tool_call_id-mismatched tool records for session "
                        f"{session_id!r}, run {run_id!r} at index {i}: "
                        f"raw tool_call_id={raw_tcid!r}, messages tool_call_id={eff_tcid!r}"
                    )

    @classmethod
    def _rebuild_effective_messages_from_jsonl(cls, jsonl_path: Path, up_to_step: int) -> "list":
        """Reconstruct the latest effective model input for a resumed state.

        Uses the last ``model_input_snapshot`` record with ``step < up_to_step``
        so incomplete/crashed in-progress steps are ignored.
        """
        if not jsonl_path.exists():
            return []

        latest: list[dict] = []
        with jsonl_path.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") != "model_input_snapshot":
                    continue
                if evt.get("step", 0) >= up_to_step:
                    continue
                latest = list(evt.get("messages") or [])

        return [dict_to_message(m) for m in latest if isinstance(m, dict)]

    @classmethod
    def wake(
        cls,
        session_id: str,
        workspace_root: str,
        strict_validation: bool = True,
    ) -> "Any":
        """Restore State from disk for the given session_id.

        Reads the session index to find the latest segment's state file.
        The state file contains only numeric counters and slots; message tracks
        are intentionally omitted (they would become stale and waste disk space).
        Both message tracks are reconstructed from JSONL:

        * ``raw_messages``: raw_* anchor records only (factual, pre-processor).
        * ``messages``:     effective records (latest delta override, else raw fallback).

        * **New format** (raw_* + delta records): raw_* anchors are used as the
          base; processor delta records override content when present.
        * **Legacy format** (user / assistant / tool records): used as-is.
        * **Last resort**: deprecated ``model_input_snapshot`` records in the
          JSONL — used only when no conversation records exist at all.

        Args:
            session_id: the session to resume.
            workspace_root: root of the workspace directory (parent of sessions/).
            strict_validation: when True (default), enforce track invariants:
                ``len(raw_messages) == len(messages)``, per-index role equality,
                and tool ``tool_call_id`` equality.

        Returns:
            Fully-populated State ready for ``harness.run(_resume_state=state)``.

        Raises:
            FileNotFoundError: if the session index, state file, or JSONL is missing.
            ValueError: if the index exists but latest_state_path cannot be resolved.
        """
        from ..core.state import State

        index_path = Path(workspace_root) / "sessions" / f"{session_id}.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Session index not found: {index_path}. Has this session run at least once with HarnessJournal?"
            )
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        state_rel = index.get("latest_state_path")
        if not state_rel:
            raise ValueError(f"Session index missing latest_state_path: {index_path}")

        state_path = Path(workspace_root) / state_rel
        if not state_path.exists():
            raise FileNotFoundError(
                f"State file not found: {state_path}. The run may have been deleted or not completed normally."
            )
        with open(state_path, encoding="utf-8") as f:
            snapshot = json.load(f)

        # Restore numeric state (step, cost, slots, …).
        # raw_messages / messages are NOT in the snapshot — they come from JSONL.
        state = State.from_snapshot(snapshot)

        # Reconstruct both message tracks from the segment JSONL.
        latest_run_id = index.get("latest_run_id") or snapshot.get("run_id", "")
        session_dir = Path(workspace_root) / "sessions" / session_id
        jsonl_path = session_dir / f"{latest_run_id}.jsonl"
        rebuilt_raw, rebuilt_effective = cls._rebuild_message_tracks_from_jsonl(jsonl_path, up_to_step=state.step)
        if rebuilt_raw or rebuilt_effective:
            state.raw_messages = rebuilt_raw
            state.messages = rebuilt_effective
        else:
            # Last-resort: deprecated model_input_snapshot records (very old sessions
            # that predate proper conversation record writing).
            rebuilt_effective = cls._rebuild_effective_messages_from_jsonl(jsonl_path, up_to_step=state.step)
            if rebuilt_effective:
                state.raw_messages = rebuilt_effective
                state.messages = list(rebuilt_effective)

        cls._validate_rebuilt_tracks(
            session_id=session_id,
            run_id=str(latest_run_id),
            raw_messages=state.raw_messages,
            messages=state.messages,
            strict=strict_validation,
        )

        return state

    @classmethod
    def wake_config(
        cls,
        session_id: str,
        workspace_root: str,
        from_run: str | None = None,
    ) -> dict:
        """Return the resolved harness config for a session or a specific run.

        Args:
            session_id:     The session identifier.
            workspace_root: Workspace root directory (parent of sessions/, configs/).
            from_run:       Optional run_id for precise reproduction.  When given,
                            reads ``sessions/{session_id}/{from_run}_state.json`` →
                            ``config_hash`` → ``configs/{hash}.yaml``.
                            When omitted, returns the latest config for the session.

        Returns:
            Raw YAML-parsed dict, loadable by ``harness_from_config()``.

        Raises:
            FileNotFoundError: session index, state file, or config file missing.
            ImportError: omegaconf not installed.
        """
        try:
            from omegaconf import OmegaConf
        except ImportError as e:
            raise ImportError("omegaconf is required for wake_config()") from e

        ws = Path(workspace_root)

        if from_run is not None:
            # Precise reproduction: resolve config via the segment's state file.
            state_path = ws / "sessions" / session_id / f"{from_run}_state.json"
            if not state_path.exists():
                raise FileNotFoundError(f"State file not found for run {from_run!r}: {state_path}")
            with open(state_path, encoding="utf-8") as f:
                state_data = json.load(f)
            config_hash = state_data.get("config_hash")
            if not config_hash:
                raise ValueError(
                    f"Run {from_run!r} has no config_hash in state file. "
                    "It may have been created before content-addressed config storage."
                )
            # Configs live at agent_home/configs/ (global). Fall back to workspace
            # for sessions created before this layout change.
            from ..home import agent_configs_dir as _acd

            config_path = _acd() / f"{config_hash}.yaml"
            if not config_path.exists():
                config_path = ws / "configs" / f"{config_hash}.yaml"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Config file missing: {_acd() / f'{config_hash}.yaml'}. "
                    f"Expected ~/.harnessx/configs/{config_hash}.yaml to exist."
                )
        else:
            # Latest config for this session (from session index).
            index_path = ws / "sessions" / f"{session_id}.json"
            if not index_path.exists():
                raise FileNotFoundError(f"Session index not found: {index_path}")
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
            config_rel = index.get("latest_config_path")
            if not config_rel:
                raise ValueError(f"Session index missing latest_config_path: {index_path}")
            # New sessions store absolute paths; old sessions stored paths relative
            # to workspace_root. Handle both for backward compatibility.
            _config_rel_path = Path(config_rel)
            if _config_rel_path.is_absolute():
                config_path = _config_rel_path
            elif config_rel.startswith("~"):
                config_path = _config_rel_path.expanduser()
            else:
                config_path = ws / config_rel  # legacy: relative to workspace root
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

        return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
