# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Optional

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sessions_dir(agent_id: str, project: str, workspace_base: str = "workspaces") -> Path:
    from harnessx.home import agent_workspace_root

    return agent_workspace_root(agent_id, project, workspace_base=workspace_base) / "sessions"


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return parsed records, ignoring malformed lines."""
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    try:
                        records.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    return records


def _extract_user_text(rec: dict) -> str:
    """Return display text for a user/raw_user/session_start record."""
    msg = rec.get("message") or {}
    content = msg.get("content", "")
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    task = rec.get("task") or ""
    return (content or task).strip()


def _extract_user_content(rec: dict, session_id: str, media_query: str = "") -> Any:
    """Return full content (str or list with media URLs) for a user message."""
    msg = rec.get("message") or {}
    content = msg.get("content", "")
    task = rec.get("task") or ""

    if isinstance(content, str):
        return (content or task).strip()

    if isinstance(content, list):
        blocks = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                blocks.append(b)
            elif b.get("type") == "image":
                source = b.get("source", {})
                media_ref = source.get("media_ref")
                if media_ref:
                    qs = f"?{media_query}" if media_query else ""
                    blocks.append(
                        {
                            "type": "image",
                            "media_url": f"/api/sessions/{session_id}/media/{media_ref.split('/')[-1]}{qs}",
                            "media_type": source.get("media_type", "image/jpeg"),
                        }
                    )
                elif source.get("data"):
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": source.get("media_type", "image/png"),
                                "data": source["data"],
                            },
                        }
                    )
        if blocks:
            return blocks

    return (str(content) if content else task).strip() or ""


def _read_first_query(session_dir: Path, run_ids: list[str]) -> str:
    """Extract the first user message text from the session's earliest JSONL."""
    for run_id in run_ids:
        for rec in _read_jsonl(session_dir / f"{run_id}.jsonl"):
            t = rec.get("type")
            if t == "session_start":
                task = rec.get("task") or ""
                if task.strip():
                    return task.strip()[:200]
            elif t in ("user", "raw_user"):
                text = _extract_user_text(rec)
                if text:
                    return text[:200]
    return ""


def _search_in_session(session_dir: Path, run_ids: list[str], query: str) -> Optional[str]:
    """Search all user messages; return a snippet around the match, or None."""
    q = query.lower()
    for run_id in run_ids:
        for rec in _read_jsonl(session_dir / f"{run_id}.jsonl"):
            if rec.get("type") not in ("user", "raw_user", "session_start"):
                continue
            text = _extract_user_text(rec)
            if q in text.lower():
                idx = text.lower().find(q)
                start = max(0, idx - 40)
                end = min(len(text), idx + len(query) + 40)
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(text) else ""
                return prefix + text[start:end] + suffix
    return None


_PRE_HOOKS = frozenset(
    {
        "step_start",
        "on_step_start",
        "before_model",
        "on_before_model",
        "on_model_start",
    }
)


def _build_run_trace_data(trace_records: list[dict]) -> dict:
    """Extract per-step traces and task-start triggers from a trace JSONL.

    Returns a dict with:
      "task_start_triggers": list[dict]  — ProcessorTrigger-shaped dicts
      "step_traces": dict[int, dict]     — step_id (0-based) → StepTrace-shaped dict
      "step_timelines": dict[int, list]  — raw ordered timeline per step (for fallback use)
    """
    task_start_triggers: list[dict] = []
    # Ordered timeline per step: processor items AND block items in arrival order.
    step_timelines: dict[int, list[dict]] = {}
    # Pre-hook triggers per step (subset of timeline, for input.on_step_start_triggers).
    step_pre_triggers: dict[int, list[dict]] = {}
    step_end_recs: dict[int, dict] = {}
    step_ctx_recs: dict[int, dict] = {}

    for rec in trace_records:
        evt = rec.get("event_type")
        step = rec.get("step", 0)

        if evt == "processor_trigger":
            hook = rec.get("hook", "")
            trigger = {
                "processor": rec.get("processor", ""),
                "hook": hook,
                "action": rec.get("action", ""),
                "detail": rec.get("detail") or {},
            }
            if hook in ("task_start", "on_task_start"):
                task_start_triggers.append(trigger)
            else:
                step_timelines.setdefault(step, []).append({"kind": "processor", "trigger": trigger})
                if hook in _PRE_HOOKS:
                    step_pre_triggers.setdefault(step, []).append(trigger)

        elif evt == "tool_call":
            step_timelines.setdefault(step, []).append(
                {
                    "kind": "block",
                    "block": {
                        "type": "tool_use",
                        "id": rec.get("tool_call_id", ""),
                        "name": rec.get("tool_name", ""),
                        "input": {},
                    },
                }
            )

        elif evt == "tool_result":
            step_timelines.setdefault(step, []).append(
                {
                    "kind": "block",
                    "block": {
                        "type": "tool_result",
                        "id": rec.get("tool_call_id", ""),
                        "name": rec.get("tool_name", ""),
                        "output": "",
                        "error": rec.get("error"),
                        "duration_ms": rec.get("duration_ms", 0),
                    },
                }
            )

        elif evt == "step_end":
            step_end_recs[step] = rec

        elif evt == "step_context":
            step_ctx_recs[step] = rec

    step_traces: dict[int, dict] = {}
    for step_num in step_end_recs:
        ctx = step_ctx_recs.get(step_num, {})
        step_traces[step_num] = {
            "step": step_num + 1,  # 1-based for UI display (mirrors SSE convention)
            "model": "",  # filled later from raw_assistant.meta
            "input_tokens": 0,
            "output_tokens": 0,
            "duration_ms": 0,
            "cost_usd": 0.0,
            "timeline": step_timelines.get(step_num, []),
            "input": {
                "tool_names": ctx.get("tool_names", []),
                "message_count": ctx.get("message_count", 0),
                "on_step_start_triggers": step_pre_triggers.get(step_num, []),
            },
        }

    return {
        "task_start_triggers": task_start_triggers,
        "step_traces": step_traces,
        "step_timelines": step_timelines,
    }


def _list_sessions_in_dir(
    sessions_dir: Path,
    agent_id: str,
    project: str,
    query: Optional[str],
) -> list[dict]:
    """Return all matching sessions as raw dicts, sorted by updated_at desc."""
    if not sessions_dir.exists():
        return []

    results = []
    for f in sessions_dir.glob("*.json"):
        try:
            with open(f, encoding="utf-8") as fh:
                idx = json.load(fh)
        except Exception:
            continue

        session_id = idx.get("session_id") or f.stem
        run_ids: list[str] = idx.get("run_ids") or []
        updated_at: str = idx.get("updated_at") or ""

        session_dir = sessions_dir / session_id

        # created_at = earliest file mtime in session dir
        created_at = updated_at
        if session_dir.exists():
            mtimes = [p.stat().st_mtime for p in session_dir.iterdir() if p.is_file()]
            if mtimes:
                ts = min(mtimes)
                created_at = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.000Z"
                )

        first_query = _read_first_query(session_dir, run_ids)

        match_snippet: Optional[str] = None
        if query:
            match_snippet = _search_in_session(session_dir, run_ids, query)
            if match_snippet is None:
                continue  # skip non-matching sessions

        results.append(
            {
                "session_id": session_id,
                "agent_id": agent_id,
                "project": project,
                "first_query": first_query,
                "created_at": created_at,
                "updated_at": updated_at,
                "run_count": len(run_ids),
                "match_snippet": match_snippet,
            }
        )

    results.sort(key=lambda x: x["updated_at"], reverse=True)
    return results


def _read_display_messages(sessions_dir: Path, session_id: str, media_query: str = "") -> list[dict]:
    """Reconstruct full conversation across all segments for read-only display.

    Handles both legacy format (type: user/assistant/tool) and the new two-track
    format (type: raw_user / raw_assistant / raw_tool + optional delta records).

    For each run segment, the companion trace JSONL is also read to populate:
    - per-step StepTrace data (model, token counts, processor trigger timeline)
    - task-start processor triggers for query_context on user messages
    """
    index_path = sessions_dir / f"{session_id}.json"
    if not index_path.exists():
        return []
    try:
        with open(index_path, encoding="utf-8") as fh:
            idx = json.load(fh)
    except Exception:
        return []

    run_ids: list[str] = idx.get("run_ids") or []
    session_dir = sessions_dir / session_id
    messages: list[dict] = []

    _NEW_RAW_TYPES = {"raw_user", "raw_assistant", "raw_tool"}

    for seg_idx, run_id in enumerate(run_ids):
        records = _read_jsonl(session_dir / f"{run_id}.jsonl")
        trace_data = _build_run_trace_data(_read_jsonl(session_dir / f"{run_id}_trace.jsonl"))
        step_traces: dict[int, dict] = trace_data["step_traces"]
        step_timelines: dict[int, list[dict]] = trace_data["step_timelines"]
        task_start_triggers: list[dict] = trace_data["task_start_triggers"]

        if not records:
            continue

        # Detect format: new (has raw_* records) or legacy.
        has_raw = any(r.get("type") in _NEW_RAW_TYPES for r in records)

        # Find context_snapshot position (written at compaction boundary).
        base_pos = -1
        for i, rec in enumerate(records):
            if rec.get("type") == "context_snapshot":
                base_pos = i

        # Insert a compact-boundary system marker for segments 2+.
        if seg_idx > 0 and base_pos >= 0:
            messages.append(
                {
                    "role": "system",
                    "content": "— Context compacted —",
                    "tool_calls": [],
                    "step_traces": [],
                    "query_context": None,
                }
            )

        # Collect system-prompt and tool-names for query_context (first occurrence wins).
        run_system: str = ""
        run_tool_names: list[str] = []
        for rec in records:
            t = rec.get("type")
            if t == "system" and not run_system:
                run_system = (rec.get("message") or {}).get("content", "")
            elif t == "tools" and not run_tool_names:
                blocks = (rec.get("message") or {}).get("content", [])
                run_tool_names = [b.get("name", "") for b in blocks if isinstance(b, dict)]

        # Per-step token/model data from raw_assistant.meta (new format).
        # Key: step_id (0-based) → meta dict.
        assistant_meta: dict[int, dict] = {}
        if has_raw:
            for rec in records:
                if rec.get("type") == "raw_assistant":
                    step_n = rec.get("step", 0)
                    meta = rec.get("meta") or {}
                    if meta:
                        assistant_meta[step_n] = meta

        # Enrich step_traces with model/token data from raw_assistant.meta.
        # For interrupted steps without a step_end record, create a minimal trace
        # so model/token info is still visible in the history detail panel.
        for step_n, meta in assistant_meta.items():
            usage = meta.get("usage") or {}
            if step_n in step_traces:
                step_traces[step_n]["model"] = meta.get("model", "")
                step_traces[step_n]["input_tokens"] = usage.get("input_tokens", 0)
                step_traces[step_n]["output_tokens"] = usage.get("output_tokens", 0)
            else:
                step_traces[step_n] = {
                    "step": step_n + 1,
                    "model": meta.get("model", ""),
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "duration_ms": 0,
                    "cost_usd": 0.0,
                    "timeline": step_timelines.get(step_n, []),
                    "input": {
                        "tool_names": [],
                        "message_count": 0,
                        "on_step_start_triggers": [],
                    },
                }

        # ── Build conversation messages ─────────────────────────────────────

        # query_context for the next user message we encounter.
        pending_query_ctx: Optional[dict] = {
            "system": run_system,
            "tool_names": run_tool_names,
            "on_task_start_triggers": task_start_triggers,
            "post_query_triggers": [],
        }
        pending_assistant: Optional[dict] = None

        def flush_assistant():
            nonlocal pending_assistant
            if pending_assistant is not None:
                messages.append(pending_assistant)
                pending_assistant = None

        for i, rec in enumerate(records):
            if seg_idx > 0 and i <= base_pos:
                continue  # already in earlier segment

            # Skip historical-context duplicates in non-first segments.
            if seg_idx > 0 and rec.get("ctx") == "history":
                continue

            t = rec.get("type")
            msg = rec.get("message") or {}

            # ── User message ──────────────────────────────────────────────────
            user_type = "raw_user" if has_raw else "user"
            if t == user_type:
                flush_assistant()
                content = _extract_user_content(rec, session_id, media_query=media_query)
                if content:
                    entry: dict = {
                        "role": "user",
                        "content": content,
                        "tool_calls": [],
                        "step_traces": [],
                        "query_context": pending_query_ctx,
                    }
                    pending_query_ctx = None  # consumed; reset after model responds
                    messages.append(entry)

            # ── Assistant message ─────────────────────────────────────────────
            elif t == ("raw_assistant" if has_raw else "assistant"):
                flush_assistant()
                content = msg.get("content") or ""
                tool_calls = [
                    {"name": tc.get("name", ""), "id": tc.get("id", "")} for tc in (msg.get("tool_calls") or [])
                ]
                step_n = rec.get("step", 0)
                traces_for_step = []
                if step_n in step_traces:
                    traces_for_step = [step_traces[step_n]]
                pending_assistant = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                    "step_traces": traces_for_step,
                    "query_context": None,
                }
                # After the assistant responds, prepare a fresh query_context
                # for the next user turn in this run.
                pending_query_ctx = {
                    "system": run_system,
                    "tool_names": run_tool_names,
                    "on_task_start_triggers": [],
                    "post_query_triggers": [],
                }

            # ── Tool result ───────────────────────────────────────────────────
            elif t == ("raw_tool" if has_raw else "tool"):
                if pending_assistant is None:
                    continue
                tool_id = msg.get("tool_call_id") or ""
                tool_name = msg.get("name") or ""
                inline = msg.get("content") or ""
                if not inline:
                    meta = rec.get("meta") or {}
                    content_ref = meta.get("content_ref")
                    if content_ref:
                        ref = session_dir / content_ref
                        try:
                            inline = ref.read_text(encoding="utf-8")[:500]
                        except Exception:
                            inline = "(large output)"
                for tc in pending_assistant["tool_calls"]:
                    if tc.get("id") == tool_id or tc.get("name") == tool_name:
                        tc["output"] = inline[:500]
                        break

        flush_assistant()

    return messages


# ── Pydantic models ───────────────────────────────────────────────────────────


class SessionMeta(BaseModel):
    session_id: str
    agent_id: str
    project: str
    first_query: str
    created_at: str
    updated_at: str
    run_count: int
    match_snippet: Optional[str] = None


class SessionListResponse(BaseModel):
    sessions: list[SessionMeta]
    total: int
    page: int
    page_size: int


class DisplayMessage(BaseModel):
    role: str
    content: Any  # str or list (Anthropic content blocks for multimodal)
    tool_calls: list[dict] = []
    step_traces: list[dict] = []  # StepTrace-shaped dicts (assistant messages only)
    query_context: Optional[dict] = None  # QueryContext-shaped dict (user messages only)


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: list[DisplayMessage]


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    workspace: str = Query("current", description="current | all"),
    agent_id: Optional[str] = Query(None),
    project: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """List sessions sorted by last activity (newest first)."""
    from harnessx.home import (
        default_agent_id,
        default_project,
        list_agents,
        list_projects,
    )

    agent_id = agent_id or default_agent_id()
    project = project or default_project()

    if workspace == "all":
        all_sessions: list[dict] = []
        for aid in list_agents():
            for proj in list_projects(aid):
                sdir = _sessions_dir(aid, proj)
                all_sessions.extend(_list_sessions_in_dir(sdir, aid, proj, q))
        all_sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    else:
        sdir = _sessions_dir(agent_id, project)
        all_sessions = _list_sessions_in_dir(sdir, agent_id, project, q)

    total = len(all_sessions)
    start = (page - 1) * page_size
    page_items = all_sessions[start : start + page_size]

    return SessionListResponse(
        sessions=[SessionMeta(**s) for s in page_items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str,
    agent_id: Optional[str] = Query(None),
    project: Optional[str] = Query(None),
    workspace_base: str = Query("workspaces"),
):
    """Return all messages in a session for read-only display."""
    from harnessx.home import default_agent_id, default_project

    agent_id = agent_id or default_agent_id()
    project = project or default_project()
    sessions_dir = _sessions_dir(agent_id, project, workspace_base=workspace_base)
    if not (sessions_dir / f"{session_id}.json").exists():
        raise HTTPException(status_code=404, detail="session not found")

    from urllib.parse import urlencode

    media_query = urlencode({"agent_id": agent_id, "project": project, "workspace_base": workspace_base})
    msgs = _read_display_messages(sessions_dir, session_id, media_query=media_query)
    return SessionMessagesResponse(
        session_id=session_id,
        messages=[DisplayMessage(**m) for m in msgs],
    )


@router.get("/sessions/{session_id}/media/{filename}")
async def get_session_media(
    session_id: str,
    filename: str,
    agent_id: Optional[str] = Query(None),
    project: Optional[str] = Query(None),
    workspace_base: str = Query("workspaces"),
):
    """Serve a media file from a session's media directory."""
    from fastapi.responses import FileResponse
    from harnessx.home import default_agent_id, default_project

    agent_id = agent_id or default_agent_id()
    project = project or default_project()
    sessions_dir = _sessions_dir(agent_id, project, workspace_base=workspace_base)
    media_path = sessions_dir / session_id / "media" / filename

    if not media_path.exists() or not media_path.is_file():
        raise HTTPException(status_code=404, detail="media file not found")

    import mimetypes

    mime = mimetypes.guess_type(str(media_path))[0] or "application/octet-stream"
    return FileResponse(media_path, media_type=mime)


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    agent_id: Optional[str] = Query(None),
    project: Optional[str] = Query(None),
    workspace_base: str = Query("workspaces"),
):
    """Permanently delete a session (index file + data directory)."""
    from harnessx.home import default_agent_id, default_project

    agent_id = agent_id or default_agent_id()
    project = project or default_project()
    sessions_dir = _sessions_dir(agent_id, project, workspace_base=workspace_base)
    index_path = sessions_dir / f"{session_id}.json"

    if not index_path.exists():
        raise HTTPException(status_code=404, detail="session not found")

    try:
        index_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete index: {e}")

    session_dir = sessions_dir / session_id
    if session_dir.exists():
        try:
            shutil.rmtree(session_dir)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete session data: {e}")

    return {"ok": True}
