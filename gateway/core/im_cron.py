# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""IM cron tool — lets the agent create and manage scheduled jobs during conversation."""

from __future__ import annotations

import json
import re

from harnessx.tools.base import tool

from .dispatch import _im_channel_var, _im_event_var, _im_is_cron_var, _im_session_id_var

_MAX_JOBS_PER_GATEWAY = 50
_MAX_PROMPT_LENGTH = 4000
_MIN_TIMEOUT = 10
_MAX_TIMEOUT = 600

_DANGEROUS_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt\s+override", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"rm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"curl\s+.*\$\w+", re.IGNORECASE),
    re.compile(r"authorized_keys", re.IGNORECASE),
    re.compile(r"eval\s*\(", re.IGNORECASE),
    re.compile(r"exec\s*\(", re.IGNORECASE),
]

_INVISIBLE_CHARS = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")


def _get_cron_manager():
    from gateway.server import _cron_manager

    return _cron_manager


def _scan_prompt(prompt: str) -> str | None:
    """Return error message if prompt contains dangerous patterns, else None."""
    if _INVISIBLE_CHARS.search(prompt):
        return "Prompt contains invisible Unicode characters which are not allowed."
    for pat in _DANGEROUS_PATTERNS:
        if pat.search(prompt):
            return f"Prompt contains potentially dangerous content (matched: {pat.pattern[:40]}...)."
    return None


def _validate_schedule(schedule: str) -> tuple[dict, str | None]:
    """Parse schedule and validate. Returns (spec_dict, error_or_none)."""
    s = schedule.strip().lower()
    s_raw = schedule.strip()
    # "every 30m", "every 2h", "30m", "1h", "2h30m"
    if re.match(r"^(every\s+)?\d+[smh]", s):
        every = re.sub(r"^every\s+", "", s)
        return {"every": every}, None
    # 5-field cron expression — validate with croniter
    if len(s_raw.split()) >= 5:
        try:
            from croniter import croniter

            croniter(s_raw)
        except (ValueError, KeyError) as e:
            return {}, f"Invalid cron expression '{s_raw}': {e}"
        return {"cron": s_raw}, None
    return {}, f"Invalid schedule format '{s_raw}'. Use interval (30m, 1h) or 5-field cron (0 9 * * *)."


@tool(
    description=(
        "Manage scheduled cron jobs. Use this when the user asks for recurring tasks, "
        "periodic reminders, or automated monitoring.\n\n"
        "Actions:\n"
        "- create: Create a new job (requires name, schedule, prompt)\n"
        "- list: List all scheduled jobs\n"
        "- get: Get details of a specific job (requires job_id)\n"
        "- pause: Pause a job (requires job_id)\n"
        "- resume: Resume a paused job (requires job_id)\n"
        "- delete: Delete a job (requires job_id)\n"
        "- run_now: Trigger a job immediately (requires job_id)\n\n"
        "Schedule formats: '30m', '1h', '2h30m' (interval), or '0 9 * * *' (5-field cron).\n"
        "Jobs automatically deliver output to the current conversation."
    )
)
async def im_cron(
    action: str,
    name: str = "",
    schedule: str = "",
    prompt: str = "",
    job_id: str = "",
    timezone: str = "",
) -> str:
    # Safety: refuse in cron sessions to prevent recursive scheduling
    if _im_is_cron_var.get():
        return json.dumps({"error": "cron tool is disabled in scheduled sessions to prevent recursive scheduling."})

    mgr = _get_cron_manager()
    if mgr is None:
        return json.dumps({"error": "CronManager not initialized. Gateway may not be fully started."})

    if action == "create":
        if not name:
            return json.dumps({"error": "name is required for create"})
        if not schedule:
            return json.dumps({"error": "schedule is required for create"})
        if not prompt:
            return json.dumps({"error": "prompt is required for create"})

        # Input validation
        if len(name) > 100:
            return json.dumps({"error": "name must be 100 characters or less"})
        if len(prompt) > _MAX_PROMPT_LENGTH:
            return json.dumps({"error": f"prompt must be {_MAX_PROMPT_LENGTH} characters or less"})

        # Prompt injection scan
        scan_err = _scan_prompt(prompt)
        if scan_err:
            return json.dumps({"error": scan_err})

        # Schedule validation
        sched_spec, sched_err = _validate_schedule(schedule)
        if sched_err:
            return json.dumps({"error": sched_err})

        # Job count limit
        existing_count = len([j for j in mgr.list_jobs() if j.get("enabled", True)])
        if existing_count >= _MAX_JOBS_PER_GATEWAY:
            return json.dumps(
                {"error": f"Maximum job limit ({_MAX_JOBS_PER_GATEWAY}) reached. Delete or pause existing jobs first."}
            )

        spec: dict = {"name": name, "prompt": prompt}
        spec.update(sched_spec)
        if timezone:
            try:
                import pytz

                pytz.timezone(timezone)
            except Exception:
                return json.dumps({"error": f"Invalid timezone '{timezone}'"})
            spec["timezone"] = timezone

        # Auto-capture origin channel/chat for delivery
        channel = _im_channel_var.get()
        event = _im_event_var.get()
        if channel is not None and event is not None:
            spec["channel"] = channel.name
            spec["chat_id"] = event.conversation.chat_id
            spec["target"] = "channel"

        session_id = _im_session_id_var.get()
        if session_id:
            spec["session_id"] = f"cron:{name.lower().replace(' ', '_')[:20]}"

        job = mgr.create_job(spec)
        return json.dumps(
            {"ok": True, "id": job.id, "name": job.name, "message": f"Job '{job.name}' created successfully."}
        )

    elif action == "list":
        jobs = mgr.list_jobs()
        summary = []
        for j in jobs:
            summary.append(
                {
                    "id": j["id"],
                    "name": j["name"],
                    "enabled": j["enabled"],
                    "schedule": j.get("cron") or j.get("every") or "",
                    "prompt": j["prompt"][:60] + ("..." if len(j["prompt"]) > 60 else ""),
                    "next_run": j.get("state", {}).get("next_run_at"),
                    "last_status": j.get("state", {}).get("last_status"),
                }
            )
        return json.dumps({"jobs": summary, "count": len(summary)})

    elif action == "get":
        if not job_id:
            return json.dumps({"error": "job_id is required for get"})
        job_data = mgr.get_job(job_id)
        if job_data is None:
            return json.dumps({"error": f"Job '{job_id}' not found"})
        return json.dumps(job_data)

    elif action == "pause":
        if not job_id:
            return json.dumps({"error": "job_id is required for pause"})
        ok = mgr.pause_job(job_id)
        if not ok:
            return json.dumps({"error": f"Failed to pause job '{job_id}' (not found or is heartbeat)"})
        return json.dumps({"ok": True, "message": f"Job '{job_id}' paused."})

    elif action == "resume":
        if not job_id:
            return json.dumps({"error": "job_id is required for resume"})
        ok = mgr.resume_job(job_id)
        if not ok:
            return json.dumps({"error": f"Failed to resume job '{job_id}' (not found or is heartbeat)"})
        return json.dumps({"ok": True, "message": f"Job '{job_id}' resumed."})

    elif action == "delete":
        if not job_id:
            return json.dumps({"error": "job_id is required for delete"})
        ok = mgr.delete_job(job_id)
        if not ok:
            return json.dumps({"error": f"Failed to delete job '{job_id}' (not found or is heartbeat)"})
        return json.dumps({"ok": True, "message": f"Job '{job_id}' deleted."})

    elif action == "run_now":
        if not job_id:
            return json.dumps({"error": "job_id is required for run_now"})
        ok = await mgr.run_now(job_id)
        if not ok:
            return json.dumps({"error": f"Failed to trigger job '{job_id}' (not found)"})
        return json.dumps({"ok": True, "message": f"Job '{job_id}' triggered."})

    else:
        return json.dumps(
            {"error": f"Unknown action '{action}'. Valid: create, list, get, pause, resume, delete, run_now"}
        )
