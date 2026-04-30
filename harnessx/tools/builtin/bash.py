# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from ..base import tool

# Default parent directory for per-run agent work. Any Bash command run without
# an active sandbox executes with cwd = <WORK_ROOT>/<run_id>/, so files the
# agent creates (e.g. `cat > verify.py`) stay out of the repo. Override with
# the HARNESSX_WORK_DIR env var.
_WORK_ROOT = Path(os.environ.get("HARNESSX_WORK_DIR", "/tmp/harnessx-work"))


def _get_work_dir() -> str:
    run_id = ""
    try:
        from ..spawn_subagent import _spawn_ctx

        run_id = _spawn_ctx.get().get("run_id", "") or ""
    except Exception:
        pass
    subdir = run_id if run_id else f"pid-{os.getpid()}"
    work_dir = _WORK_ROOT / subdir
    work_dir.mkdir(parents=True, exist_ok=True)
    return str(work_dir)


_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "The bash command to execute"},
        "timeout": {
            "type": "integer",
            "description": "Timeout in milliseconds (default 120000, max 600000)",
        },
    },
    "required": ["command"],
}

_DEFAULT_TIMEOUT_MS = 120_000
_MAX_TIMEOUT_MS = 600_000

# Commands blocked unconditionally — no prompting, just an error return.
# Patterns are checked against the full command string (case-insensitive).
_BLOCKED: list[tuple[re.Pattern, str]] = [
    # Recursive delete of filesystem root or home root
    (
        re.compile(r"\brm\b.*\s+-[^\s]*r[^\s]*\s+/\s*$", re.I),
        "Blocked: recursive delete of / is not allowed.",
    ),
    (
        re.compile(r"\brm\b.*\s+-[^\s]*r[^\s]*\s+/\*", re.I),
        "Blocked: recursive delete of /* is not allowed.",
    ),
    (
        re.compile(r"\brm\b.*\s+-[^\s]*r[^\s]*\s+~\s*$", re.I),
        "Blocked: recursive delete of ~ is not allowed.",
    ),
    # Fork bomb
    (re.compile(r":\(\)\s*\{", re.I), "Blocked: fork bomb pattern detected."),
    # Overwrite block devices (dd to disk, redirect to /dev/sd*)
    (
        re.compile(r"\bdd\b.*\bof=/dev/(sd|hd|nvme|xvd|vd)", re.I),
        "Blocked: writing directly to a block device is not allowed.",
    ),
    (
        re.compile(r">\s*/dev/(sd|hd|nvme|xvd|vd)", re.I),
        "Blocked: redirecting output to a block device is not allowed.",
    ),
    # Disk format
    (
        re.compile(r"\bmkfs\b", re.I),
        "Blocked: disk formatting commands are not allowed.",
    ),
    # Kernel / sysctl destructive
    (
        re.compile(r"\bsysrq\b.*\bb\b", re.I),
        "Blocked: forced kernel reboot via sysrq is not allowed.",
    ),
]


def _check_blocked(command: str) -> str | None:
    """Return an error string if the command matches a blocked pattern, else None."""
    for pattern, msg in _BLOCKED:
        if pattern.search(command):
            return msg
    return None


@tool(
    name="Bash",
    description="Execute a shell command and return stdout+stderr. Use for running scripts, checking files, and system commands.",
    input_schema=_SCHEMA,
)
async def bash_tool(command: str, timeout: int = _DEFAULT_TIMEOUT_MS) -> str:
    blocked_msg = _check_blocked(command)
    if blocked_msg:
        return f"Error: {blocked_msg}"

    timeout_sec = min(timeout, _MAX_TIMEOUT_MS) / 1000
    from ...sandbox.base import get_current_sandbox

    _sandbox = get_current_sandbox()

    if _sandbox is not None:
        # Route through sandbox (handles LocalSandbox and remote sandboxes like Harbor).
        try:
            return await asyncio.wait_for(
                _sandbox.exec(command, timeout=timeout_sec),
                timeout=timeout_sec + 5,
            )
        except asyncio.TimeoutError:
            # Kill orphaned container processes so they don't OOM the sandbox.
            try:
                await asyncio.shield(_sandbox.kill_running())
            except Exception:
                pass
            return f"Error: command timed out after {timeout_sec:.0f}s"

    # No sandbox — direct local subprocess, confined to a per-run work dir
    # so the agent's `cat > file` / `python3 script.py` don't pollute the repo.
    cwd = _get_work_dir()
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"Error: command timed out after {timeout_sec:.0f}s"
    output = stdout.decode("utf-8", errors="replace")
    errors = stderr.decode("utf-8", errors="replace")
    if errors:
        return f"{output}\nSTDERR: {errors}"
    return output
