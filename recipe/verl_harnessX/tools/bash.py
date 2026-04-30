# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""
Bash tool — execute shell commands and return stdout+stderr.

Bash tool for verl_harnessX tool registry.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path

from .base import tool

logger = logging.getLogger(__name__)

_success_count: int = 0
_fail_count: int = 0

_WORK_ROOT = Path(os.environ.get("HARNESSX_WORK_DIR", "/tmp/harnessx-work"))


def _get_work_dir() -> str:
    subdir = f"pid-{os.getpid()}"
    work_dir = _WORK_ROOT / subdir
    work_dir.mkdir(parents=True, exist_ok=True)
    return str(work_dir)


def _get_sandbox():
    try:
        from harnessx.sandbox.base import get_current_sandbox

        return get_current_sandbox()
    except ImportError:
        return None


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

_DEFAULT_TIMEOUT_MS = int(os.environ.get("VERL_BASH_TIMEOUT_MS", "240000"))
_MAX_TIMEOUT_MS = 600_000

_BLOCKED: list[tuple[re.Pattern, str]] = [
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
    (re.compile(r":\(\)\s*\{", re.I), "Blocked: fork bomb pattern detected."),
    (
        re.compile(r"\bdd\b.*\bof=/dev/(sd|hd|nvme|xvd|vd)", re.I),
        "Blocked: writing directly to a block device is not allowed.",
    ),
    (
        re.compile(r">\s*/dev/(sd|hd|nvme|xvd|vd)", re.I),
        "Blocked: redirecting output to a block device is not allowed.",
    ),
    (
        re.compile(r"\bmkfs\b", re.I),
        "Blocked: disk formatting commands are not allowed.",
    ),
    (
        re.compile(r"\bsysrq\b.*\bb\b", re.I),
        "Blocked: forced kernel reboot via sysrq is not allowed.",
    ),
]


def _check_blocked(command: str) -> str | None:
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
    global _success_count, _fail_count
    t0 = time.monotonic()

    blocked_msg = _check_blocked(command)
    if blocked_msg:
        _fail_count += 1
        logger.warning(
            "Bash BLOCKED cmd=%s [total: %d ok, %d fail]",
            command[:80],
            _success_count,
            _fail_count,
        )
        return f"Error: {blocked_msg}"

    timeout_sec = min(timeout, _MAX_TIMEOUT_MS) / 1000

    _sandbox = _get_sandbox()

    if _sandbox is not None:
        try:
            result = await asyncio.wait_for(
                _sandbox.exec(command, timeout=timeout_sec),
                timeout=timeout_sec + 5,
            )
            elapsed = time.monotonic() - t0
            _success_count += 1
            logger.warning(
                "Bash OK (%.1fs) cmd=%s [total: %d ok, %d fail]",
                elapsed,
                command[:80],
                _success_count,
                _fail_count,
            )
            return result
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            _fail_count += 1
            logger.warning(
                "Bash TIMEOUT (%.1fs) cmd=%s [total: %d ok, %d fail]",
                elapsed,
                command[:80],
                _success_count,
                _fail_count,
            )
            return f"Error: command timed out after {timeout_sec:.0f}s"

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
        elapsed = time.monotonic() - t0
        _fail_count += 1
        logger.warning(
            "Bash TIMEOUT (%.1fs) cmd=%s [total: %d ok, %d fail]",
            elapsed,
            command[:80],
            _success_count,
            _fail_count,
        )
        return f"Error: command timed out after {timeout_sec:.0f}s"

    elapsed = time.monotonic() - t0
    output = stdout.decode("utf-8", errors="replace")
    errors = stderr.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        _fail_count += 1
        logger.warning(
            "Bash FAILED rc=%d (%.1fs) cmd=%s [total: %d ok, %d fail]",
            proc.returncode,
            elapsed,
            command[:80],
            _success_count,
            _fail_count,
        )
    else:
        _success_count += 1
        if elapsed > 10:
            logger.warning(
                "Bash OK (%.1fs, slow) cmd=%s [total: %d ok, %d fail]",
                elapsed,
                command[:80],
                _success_count,
                _fail_count,
            )
        else:
            logger.warning(
                "Bash OK (%.1fs) cmd=%s [total: %d ok, %d fail]",
                elapsed,
                command[:80],
                _success_count,
                _fail_count,
            )

    if errors:
        return f"{output}\nSTDERR: {errors}"
    return output
