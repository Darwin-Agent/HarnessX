# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import dataclasses
import logging
import re

from harnessx.core.events import ToolCallEvent
from harnessx.core.processor import MultiHookProcessor

logger = logging.getLogger(__name__)

_CONFIRM_TIMEOUT = 60.0  # seconds to wait for user confirmation

# ---------------------------------------------------------------------------
# Dangerous pattern rules
# ---------------------------------------------------------------------------

# Each rule: (description, compiled regex applied to the full command string)
_SHELL_RULES: list[tuple[str, re.Pattern[str]]] = [
    # Recursive force-delete at dangerous paths
    (
        "recursive force-delete at root/home level",
        re.compile(
            r"\brm\b.*-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+(/|~|\$HOME|\$\{HOME\}|/home/|/root)",
            re.IGNORECASE,
        ),
    ),
    (
        "recursive force-delete at root/home level (reversed flags)",
        re.compile(
            r"\brm\b.*-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+(/|~|\$HOME|\$\{HOME\}|/home/|/root)",
            re.IGNORECASE,
        ),
    ),
    # Fork bomb
    (
        "fork bomb",
        re.compile(r":\s*\(\s*\)\s*\{.*:\s*\|.*:\s*&", re.DOTALL),
    ),
    # Disk wipe via dd
    (
        "disk wipe via dd",
        re.compile(
            r"\bdd\b.*\bof\s*=\s*/dev/(sd[a-z]|hd[a-z]|nvme\d|disk\d|vd[a-z])\b",
            re.IGNORECASE,
        ),
    ),
    # Writing raw zeros/random to a block device
    (
        "write zeros/random to block device",
        re.compile(
            r"\b(dd|shred|wipe)\b.*\bof\s*=\s*/dev/(sd|hd|nvme|vd)",
            re.IGNORECASE,
        ),
    ),
    # Access to sensitive credential files
    (
        "access to SSH private key or AWS credentials",
        re.compile(
            r"(~|/root|/home/\w+)/\.ssh/id_|/\.aws/credentials",
            re.IGNORECASE,
        ),
    ),
    # System shadow/password files
    (
        "access to system password/shadow files",
        re.compile(r"\b(/etc/shadow|/etc/passwd)\b"),
    ),
    # Overwrite shell rc files with dangerous content
    (
        "overwrite shell startup file",
        re.compile(
            r">\s*(~|/root|/home/\w+)/\.(bash_profile|bashrc|zshrc|profile|zprofile)",
            re.IGNORECASE,
        ),
    ),
]

# Sensitive paths that should never be written to from IM context
_WRITE_SENSITIVE_PATHS: list[re.Pattern[str]] = [
    re.compile(r"(~|/root|/home/\w+)/\.ssh/"),
    re.compile(r"(~|/root|/home/\w+)/\.aws/"),
    re.compile(r"/etc/(shadow|passwd|sudoers)"),
]


def _check_command(cmd: str) -> str | None:
    """Return a description of the first matched dangerous pattern, or None."""
    for description, pattern in _SHELL_RULES:
        if pattern.search(cmd):
            return description
    return None


def _check_write_path(path: str) -> str | None:
    for pattern in _WRITE_SENSITIVE_PATHS:
        if pattern.search(path):
            return f"write to sensitive path: {path[:80]}"
    return None


class IMToolGuardProcessor(MultiHookProcessor):
    """Block dangerous shell commands and sensitive file writes in IM sessions.

    When a dangerous pattern is detected, asks the user for explicit confirmation
    via the IM channel before allowing the tool to proceed. Falls back to a hard
    block if ContextVars are unavailable (non-IM context).
    """

    _singleton_group = "im_tool_guard"
    _order = 2  # before ToolWhitelistProcessor (10)

    async def on_before_tool(self, event: ToolCallEvent):
        reason = None
        tool_preview = ""

        if event.tool_name == "Bash":
            cmd = event.tool_input.get("command", "")
            reason = _check_command(cmd)
            tool_preview = cmd[:120]
        elif event.tool_name in ("Write", "Edit"):
            path = event.tool_input.get("file_path", "")
            reason = _check_write_path(path)
            tool_preview = path[:120]

        if not reason:
            yield event
            return

        # Lazy import to avoid circular dependency at module load time
        from ..dispatch import (
            _im_channel_var,
            _im_event_var,
            _im_session_id_var,
            _im_confirm_registry_var,
        )

        channel = _im_channel_var.get()
        msg_event = _im_event_var.get()
        session_id = _im_session_id_var.get()
        registry = _im_confirm_registry_var.get()

        if channel and msg_event and session_id is not None and registry is not None:
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[bool] = loop.create_future()
            registry[session_id] = fut

            target = channel.make_reply_target(msg_event)
            question = (
                f"⚠️ **危险操作确认**\n"
                f"原因：{reason}\n"
                f"操作：`{tool_preview}`\n\n"
                f"回复 `确认执行` 继续，其他任意内容取消（{int(_CONFIRM_TIMEOUT)} 秒超时）"
            )
            sent = False
            try:
                await channel.send(target, question)
                sent = True
            except Exception as e:
                logger.warning("[im_tool_guard] failed to send confirmation question: %s", e)
                registry.pop(session_id, None)

            if sent:
                try:
                    approved = await asyncio.wait_for(fut, timeout=_CONFIRM_TIMEOUT)
                except asyncio.TimeoutError:
                    registry.pop(session_id, None)
                    logger.warning(
                        "[im_tool_guard] confirmation timed out tool=%r reason=%r",
                        event.tool_name,
                        reason,
                    )
                    yield dataclasses.replace(
                        event,
                        approved=False,
                        synthetic_result=(
                            f"[TIMEOUT] User did not confirm within {int(_CONFIRM_TIMEOUT)}s. "
                            f"{reason} operation was cancelled."
                        ),
                    )
                    return
                except asyncio.CancelledError:
                    registry.pop(session_id, None)
                    raise

                if approved:
                    logger.info(
                        "[im_tool_guard] user confirmed tool=%r reason=%r",
                        event.tool_name,
                        reason,
                    )
                    yield event
                else:
                    logger.info(
                        "[im_tool_guard] user declined tool=%r reason=%r",
                        event.tool_name,
                        reason,
                    )
                    yield dataclasses.replace(
                        event,
                        approved=False,
                        synthetic_result=f"[DECLINED] User cancelled the {reason} operation.",
                    )
                return

        # Fallback: hard block (ContextVars not set — non-IM context)
        logger.warning(
            "[im_tool_guard] blocked tool=%r reason=%r input=%r",
            event.tool_name,
            reason,
            str(event.tool_input)[:120],
        )
        yield dataclasses.replace(
            event,
            approved=False,
            synthetic_result=(
                f"[BLOCKED by security policy] {reason}. "
                "This operation is not permitted in IM sessions. "
                "Please confirm intent with the user before attempting equivalent steps."
            ),
        )
