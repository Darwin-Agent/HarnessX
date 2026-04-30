# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import uuid
from typing import TYPE_CHECKING, Any

from ..base import HarnessPlugin

if TYPE_CHECKING:
    from ...core.harness import Harness


class SessionPlugin(HarnessPlugin):
    """Built-in plugin providing session lifecycle slash commands.

    Pure-output commands (/help, /session, /quit) are handled directly by
    the registry dispatch. Slot-based commands (/compact) set a pending slot
    on the harness so SlashCommandProcessor can handle them at task_start.
    """

    name = "_builtin.session"
    version = "0.1.0"
    description = "Built-in session management commands"

    slash_commands: dict[str, str | None] = {
        "/new": None,
        "/compact": "_force_compact",
        "/session": None,
        "/help": None,
        "/quit": None,
        "/exit": None,
        "/q": None,
    }

    commands = [
        {
            "name": "new",
            "description": "Start a fresh session (new session-id, blank history)",
        },
        {
            "name": "compact",
            "description": "Summarise and compress the current context in-place",
        },
        {"name": "session", "description": "Print the current session-id"},
        {"name": "help", "description": "Show this message"},
        {"name": "quit", "description": "Exit (same as blank line or Ctrl-C)"},
    ]

    def _handle_new(
        self,
        args: list[str],
        session_id: str,
        harness: "Harness",
        make_harness_fn: Any,
    ) -> None:
        new_sid = str(uuid.uuid4())
        if make_harness_fn is not None:
            harness.config = make_harness_fn(new_sid).config
        harness._new_session_id = new_sid  # type: ignore[attr-defined]
        print(f"New session started: {new_sid}", file=sys.stderr)

    def _handle_session(
        self,
        args: list[str],
        session_id: str,
        harness: "Harness",
        make_harness_fn: Any,
    ) -> None:
        print(f"Session: {session_id}", file=sys.stderr)

    def _handle_help(
        self,
        args: list[str],
        session_id: str,
        harness: "Harness",
        make_harness_fn: Any,
    ) -> None:
        from ...plugins.registry import plugin_registry

        print(plugin_registry.help_text(), file=sys.stderr)

    def _handle_quit(
        self,
        args: list[str],
        session_id: str,
        harness: "Harness",
        make_harness_fn: Any,
    ) -> None:
        harness._quit_requested = True  # type: ignore[attr-defined]

    # aliases
    _handle_exit = _handle_quit
    _handle_q = _handle_quit
