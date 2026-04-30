# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import uuid
from typing import TYPE_CHECKING, Any

from ..base import HarnessPlugin

if TYPE_CHECKING:
    from ...core.harness import Harness


class AgentContextPlugin(HarnessPlugin):
    """Built-in plugin providing /agent, /project, and /home slash commands."""

    name = "_builtin.agent_ctx"
    version = "0.1.0"
    description = "Switch agent/project context and inspect AGENT_HOME"

    slash_commands: dict[str, str | None] = {
        "/agent": None,
        "/project": None,
        "/home": None,
    }

    commands = [
        {
            "name": "agent",
            "description": "Print or switch the active agent  (/agent [name])",
        },
        {
            "name": "project",
            "description": "Print or switch the active project (/project [name])",
        },
        {"name": "home", "description": "Show AGENT_HOME and current workspace path"},
    ]

    # ── helpers ────────────────────────────────────────────────────────────────

    def _current_workspace(self, harness: "Harness"):
        """Return the Workspace attached to harness config, or None."""
        return getattr(harness.config, "workspace", None)

    def _restart_with_workspace(
        self,
        agent_id: str,
        project: str,
        harness: "Harness",
        make_harness_fn: Any,
    ) -> None:
        """Switch workspace and restart with a fresh session."""
        from harnessx.home import agent_home
        from harnessx.workspace.workspace import Workspace

        new_ws = Workspace(
            agent_id=agent_id,
            project=project,
            home=agent_home(),
            mode=None,  # CLI: full filesystem access
        )
        from harnessx.core.harness import _runtime_workspace_to_config

        new_config = harness.config.copy(workspace=_runtime_workspace_to_config(new_ws))
        new_sid = str(uuid.uuid4())
        if make_harness_fn is not None:
            _new_harness = make_harness_fn(new_sid)
            harness.config = new_config
        else:
            harness.config = new_config
        harness._new_session_id = new_sid  # type: ignore[attr-defined]
        print(
            f"Switched to agent='{agent_id}' project='{project}' → workspace: {new_ws.root}",
            file=sys.stderr,
        )

    # ── handlers ───────────────────────────────────────────────────────────────

    def _handle_agent(
        self,
        args: list[str],
        session_id: str,
        harness: "Harness",
        make_harness_fn: Any,
    ) -> None:
        ws = self._current_workspace(harness)
        if not args:
            current = ws.agent_id if ws else "(unknown)"
            print(f"Agent: {current}", file=sys.stderr)
            return
        new_agent = args[0]
        current_project = ws.project if ws is not None else "default"
        self._restart_with_workspace(new_agent, current_project, harness, make_harness_fn)

    def _handle_project(
        self,
        args: list[str],
        session_id: str,
        harness: "Harness",
        make_harness_fn: Any,
    ) -> None:
        ws = self._current_workspace(harness)
        if not args:
            current_project = ws.project if ws is not None else "default"
            print(f"Project: {current_project}", file=sys.stderr)
            return
        new_project = args[0]
        current_agent = ws.agent_id if ws else "default"
        self._restart_with_workspace(current_agent, new_project, harness, make_harness_fn)

    def _handle_home(
        self,
        args: list[str],
        session_id: str,
        harness: "Harness",
        make_harness_fn: Any,
    ) -> None:
        from harnessx.home import agent_home

        home = agent_home()
        ws = self._current_workspace(harness)
        print(f"AGENT_HOME : {home}", file=sys.stderr)
        print(f"Agent      : {ws.agent_id if ws else '(none)'}", file=sys.stderr)
        print(f"Project    : {ws.project if ws else '(none)'}", file=sys.stderr)
        print(f"Workspace  : {ws.root if ws else '(none)'}", file=sys.stderr)
