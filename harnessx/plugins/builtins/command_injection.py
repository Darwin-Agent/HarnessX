# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, AsyncIterator

from ...core.processor import MultiHookProcessor

if TYPE_CHECKING:
    from ...core.events import TaskStartEvent


class CommandInjectionProcessor(MultiHookProcessor):
    """Prepend a pending command prompt to ``event.system_prompt``.

    One shared instance per harness (``_singleton_group``); each plugin's
    commands are merged in via ``add_commands()``.
    """

    _singleton_group = "_builtin.command_injection"
    _order = 1  # fires before context processors (order > 0) but first among them

    def __init__(self) -> None:
        self._command_map: dict[str, dict] = {}

    def add_commands(self, commands: list[dict]) -> None:
        """Register commands; only entries with a prompt are indexed."""
        for cmd in commands:
            name = cmd.get("name", "")
            prompt = cmd.get("prompt", "")
            if name and prompt:
                self._command_map[name] = dict(cmd)

    def get_prompt(self, name: str) -> str | None:
        """Return the prompt text for *name*, or ``None`` if not registered."""
        entry = self._command_map.get(name)
        return entry.get("prompt") if entry else None

    def get_allowed_tools(self, name: str) -> list[str] | None:
        """Return the ``allowed_tools`` list for *name*, or ``None`` if unrestricted."""
        entry = self._command_map.get(name)
        if entry is None:
            return None
        tools = entry.get("allowed_tools")
        return list(tools) if tools else None

    async def on_task_start(self, event: "TaskStartEvent") -> AsyncIterator:
        cmd_prompt: str | None = None
        state = getattr(event, "state", None)
        if state is not None:
            cmd_prompt = getattr(state, "_pending_command_prompt", None)
            if cmd_prompt is not None:
                try:
                    del state._pending_command_prompt  # type: ignore[attr-defined]
                except AttributeError:
                    pass

        if cmd_prompt:
            separator = "\n\n" if event.system_prompt else ""
            new_system = cmd_prompt + separator + event.system_prompt
            try:
                event = dataclasses.replace(event, system_prompt=new_system)
            except TypeError:
                event.system_prompt = new_system

        yield event
