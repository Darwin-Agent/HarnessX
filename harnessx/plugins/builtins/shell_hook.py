# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from ...core.processor import MultiHookProcessor

if TYPE_CHECKING:
    from ...core.events import TaskEndEvent, ToolCallEvent, ToolResultEvent


# Claude Code event name → HarnessX on_* method
_CC_TO_OH: dict[str, str] = {
    "Stop": "task_end",
    "PreToolUse": "before_tool",
    "PostToolUse": "after_tool",
}

# Type alias: list of (matcher_regex | None, shell_command) pairs
_HookEntries = list[tuple[str | None, str]]


class ShellHookProcessor(MultiHookProcessor):
    """Execute plugin shell hooks at HarnessX lifecycle events.

    Created by ``HarnessBuilder.plugin()`` when a plugin declares non-empty
    ``lifecycle_hooks``.  One instance per plugin (singleton group includes the
    plugin name to allow multiple plugins with hooks).

    Each hook entry is a ``(matcher, command)`` pair where *matcher* is either
    ``None`` (run unconditionally) or a regex string tested against the
    relevant event field:

    - ``task_end``    → ``TaskEndEvent.exit_reason``
    - ``before_tool`` → ``ToolCallEvent.tool_name``
    - ``after_tool``  → ``ToolResultEvent.tool_name``
    """

    _order = 99  # run after other processors at the same hook point

    def __init__(self, hooks: dict, plugin_root: Path, plugin_name: str) -> None:
        """
        Args:
            hooks:       Hook dict mapping HarnessX hook keys to entry lists.
                         Each entry is a ``(matcher_regex | None, cmd_string)`` tuple.
                         Example: ``{"task_end": [(None, "bash ./stop.sh")],
                                     "before_tool": [("Bash", "bash ./pre-bash.sh")]}``
            plugin_root: Absolute path to the plugin directory.
            plugin_name: Used to build a unique ``_singleton_group``.
        """
        self._hooks: dict[str, _HookEntries] = hooks
        self._plugin_root = plugin_root
        self._singleton_group = f"_shell_hook.{plugin_name}"

    def _run_hooks(self, event_key: str, match_value: str = "") -> None:
        """Run all hook commands for *event_key* whose matcher matches *match_value*.

        Args:
            event_key:   Internal hook key (``"task_end"``, ``"before_tool"``,
                         ``"after_tool"``).
            match_value: The value to test against each entry's matcher regex.
                         Empty string is used when there is nothing to match
                         (e.g. some TaskEndEvent exit reasons may be blank).
        """
        entries: _HookEntries = self._hooks.get(event_key, [])
        if not entries:
            return
        env = {**os.environ, "CLAUDE_PLUGIN_ROOT": str(self._plugin_root)}
        for matcher, cmd in entries:
            # Skip if matcher is set and doesn't match the relevant event field
            if matcher and not re.search(matcher, match_value):
                continue
            try:
                subprocess.run(cmd, shell=True, env=env, check=False)
            except Exception as exc:
                import warnings

                warnings.warn(
                    f"ShellHookProcessor: hook '{event_key}' failed: {exc}",
                    stacklevel=2,
                )

    async def on_task_end(self, event: "TaskEndEvent") -> AsyncIterator:
        # matcher matched against exit_reason ("done", "budget_exceeded", "error", …)
        self._run_hooks("task_end", getattr(event, "exit_reason", ""))
        yield event

    async def on_before_tool(self, event: "ToolCallEvent") -> AsyncIterator:
        self._run_hooks("before_tool", getattr(event, "tool_name", ""))
        yield event

    async def on_after_tool(self, event: "ToolResultEvent") -> AsyncIterator:
        self._run_hooks("after_tool", getattr(event, "tool_name", ""))
        yield event


def build_shell_hook_processor(lifecycle_hooks: dict, plugin_root: Path, plugin_name: str) -> ShellHookProcessor | None:
    """Build a ``ShellHookProcessor`` from a plugin's ``lifecycle_hooks`` dict.

    Normalises Claude Code event names and both entry formats (direct command
    objects and nested matcher objects) into ``(matcher, cmd)`` pairs.

    Returns ``None`` if no supported hooks are found.
    """
    normalised: dict[str, _HookEntries] = {}

    for cc_event, oh_key in _CC_TO_OH.items():
        entries = lifecycle_hooks.get(cc_event, [])
        pairs: _HookEntries = []
        for entry in entries:
            if isinstance(entry, str):
                # Plain command string — no matcher
                pairs.append((None, entry))
            elif isinstance(entry, dict):
                # Direct command object: {"type": "command", "command": "...", "matcher": "..."}
                cmd = entry.get("command", "")
                matcher: str | None = entry.get("matcher") or None
                if cmd:
                    pairs.append((matcher, cmd))
                # Nested Claude Code real format:
                # {"matcher": "Bash", "hooks": [{"type": "command", "command": "..."}]}
                # The outer matcher applies to all inner commands.
                outer_matcher: str | None = entry.get("matcher") or None
                for sub in entry.get("hooks", []):
                    if isinstance(sub, dict):
                        sub_cmd = sub.get("command", "")
                        if sub_cmd:
                            pairs.append((outer_matcher, sub_cmd))
        if pairs:
            normalised[oh_key] = pairs

    if not normalised:
        return None

    return ShellHookProcessor(normalised, plugin_root, plugin_name)
