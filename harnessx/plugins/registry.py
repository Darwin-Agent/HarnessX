# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .base import HarnessPlugin

if TYPE_CHECKING:
    from ..core.harness import Harness

_EVENT_HOOK_MAP: dict[str, str] = {
    "on_task_start": "task_start",
    "on_step_start": "step_start",
    "on_before_model": "before_model",
    "on_after_model": "after_model",
    "on_before_tool": "before_tool",
    "on_after_tool": "after_tool",
    "on_step_end": "step_end",
    "on_task_end": "task_end",
}


class PluginRegistry:
    """Central registry for HarnessPlugin instances.

    Responsibilities:
    - Store registered plugins.
    - Dispatch slash commands: either handle directly (pure-output commands)
      or set a slot on the State and let a processor handle it at task_start.
    - Aggregate processors and tools from all registered plugins (for P1 builder integration).
    """

    def __init__(self) -> None:
        self._plugins: list[HarnessPlugin] = []
        # Flat map: "/cmd" → (plugin, slot_key)
        self._slash_map: dict[str, tuple[HarnessPlugin, str | None]] = {}
        # Flat map: "/cmd" → full command dict (prompt, allowed_tools, hidden, …)
        self._command_map: dict[str, dict] = {}

    def register(self, plugin: HarnessPlugin) -> None:
        """Register a plugin.  Duplicate registrations (same name) are silently ignored."""
        if any(p.name == plugin.name for p in self._plugins):
            return
        self._plugins.append(plugin)
        for cmd, slot_key in plugin.slash_commands.items():
            cmd_lower = cmd.lower()
            if cmd_lower not in self._slash_map:
                self._slash_map[cmd_lower] = (plugin, slot_key)
        for cmd_entry in plugin.commands:
            name = cmd_entry.get("name", "")
            prompt = cmd_entry.get("prompt", "")
            if name and prompt:
                self._command_map[f"/{name}"] = dict(cmd_entry)

    def unregister(self, name: str) -> None:
        """Remove a plugin by name."""
        target = next((p for p in self._plugins if p.name == name), None)
        self._plugins = [p for p in self._plugins if p.name != name]
        self._slash_map = {cmd: (pl, sk) for cmd, (pl, sk) in self._slash_map.items() if pl.name != name}
        if target is not None:
            for cmd_entry in getattr(target, "commands", []):
                cmd_key = f"/{cmd_entry.get('name', '')}"
                self._command_map.pop(cmd_key, None)

    def dispatch_slash(
        self,
        raw: str,
        session_id: str,
        harness: "Harness",
        make_harness_fn: Callable[[str], "Harness"] | None = None,
    ) -> bool:
        """Dispatch a slash command.

        Returns True if the command was handled (caller should ``continue``),
        False if the command is unknown.

        Pure-output commands (/help, /session, /quit) are handled immediately.
        Slot-based commands (/compact) set a slot on the harness's next State
        via a marker; the corresponding processor picks it up at task_start.

        Args:
            raw:            The raw input string (e.g. "/compact now").
            session_id:     Current session ID (used by /session, /new).
            harness:        Current Harness instance.
            make_harness_fn: Callable(session_id) → Harness; required for /new.
        """
        parts = raw.strip().split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        args_str = " ".join(args)

        if cmd not in self._slash_map:
            if cmd in self._command_map:
                self._inject_command_prompt(harness, cmd, args_str)
                return True
            return False

        plugin, slot_key = self._slash_map[cmd]

        handler_name = f"_handle_{cmd.lstrip('/')}"
        if hasattr(plugin, handler_name):
            getattr(plugin, handler_name)(args, session_id, harness, make_harness_fn)
            return True

        if cmd in self._command_map:
            self._inject_command_prompt(harness, cmd, args_str)
            return True

        if slot_key is not None:
            self._set_pending_slot(harness, slot_key, args)
        return True

    def _inject_command_prompt(self, harness: "Harness", cmd: str, args_str: str) -> None:
        entry = self._command_map.get(cmd, {})
        prompt = entry.get("prompt", "")
        if args_str:
            prompt = prompt.replace("$ARGUMENTS", args_str)
        harness._pending_command_prompt = prompt  # type: ignore[attr-defined]

        allowed_tools = entry.get("allowed_tools")
        if allowed_tools:
            harness._pending_command_allowed_tools = list(allowed_tools)  # type: ignore[attr-defined]
        else:
            try:
                del harness._pending_command_allowed_tools  # type: ignore[attr-defined]
            except AttributeError:
                pass

    def _set_pending_slot(self, harness: "Harness", slot_key: str, args: list[str]) -> None:
        pending = getattr(harness, "_pending_slash_slots", {})
        pending[slot_key] = " ".join(args) if args else True
        harness._pending_slash_slots = pending  # type: ignore[attr-defined]

    def get_processors(self) -> dict[str, list[Any]]:
        """Collect all processor instances from registered plugins, keyed by hook name."""
        from ..core.processor import MultiHookProcessor

        result: dict[str, list] = {}
        for plugin in self._plugins:
            for proc in plugin.processors:
                if not isinstance(proc, MultiHookProcessor):
                    continue
                hooks_for_proc = [
                    hook_key
                    for method_name, hook_key in _EVENT_HOOK_MAP.items()
                    if _has_override(type(proc), method_name)
                ]
                for key in hooks_for_proc or ["*"]:
                    result.setdefault(key, []).append(proc)
        return result

    def get_tools(self) -> list[Any]:
        """Collect all tools from registered plugins."""
        tools: list = []
        for plugin in self._plugins:
            tools.extend(plugin.tools)
        return tools

    def all_commands(self) -> list[str]:
        """Return all registered slash command names (excluding hidden ones)."""
        visible_slash = [cmd for cmd in self._slash_map if not self._is_hidden(cmd)]
        visible_inject = [
            cmd for cmd in self._command_map if cmd not in self._slash_map and not self._command_map[cmd].get("hidden")
        ]
        return sorted(set(visible_slash) | set(visible_inject))

    def _is_hidden(self, cmd: str) -> bool:
        if cmd in self._command_map and self._command_map[cmd].get("hidden"):
            return True
        if cmd in self._slash_map:
            plugin, _ = self._slash_map[cmd]
            for entry in plugin.commands:
                if f"/{entry.get('name', '')}" == cmd:
                    return bool(entry.get("hidden"))
        return False

    def help_text(self) -> str:
        """Build a /help string from all registered plugins (hidden commands omitted)."""
        lines = ["Slash commands (not sent to the model):"]
        for cmd in self.all_commands():
            desc = ""
            if cmd in self._command_map:
                desc = self._command_map[cmd].get("description", "")
            elif cmd in self._slash_map:
                plugin, _ = self._slash_map[cmd]
                for entry in plugin.commands:
                    if entry.get("name") == cmd.lstrip("/"):
                        desc = entry.get("description", "")
                        break
            lines.append(f"  {cmd}" + (f"  —  {desc}" if desc else ""))
        return "\n".join(lines)

    @property
    def plugins(self) -> list[HarnessPlugin]:
        return list(self._plugins)


def _has_override(cls: type, method_name: str) -> bool:
    """Return True if ``cls`` or any of its non-MultiHookProcessor ancestors
    defines ``method_name`` as a real override (not the base no-op)."""
    from ..core.processor import MultiHookProcessor

    base_method = getattr(MultiHookProcessor, method_name, None)
    cls_method = getattr(cls, method_name, None)
    return cls_method is not None and cls_method is not base_method


# Global default registry (used by CLI and HarnessBuilder.plugin())
plugin_registry = PluginRegistry()
