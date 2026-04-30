# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import uuid
from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest

from harnessx.plugins import load_plugin
from harnessx.plugins.base import HarnessPlugin
from harnessx.plugins.registry import PluginRegistry
from harnessx.plugins.loader import load_from_manifest, load_from_directory
from harnessx.plugins.builtins.session import SessionPlugin
from harnessx.core.processor import MultiHookProcessor


# ── Fixtures & helpers ────────────────────────────────────────────────────────


class EchoProcessor(MultiHookProcessor):
    _singleton_group = "echo_test"
    _order = 99

    def __init__(self):
        self.called_task_start = False

    async def on_task_start(self, event) -> AsyncIterator:
        self.called_task_start = True
        yield event


class MyTestPlugin(HarnessPlugin):
    name = "test-plugin"
    version = "0.1.0"
    description = "A test plugin"
    processors = []  # set per-test to avoid class-level mutation
    tools = []
    slash_commands = {"/mycommand": "_my_slot"}
    commands = [{"name": "mycommand", "description": "Do something"}]


def _make_registry() -> PluginRegistry:
    return PluginRegistry()


# ── HarnessPlugin base ────────────────────────────────────────────────────────


class TestHarnessPluginBase:
    def test_default_attrs(self):
        plugin = HarnessPlugin()
        assert plugin.name == ""
        assert plugin.version == "0.1.0"
        assert plugin.processors == []
        assert plugin.tools == []
        assert plugin.slash_commands == {}
        assert plugin.commands == []

    def test_setup_and_stop_are_noops(self):
        plugin = HarnessPlugin()
        plugin.setup(None)  # should not raise
        plugin.stop()  # should not raise

    def test_repr(self):
        plugin = MyTestPlugin()
        assert "MyTestPlugin" in repr(plugin)
        assert "test-plugin" in repr(plugin)


# ── PluginRegistry ────────────────────────────────────────────────────────────


class TestPluginRegistry:
    def test_register_and_list(self):
        reg = _make_registry()
        p = MyTestPlugin()
        reg.register(p)
        assert p in reg.plugins

    def test_duplicate_registration_ignored(self):
        reg = _make_registry()
        p1 = MyTestPlugin()
        p2 = MyTestPlugin()  # same name
        reg.register(p1)
        reg.register(p2)
        assert len(reg.plugins) == 1

    def test_all_commands(self):
        reg = _make_registry()
        reg.register(MyTestPlugin())
        assert "/mycommand" in reg.all_commands()

    def test_dispatch_slash_unknown_returns_false(self):
        reg = _make_registry()
        harness = MagicMock()
        handled = reg.dispatch_slash("/unknown", "sid", harness)
        assert handled is False

    def test_dispatch_slash_sets_pending_slot(self):
        reg = _make_registry()
        reg.register(MyTestPlugin())
        harness = MagicMock(spec=[])  # bare mock without _pending_slash_slots
        handled = reg.dispatch_slash("/mycommand extra_arg", "sid", harness)
        assert handled is True
        assert hasattr(harness, "_pending_slash_slots")
        assert "_my_slot" in harness._pending_slash_slots

    def test_dispatch_slash_quit_via_direct_handler(self):
        reg = _make_registry()
        reg.register(SessionPlugin())
        harness = MagicMock(spec=[])
        handled = reg.dispatch_slash("/quit", "sid", harness)
        assert handled is True
        assert getattr(harness, "_quit_requested", False) is True

    def test_dispatch_slash_session(self, capsys):
        reg = _make_registry()
        reg.register(SessionPlugin())
        harness = MagicMock(spec=[])
        handled = reg.dispatch_slash("/session", "my-session-123", harness)
        assert handled is True
        captured = capsys.readouterr()
        assert "my-session-123" in captured.err

    def test_dispatch_slash_new_generates_uuid(self):
        reg = _make_registry()
        reg.register(SessionPlugin())
        harness = MagicMock(spec=[])
        make_fn = MagicMock(return_value=MagicMock())
        handled = reg.dispatch_slash("/new", "old-session", harness, make_fn)
        assert handled is True
        assert hasattr(harness, "_new_session_id")
        new_sid = harness._new_session_id
        # Should be a valid UUID
        uuid.UUID(new_sid)

    def test_help_text_includes_registered_commands(self):
        reg = _make_registry()
        reg.register(SessionPlugin())
        text = reg.help_text()
        assert "/new" in text
        assert "/compact" in text
        assert "/quit" in text

    def test_get_processors_aggregates_by_hook(self):
        proc = EchoProcessor()
        plugin = MyTestPlugin()
        plugin.processors = [proc]
        reg = _make_registry()
        reg.register(plugin)
        by_hook = reg.get_processors()
        # EchoProcessor overrides on_task_start → should be under "task_start"
        assert "task_start" in by_hook
        assert proc in by_hook["task_start"]

    def test_get_tools_aggregates(self):
        def fake_tool():
            pass

        plugin = MyTestPlugin()
        plugin.tools = [fake_tool]
        reg = _make_registry()
        reg.register(plugin)
        assert fake_tool in reg.get_tools()

    def test_unregister(self):
        reg = _make_registry()
        reg.register(MyTestPlugin())
        reg.unregister("test-plugin")
        assert not reg.plugins
        assert "/mycommand" not in reg.all_commands()


# ── PluginLoader ──────────────────────────────────────────────────────────────


class TestPluginLoader:
    def test_load_instance(self):
        plugin = MyTestPlugin()
        loaded = load_plugin(plugin)
        assert loaded is plugin

    def test_load_class(self):
        loaded = load_plugin(MyTestPlugin)
        assert isinstance(loaded, MyTestPlugin)

    def test_load_from_minimal_manifest(self):
        """A minimal (Claude Code compatible) manifest loads without errors."""
        manifest = {
            "name": "minimal-plugin",
            "version": "0.1.0",
            "description": "Test",
            "commands": [
                {
                    "name": "recall",
                    "description": "Recall something",
                    "prompt": "Recall: {{input}}",
                }
            ],
        }
        plugin = load_from_manifest(manifest)
        assert plugin.name == "minimal-plugin"
        assert len(plugin.commands) == 1
        assert plugin.commands[0]["name"] == "recall"
        assert plugin.processors == []
        assert plugin.tools == []
        assert plugin.slash_commands == {}

    def test_load_from_extended_manifest(self):
        """An extended manifest with slash_commands parses correctly."""
        manifest = {
            "name": "ext-plugin",
            "version": "0.1.0",
            "slash_commands": [{"command": "/recall", "slot": "_force_recall"}],
        }
        plugin = load_from_manifest(manifest)
        assert plugin.slash_commands == {"/recall": "_force_recall"}

    def test_load_from_directory(self, tmp_path):
        manifest = {"name": "dir-plugin", "version": "0.1.0"}
        (tmp_path / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
        plugin = load_from_directory(tmp_path)
        assert plugin.name == "dir-plugin"

    def test_load_from_directory_missing_manifest(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="plugin.json"):
            load_from_directory(tmp_path)

    def test_load_command_prompt_from_file(self, tmp_path):
        """Commands with file-based prompts should be resolved to text content."""
        commands_dir = tmp_path / "commands"
        commands_dir.mkdir()
        (commands_dir / "recall.md").write_text("Recall: {{input}}", encoding="utf-8")
        manifest = {
            "name": "file-prompt-plugin",
            "commands": [
                {
                    "name": "recall",
                    "description": "Recall",
                    "prompt": "./commands/recall.md",
                }
            ],
        }
        plugin = load_from_manifest(manifest, base_dir=tmp_path)
        assert plugin.commands[0]["prompt"] == "Recall: {{input}}"

    def test_load_plugin_invalid_type(self):
        with pytest.raises(TypeError):
            load_plugin(12345)


# ── SessionPlugin ─────────────────────────────────────────────────────────────


class TestSessionPlugin:
    def test_slash_commands_declared(self):
        p = SessionPlugin()
        assert "/new" in p.slash_commands
        assert "/compact" in p.slash_commands
        assert "/session" in p.slash_commands
        assert "/help" in p.slash_commands
        assert "/quit" in p.slash_commands

    def test_compact_uses_slot(self):
        """_force_compact slot key declared (not a direct handler)."""
        p = SessionPlugin()
        assert p.slash_commands["/compact"] == "_force_compact"

    def test_new_session_id_assigned_to_harness(self):
        p = SessionPlugin()
        harness = MagicMock(spec=[])
        make_fn = MagicMock(return_value=MagicMock())
        p._handle_new([], "old-sid", harness, make_fn)
        assert hasattr(harness, "_new_session_id")
        uuid.UUID(harness._new_session_id)  # valid UUID

    def test_quit_sets_flag(self):
        p = SessionPlugin()
        harness = MagicMock(spec=[])
        p._handle_quit([], "sid", harness, None)
        assert harness._quit_requested is True
