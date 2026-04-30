# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json
import sys
import warnings
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from fixtures.mock_provider import MockProvider
from fixtures.mock_tools import make_registry

from harnessx import BaseTask
from harnessx.core.builder import HarnessBuilder
from harnessx.core.events import TaskStartEvent
from harnessx.core.processor import MultiHookProcessor
from harnessx.plugins.base import HarnessPlugin
from harnessx.plugins.builtins.command_injection import CommandInjectionProcessor
from harnessx.plugins.builtins.shell_hook import (
    ShellHookProcessor,
    build_shell_hook_processor,
)
from harnessx.plugins.dimensions.mcp_runtime import McpRuntimePlugin
from harnessx.plugins.loader import (
    load_from_directory,
    _find_skill_dirs,
    _load_hooks_json,
    _normalise_mcp_servers,
)
from harnessx.plugins.registry import PluginRegistry
from harnessx.tracing.null_tracer import NullTracer


# ══════════════════════════════════════════════════════════════════════════════
# CommandInjectionProcessor
# ══════════════════════════════════════════════════════════════════════════════


class TestCommandInjectionProcessor:
    def _make_event(self, system_prompt="", harness=None):
        """Build a minimal TaskStartEvent-like object."""
        event = MagicMock()
        event.system_prompt = system_prompt
        event._harness = harness
        event.state = MagicMock()
        event.state.slots = {}
        # Explicitly None so getattr(state, "_pending_command_prompt", None) returns None,
        # not a truthy MagicMock (MagicMock auto-creates attributes on access).
        event.state._pending_command_prompt = None

        # dataclasses.replace support
        def replace(**kwargs):
            for k, v in kwargs.items():
                setattr(event, k, v)
            return event

        event.__class__ = TaskStartEvent  # duck-type
        return event

    @pytest.mark.asyncio
    async def test_no_pending_prompt_passes_through(self):
        proc = CommandInjectionProcessor()
        proc.add_commands([{"name": "enhance", "prompt": "ENHANCE"}])
        event = self._make_event("original")
        results = [e async for e in proc.on_task_start(event)]
        assert results[0].system_prompt == "original"

    @pytest.mark.asyncio
    async def test_injects_pending_prompt(self):
        proc = CommandInjectionProcessor()
        proc.add_commands([{"name": "enhance", "prompt": "PREPEND"}])

        class FakeState:
            def __init__(self):
                self._pending_command_prompt = "PREPEND"  # instance attr → del works

        class FakeEvent:
            system_prompt = "ORIGINAL"
            _harness = None

        fe = FakeEvent()
        fe.state = FakeState()

        results = []
        async for r in proc.on_task_start(fe):
            results.append(r)

        assert "PREPEND" in results[0].system_prompt
        assert "ORIGINAL" in results[0].system_prompt
        # After processing the attribute must be consumed (deleted from instance)
        assert (
            not hasattr(fe.state, "_pending_command_prompt")
            or getattr(fe.state, "_pending_command_prompt", None) is None
        )

    def test_add_commands_registers_prompts(self):
        proc = CommandInjectionProcessor()
        proc.add_commands(
            [
                {"name": "recall", "prompt": "Please recall memory."},
                {"name": "summarise", "description": "no prompt here"},  # no prompt
            ]
        )
        assert proc.get_prompt("recall") == "Please recall memory."
        assert proc.get_prompt("summarise") is None
        assert proc.get_prompt("unknown") is None

    def test_multiple_plugins_merge_commands(self):
        proc = CommandInjectionProcessor()
        proc.add_commands([{"name": "cmd1", "prompt": "P1"}])
        proc.add_commands([{"name": "cmd2", "prompt": "P2"}])
        assert proc.get_prompt("cmd1") == "P1"
        assert proc.get_prompt("cmd2") == "P2"


# ══════════════════════════════════════════════════════════════════════════════
# ShellHookProcessor
# ══════════════════════════════════════════════════════════════════════════════


class TestShellHookProcessor:
    def test_build_from_empty_hooks_returns_none(self):
        proc = build_shell_hook_processor({}, Path("/tmp"), "test")
        assert proc is None

    def test_build_from_unsupported_events_returns_none(self):
        proc = build_shell_hook_processor(
            {"UnknownEvent": [{"type": "command", "command": "echo hi"}]},
            Path("/tmp"),
            "test",
        )
        assert proc is None

    def test_build_from_stop_hook(self, tmp_path):
        hooks = {"Stop": [{"type": "command", "command": "echo stopped"}]}
        proc = build_shell_hook_processor(hooks, tmp_path, "my-plugin")
        assert proc is not None
        assert isinstance(proc, ShellHookProcessor)
        assert proc._singleton_group == "_shell_hook.my-plugin"
        assert "task_end" in proc._hooks

    def test_build_string_command_entries(self, tmp_path):
        hooks = {"PreToolUse": ["bash ./pre.sh"], "PostToolUse": ["bash ./post.sh"]}
        proc = build_shell_hook_processor(hooks, tmp_path, "p")
        assert proc is not None
        assert "before_tool" in proc._hooks
        assert "after_tool" in proc._hooks

    @pytest.mark.asyncio
    async def test_stop_hook_runs_on_task_end(self, tmp_path):
        _ran = []
        hooks = {"Stop": [{"type": "command", "command": f"touch {tmp_path}/ran.txt"}]}
        proc = build_shell_hook_processor(hooks, tmp_path, "test")
        event = MagicMock()
        _results = [e async for e in proc.on_task_end(event)]
        assert (tmp_path / "ran.txt").exists()

    @pytest.mark.asyncio
    async def test_plugin_root_env_var_set(self, tmp_path):
        env_log = tmp_path / "env.txt"
        hooks = {"Stop": [{"type": "command", "command": f"echo $CLAUDE_PLUGIN_ROOT > {env_log}"}]}
        proc = build_shell_hook_processor(hooks, tmp_path, "test")
        event = MagicMock()
        [e async for e in proc.on_task_end(event)]
        content = env_log.read_text().strip()
        assert str(tmp_path) in content


# ══════════════════════════════════════════════════════════════════════════════
# MCP Runtime Plugin
# ══════════════════════════════════════════════════════════════════════════════


class TestMcpRuntimePlugin:
    @pytest.mark.asyncio
    async def test_runtime_processor_loads_and_connects_servers(self):
        from harnessx.tools.inmemory import InMemoryToolRegistry

        registry = InMemoryToolRegistry()
        servers = [
            {"name": "wiki", "transport": "http", "url": "http://localhost:9000/mcp"},
        ]
        plugin = McpRuntimePlugin(mcp_config={"source": "inline", "servers": servers}, ensure_primary=False)
        plugin.setup(MagicMock(tool_registry=registry))
        runtime_proc = plugin.processors[0]

        mock_client = AsyncMock()
        mock_client.list_tools.return_value = []
        event = TaskStartEvent(
            run_id="r1",
            step_id=0,
            task_description="hello",
            model="mock",
            tools=tuple(),
        )

        with patch("harnessx.plugins.dimensions.mcp_runtime.plugin.MCPClient", return_value=mock_client):
            [e async for e in runtime_proc.on_task_start(event)]
            assert mock_client.connect.await_count == 1
            [e async for e in runtime_proc.on_task_start(event)]
            assert mock_client.connect.await_count == 1
            await plugin.stop()
            mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connection_failure_retries_then_gives_up(self):
        from harnessx.plugins.dimensions.mcp_runtime import plugin as runtime_plugin_mod

        plugin = McpRuntimePlugin(
            mcp_config={"source": "inline", "servers": [{"name": "fail", "transport": "stdio", "command": "false"}]},
            ensure_primary=False,
        )
        plugin.setup(MagicMock(tool_registry=MagicMock()))
        runtime_proc = plugin.processors[0]
        event = TaskStartEvent(run_id="r1", step_id=0, task_description="t", model="mock", tools=tuple())

        mock_client = AsyncMock()
        mock_client.connect.side_effect = ConnectionRefusedError("mock: connection refused")

        with (
            patch("harnessx.plugins.dimensions.mcp_runtime.plugin.MCPClient", return_value=mock_client),
            patch("harnessx.plugins.dimensions.mcp_runtime.plugin.asyncio.sleep", new_callable=AsyncMock),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                for _ in range(runtime_plugin_mod._MAX_RETRIES):
                    [e async for e in runtime_proc.on_task_start(event)]

        state = next(iter(plugin._servers.values()))
        assert state.give_up
        assert state.client is None
        assert len(w) == runtime_plugin_mod._MAX_RETRIES
        assert any("giving up" in str(ww.message).lower() for ww in w)

        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            [e async for e in runtime_proc.on_task_start(event)]
        assert len(w2) == 0

    @pytest.mark.asyncio
    async def test_registers_tools_and_refreshes_task_start_tools(self):
        from harnessx.tools.inmemory import InMemoryToolRegistry

        registry = InMemoryToolRegistry()
        plugin = McpRuntimePlugin(
            mcp_config={
                "source": "inline",
                "servers": [{"name": "wiki", "transport": "http", "url": "http://localhost:9000/mcp"}],
            },
            ensure_primary=False,
        )
        plugin.setup(MagicMock(tool_registry=registry))
        runtime_proc = plugin.processors[0]

        mock_client = AsyncMock()
        mock_client.list_tools.return_value = [
            {
                "name": "search_docs",
                "description": "Search docs",
                "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ]
        event = TaskStartEvent(run_id="r1", step_id=0, task_description="t", model="mock", tools=tuple())

        with patch("harnessx.plugins.dimensions.mcp_runtime.plugin.MCPClient", return_value=mock_client):
            out = [e async for e in runtime_proc.on_task_start(event)]
            tool_names = {schema.name for schema in out[0].tools}
            assert "search_docs" in tool_names
            assert "search_docs" in registry.list_names()

            await plugin.stop()
            assert "search_docs" not in registry.list_names()

    @pytest.mark.asyncio
    async def test_warmup_summary_reports_servers_and_tools(self):
        from harnessx.tools.inmemory import InMemoryToolRegistry

        registry = InMemoryToolRegistry()
        plugin = McpRuntimePlugin(
            mcp_config={
                "source": "inline",
                "servers": [{"name": "wiki", "transport": "http", "url": "http://localhost:9000/mcp"}],
            },
            ensure_primary=False,
        )
        plugin.setup(MagicMock(tool_registry=registry))

        mock_client = AsyncMock()
        mock_client.list_tools.return_value = [
            {"name": "a", "description": "", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "b", "description": "", "inputSchema": {"type": "object", "properties": {}}},
        ]

        with patch("harnessx.plugins.dimensions.mcp_runtime.plugin.MCPClient", return_value=mock_client):
            summary = await plugin.warmup_summary()
            assert summary["servers"] == 1
            assert summary["connected_servers"] == 1
            assert summary["tools"] == 2

    @pytest.mark.asyncio
    async def test_warmup_cancelled_error_is_swallowed(self):
        from harnessx.tools.inmemory import InMemoryToolRegistry

        registry = InMemoryToolRegistry()
        plugin = McpRuntimePlugin(
            mcp_config={
                "source": "inline",
                "servers": [{"name": "wiki", "transport": "http", "url": "http://localhost:9000/mcp"}],
            },
            ensure_primary=False,
        )
        plugin.setup(MagicMock(tool_registry=registry))

        mock_client = AsyncMock()
        mock_client.connect.side_effect = asyncio.CancelledError("Cancelled via cancel scope")

        with patch("harnessx.plugins.dimensions.mcp_runtime.plugin.MCPClient", return_value=mock_client):
            summary = await plugin.warmup_summary()
            assert summary["servers"] == 1
            assert summary["connected_servers"] == 0
            assert summary["tools"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# Loader: capability discovery
# ══════════════════════════════════════════════════════════════════════════════


class TestLoaderCapabilityDiscovery:
    def _make_plugin_dir(self, tmp_path: Path, with_cc_layout=True) -> Path:
        """Create a realistic Claude Code plugin directory."""
        plugin_dir = tmp_path / "my-plugin"
        if with_cc_layout:
            (plugin_dir / ".claude-plugin").mkdir(parents=True)
            manifest = {"name": "my-plugin", "version": "0.1.0"}
            (plugin_dir / ".claude-plugin" / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
        else:
            plugin_dir.mkdir()
            manifest = {"name": "my-plugin", "version": "0.1.0"}
            (plugin_dir / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
        return plugin_dir

    def test_find_skill_dirs(self, tmp_path):
        plugin_dir = self._make_plugin_dir(tmp_path)
        # no skills/ dir → empty
        assert _find_skill_dirs(plugin_dir) == []
        # subdir without SKILL.md → still empty
        (plugin_dir / "skills" / "not-a-skill").mkdir(parents=True)
        assert _find_skill_dirs(plugin_dir) == []
        # subdir with SKILL.md → discovered
        skill_dir = plugin_dir / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nContent", encoding="utf-8")
        dirs = _find_skill_dirs(plugin_dir)
        assert len(dirs) == 1 and dirs[0] == skill_dir

    def test_load_hooks_json(self, tmp_path):
        plugin_dir = self._make_plugin_dir(tmp_path)
        # missing file → empty dict
        assert _load_hooks_json(plugin_dir) == {}
        # present file → parses hooks key
        hooks_dir = plugin_dir / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "hooks.json").write_text(
            json.dumps({"hooks": {"Stop": [{"type": "command", "command": "bash stop.sh"}]}}),
            encoding="utf-8",
        )
        result = _load_hooks_json(plugin_dir)
        assert "Stop" in result and result["Stop"][0]["command"] == "bash stop.sh"

    def test_normalise_mcp_servers(self):
        # empty / None inputs
        assert _normalise_mcp_servers({}) == []
        assert _normalise_mcp_servers(None) == []
        assert _normalise_mcp_servers([]) == []
        # list passthrough
        raw_list = [{"name": "x", "transport": "stdio", "command": "cmd"}]
        assert _normalise_mcp_servers(raw_list)[0]["name"] == "x"
        # dict-of-dicts: transport inferred from keys
        raw = {
            "sqlite": {
                "command": "uvx mcp-server-sqlite",
                "args": ["--db", "db.sqlite"],
            },
            "fetch": {"url": "http://localhost:3000"},
        }
        result = _normalise_mcp_servers(raw)
        assert {s["name"] for s in result} == {"sqlite", "fetch"}
        assert next(s for s in result if s["name"] == "sqlite")["transport"] == "stdio"
        assert next(s for s in result if s["name"] == "fetch")["transport"] == "http"

    def test_load_from_directory(self, tmp_path):
        # skill_dirs
        plugin_dir = self._make_plugin_dir(tmp_path, with_cc_layout=True)
        skill_dir = plugin_dir / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nContent")
        hooks_dir = plugin_dir / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "hooks.json").write_text(
            json.dumps({"hooks": {"Stop": [{"type": "command", "command": "echo done"}]}})
        )
        plugin = load_from_directory(plugin_dir)
        assert len(plugin.skill_dirs) == 1 and plugin.skill_dirs[0] == skill_dir
        assert "Stop" in plugin.lifecycle_hooks
        assert plugin._plugin_root == plugin_dir.resolve()
        # mcp_servers populated from manifest mcpServers key
        mcp_dir = tmp_path / "mcp-plugin"
        mcp_dir.mkdir()
        (mcp_dir / "plugin.json").write_text(
            json.dumps(
                {
                    "name": "mcp-plugin",
                    "mcpServers": {"sqlite": {"command": "uvx mcp-server-sqlite"}},
                }
            )
        )
        mcp_plugin = load_from_directory(mcp_dir)
        assert len(mcp_plugin.mcp_servers) == 1 and mcp_plugin.mcp_servers[0]["name"] == "sqlite"


# ══════════════════════════════════════════════════════════════════════════════
# PluginRegistry: prompt-injection dispatch
# ══════════════════════════════════════════════════════════════════════════════


class TestRegistryPromptInjection:
    def _make_plugin_with_command(self, cmd_name="enhance", prompt="DO THIS: $ARGUMENTS"):
        class TestPlugin(HarnessPlugin):
            name = "test-plugin"
            commands = [{"name": cmd_name, "description": "test", "prompt": prompt}]

        return TestPlugin()

    def test_dispatch_prompt_injection_command(self):
        plugin = self._make_plugin_with_command("recall", "RECALL: $ARGUMENTS")
        reg = PluginRegistry()
        reg.register(plugin)

        harness = MagicMock()
        handled = reg.dispatch_slash("/recall topic", "sid", harness)
        assert handled is True
        assert hasattr(harness, "_pending_command_prompt")
        assert "RECALL:" in harness._pending_command_prompt
        assert "topic" in harness._pending_command_prompt

    def test_arguments_substituted(self):
        plugin = self._make_plugin_with_command("go", "Run: $ARGUMENTS now")
        reg = PluginRegistry()
        reg.register(plugin)

        harness = MagicMock()
        reg.dispatch_slash("/go fast and furious", "sid", harness)
        assert "fast and furious" in harness._pending_command_prompt

    def test_unknown_command_returns_false(self):
        reg = PluginRegistry()
        harness = MagicMock()
        assert reg.dispatch_slash("/unknown cmd", "sid", harness) is False

    def test_command_without_prompt_not_injected(self):
        """A command with no prompt body doesn't get stored as prompt injection."""

        class NoPromptPlugin(HarnessPlugin):
            name = "no-prompt"
            commands = [{"name": "noop", "description": "nothing"}]  # no prompt key

        reg = PluginRegistry()
        reg.register(NoPromptPlugin())
        harness = MagicMock()
        result = reg.dispatch_slash("/noop", "sid", harness)
        # No prompt → not in _command_map → not handled
        assert result is False


# ══════════════════════════════════════════════════════════════════════════════
# HarnessBuilder.plugin() capability wiring
# ══════════════════════════════════════════════════════════════════════════════


class TestBuilderPluginCapabilities:
    def _base_builder(self):
        return HarnessBuilder().slot(
            tool_registry=make_registry(),
            tracer=NullTracer(),
        )

    def _make_harness(self, config):
        from harnessx.core.model_config import ModelConfig

        return ModelConfig(main=MockProvider(responses=["Done."])).agentic(config)

    def test_commands_create_command_injection_processor(self):
        class CmdPlugin(HarnessPlugin):
            name = "cmd-plugin"
            commands = [{"name": "go", "description": "Go", "prompt": "Please go now."}]

        builder = self._base_builder().plugin(CmdPlugin())
        procs = [e.processor for e in builder._entries]
        cmd_procs = [p for p in procs if isinstance(p, CommandInjectionProcessor)]
        assert len(cmd_procs) == 1
        assert cmd_procs[0].get_prompt("go") == "Please go now."

    def test_multiple_plugins_share_command_injection_processor(self):
        class P1(HarnessPlugin):
            name = "p1"
            commands = [{"name": "cmd1", "prompt": "PROMPT1"}]

        class P2(HarnessPlugin):
            name = "p2"
            commands = [{"name": "cmd2", "prompt": "PROMPT2"}]

        builder = self._base_builder().plugin(P1()).plugin(P2())
        procs = [e.processor for e in builder._entries]
        cmd_procs = [p for p in procs if isinstance(p, CommandInjectionProcessor)]
        assert len(cmd_procs) == 1, "Should share one CommandInjectionProcessor"
        assert cmd_procs[0].get_prompt("cmd1") == "PROMPT1"
        assert cmd_procs[0].get_prompt("cmd2") == "PROMPT2"

    def test_mcp_servers_mount_mcp_runtime_plugin(self):
        class McpPlugin(HarnessPlugin):
            name = "mcp-plugin"
            mcp_servers = [
                {
                    "name": "sqlite",
                    "transport": "stdio",
                    "command": "uvx mcp-server-sqlite",
                },
            ]

        builder = self._base_builder().plugin(McpPlugin())
        procs = [e.processor for e in builder._entries]
        groups = {getattr(type(p), "_singleton_group", None) for p in procs}
        assert "_mcp_runtime" in groups
        runtime_plugins = [p for p in builder._plugins if isinstance(p, McpRuntimePlugin)]
        assert len(runtime_plugins) == 1
        servers = runtime_plugins[0]._load_servers()
        assert [s.get("name") for s in servers] == ["sqlite"]

    def test_lifecycle_hooks_create_shell_hook_processor(self, tmp_path):
        class HookPlugin(HarnessPlugin):
            name = "hook-plugin"
            lifecycle_hooks = {"Stop": [{"type": "command", "command": "echo done"}]}

        p = HookPlugin()
        p._plugin_root = tmp_path  # type: ignore[attr-defined]

        builder = self._base_builder().plugin(p)
        procs = [e.processor for e in builder._entries]
        hook_procs = [p for p in procs if isinstance(p, ShellHookProcessor)]
        assert len(hook_procs) == 1

    def test_skill_dirs_preserved_on_plugin(self, tmp_path):
        """Plugin skill_dirs are preserved on the plugin object (used by Lab API for display)."""
        skill_dir = tmp_path / "skills" / "recall"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: recall\n---\n")

        class SkillPlugin(HarnessPlugin):
            name = "skill-plugin"
            skill_dirs = [skill_dir]

        plugin = SkillPlugin()
        _builder = self._base_builder().plugin(plugin)
        # skill_dirs no longer wires SkillInstallProcessor — skills are
        # discovered at runtime by SkillIndex via collect_plugin_skill_dirs().
        # The field is preserved on the plugin for Lab API display.
        assert len(plugin.skill_dirs) == 1

    @pytest.mark.asyncio
    async def test_command_injection_fires_during_run(self):
        """Full integration: /enhance → CommandInjectionProcessor prepends to system_prompt."""
        injected_prompts = []

        class RecordingProcessor(MultiHookProcessor):
            _order = 99

            async def on_task_start(self, event) -> AsyncIterator:
                injected_prompts.append(event.system_prompt)
                yield event

        class GoPlugin(HarnessPlugin):
            name = "go-plugin"
            commands = [{"name": "go", "description": "Go!", "prompt": "INJECTED_PREAMBLE"}]

        config = self._base_builder().plugin(GoPlugin()).add(RecordingProcessor()).build()

        harness = self._make_harness(config)
        harness._pending_command_prompt = "INJECTED_PREAMBLE"
        _result = await harness.run(BaseTask("Do the thing"))

        assert any("INJECTED_PREAMBLE" in p for p in injected_prompts), (
            f"Expected INJECTED_PREAMBLE in system_prompt, got: {injected_prompts}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Harness lifecycle cleanup
# ══════════════════════════════════════════════════════════════════════════════


class TestHarnessLifecycleCleanup:
    @pytest.mark.asyncio
    async def test_plugin_stop_runs_on_cleanup_not_each_task(self):
        class CountingPlugin(HarnessPlugin):
            name = "counting"

            def __init__(self):
                super().__init__()
                self.setup_calls = 0
                self.stop_calls = 0

            def setup(self, config):
                self.setup_calls += 1

            def stop(self):
                self.stop_calls += 1

        plugin = CountingPlugin()
        config = (
            HarnessBuilder()
            .slot(
                tool_registry=make_registry(),
                tracer=NullTracer(),
            )
            .build()
            .copy(plugins=[plugin])
        )
        from harnessx.core.model_config import ModelConfig

        harness = ModelConfig(main=MockProvider(responses=["Done.", "Done."])).agentic(config)

        await harness.run(BaseTask("task 1"))
        await harness.run(BaseTask("task 2"))

        assert plugin.setup_calls == 1
        assert plugin.stop_calls == 0

        await harness.cleanup()
        assert plugin.stop_calls == 1

    @pytest.mark.asyncio
    async def test_sandbox_released_on_cleanup(self):
        class FakeSandboxProvider:
            def __init__(self):
                self.acquire_calls = 0
                self.release_calls = 0

            async def acquire(self, hint_id=None, workspace=None):
                self.acquire_calls += 1
                return object()

            async def release(self, sandbox):
                self.release_calls += 1

        provider = FakeSandboxProvider()
        config = (
            HarnessBuilder()
            .slot(
                tool_registry=make_registry(),
                tracer=NullTracer(),
                sandbox_provider=provider,
            )
            .build()
        )
        from harnessx.core.model_config import ModelConfig

        harness = ModelConfig(main=MockProvider(responses=["Done.", "Done."])).agentic(config)

        await harness.run(BaseTask("task 1"))
        await harness.run(BaseTask("task 2"))

        assert provider.acquire_calls == 1
        assert provider.release_calls == 0

        await harness.cleanup()
        assert provider.release_calls == 1


# ══════════════════════════════════════════════════════════════════════════════
# allowed_tools — command-level tool restriction
# ══════════════════════════════════════════════════════════════════════════════


class TestAllowedTools:
    """allowed_tools restricts the visible tool set for the task triggered by a command."""

    def test_command_injection_processor_stores_allowed_tools(self):
        proc = CommandInjectionProcessor()
        proc.add_commands(
            [
                {
                    "name": "search",
                    "prompt": "Search for...",
                    "allowed_tools": ["WebSearch", "Read"],
                },
            ]
        )
        assert proc.get_allowed_tools("search") == ["WebSearch", "Read"]
        assert proc.get_allowed_tools("unknown") is None

    def test_command_without_allowed_tools_returns_none(self):
        proc = CommandInjectionProcessor()
        proc.add_commands([{"name": "go", "prompt": "Go now."}])
        assert proc.get_allowed_tools("go") is None

    def test_registry_sets_pending_allowed_tools_on_harness(self):
        from harnessx.plugins.registry import PluginRegistry

        class SearchPlugin(HarnessPlugin):
            name = "search-plugin"
            commands = [
                {
                    "name": "search",
                    "prompt": "Search the web.",
                    "allowed_tools": ["WebSearch"],
                }
            ]

        reg = PluginRegistry()
        reg.register(SearchPlugin())
        harness = MagicMock(spec=[])
        reg.dispatch_slash("/search query", "sid", harness)
        assert hasattr(harness, "_pending_command_allowed_tools")
        assert harness._pending_command_allowed_tools == ["WebSearch"]

    def test_registry_clears_allowed_tools_when_command_has_none(self):
        """Commands without allowed_tools must clear any stale restriction."""
        from harnessx.plugins.registry import PluginRegistry

        class FreePlugin(HarnessPlugin):
            name = "free-plugin"
            commands = [{"name": "free", "prompt": "Do anything."}]

        reg = PluginRegistry()
        reg.register(FreePlugin())
        harness = MagicMock(spec=[])
        harness._pending_command_allowed_tools = ["OldTool"]  # stale
        reg.dispatch_slash("/free", "sid", harness)
        # Should not have allowed_tools set (either absent or deleted)
        assert (
            not hasattr(harness, "_pending_command_allowed_tools")
            or getattr(harness, "_pending_command_allowed_tools", None) is None
        )

    @pytest.mark.asyncio
    async def test_allowed_tools_filters_steps_during_run(self):
        """Full integration: command with allowed_tools=[ReadTool] only shows ReadTool."""
        from harnessx.tools.base import Tool

        seen_tool_names: list[set] = []

        class ToolRecordingProcessor(MultiHookProcessor):
            _order = 50

            async def on_step_start(self, event) -> AsyncIterator:
                seen_tool_names.append({s.name for s in event.tools})
                yield event

        # Build a registry with two tools
        registry = make_registry()  # has mock tools
        extra_tool = Tool(
            name="ReadTool",
            description="Read a file",
            input_schema={"type": "object", "properties": {}},
            fn=AsyncMock(return_value="content"),
            tags=[],
        )
        registry.register(extra_tool)

        class RestrictedPlugin(HarnessPlugin):
            name = "restricted"
            commands = [
                {
                    "name": "read-only",
                    "prompt": "Read only.",
                    "allowed_tools": ["ReadTool"],
                }
            ]

        config = (
            HarnessBuilder()
            .slot(
                tool_registry=registry,
                tracer=NullTracer(),
            )
            .plugin(RestrictedPlugin())
            .add(ToolRecordingProcessor())
            .build()
        )

        from harnessx.core.model_config import ModelConfig

        harness = ModelConfig(main=MockProvider(responses=["Done."])).agentic(config)
        harness._pending_command_prompt = "Read only."
        harness._pending_command_allowed_tools = ["ReadTool"]
        await harness.run(BaseTask("Read something"))

        assert seen_tool_names, "ToolRecordingProcessor never fired"
        for step_tools in seen_tool_names:
            assert step_tools == {"ReadTool"}, f"Expected only ReadTool, got: {step_tools}"

    @pytest.mark.asyncio
    async def test_no_allowed_tools_shows_all_tools(self):
        """Without allowed_tools restriction, all registered tools are visible."""
        seen_tool_counts: list[int] = []

        class ToolCountingProcessor(MultiHookProcessor):
            _order = 50

            async def on_step_start(self, event) -> AsyncIterator:
                seen_tool_counts.append(len(event.tools))
                yield event

        registry = make_registry()
        expected_count = len(list(registry.get_schemas()))

        config = (
            HarnessBuilder()
            .slot(
                tool_registry=registry,
                tracer=NullTracer(),
            )
            .add(ToolCountingProcessor())
            .build()
        )

        from harnessx.core.model_config import ModelConfig

        harness = ModelConfig(main=MockProvider(responses=["Done."])).agentic(config)
        await harness.run(BaseTask("Do something"))

        assert all(c == expected_count for c in seen_tool_counts), (
            f"Expected {expected_count} tools each step, got: {seen_tool_counts}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# hidden — commands omitted from /help
# ══════════════════════════════════════════════════════════════════════════════


class TestHiddenCommands:
    def test_hidden_command_not_in_all_commands(self):
        from harnessx.plugins.registry import PluginRegistry

        class HiddenPlugin(HarnessPlugin):
            name = "hidden-plugin"
            commands = [
                {
                    "name": "visible",
                    "prompt": "Visible cmd.",
                    "description": "I show up",
                },
                {"name": "secret", "prompt": "Secret cmd.", "hidden": True},
            ]

        reg = PluginRegistry()
        reg.register(HiddenPlugin())
        cmds = reg.all_commands()
        assert "/visible" in cmds
        assert "/secret" not in cmds

    def test_hidden_command_not_in_help_text(self):
        from harnessx.plugins.registry import PluginRegistry

        class HiddenPlugin(HarnessPlugin):
            name = "hidden-plugin2"
            commands = [
                {"name": "shown", "prompt": "P.", "description": "Public"},
                {
                    "name": "hidden",
                    "prompt": "P.",
                    "hidden": True,
                    "description": "Internal",
                },
            ]

        reg = PluginRegistry()
        reg.register(HiddenPlugin())
        text = reg.help_text()
        assert "/shown" in text
        assert "/hidden" not in text
        assert "Internal" not in text

    def test_hidden_command_still_dispatches(self):
        """Hidden commands can still be called — they're just not listed."""
        from harnessx.plugins.registry import PluginRegistry

        class HiddenPlugin(HarnessPlugin):
            name = "hidden-plugin3"
            commands = [{"name": "internal", "prompt": "Internal cmd.", "hidden": True}]

        reg = PluginRegistry()
        reg.register(HiddenPlugin())
        harness = MagicMock(spec=[])
        result = reg.dispatch_slash("/internal", "sid", harness)
        assert result is True  # dispatched successfully
        assert hasattr(harness, "_pending_command_prompt")


# ══════════════════════════════════════════════════════════════════════════════
# harnessx plugin remove
# ══════════════════════════════════════════════════════════════════════════════


class TestPluginRemoveCLI:
    """harnessx plugin remove deletes an installed plugin directory."""

    def _make_installed_plugin(self, install_dir: Path, dir_name: str, plugin_name: str) -> Path:
        plugin_dir = install_dir / dir_name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.json").write_text(
            json.dumps({"name": plugin_name, "version": "0.1.0"}),
            encoding="utf-8",
        )
        return plugin_dir

    def test_remove_by_directory_name(self, tmp_path):
        install_dir = tmp_path / "plugins"
        plugin_dir = self._make_installed_plugin(install_dir, "my-plugin", "my-plugin")

        # Patch agent_home so _agent_home() / "plugins" resolves to tmp_path / "plugins".
        with patch("harnessx.home.agent_home", return_value=tmp_path):
            args = MagicMock()
            args.plugin_command = "remove"
            args.name = "my-plugin"
            args.yes = True

            import harnessx.cli as cli_mod

            cli_mod._plugin(args)

        assert not plugin_dir.exists(), "Plugin directory should be removed"

    def test_remove_nonexistent_exits_with_error(self, tmp_path):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "harnessx",
                "plugin",
                "remove",
                "does-not-exist",
                "--yes",
            ],
            capture_output=True,
            text=True,
        )
        # Should exit with non-zero status
        assert result.returncode != 0


# ══════════════════════════════════════════════════════════════════════════════
# Real Claude Code plugin: skills / hooks / mcpServers discovery
# ══════════════════════════════════════════════════════════════════════════════

INSTALLED_PLUGINS_JSON = Path.home() / ".claude" / "plugins" / "installed_plugins.json"
_HAS_REAL_PLUGINS = INSTALLED_PLUGINS_JSON.exists()

requires_real = pytest.mark.skipif(not _HAS_REAL_PLUGINS, reason="No Claude plugins installed")


class TestPluginCapabilities:
    @requires_real
    def test_frontend_design_has_skill_dirs(self):
        """frontend-design plugin has skills/frontend-design/SKILL.md."""
        cache = Path.home() / ".claude" / "plugins" / "cache"
        fd_dirs = list(cache.glob("*/frontend-design/*/"))
        if not fd_dirs:
            pytest.skip("frontend-design not in cache")
        plugin = load_from_directory(fd_dirs[0])
        assert len(plugin.skill_dirs) >= 1, f"Expected skill_dirs, got {plugin.skill_dirs}"
        assert any("frontend-design" in str(d) for d in plugin.skill_dirs)

    @requires_real
    def test_ralph_loop_has_lifecycle_hooks(self):
        """ralph-loop plugin has hooks/hooks.json with a Stop hook."""
        cache = Path.home() / ".claude" / "plugins" / "cache"
        rl_dirs = list(cache.glob("*/ralph-loop/*/"))
        if not rl_dirs:
            pytest.skip("ralph-loop not in cache")
        plugin = load_from_directory(rl_dirs[0])
        assert plugin.lifecycle_hooks, f"Expected lifecycle_hooks, got {plugin.lifecycle_hooks}"
        assert "Stop" in plugin.lifecycle_hooks

    @requires_real
    def test_builder_wires_real_plugin_hooks(self):
        """HarnessBuilder.plugin() creates a ShellHookProcessor for ralph-loop."""
        cache = Path.home() / ".claude" / "plugins" / "cache"
        rl_dirs = list(cache.glob("*/ralph-loop/*/"))
        if not rl_dirs:
            pytest.skip("ralph-loop not in cache")

        builder = (
            HarnessBuilder()
            .slot(
                tool_registry=make_registry(),
                tracer=NullTracer(),
            )
            .plugin(rl_dirs[0])
        )
        procs = [e.processor for e in builder._entries]
        hook_procs = [p for p in procs if isinstance(p, ShellHookProcessor)]
        assert len(hook_procs) >= 1, "Expected ShellHookProcessor for ralph-loop Stop hook"

    @requires_real
    def test_builder_wires_real_plugin_skills(self):
        """HarnessBuilder.plugin() preserves skill_dirs on the plugin (frontend-design)."""
        cache = Path.home() / ".claude" / "plugins" / "cache"
        fd_dirs = list(cache.glob("*/frontend-design/*/"))
        if not fd_dirs:
            pytest.skip("frontend-design not in cache")

        from harnessx.plugins.loader import load_plugin

        plugin = load_plugin(fd_dirs[0])
        assert len(plugin.skill_dirs) >= 1, "Expected skill_dirs for frontend-design"
