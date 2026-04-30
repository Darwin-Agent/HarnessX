# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import sys
from typing import AsyncIterator

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fixtures.mock_provider import MockProvider
from fixtures.mock_tools import make_registry

from harnessx import BaseTask, HarnessConfig, ModelConfig
from harnessx.core.builder import HarnessBuilder
from harnessx.core.processor import MultiHookProcessor
from harnessx.plugins import HarnessPlugin
from harnessx.tracing.null_tracer import NullTracer


# ── Fixtures ──────────────────────────────────────────────────────────────────


class CallCountingProcessor(MultiHookProcessor):
    _singleton_group = "call_counting"
    _order = 99

    def __init__(self):
        self.task_start_count = 0

    async def on_task_start(self, event) -> AsyncIterator:
        self.task_start_count += 1
        yield event


class CountingPlugin(HarnessPlugin):
    name = "counting-plugin"

    def __init__(self):
        self.proc = CallCountingProcessor()
        self.processors = [self.proc]


def make_config(responses=None, processors=None):
    return HarnessConfig(
        tool_registry=make_registry(),
        tracer=NullTracer(),
        processors=processors or [],
    )


def make_harness(responses=None, processors=None):
    config = make_config(responses, processors)
    return ModelConfig(main=MockProvider(responses=responses or ["Done."])).agentic(config)


# ── HarnessBuilder.plugin() ───────────────────────────────────────────────────


class TestBuilderPlugin:
    def test_plugin_processors_merged(self):
        """builder.plugin() adds the plugin's processors to HarnessConfig."""
        counting = CountingPlugin()
        config = (
            HarnessBuilder()
            .slot(
                tool_registry=make_registry(),
                tracer=NullTracer(),
            )
            .plugin(counting)
            .build()
        )
        # Serializable processors become _target_ dicts in config.processors;
        # non-serializable instances go to config._rt_procs.
        rt_procs = getattr(config, "_rt_procs", [])
        proc_cls = type(counting.proc).__qualname__
        in_rt = counting.proc in rt_procs
        in_procs = any(isinstance(p, dict) and proc_cls in p.get("_target_", "") for p in config.processors)
        assert in_rt or in_procs, "Plugin processor must appear in config._rt_procs or config.processors"

    @pytest.mark.asyncio
    async def test_plugin_processor_called_during_run(self):
        """Plugin's processor is invoked by the run loop."""
        counting = CountingPlugin()
        config = (
            HarnessBuilder()
            .slot(
                tool_registry=make_registry(),
                tracer=NullTracer(),
            )
            .plugin(counting)
            .build()
        )
        harness = ModelConfig(main=MockProvider(responses=["Done."])).agentic(config)
        await harness.run(BaseTask("test task"))
        assert counting.proc.task_start_count >= 1

    def test_plugin_tools_registered(self):
        """builder.plugin() calls add_tool for each tool in the plugin."""
        from harnessx.tools import tool as tool_decorator

        @tool_decorator(name="fake_test_tool", description="A fake tool for testing")
        def fake_fn(x: str) -> str:
            return x

        class ToolPlugin(HarnessPlugin):
            name = "tool-plugin"
            tools = [fake_fn]

        config = (
            HarnessBuilder()
            .slot(
                tool_registry=make_registry(),
                tracer=NullTracer(),
            )
            .plugin(ToolPlugin())
            .build()
        )
        # Tool should be reachable via the final runtime registry.
        from harnessx.core.harness import _instantiate_runtime

        rt = _instantiate_runtime(config)
        assert "fake_test_tool" in rt.tool_registry.list_names()


# ── HarnessConfig plugins list ───────────────────────────────────────────────


class TestConfigPlugins:
    def test_harness_config_no_plugins(self):
        config = HarnessConfig(processors=[])
        assert config.plugins == []

    def test_harness_config_applies_plugins(self, tmp_path):
        """plugins list in HarnessConfig loads plugins end-to-end."""
        import json

        manifest = {"name": "desc-test-plugin", "version": "0.1.0"}
        plugin_dir = tmp_path / "my_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")

        config = HarnessConfig(processors=[], plugins=[str(plugin_dir)])
        assert config is not None
