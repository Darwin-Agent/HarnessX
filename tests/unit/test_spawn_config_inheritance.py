# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import sys
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harnessx import BaseTask, HarnessConfig, ModelConfig
from harnessx.core.state import State
from harnessx.processors.context.system_prompt import SystemPromptProcessor
from harnessx.processors.context.strategies.system_prompt.default import (
    DefaultSystemPromptBuilder,
)
from harnessx.tools.base import Tool
from harnessx.tools.inmemory import InMemoryToolRegistry
from harnessx.tools.spawn_subagent import (
    SPAWN_TOOL_NAME,
    _default_child_config,
    _spawn_ctx,
    spawn_subagent_tool,
)
from harnessx.tracing.null_tracer import NullTracer

from fixtures.mock_provider import MockProvider
from fixtures.mock_tools import add_tool, echo_tool, fail_tool, make_registry


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _proc_instances(config, cls=None):
    """Return all processor instances from _rt_procs (and any dicts skipped).

    After the processors-gate refactor, runtime processor instances live in
    config._rt_procs, not config.processors (which holds only _target_ dicts).
    """
    all_instances = list(getattr(config, "_rt_procs", None) or [])
    if cls is None:
        return all_instances
    return [p for p in all_instances if isinstance(p, cls)]


def _make_overrides(
    model: str = "",
    system_prompt: str = "",
    tools: list | None = None,
) -> dict:
    return {"model": model, "system_prompt": system_prompt, "tools": tools or []}


def _parent_config(**kwargs) -> HarnessConfig:
    defaults = dict(tool_registry=InMemoryToolRegistry(), tracer=NullTracer())
    defaults.update(kwargs)
    return HarnessConfig(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1.5: Config Inheritance
# ═══════════════════════════════════════════════════════════════════════════════


# ── 1.5.1 init_workspace suppression ─────────────────────────────────────────


def _make_capture_tool() -> tuple[Tool, dict]:
    """Create a tool that captures _spawn_ctx when executed."""
    captured: dict = {}

    async def _capture(**kwargs):
        ctx = _spawn_ctx.get()
        captured.update(ctx)
        return "captured"

    t = Tool(
        name="capture_ctx",
        description="Captures _spawn_ctx for testing",
        input_schema={"type": "object", "properties": {}, "required": []},
        fn=_capture,
    )
    return t, captured


class TestSpawnConfigInheritance:
    def test_child_init_workspace_suppressed(self):
        parent = _parent_config(init_workspace=True)
        assert parent.init_workspace is True

        child = _default_child_config(
            parent,
            _make_overrides(),
            child_depth=1,
            max_depth=3,
        )
        assert child.init_workspace is False

    def test_child_init_workspace_already_false(self):
        parent = _parent_config(init_workspace=False)
        child = _default_child_config(
            parent,
            _make_overrides(),
            child_depth=1,
            max_depth=3,
        )
        assert child.init_workspace is False

    # ── 1.5.2 & 1.5.3 Tool restriction ──────────────────────────────────────────

    def test_child_tools_restricted(self):
        """Only allowed tools survive restriction from parent registry."""
        registry = make_registry(add_tool, echo_tool)
        parent = _parent_config(tool_registry=registry)

        child = _default_child_config(
            parent,
            _make_overrides(tools=["add"]),
            child_depth=1,
            max_depth=3,
        )
        child_names = set(child.tool_registry.list_names())
        assert "add" in child_names
        assert "echo" not in child_names
        assert child.tool_registry is not parent.tool_registry

    def test_child_tools_restricted_with_spawn_in_parent(self):
        """When spawn_subagent is already in parent registry, it survives restriction."""
        registry = make_registry(add_tool, echo_tool)
        registry.register(spawn_subagent_tool)
        parent = _parent_config(tool_registry=registry)

        child = _default_child_config(
            parent,
            _make_overrides(tools=["add"]),
            child_depth=1,
            max_depth=3,
        )
        child_names = set(child.tool_registry.list_names())
        assert child_names == {"add", SPAWN_TOOL_NAME}

    def test_child_tools_inherited_when_empty(self):
        """Empty tools list means no restriction — child inherits parent registry."""
        registry = make_registry(add_tool, echo_tool)
        parent = _parent_config(tool_registry=registry)

        child = _default_child_config(
            parent,
            _make_overrides(tools=[]),
            child_depth=1,
            max_depth=3,
        )
        # No restriction applied, so child registry is the same object
        assert child.tool_registry is parent.tool_registry

    # ── 1.5.4 & 1.5.5 & 1.5.6 Tracer nesting ───────────────────────────────────

    def test_child_tracer_nested_with_parent_run_id(self, tmp_path):
        from harnessx.core.config_schema import TracerConfig

        base = str(tmp_path / "sessions")
        from harnessx.tracing.journal import HarnessJournal

        tracer = HarnessJournal(base_dir=base, export_jsonl=False, silent=True)
        parent = _parent_config(tracer=tracer)

        child = _default_child_config(
            parent,
            _make_overrides(),
            child_depth=1,
            max_depth=3,
            runtime_tracer=tracer,
            parent_run_id="run-abc",
        )
        expected = os.path.join(base, "run-abc", "subagents")
        # HarnessJournal is auto-converted to TracerConfig by __post_init__
        assert isinstance(child.tracer, TracerConfig)
        assert child.tracer._target_ == "harnessx.tracing.journal.HarnessJournal"
        assert child.tracer.base_dir == expected
        assert child.tracer is not tracer  # TracerConfig != HarnessJournal — always different

    def test_child_tracer_sibling_without_parent_run_id(self, tmp_path):
        from harnessx.core.config_schema import TracerConfig

        base = str(tmp_path / "sessions")
        from harnessx.tracing.journal import HarnessJournal

        tracer = HarnessJournal(base_dir=base, export_jsonl=False, silent=True)
        parent = _parent_config(tracer=tracer)

        child = _default_child_config(
            parent,
            _make_overrides(),
            child_depth=1,
            max_depth=3,
            runtime_tracer=tracer,
            parent_run_id="",
        )
        # HarnessJournal is auto-converted to TracerConfig by __post_init__
        assert isinstance(child.tracer, TracerConfig)
        assert child.tracer._target_ == "harnessx.tracing.journal.HarnessJournal"
        assert child.tracer.base_dir == base

    def test_child_tracer_custom_passthrough(self):
        """Non-HarnessJournal tracer is passed through as-is."""
        custom_tracer = MagicMock()
        parent = _parent_config(tracer=custom_tracer)

        child = _default_child_config(
            parent,
            _make_overrides(),
            child_depth=1,
            max_depth=3,
            runtime_tracer=custom_tracer,
        )
        assert child.tracer is custom_tracer

    # ── 1.5.7 & 1.5.8 & 1.5.9 Processor patching ───────────────────────────────

    def test_child_processor_default_builder_patched(self):
        builder = DefaultSystemPromptBuilder()
        proc = SystemPromptProcessor(builder)
        parent = _parent_config(processors=[proc])

        child = _default_child_config(
            parent,
            _make_overrides(system_prompt=""),
            child_depth=2,
            max_depth=3,
        )
        child_procs = _proc_instances(child, SystemPromptProcessor)
        assert len(child_procs) == 1
        child_builder = child_procs[0].system_builder
        assert isinstance(child_builder, DefaultSystemPromptBuilder)
        assert child_builder.spawn_depth == 2
        assert child_builder.max_spawn_depth == 3

    @pytest.mark.asyncio
    async def test_child_processor_system_prompt_override(self):
        builder = DefaultSystemPromptBuilder()
        proc = SystemPromptProcessor(builder)
        parent = _parent_config(processors=[proc])

        child = _default_child_config(
            parent,
            _make_overrides(system_prompt="You are a helper."),
            child_depth=1,
            max_depth=3,
        )
        child_procs = _proc_instances(child, SystemPromptProcessor)
        assert len(child_procs) == 1
        result = await child_procs[0].system_builder.build()
        assert result == "You are a helper."

    def test_child_processor_custom_builder_unchanged(self):
        custom_builder = MagicMock()
        proc = SystemPromptProcessor(custom_builder)
        parent = _parent_config(processors=[proc])

        child = _default_child_config(
            parent,
            _make_overrides(system_prompt=""),
            child_depth=1,
            max_depth=3,
        )
        child_procs = _proc_instances(child, SystemPromptProcessor)
        assert len(child_procs) == 1
        assert child_procs[0].system_builder is custom_builder

    def test_child_non_system_prompt_processors_preserved(self):
        """Non-SystemPromptProcessor processors pass through unchanged."""
        other_proc = MagicMock()
        parent = _parent_config(processors=[other_proc])

        child = _default_child_config(
            parent,
            _make_overrides(),
            child_depth=1,
            max_depth=3,
        )
        assert other_proc in _proc_instances(child)

    # ── 1.5.10 & 1.5.11 Model override ──────────────────────────────────────────

    def test_model_override_mutates_provider(self):
        """Overriding model creates a new ModelConfig with mutated provider."""
        import copy

        provider = MagicMock()
        provider.model = "claude-sonnet-4-6"
        parent_mc = ModelConfig(main=provider)

        new_main = copy.copy(provider)
        new_main.model = "gpt-4o"
        child_mc = parent_mc.copy(main=new_main)
        assert child_mc.main.model == "gpt-4o"
        assert child_mc.main is not parent_mc.main

    def test_model_no_override_inherits_parent(self):
        provider = MagicMock()
        provider.model = "claude-sonnet-4-6"
        _parent_mc = ModelConfig(main=provider)

        # When model="" (empty), spawn_subagent uses parent_model_config as-is
        overrides = _make_overrides(model="")
        assert not overrides.get("model")  # falsy → inherits parent

    # ═══════════════════════════════════════════════════════════════════════════════
    # Group 1.7: _spawn_ctx Propagation in RunLoop
    # ═══════════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_spawn_ctx_set_during_tool_execution(self):
        """RunLoop sets _spawn_ctx with correct values before tool execution."""
        capture_tool, captured = _make_capture_tool()
        registry = make_registry(capture_tool)

        responses = [
            {"tool_calls": [{"id": "c1", "name": "capture_ctx", "input": {}}]},
            "Done.",
        ]
        config = HarnessConfig(
            tool_registry=registry,
            tracer=NullTracer(),
            processors=[],
        )
        harness = ModelConfig(main=MockProvider(responses=responses)).agentic(config)
        result = await harness.run(BaseTask(description="test", max_steps=5))

        assert result.exit_reason == "done"
        # _spawn_ctx should have been set with these keys during tool execution
        assert "run_id" in captured
        assert "step_id" in captured
        assert "spawn_depth" in captured
        assert "state" in captured
        assert "tracer" in captured
        # spawn_depth defaults to 0 for fresh state
        assert captured["spawn_depth"] == 0
        assert captured["step_id"] == 0
        assert isinstance(captured["state"], State)
        assert isinstance(captured["tracer"], NullTracer)

    @pytest.mark.asyncio
    async def test_spawn_ctx_reset_after_tool_execution(self):
        """After harness run completes, _spawn_ctx should be reset to default."""
        capture_tool, _ = _make_capture_tool()
        registry = make_registry(capture_tool)

        responses = [
            {"tool_calls": [{"id": "c1", "name": "capture_ctx", "input": {}}]},
            "Done.",
        ]
        config = HarnessConfig(
            tool_registry=registry,
            tracer=NullTracer(),
            processors=[],
        )
        harness = ModelConfig(main=MockProvider(responses=responses)).agentic(config)
        await harness.run(BaseTask(description="test", max_steps=5))

        # After run, ctx should be back to default empty dict
        assert _spawn_ctx.get() == {}

    @pytest.mark.asyncio
    async def test_spawn_ctx_reset_after_tool_error(self):
        """_spawn_ctx is reset even when a tool raises an exception."""
        # fail_tool always raises RuntimeError
        registry = make_registry(fail_tool)

        responses = [
            {"tool_calls": [{"id": "c1", "name": "fail_tool", "input": {"message": "boom"}}]},
            "Done despite error.",
        ]
        config = HarnessConfig(
            tool_registry=registry,
            tracer=NullTracer(),
            processors=[],
        )
        harness = ModelConfig(main=MockProvider(responses=responses)).agentic(config)
        result = await harness.run(BaseTask(description="test", max_steps=5))

        assert result.exit_reason == "done"
        # _spawn_ctx must still be reset after failed tool
        assert _spawn_ctx.get() == {}

    @pytest.mark.asyncio
    async def test_spawn_ctx_state_spawn_depth_forwarded(self):
        """State.spawn_depth is propagated into _spawn_ctx during tool execution."""
        capture_tool, captured = _make_capture_tool()
        registry = make_registry(capture_tool)

        responses = [
            {"tool_calls": [{"id": "c1", "name": "capture_ctx", "input": {}}]},
            "Done.",
        ]
        config = HarnessConfig(
            tool_registry=registry,
            tracer=NullTracer(),
            processors=[],
        )
        harness = ModelConfig(main=MockProvider(responses=responses)).agentic(config)

        # Create a state with spawn_depth=5 to simulate a deeply nested agent
        from harnessx.core.events import make_run_id

        resume_state = State(run_id=make_run_id(), spawn_depth=5)

        result = await harness.run(
            BaseTask(description="test", max_steps=5),
            _resume_state=resume_state,
        )

        assert result.exit_reason == "done"
        assert captured["spawn_depth"] == 5

    @pytest.mark.asyncio
    async def test_spawn_ctx_run_id_matches_state(self):
        """The run_id in _spawn_ctx matches the current state.run_id."""
        capture_tool, captured = _make_capture_tool()
        registry = make_registry(capture_tool)

        responses = [
            {"tool_calls": [{"id": "c1", "name": "capture_ctx", "input": {}}]},
            "Done.",
        ]
        config = HarnessConfig(
            tool_registry=registry,
            tracer=NullTracer(),
            processors=[],
        )
        harness = ModelConfig(main=MockProvider(responses=responses)).agentic(config)
        result = await harness.run(BaseTask(description="test", max_steps=5))

        assert captured["run_id"] == result.run_id
