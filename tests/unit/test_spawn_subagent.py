# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock

from harnessx.core.model_config import ModelConfig
from harnessx.core.state import State, PendingSubagent
from harnessx.tools.spawn_subagent import (
    _spawn_ctx,
    spawn_subagent,
    spawn_subagent_tool,
    SPAWN_TOOL_NAME,
)
from harnessx.processors.context.strategies.system_prompt.default import (
    DefaultSystemPromptBuilder,
)


# ---------------------------------------------------------------------------
# State: PendingSubagent + snapshot round-trip
# ---------------------------------------------------------------------------


class TestSpawnSubagent:
    def test_state_spawn_depth_default(self):
        s = State(run_id="r1")
        assert s.spawn_depth == 0
        assert s.pending_subagents == {}

    def test_state_pending_subagents_snapshot_roundtrip(self):
        s = State(run_id="r1", spawn_depth=2)
        s.pending_subagents["worker-1"] = PendingSubagent(
            label="worker-1",
            task="do something",
            run_id="child-abc",
            model="claude-haiku-4-5-20251001",
            system_prompt="",
            tools=["Read"],
        )
        snap = s.snapshot()
        s2 = State.from_snapshot(snap)
        assert s2.spawn_depth == 2
        assert "worker-1" in s2.pending_subagents
        pa = s2.pending_subagents["worker-1"]
        assert pa.task == "do something"
        assert pa.run_id == "child-abc"
        assert pa.tools == ["Read"]

    def test_state_empty_pending_snapshot(self):
        s = State(run_id="r1")
        snap = s.snapshot()
        s2 = State.from_snapshot(snap)
        assert s2.pending_subagents == {}
        assert s2.spawn_depth == 0

    # ---------------------------------------------------------------------------
    # DefaultSystemPromptBuilder: spawn section
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_spawn_section_not_present_by_default(self):
        b = DefaultSystemPromptBuilder()
        prompt = await b.build()
        assert "spawn_subagent" not in prompt

    @pytest.mark.asyncio
    async def test_spawn_section_root_agent(self):
        b = DefaultSystemPromptBuilder(spawn_depth=0, max_spawn_depth=3)
        prompt = await b.build()
        assert "spawn_subagent" in prompt
        assert "wait=true" in prompt
        assert "leaf worker" not in prompt

    @pytest.mark.asyncio
    async def test_spawn_section_leaf_worker(self):
        b = DefaultSystemPromptBuilder(spawn_depth=3, max_spawn_depth=3)
        prompt = await b.build()
        assert "leaf worker" in prompt
        assert "wait=true" not in prompt

    @pytest.mark.asyncio
    async def test_spawn_section_mid_depth(self):
        b = DefaultSystemPromptBuilder(spawn_depth=2, max_spawn_depth=3)
        prompt = await b.build()
        assert "spawn_subagent" in prompt
        assert "leaf worker" not in prompt

    # ---------------------------------------------------------------------------
    # spawn_subagent tool: depth limit
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_spawn_depth_limit(self):
        """Spawn at max depth returns error string, does not raise."""
        from harnessx.core.harness import HarnessConfig
        from harnessx.tools.inmemory import InMemoryToolRegistry
        from harnessx.tools.spawn_subagent import _MAX_SPAWN_DEPTH

        model_config = ModelConfig(main=MagicMock())
        config = HarnessConfig(tool_registry=InMemoryToolRegistry())

        token = _spawn_ctx.set(
            {
                "run_id": "r1",
                "step_id": 0,
                "spawn_depth": _MAX_SPAWN_DEPTH,
                "state": None,
                "model_config": model_config,
                "harness_config": config,
            }
        )
        try:
            result = await spawn_subagent(task="sub-task")
        finally:
            _spawn_ctx.reset(token)

        assert "Cannot spawn" in result
        assert str(_MAX_SPAWN_DEPTH) in result

    # ---------------------------------------------------------------------------
    # spawn_subagent tool: sync mode (mock harness)
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_spawn_sync_returns_output(self, monkeypatch):
        """Sync spawn awaits child harness and returns final_output."""
        from harnessx.core.harness import HarnessConfig
        from harnessx.tools.inmemory import InMemoryToolRegistry
        from harnessx.core.events import make_run_id

        model_config = ModelConfig(main=MagicMock())
        config = HarnessConfig(tool_registry=InMemoryToolRegistry())

        class _FakeResult:
            final_output = "child result"
            run_id = make_run_id()

        async def _fake_run(self, subtask, parent_run_id=None, run_id=None):
            return _FakeResult()

        import harnessx.core.harness as harness_mod

        monkeypatch.setattr(harness_mod.Harness, "run", _fake_run)

        token = _spawn_ctx.set(
            {
                "run_id": "parent-1",
                "step_id": 0,
                "spawn_depth": 0,
                "state": None,
                "tracer": None,
                "model_config": model_config,
                "harness_config": config,
            }
        )
        try:
            result = await spawn_subagent(task="do work", wait=True)
        finally:
            _spawn_ctx.reset(token)

        assert result == "child result"

    # ---------------------------------------------------------------------------
    # spawn_subagent tool: async mode injects completion message
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_spawn_async_injects_completion_message(self, monkeypatch):
        """Async spawn returns accepted JSON and pushes completion into parent state."""
        from harnessx.core.harness import HarnessConfig
        from harnessx.core.state import State
        from harnessx.tools.inmemory import InMemoryToolRegistry

        model_config = ModelConfig(main=MagicMock())
        config = HarnessConfig(tool_registry=InMemoryToolRegistry())

        class _FakeResult:
            final_output = "async result"
            run_id = "child-async-run"

        async def _fake_run(self, subtask, parent_run_id=None, run_id=None):
            return _FakeResult()

        import harnessx.core.harness as harness_mod

        monkeypatch.setattr(harness_mod.Harness, "run", _fake_run)

        parent_state = State(run_id="parent-1")
        token = _spawn_ctx.set(
            {
                "run_id": "parent-1",
                "step_id": 0,
                "spawn_depth": 0,
                "state": parent_state,
                "tracer": None,
                "model_config": model_config,
                "harness_config": config,
            }
        )
        try:
            result = await spawn_subagent(task="async work", wait=False, label="worker-a")
        finally:
            _spawn_ctx.reset(token)

        import json

        data = json.loads(result)
        assert data["status"] == "accepted"
        assert data["label"] == "worker-a"

        # pending_subagents registered immediately
        assert "worker-a" in parent_state.pending_subagents

        # Wait for the background task to complete and inject the message
        await asyncio.sleep(0.05)

        # Completion message injected, pending cleared
        assert "worker-a" not in parent_state.pending_subagents
        user_msgs = [m for m in parent_state.messages if m.role == "user"]
        assert any("async result" in m.content for m in user_msgs)

    # ---------------------------------------------------------------------------
    # Tool schema
    # ---------------------------------------------------------------------------

    def test_spawn_tool_schema(self):
        assert spawn_subagent_tool.name == SPAWN_TOOL_NAME
        assert "task" in spawn_subagent_tool.input_schema["required"]
        props = spawn_subagent_tool.input_schema["properties"]
        for key in ("task", "model", "system_prompt", "tools", "max_steps", "max_cost_usd", "wait", "label"):
            assert key in props
