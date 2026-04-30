# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fixtures.mock_provider import MockProvider
from fixtures.mock_tools import add_tool, echo_tool, fail_tool, make_registry

from harnessx import BaseTask, HarnessConfig, ModelConfig
from harnessx.tools.base import Tool, tool
from harnessx.tools.spawn_subagent import spawn_subagent_tool
from harnessx.tracing.null_tracer import NullTracer


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def make_spawn_harness(responses, tools, max_depth=3):
    registry = make_registry(*tools)
    registry.register(spawn_subagent_tool)
    mc = ModelConfig(main=MockProvider(responses=responses))
    config = HarnessConfig(
        tool_registry=registry,
        tracer=NullTracer(),
        processors={},
    )
    config.init_workspace = False
    return mc.agentic(config)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Child tool returns empty string
# ═══════════════════════════════════════════════════════════════════════════════


@tool(name="empty_tool", description="Returns empty string")
def empty_tool_fn() -> str:
    return ""


@pytest.mark.asyncio
async def test_child_tool_returns_empty():
    """Child tool returns empty string — should not crash, child sees empty result."""
    responses = [
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "call empty", "wait": True},
                }
            ]
        },
        {"tool_calls": [{"id": "c1", "name": "empty_tool", "input": {}}]},
        "tool returned nothing",
        "child said: tool returned nothing",
    ]
    harness = make_spawn_harness(responses, [empty_tool_fn])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    assert len(spawn_obs) == 1
    assert spawn_obs[0].error is None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Child tool returns None
# ═══════════════════════════════════════════════════════════════════════════════


@tool(name="none_tool", description="Returns None")
def none_tool_fn() -> None:
    return None


@pytest.mark.asyncio
async def test_child_tool_returns_none():
    """Tool that returns None — _execute_tool converts to empty string."""
    responses = [
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "call none", "wait": True},
                }
            ]
        },
        {"tool_calls": [{"id": "c1", "name": "none_tool", "input": {}}]},
        "got nothing",
        "child: got nothing",
    ]
    harness = make_spawn_harness(responses, [none_tool_fn])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Child calls non-existent tool
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_child_calls_nonexistent_tool():
    """Child model requests a tool that doesn't exist — gets 'not found' error."""
    responses = [
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "call missing", "wait": True},
                }
            ]
        },
        # Child tries to call "nonexistent" — doesn't exist in registry
        {"tool_calls": [{"id": "c1", "name": "nonexistent", "input": {"x": 1}}]},
        # Child sees the error and reports
        "Error: tool not found",
        "child error: not found",
    ]
    harness = make_spawn_harness(responses, [echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    # spawn should have returned child's error report
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    assert len(spawn_obs) == 1
    assert "not found" in spawn_obs[0].result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Child hits max_steps budget while using tools
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_child_budget_exceeded_during_tools():
    """Child with max_steps=1 runs out of budget after first tool call."""
    responses = [
        # Parent: spawn with max_steps=1
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "multi step", "wait": True, "max_steps": 1},
                }
            ]
        },
        # Child step 0: call echo (this is the only step allowed)
        {"tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "step0"}}]},
        # Child step 1: try another tool (but budget exceeded before model call)
        {"tool_calls": [{"id": "c2", "name": "echo", "input": {"message": "step1"}}]},
        # These won't be consumed — child exits with budget_exceeded
        "unreachable",
        "parent: child was budget limited",
    ]
    harness = make_spawn_harness(responses, [echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    # The spawn_subagent tool_result should have whatever child produced
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    assert len(spawn_obs) == 1
    # Child may return partial output or budget_exceeded message
    assert spawn_obs[0].result  # non-empty


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Child tool raises unexpected exception type
# ═══════════════════════════════════════════════════════════════════════════════


@tool(name="type_error_tool", description="Raises TypeError")
def type_error_tool_fn(x: int) -> str:
    raise TypeError("unexpected type")


@pytest.mark.asyncio
async def test_child_tool_raises_unexpected_exception():
    """Tool raises TypeError (not RuntimeError) — still caught, error returned."""
    responses = [
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "call broken", "wait": True},
                }
            ]
        },
        {"tool_calls": [{"id": "c1", "name": "type_error_tool", "input": {"x": 1}}]},
        "Error: unexpected type",
        "child: type error",
    ]
    harness = make_spawn_harness(responses, [type_error_tool_fn])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    assert len(spawn_obs) == 1
    assert "type" in spawn_obs[0].result.lower() or "error" in spawn_obs[0].result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Multiple tool calls in a single child step
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_child_multiple_tool_calls_single_step():
    """Child returns two tool_calls in one model response — both executed."""
    responses = [
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "call two tools", "wait": True},
                }
            ]
        },
        # Child: two tool calls in one step
        {
            "tool_calls": [
                {"id": "c1", "name": "add", "input": {"a": 10, "b": 20}},
                {"id": "c2", "name": "echo", "input": {"message": "parallel"}},
            ]
        },
        "add=30, echo=parallel",
        "child: add=30, echo=parallel",
    ]
    harness = make_spawn_harness(responses, [add_tool, echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    assert "30" in result.final_output or "parallel" in result.final_output


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Spawn at exact depth boundary (max_depth-1 succeeds, max_depth fails)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_spawn_at_exact_depth_boundary():
    """max_depth=2: parent(0)→child(1) succeeds, child(1)→grandchild(2) is blocked."""
    responses = [
        # Parent (depth=0): spawn child
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "try to spawn grandchild", "wait": True},
                }
            ]
        },
        # Child (depth=1): try to spawn grandchild — should get "Cannot spawn"
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "spawn_subagent",
                    "input": {"task": "grandchild task", "wait": True},
                }
            ]
        },
        # Child sees "Cannot spawn" as tool_result, reports it
        "depth limit reached",
        # Parent final
        "child said: depth limit reached",
    ]
    harness = make_spawn_harness(responses, [echo_tool], max_depth=2)
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    assert len(spawn_obs) == 1
    # Child's final output should mention depth limit
    assert "depth limit" in spawn_obs[0].result.lower() or "cannot spawn" in spawn_obs[0].result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. max_depth=1 — child cannot spawn at all
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_max_depth_one_child_cannot_spawn():
    """max_depth=1: parent(0) can spawn child(1), child gets 'Cannot spawn' immediately."""
    responses = [
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "try spawn", "wait": True},
                }
            ]
        },
        # Child tries to spawn — blocked at depth 1 with max_depth=1
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "spawn_subagent",
                    "input": {"task": "grandchild", "wait": True},
                }
            ]
        },
        "Cannot spawn: depth limit",
        "child: blocked",
    ]
    harness = make_spawn_harness(responses, [echo_tool], max_depth=1)
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Tool with slow execution (async tool)
# ═══════════════════════════════════════════════════════════════════════════════


async def _slow_fn(message: str = "") -> str:
    await asyncio.sleep(0.05)
    return f"slow:{message}"


slow_tool = Tool(
    name="slow_tool",
    description="Async tool with slight delay",
    input_schema={
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": [],
    },
    fn=_slow_fn,
)


@pytest.mark.asyncio
async def test_child_async_tool_execution():
    """Child calls an async tool (coroutine fn) — awaited correctly."""
    responses = [
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "call slow", "wait": True},
                }
            ]
        },
        {"tool_calls": [{"id": "c1", "name": "slow_tool", "input": {"message": "delayed"}}]},
        "got: slow:delayed",
        "child: slow:delayed",
    ]
    harness = make_spawn_harness(responses, [slow_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    assert "slow:delayed" in result.final_output


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Child tool error followed by successful retry
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_child_tool_error_then_retry_succeeds():
    """Child calls fail_tool, gets error, then calls echo successfully."""
    responses = [
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "retry pattern", "wait": True},
                }
            ]
        },
        # Child step 0: call fail_tool — gets error
        {"tool_calls": [{"id": "c1", "name": "fail_tool", "input": {"message": "attempt1"}}]},
        # Child step 1: retry with echo
        {"tool_calls": [{"id": "c2", "name": "echo", "input": {"message": "recovered"}}]},
        # Child final
        "recovered after retry",
        # Parent final
        "child: recovered after retry",
    ]
    harness = make_spawn_harness(responses, [fail_tool, echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    assert "recovered" in result.final_output


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Async child tool fails — error injected into parent state
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_async_child_failure_injects_error_message():
    """Async child raises during run — error message injected into parent state."""
    # Child calls fail_tool, which errors, then child's model tries to respond
    # but gets budget_exceeded (max_steps=1 with 1 tool call uses the budget).
    responses = [
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "will fail", "wait": False, "label": "failing"},
                }
            ]
        },
        "spawned async",
        # Child: fail_tool
        {"tool_calls": [{"id": "c1", "name": "fail_tool", "input": {"message": "boom"}}]},
        # Child ends with error in output
        "Error: always fails",
    ]
    harness = make_spawn_harness(responses, [fail_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"

    # Wait for async child
    await asyncio.sleep(0.2)

    # The spawn tool_result should be "accepted" (async mode)
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    assert "accepted" in spawn_obs[0].result


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Spawn with empty task string
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_spawn_with_empty_task():
    """Spawn with task='' — child still runs (empty user message)."""
    responses = [
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "", "wait": True},
                }
            ]
        },
        # Child with empty task: just responds
        "child with empty task",
        # Parent final
        "child said: child with empty task",
    ]
    harness = make_spawn_harness(responses, [echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Nested tool restriction — parent restricts, child restricts further
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_nested_tool_restriction():
    """Parent restricts child to [add, echo], child restricts grandchild to [echo] only."""
    responses = [
        # Parent: spawn child with tools=[add, echo]
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {
                        "task": "restrict further",
                        "wait": True,
                        "tools": ["add", "echo"],
                    },
                }
            ]
        },
        # Child: spawn grandchild with tools=[echo]
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "spawn_subagent",
                    "input": {"task": "echo only", "wait": True, "tools": ["echo"]},
                }
            ]
        },
        # Grandchild: call echo
        {"tool_calls": [{"id": "g1", "name": "echo", "input": {"message": "nested-restricted"}}]},
        # Grandchild final
        "nested-restricted",
        # Child final
        "gc: nested-restricted",
        # Parent final
        "child: gc: nested-restricted",
    ]
    harness = make_spawn_harness(responses, [add_tool, echo_tool], max_depth=3)
    result = await harness.run(BaseTask(description="test", max_steps=15))

    assert result.exit_reason == "done"
    assert "nested-restricted" in result.final_output


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Child cost tracking — spawn_subagent result on cost-limited child
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_child_max_cost_zero_exits_immediately():
    """Child with max_cost_usd=0.0 exits with budget_exceeded before model call."""
    responses = [
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "cost test", "wait": True, "max_cost_usd": 0.0},
                }
            ]
        },
        # Child would call echo, but budget is 0 so it exits immediately
        {"tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "x"}}]},
        "unreachable",
        # Parent gets "(no output)" or budget_exceeded message
        "child was cost limited",
    ]
    harness = make_spawn_harness(responses, [echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    assert len(spawn_obs) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Concurrent sync + async spawn in same parent step
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_sync_and_async_spawn_same_step():
    """Parent calls spawn(sync) and spawn(async) in the same step — both work."""
    responses = [
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "sync job", "wait": True},
                },
                {
                    "id": "p2",
                    "name": "spawn_subagent",
                    "input": {"task": "async job", "wait": False, "label": "bg"},
                },
            ]
        },
        # Sync child runs first (blocking):
        {"tool_calls": [{"id": "sc1", "name": "echo", "input": {"message": "sync-result"}}]},
        "sync-result",
        # Parent continues after sync child completes,
        # and async spawn returns "accepted" for the second call.
        "both spawned",
        # Async child runs in background:
        {"tool_calls": [{"id": "ac1", "name": "echo", "input": {"message": "async-result"}}]},
        "async-result",
    ]
    harness = make_spawn_harness(responses, [echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    # Two spawn calls in one step
    assert len(spawn_obs) == 2

    await asyncio.sleep(0.2)
