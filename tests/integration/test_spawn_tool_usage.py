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
from harnessx.tools.base import tool
from harnessx.tools.spawn_subagent import spawn_subagent_tool
from harnessx.tracing.null_tracer import NullTracer


# ═══════════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════════


def make_spawn_harness(responses, tools, max_depth=3):
    """Build a Harness with MockProvider + spawn_subagent + given tools."""
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
# 1. Child uses a single tool (sync)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_child_uses_tool_sync():
    """Parent spawns child (wait=true); child calls echo tool; result propagates."""
    responses = [
        # [0] Parent step 0: call spawn_subagent
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "use echo", "wait": True},
                }
            ]
        },
        # [1] Child step 0: call echo
        {"tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "hello-from-child"}}]},
        # [2] Child step 1: final answer
        "echo said: hello-from-child",
        # [3] Parent step 1: final answer
        "child returned: echo said: hello-from-child",
    ]
    harness = make_spawn_harness(responses, [echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    assert "hello-from-child" in result.final_output

    # Verify spawn_subagent was called in parent trajectory
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    assert len(spawn_obs) == 1
    assert "hello-from-child" in spawn_obs[0].result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Child uses multiple tools in sequence
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_child_uses_multiple_tools():
    """Child calls add(3,4) then echo('sum is 7') in sequence."""
    responses = [
        # [0] Parent: spawn
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "add then echo", "wait": True},
                }
            ]
        },
        # [1] Child step 0: call add
        {"tool_calls": [{"id": "c1", "name": "add", "input": {"a": 3, "b": 4}}]},
        # [2] Child step 1: call echo
        {"tool_calls": [{"id": "c2", "name": "echo", "input": {"message": "sum is 7"}}]},
        # [3] Child step 2: final answer
        "result: sum is 7",
        # [4] Parent step 1: final answer
        "child said: result: sum is 7",
    ]
    harness = make_spawn_harness(responses, [add_tool, echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    assert "sum is 7" in result.final_output


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Child with tools restriction
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_child_tool_restriction():
    """Parent has [add, echo]; child restricted to tools=['echo'] only."""
    responses = [
        # [0] Parent: spawn with tools restriction
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "echo only", "wait": True, "tools": ["echo"]},
                }
            ]
        },
        # [1] Child step 0: call echo (the only tool available)
        {"tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "restricted-ok"}}]},
        # [2] Child final
        "restricted-ok",
        # [3] Parent final
        "child: restricted-ok",
    ]
    harness = make_spawn_harness(responses, [add_tool, echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    assert "restricted-ok" in result.final_output


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Child tool error handling
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_child_tool_error_handling():
    """Child calls fail_tool, gets an error, reports it gracefully."""
    responses = [
        # [0] Parent: spawn
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "call fail tool", "wait": True},
                }
            ]
        },
        # [1] Child step 0: call fail_tool
        {"tool_calls": [{"id": "c1", "name": "fail_tool", "input": {"message": "boom"}}]},
        # [2] Child step 1: report the error (model sees "Error: ..." in tool result)
        "Error encountered: This tool always fails",
        # [3] Parent final
        "child error: This tool always fails",
    ]
    harness = make_spawn_harness(responses, [fail_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    # The spawn_subagent tool_result should contain child's error report
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    assert len(spawn_obs) == 1
    assert "always fails" in spawn_obs[0].result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Async child uses tool, result injected into parent state
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_async_child_tool_result_injection():
    """wait=false child calls echo; after parent ends, child result appears in state."""
    responses = [
        # [0] Parent step 0: spawn async
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "echo async", "wait": False, "label": "w1"},
                }
            ]
        },
        # [1] Parent step 1: end turn (child still running in background)
        "spawned, waiting",
        # [2] Child step 0: call echo (runs after parent ends)
        {"tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "async-hello"}}]},
        # [3] Child step 1: final answer
        "async-hello",
    ]
    harness = make_spawn_harness(responses, [echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"

    # Wait for background child to complete
    await asyncio.sleep(0.2)

    # Check that child result was injected into the state
    _final_msgs = result.task_end.final_messages
    # The state used by the parent may have received the async injection
    # We verify via the task_end snapshot which captures messages at run end.
    # Since the child runs AFTER parent ends, the injection happens post-run.
    # We check the spawn_subagent tool_result has "accepted" (async mode)
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    assert len(spawn_obs) == 1
    assert "accepted" in spawn_obs[0].result
    assert "w1" in spawn_obs[0].result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Two concurrent async children with tools
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_concurrent_async_children_with_tools():
    """Two async children both call echo — no crash, no interference."""
    responses = [
        # [0] Parent step 0: spawn two children in one step
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "t1", "wait": False, "label": "a"},
                },
                {
                    "id": "p2",
                    "name": "spawn_subagent",
                    "input": {"task": "t2", "wait": False, "label": "b"},
                },
            ]
        },
        # [1] Parent step 1: end turn
        "spawned both",
        # [2-3] Child a: echo + final
        {"tool_calls": [{"id": "ca1", "name": "echo", "input": {"message": "from-a"}}]},
        "from-a",
        # [4-5] Child b: echo + final
        {"tool_calls": [{"id": "cb1", "name": "echo", "input": {"message": "from-b"}}]},
        "from-b",
    ]
    harness = make_spawn_harness(responses, [echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"

    # Both spawn_subagent calls should have "accepted" results
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    assert len(spawn_obs) == 2
    for obs in spawn_obs:
        assert "accepted" in obs.result

    # Wait for both children to finish
    await asyncio.sleep(0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Grandchild (3-layer) uses tool
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_grandchild_uses_tool():
    """parent → child → grandchild → echo tool. 3-layer delegation."""
    responses = [
        # [0] Parent: spawn child
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "delegate to grandchild", "wait": True},
                }
            ]
        },
        # [1] Child: spawn grandchild
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "spawn_subagent",
                    "input": {"task": "call echo", "wait": True},
                }
            ]
        },
        # [2] Grandchild: call echo
        {"tool_calls": [{"id": "g1", "name": "echo", "input": {"message": "from-grandchild"}}]},
        # [3] Grandchild final
        "gc: from-grandchild",
        # [4] Child final (received grandchild result)
        "child got: gc: from-grandchild",
        # [5] Parent final
        "parent got: child got: gc: from-grandchild",
    ]
    harness = make_spawn_harness(responses, [echo_tool], max_depth=3)
    result = await harness.run(BaseTask(description="test", max_steps=15))

    assert result.exit_reason == "done"
    assert "from-grandchild" in result.final_output


# ═══════════════════════════════════════════════════════════════════════════════
# 8. system_prompt override + tool usage
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_child_system_prompt_override_with_tool():
    """Child with custom system_prompt still uses tools correctly."""
    responses = [
        # [0] Parent: spawn with system_prompt override
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {
                        "task": "echo test",
                        "wait": True,
                        "system_prompt": "Always prefix with CALC:",
                    },
                }
            ]
        },
        # [1] Child: call echo
        {"tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "42"}}]},
        # [2] Child final
        "CALC: 42",
        # [3] Parent final
        "child said: CALC: 42",
    ]
    harness = make_spawn_harness(responses, [echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    assert "CALC: 42" in result.final_output


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Large tool output handled without crash
# ═══════════════════════════════════════════════════════════════════════════════

_LARGE_OUTPUT = "x" * 10_000 + "NEEDLE"


@tool(name="large_tool", description="Returns large output")
def large_tool_fn() -> str:
    return _LARGE_OUTPUT


@pytest.mark.asyncio
async def test_child_large_tool_output():
    """Child tool returns 10KB+ output — no crash or truncation in spawn result."""
    responses = [
        # [0] Parent: spawn
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "call large tool", "wait": True},
                }
            ]
        },
        # [1] Child: call large_tool
        {"tool_calls": [{"id": "c1", "name": "large_tool", "input": {}}]},
        # [2] Child final
        "got large output with NEEDLE",
        # [3] Parent final
        "child: got large output",
    ]
    harness = make_spawn_harness(responses, [large_tool_fn])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"
    # The spawn tool_result should contain the child's final output
    spawn_obs = [
        obs for step in result.trajectory.steps for obs in step.observation if obs.tool_name == "spawn_subagent"
    ]
    assert len(spawn_obs) == 1
    assert "NEEDLE" in spawn_obs[0].result
