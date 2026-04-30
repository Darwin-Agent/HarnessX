# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fixtures.mock_provider import MockProvider
from fixtures.mock_tools import echo_tool, make_registry

from harnessx import BaseTask, HarnessConfig, ModelConfig
from harnessx.api.sse_tracer import SSETracer
from harnessx.tools.spawn_subagent import spawn_subagent_tool
from harnessx.tracing.null_tracer import NullTracer


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_sse_events(queue: asyncio.Queue) -> list[dict]:
    """Drain the queue and parse all SSE data frames."""
    events = []
    while not queue.empty():
        line = queue.get_nowait()
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _events_by_type(events: list[dict]) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = {}
    for e in events:
        by_type.setdefault(e.get("type", "?"), []).append(e)
    return by_type


def make_sse_spawn_harness(responses, tools, api_run_id="test-run-id", max_depth=3):
    """Build Harness with MockProvider + SSETracer + spawn_subagent."""
    registry = make_registry(*tools)
    mc = ModelConfig(main=MockProvider(responses=responses))

    queue = asyncio.Queue()
    sse_tracer = SSETracer(queue=queue, inner=NullTracer(), api_run_id=api_run_id)

    config = HarnessConfig(
        tool_registry=registry,
        tracer=sse_tracer,
        processors={},
    )
    config.init_workspace = False

    registry.register(spawn_subagent_tool)
    harness = mc.agentic(config)
    return harness, queue


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Sync spawn emits child_start + tool_use + tool_result events
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_sync_spawn_emits_child_start_event():
    """SSETracer emits child_start when spawn_subagent (sync) runs."""
    responses = [
        # Parent: call spawn_subagent
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "say hi", "wait": True},
                }
            ]
        },
        # Child: call echo
        {"tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "hello"}}]},
        # Child: final
        "child done: hello",
        # Parent: final
        "parent done",
    ]
    harness, queue = make_sse_spawn_harness(responses, [echo_tool])
    result = await harness.run(BaseTask(description="test", max_steps=10))

    assert result.exit_reason == "done"

    events = _parse_sse_events(queue)
    by_type = _events_by_type(events)

    # Must have child_start event
    assert len(by_type.get("child_start", [])) >= 1
    cs = by_type["child_start"][0]
    assert cs["parent_run_id"]  # should be the API run_id
    assert cs["child_run_id"]
    assert "say hi" in cs.get("task", "")

    # Must have tool_use for spawn_subagent from parent
    spawn_tool_uses = [e for e in by_type.get("tool_use", []) if e.get("name") == "spawn_subagent"]
    assert len(spawn_tool_uses) >= 1

    # Must have tool_result for spawn_subagent
    spawn_tool_results = [e for e in by_type.get("tool_result", []) if e.get("name") == "spawn_subagent"]
    assert len(spawn_tool_results) >= 1
    assert "hello" in spawn_tool_results[0].get("output", "")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Child tool events appear with child run_id in SSE stream
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_child_tool_events_have_child_run_id():
    """Child's tool_use and tool_result events have a different run_id than parent."""
    api_run_id = "api-parent-123"
    responses = [
        # Parent: spawn
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "use echo", "wait": True},
                }
            ]
        },
        # Child: echo
        {"tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "from-child"}}]},
        # Child: final
        "from-child",
        # Parent: final
        "got it",
    ]
    harness, queue = make_sse_spawn_harness(responses, [echo_tool], api_run_id=api_run_id)
    _result = await harness.run(BaseTask(description="test", max_steps=10))

    events = _parse_sse_events(queue)
    by_type = _events_by_type(events)

    # Child echo tool_use must have a run_id != api_run_id (child's own run_id)
    echo_uses = [e for e in by_type.get("tool_use", []) if e.get("name") == "echo"]
    assert len(echo_uses) >= 1
    child_run_id = echo_uses[0]["run_id"]
    assert child_run_id != api_run_id, "Child tool events must have child run_id"

    # Child echo tool_result also has child run_id
    echo_results = [e for e in by_type.get("tool_result", []) if e.get("name") == "echo"]
    assert len(echo_results) >= 1
    assert echo_results[0]["run_id"] == child_run_id

    # Parent spawn_subagent tool events have the api_run_id
    spawn_uses = [e for e in by_type.get("tool_use", []) if e.get("name") == "spawn_subagent"]
    assert len(spawn_uses) >= 1
    assert spawn_uses[0]["run_id"] == api_run_id


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Async spawn emits child_start
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_async_spawn_emits_child_start():
    """Async spawn (wait=false) still emits child_start event."""
    responses = [
        # Parent: spawn async
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "async job", "wait": False, "label": "w1"},
                }
            ]
        },
        # Parent: end turn
        "spawned async",
        # Child (background): echo + final
        {"tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "async-output"}}]},
        "async-output",
    ]
    harness, queue = make_sse_spawn_harness(responses, [echo_tool])
    _result = await harness.run(BaseTask(description="test", max_steps=10))

    # Wait for async child to complete
    await asyncio.sleep(0.2)

    events = _parse_sse_events(queue)
    by_type = _events_by_type(events)

    # child_start must be present
    assert len(by_type.get("child_start", [])) >= 1
    cs = by_type["child_start"][0]
    assert "async job" in cs.get("task", "")

    # spawn_subagent tool_result should contain "accepted"
    spawn_results = [e for e in by_type.get("tool_result", []) if e.get("name") == "spawn_subagent"]
    assert len(spawn_results) >= 1
    assert "accepted" in spawn_results[0].get("output", "")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Multiple children produce separate child_start events
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_multiple_children_emit_separate_child_start():
    """Two sync spawns produce two distinct child_start events with different child_run_ids."""
    responses = [
        # Parent: first spawn
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "child-A task", "wait": True},
                }
            ]
        },
        # Child A: echo + final
        {"tool_calls": [{"id": "ca1", "name": "echo", "input": {"message": "A"}}]},
        "A",
        # Parent: second spawn
        {
            "tool_calls": [
                {
                    "id": "p2",
                    "name": "spawn_subagent",
                    "input": {"task": "child-B task", "wait": True},
                }
            ]
        },
        # Child B: echo + final
        {"tool_calls": [{"id": "cb1", "name": "echo", "input": {"message": "B"}}]},
        "B",
        # Parent: final
        "both done",
    ]
    harness, queue = make_sse_spawn_harness(responses, [echo_tool])
    _result = await harness.run(BaseTask(description="test", max_steps=15))

    events = _parse_sse_events(queue)
    by_type = _events_by_type(events)

    child_starts = by_type.get("child_start", [])
    assert len(child_starts) == 2

    # Two distinct child_run_ids
    child_ids = {cs["child_run_id"] for cs in child_starts}
    assert len(child_ids) == 2

    # Tasks match
    tasks = {cs.get("task", "") for cs in child_starts}
    assert any("child-A" in t for t in tasks)
    assert any("child-B" in t for t in tasks)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SSE event ordering: child events nested between parent spawn/result
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_sse_event_ordering():
    """child_start and child events appear between parent's tool_use and tool_result for spawn."""
    api_run_id = "ordering-test"
    responses = [
        # Parent: spawn
        {
            "tool_calls": [
                {
                    "id": "p1",
                    "name": "spawn_subagent",
                    "input": {"task": "ordering test", "wait": True},
                }
            ]
        },
        # Child: echo
        {"tool_calls": [{"id": "c1", "name": "echo", "input": {"message": "child-output"}}]},
        # Child: final
        "child-output",
        # Parent: final
        "parent-final",
    ]
    harness, queue = make_sse_spawn_harness(responses, [echo_tool], api_run_id=api_run_id)
    await harness.run(BaseTask(description="test", max_steps=10))

    events = _parse_sse_events(queue)

    # Find indices of key events
    spawn_use_idx = None
    child_start_idx = None
    child_echo_idx = None
    spawn_result_idx = None

    for i, e in enumerate(events):
        if e.get("type") == "tool_use" and e.get("name") == "spawn_subagent":
            spawn_use_idx = i
        elif e.get("type") == "child_start":
            child_start_idx = i
        elif e.get("type") == "tool_use" and e.get("name") == "echo":
            child_echo_idx = i
        elif e.get("type") == "tool_result" and e.get("name") == "spawn_subagent":
            spawn_result_idx = i

    assert spawn_use_idx is not None, "spawn tool_use event missing"
    assert child_start_idx is not None, "child_start event missing"
    assert spawn_result_idx is not None, "spawn tool_result event missing"

    # Ordering: spawn_use < child_start < spawn_result
    assert spawn_use_idx < child_start_idx < spawn_result_idx, (
        f"Expected spawn_use({spawn_use_idx}) < child_start({child_start_idx}) < spawn_result({spawn_result_idx})"
    )

    # Child echo events should also be between spawn_use and spawn_result
    if child_echo_idx is not None:
        assert spawn_use_idx < child_echo_idx < spawn_result_idx
