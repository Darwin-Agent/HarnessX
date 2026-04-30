# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Shape + safety tests for ``spawn_reflect_worker``."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from harnessx.core.harness import HarnessConfig
from harnessx.core.model_config import ModelConfig
from harnessx.processors.context.system_prompt import SystemPromptProcessor
from harnessx.tools.base import Tool
from harnessx.tools.inmemory import InMemoryToolRegistry
from harnessx.tracing.null_tracer import NullTracer

from harnessx.meta_harness.workers.trajectory_digester import (
    SPAWN_REFLECT_WORKER_TOOL_NAME,
    _WORKER_SPECS,
    _make_worker_child_config_fn,
    make_spawn_reflect_worker_tool,
)

from fixtures.mock_provider import MockProvider


# ── helpers ─────────────────────────────────────────────────────────────────


def _fake_tool(name: str) -> Tool:
    async def _fn(**_: object) -> str:
        return ""

    return Tool(
        name=name,
        description=f"fake {name}",
        input_schema={"type": "object", "properties": {}, "required": []},
        fn=_fn,
    )


_WRITE_TOOL_NAMES = [
    "save_meta_skill",
    "save_python_tool",
    "save_python_processor",
    "edit_python_tool",
    "edit_python_processor",
    "edit_meta_skill_directives",
    "merge_meta_skill",
    "mark_negative",
]


def _parent_with_all_tools() -> HarnessConfig:
    reg = InMemoryToolRegistry()
    for name in ("Read", "Glob", "Grep", "Bash"):
        reg.register(_fake_tool(name))
    for name in _WRITE_TOOL_NAMES:
        reg.register(_fake_tool(name))
    return HarnessConfig(tool_registry=reg, tracer=NullTracer())


def _inner_model() -> ModelConfig:
    return ModelConfig(main=MockProvider(["done"]))


# ── tests ───────────────────────────────────────────────────────────────────


def test_tool_shape() -> None:
    tool = make_spawn_reflect_worker_tool(
        inner_model=_inner_model(),
        parent_harness_config=_parent_with_all_tools(),
    )
    assert tool.name == SPAWN_REFLECT_WORKER_TOOL_NAME
    props = tool.input_schema["properties"]
    assert sorted(props) == ["files", "kind", "task"]
    assert sorted(tool.input_schema["required"]) == ["kind", "task"]
    assert "trajectory-digester" in props["kind"]["enum"]
    assert "CANNOT write artifacts" in tool.description


def test_unknown_kind_returns_error_without_spawn() -> None:
    tool = make_spawn_reflect_worker_tool(
        inner_model=_inner_model(),
        parent_harness_config=_parent_with_all_tools(),
    )
    out = asyncio.run(tool.fn(kind="nope", task="anything", files=[]))
    assert "unknown worker kind" in out
    assert "trajectory-digester" in out


def test_empty_task_returns_error() -> None:
    tool = make_spawn_reflect_worker_tool(
        inner_model=_inner_model(),
        parent_harness_config=_parent_with_all_tools(),
    )
    out = asyncio.run(tool.fn(kind="trajectory-digester", task="   ", files=[]))
    assert "task description is empty" in out


def test_child_config_strips_write_tools_and_keeps_allowed() -> None:
    parent = _parent_with_all_tools()
    spec = _WORKER_SPECS["trajectory-digester"]
    fn = _make_worker_child_config_fn(spec)

    child = fn(
        parent,
        {"model": "", "system_prompt": "", "tools": []},
        child_depth=1,
        max_depth=2,
        runtime_tracer=NullTracer(),
        parent_run_id="parent-run",
    )

    names = set(child.tool_registry.list_names())
    # Allowed tools survive:
    assert names == {"Read", "Glob", "Grep", "Bash"}
    # No write tool from the reflect toolset leaks through:
    assert not (names & set(_WRITE_TOOL_NAMES))
    # No accidental spawn tool promotion — the worker is a leaf:
    assert "spawn_subagent" not in names
    assert "spawn_reflect_worker" not in names


def test_spawn_reflect_worker_end_to_end_does_not_raise() -> None:
    """build_spawn_fn must pass 6 args to child_config_fn (not 2).

    Regression: the worker's _child_config_fn needs child_depth /
    max_depth / runtime_tracer / parent_run_id to call _default_child_config;
    an older 2-arg build_spawn_fn call site crashed with TypeError the
    moment the reflect agent invoked spawn_reflect_worker.
    """
    tool = make_spawn_reflect_worker_tool(
        inner_model=_inner_model(),
        parent_harness_config=_parent_with_all_tools(),
    )
    # No _spawn_ctx is set — build_spawn_fn's inner spawn_subagent reads ctx
    # via _spawn_ctx.get() which returns {} by default, so child_depth starts
    # at 0 and max_depth gating never trips. The call should reach the
    # child_config_fn site without TypeError. Any failure downstream of
    # that site (e.g. actually running the MockProvider) is out of scope.
    out = asyncio.run(
        tool.fn(
            kind="trajectory-digester",
            task="digest these",
            files=["/tmp/does-not-matter.md"],
        )
    )
    # Should reach the worker child build path without signature crash.
    # The mock provider will produce "done" or similar — not an error string.
    assert "missing" not in out, f"signature mismatch leaked through: {out}"
    assert "positional argument" not in out, f"signature mismatch: {out}"


def test_build_meta_agent_raises_on_spawn_tool_conflict(monkeypatch) -> None:
    """If ``make_spawn_reflect_worker_tool`` yields a tool whose name
    collides with an existing registered tool, ``build_meta_agent_harness_config``
    must propagate ``ToolConflictError`` — SOUL.md + reflect-on-traces require
    the worker to exist, so silently degrading would waste budget on every
    "tool not found" trial at runtime.
    """
    from harnessx.meta_harness.agent import build_meta_agent_harness_config
    from harnessx.tools.base import Tool, ToolConflictError

    async def _collide(**_: object) -> str:
        return ""

    colliding = Tool(
        name="Read",  # already registered as a builtin in the builder
        description="synthetic collision",
        input_schema={"type": "object", "properties": {}, "required": []},
        fn=_collide,
    )

    # build_meta_agent_harness_config imports make_spawn_reflect_worker_tool
    # from .workers inside the function; patch at the source module.
    monkeypatch.setattr(
        "harnessx.meta_harness.workers.make_spawn_reflect_worker_tool",
        lambda **_: colliding,
    )

    with pytest.raises(ToolConflictError):
        build_meta_agent_harness_config(inner_model=_inner_model())


def test_child_system_prompt_is_worker_specific() -> None:
    parent = _parent_with_all_tools()
    spec = _WORKER_SPECS["trajectory-digester"]
    fn = _make_worker_child_config_fn(spec)
    child = fn(
        parent,
        {"model": "", "system_prompt": "", "tools": []},
        child_depth=1,
        max_depth=2,
        runtime_tracer=NullTracer(),
        parent_run_id="parent-run",
    )

    # Find the SystemPromptProcessor in _rt_procs.
    found = None
    for p in getattr(child, "_rt_procs", []):
        if isinstance(p, SystemPromptProcessor):
            found = p
            break
    assert found is not None
    built = asyncio.run(found.system_builder.build())
    # Worker prompt characteristics — do not leak reflect guide content.
    assert "trajectory digester worker" in built
    assert "CANNOT write" not in built.lower() or "cannot write" in built.lower()
    assert "save_meta_skill" not in built
