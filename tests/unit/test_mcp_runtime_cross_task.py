# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Cross-task lifecycle tests for the MCP runtime supervisor + anyio patch."""

from __future__ import annotations

import asyncio
import warnings

import anyio
import pytest

import harnessx  # noqa: F401  — applies the anyio patch on import
from harnessx._anyio_patch import _PATCHED_FLAG
from harnessx.plugins.dimensions.mcp_runtime.plugin import (
    McpRuntimePlugin,
    _LifecycleSupervisor,
)


# ── anyio CancelScope patch ─────────────────────────────────────────────────


class TestAnyioPatch:
    def test_patch_marker_is_present(self) -> None:
        from anyio._backends._asyncio import CancelScope

        assert getattr(CancelScope.__exit__, _PATCHED_FLAG, False) is True

    def test_cross_task_exit_cleans_up_orphan_state(self) -> None:
        """When a CancelScope is exited from a different task, the patch must
        leave no _tasks / _cancel_handle / parent _child_scopes residue, even
        though the original RuntimeError still propagates."""

        async def run() -> tuple[anyio.CancelScope, anyio.CancelScope]:
            entered = asyncio.Event()
            release = asyncio.Event()
            captured: dict = {}

            async def enter_then_yield() -> None:
                outer = anyio.CancelScope()
                captured["outer"] = outer
                with outer:
                    inner = anyio.CancelScope()
                    captured["inner"] = inner
                    inner.__enter__()
                    captured["entered"] = True
                    entered.set()
                    await release.wait()

            entering_task = asyncio.create_task(enter_then_yield())
            await entered.wait()

            inner: anyio.CancelScope = captured["inner"]
            with pytest.raises(RuntimeError, match="different task"):
                inner.__exit__(None, None, None)

            release.set()
            entering_task.cancel()
            try:
                await entering_task
            except BaseException:
                pass
            return captured["outer"], inner

        outer, inner = asyncio.run(run())

        assert getattr(inner, "_active", True) is False
        assert getattr(inner, "_cancel_handle", "missing") is None
        host = getattr(inner, "_host_task", None)
        if host is not None:
            assert host not in getattr(inner, "_tasks", set())
        children = getattr(outer, "_child_scopes", set())
        assert inner not in children


# ── Lifecycle supervisor ────────────────────────────────────────────────────


class _RecordingClient:
    """Minimal MCPClient stand-in that records which task ran each call."""

    def __init__(self) -> None:
        self.connect_task: asyncio.Task | None = None
        self.disconnect_task: asyncio.Task | None = None

    async def connect(self) -> None:
        self.connect_task = asyncio.current_task()
        await asyncio.sleep(0)

    async def disconnect(self) -> None:
        self.disconnect_task = asyncio.current_task()
        await asyncio.sleep(0)


class TestLifecycleSupervisor:
    @pytest.mark.asyncio
    async def test_runs_all_submissions_on_one_task(self) -> None:
        sup = _LifecycleSupervisor()
        seen: list[asyncio.Task] = []

        async def record() -> None:
            seen.append(asyncio.current_task())  # type: ignore[arg-type]

        async def submit_from(label: str) -> None:
            await sup.submit(label, record)

        try:
            await asyncio.gather(
                asyncio.create_task(submit_from("a"), name="caller-a"),
                asyncio.create_task(submit_from("b"), name="caller-b"),
                asyncio.create_task(submit_from("c"), name="caller-c"),
            )
            assert len(seen) == 3
            assert all(t is seen[0] for t in seen)
            assert seen[0] is sup._task
            assert sup._task is not None
            assert sup._task.get_name() == "mcp-lifecycle-supervisor"
        finally:
            await sup.stop()

    @pytest.mark.asyncio
    async def test_caller_cancellation_does_not_abort_inflight_work(self) -> None:
        """If the caller is cancelled while awaiting a submitted job, the job
        must still complete on the supervisor — disconnect() leaks otherwise."""
        sup = _LifecycleSupervisor()
        completed = asyncio.Event()
        gate = asyncio.Event()

        async def slow_job() -> None:
            await gate.wait()
            completed.set()

        async def caller() -> None:
            await sup.submit("slow", slow_job)

        task = asyncio.create_task(caller())
        await asyncio.sleep(0)  # let supervisor start
        await asyncio.sleep(0)  # let job begin awaiting gate
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set()
        try:
            await asyncio.wait_for(completed.wait(), timeout=1.0)
        finally:
            await sup.stop()
        assert completed.is_set()

    @pytest.mark.asyncio
    async def test_stop_drains_supervisor(self) -> None:
        sup = _LifecycleSupervisor()
        await sup.submit("noop", lambda: asyncio.sleep(0))
        assert sup._task is not None
        await sup.stop()
        assert sup._task is None


# ── Plugin: connect/disconnect run on supervisor regardless of caller task ──


class TestPluginCrossTask:
    @pytest.mark.asyncio
    async def test_connect_and_disconnect_share_supervisor_task(self) -> None:
        plugin = McpRuntimePlugin(mcp_config={"servers": []}, ensure_primary=False)
        client = _RecordingClient()

        async def run_connect() -> None:
            await plugin._run_connect("rec", client)  # type: ignore[arg-type]

        async def run_disconnect() -> None:
            await plugin._run_disconnect("rec", client)  # type: ignore[arg-type]

        try:
            connect_caller = asyncio.create_task(run_connect(), name="caller-connect")
            await connect_caller
            disconnect_caller = asyncio.create_task(run_disconnect(), name="caller-disconnect")
            await disconnect_caller

            assert client.connect_task is not None
            assert client.disconnect_task is not None
            assert client.connect_task is client.disconnect_task
            assert client.connect_task is plugin._supervisor._task
            assert client.connect_task is not connect_caller
            assert client.disconnect_task is not disconnect_caller
        finally:
            await plugin._supervisor.stop()

    @pytest.mark.asyncio
    async def test_disconnect_swallows_failures_with_warning(self) -> None:
        plugin = McpRuntimePlugin(mcp_config={"servers": []}, ensure_primary=False)

        class _FailingClient:
            async def connect(self) -> None:
                pass

            async def disconnect(self) -> None:
                raise RuntimeError("boom")

        client = _FailingClient()

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                await plugin._run_disconnect("bad", client)  # type: ignore[arg-type]
            assert any("disconnect failed" in str(w.message) for w in caught)
        finally:
            await plugin._supervisor.stop()
