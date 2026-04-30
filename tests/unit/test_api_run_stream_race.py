# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio

import pytest

from fastapi import HTTPException
from harnessx.api.models import RunRequest
from harnessx.api.routes import run as run_route


async def _run_case(monkeypatch) -> None:
    run_id = "run-race-test"
    session_id = "session-race-test"
    queue: asyncio.Queue[str] = asyncio.Queue()

    req = RunRequest(task="hello", provider_config={})

    def _boom(_req, _session_id):
        raise RuntimeError("boom")

    class _FakeLoop:
        async def run_in_executor(self, _executor, fn, *args):
            return fn(*args)

    monkeypatch.setattr(run_route, "_build_config", _boom)
    monkeypatch.setattr(run_route.asyncio, "get_event_loop", lambda: _FakeLoop())
    run_route._runs.clear()
    run_route._runs[run_id] = queue

    await run_route._execute_run(run_id, session_id, req, queue)

    # Mapping must still exist so GET /run/{id}/stream can attach.
    assert run_id in run_route._runs

    agen = run_route._sse_generator(run_id, queue)
    first = await anext(agen)
    assert '"type": "error"' in first

    # Terminal event should end stream and clean up mapping.
    with pytest.raises(StopAsyncIteration):
        await anext(agen)
    assert run_id not in run_route._runs


async def _cancel_run_case() -> None:
    run_id = "run-cancel-test"
    run_route._run_tasks.clear()

    sleeper = asyncio.create_task(asyncio.sleep(10))
    run_route._run_tasks[run_id] = sleeper

    resp = await run_route.cancel_run(run_id)
    assert resp["ok"] is True
    assert resp["run_id"] == run_id

    with pytest.raises(asyncio.CancelledError):
        await sleeper

    run_route._run_tasks.clear()


async def _cancel_run_not_found_case() -> None:
    run_route._run_tasks.clear()
    with pytest.raises(HTTPException) as exc:
        await run_route.cancel_run("missing-run")
    assert exc.value.status_code == 404


class TestApiRunStreamRace:
    def test_execute_run_keeps_mapping_until_stream_consumes_terminal_event(self, monkeypatch):
        """Prevent race: stream endpoint must remain connectable after fast failure."""
        asyncio.run(_run_case(monkeypatch))

    def test_cancel_run_endpoint_cancels_background_task(self):
        asyncio.run(_cancel_run_case())

    def test_cancel_run_endpoint_missing_run_returns_404(self):
        asyncio.run(_cancel_run_not_found_case())
