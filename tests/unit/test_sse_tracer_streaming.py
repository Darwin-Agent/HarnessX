# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json

from harnessx.api.sse_tracer import SSETracer
from harnessx.core.events import ModelResponseEvent, StepStartEvent


def _parse_sse(line: str) -> dict:
    assert line.startswith("data: ")
    return json.loads(line[len("data: ") :].strip())


class TestSseTracerStreaming:
    def test_sse_tracer_emits_full_token_when_no_stream_deltas(self) -> None:
        async def _case() -> None:
            q: asyncio.Queue[str] = asyncio.Queue()
            t = SSETracer(queue=q, api_run_id="api-run-1")

            await t.on_event(StepStartEvent(run_id="h-root-1", step_id=0))
            await t.on_event(ModelResponseEvent(run_id="h-root-1", step_id=0, content="hello"))

            _ = _parse_sse(await q.get())  # step_start
            tok = _parse_sse(await q.get())
            assert tok["type"] == "token"
            assert tok["run_id"] == "api-run-1"
            assert tok["content"] == "hello"

        asyncio.run(_case())

    def test_sse_tracer_avoids_duplicate_full_token_after_stream_deltas(self) -> None:
        async def _case() -> None:
            q: asyncio.Queue[str] = asyncio.Queue()
            t = SSETracer(queue=q, api_run_id="api-run-2")

            await t.on_event(StepStartEvent(run_id="h-root-2", step_id=0))
            t.emit_stream_delta("api-run-2", "he")
            t.emit_stream_delta("api-run-2", "llo")
            await t.on_event(ModelResponseEvent(run_id="h-root-2", step_id=0, content="hello"))

            msgs = [_parse_sse(await q.get()) for _ in range(3)]
            # 1) step_start 2) delta "he" 3) delta "llo"
            assert msgs[0]["type"] == "step_start"
            assert msgs[1]["type"] == "token" and msgs[1]["content"] == "he"
            assert msgs[2]["type"] == "token" and msgs[2]["content"] == "llo"
            assert q.empty()

        asyncio.run(_case())

    def test_sse_tracer_avoids_duplicate_full_thinking_after_stream_deltas(
        self,
    ) -> None:
        async def _case() -> None:
            q: asyncio.Queue[str] = asyncio.Queue()
            t = SSETracer(queue=q, api_run_id="api-run-3")

            await t.on_event(StepStartEvent(run_id="h-root-3", step_id=0))
            t.emit_stream_delta("api-run-3", "thin", kind="thinking")
            t.emit_stream_delta("api-run-3", "king", kind="thinking")
            await t.on_event(ModelResponseEvent(run_id="h-root-3", step_id=0, thinking="thinking"))

            msgs = [_parse_sse(await q.get()) for _ in range(3)]
            assert msgs[0]["type"] == "step_start"
            assert msgs[1]["type"] == "thinking" and msgs[1]["content"] == "thin"
            assert msgs[2]["type"] == "thinking" and msgs[2]["content"] == "king"
            assert q.empty()

        asyncio.run(_case())

    def test_sse_tracer_kind_dedupe_is_independent_between_token_and_thinking(
        self,
    ) -> None:
        async def _case() -> None:
            q: asyncio.Queue[str] = asyncio.Queue()
            t = SSETracer(queue=q, api_run_id="api-run-4")

            await t.on_event(StepStartEvent(run_id="h-root-4", step_id=0))
            t.emit_stream_delta("api-run-4", "hello ", kind="token")
            await t.on_event(
                ModelResponseEvent(
                    run_id="h-root-4",
                    step_id=0,
                    content="hello world",
                    thinking="chain",
                )
            )

            msgs = [_parse_sse(await q.get()) for _ in range(3)]
            assert msgs[0]["type"] == "step_start"
            assert msgs[1]["type"] == "token" and msgs[1]["content"] == "hello "
            # token full-content is suppressed, but thinking full-content still emitted.
            assert msgs[2]["type"] == "thinking" and msgs[2]["content"] == "chain"
            assert q.empty()

        asyncio.run(_case())
