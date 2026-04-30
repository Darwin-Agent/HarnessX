# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from harnessx.core.events import (
    Event,
    ModelResponseEvent,
    ToolResultEvent,
    StepStartEvent,
    StepEndEvent,
    SegmentBoundaryEvent,
    SpawnSubAgentEvent,
    ProcessorTriggerEvent,
    TaskStartEvent,
    BeforeModelEvent,
)

if TYPE_CHECKING:
    from harnessx.tracing.base import BaseTracer


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class SSETracer:
    """Intercepts harness events and pushes SSE-serialised payloads to a queue.

    Wraps an optional inner tracer (e.g. HarnessJournal) so normal tracing
    is preserved alongside the live SSE stream.

    All SSE payloads include ``run_id`` so the frontend can route events to
    the correct agent (root or a spawned child).  Per-run cost deltas are
    tracked independently per run_id to avoid mixing parent/child cumulative
    costs.

    ``api_run_id`` is the UUID returned to the frontend by ``POST /api/run``.
    The harness generates its own internal run_id independently, so we map the
    first-seen harness run_id → api_run_id so the frontend can correlate events.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        inner: "BaseTracer | None" = None,
        api_run_id: str | None = None,
    ) -> None:
        self._queue = queue
        self._inner = inner
        self._api_run_id = api_run_id
        self._harness_root_run_id: str | None = None  # set on first event
        self._last_cost: dict[str, float] = {}  # run_id → last seen cumulative cost
        self._last_model: dict[str, str] = {}  # run_id → model name from last ModelResponseEvent
        self._run_start_ts: float | None = None  # wall-clock ts of first event (from Event.ts)
        # run_id -> streamed chunk kinds ("token"/"thinking") observed via callback
        # in the current model call; used to suppress duplicate full-content events.
        self._has_stream_deltas: dict[str, set[str]] = {}

    def emit_stream_delta(self, run_id: str, content: str, kind: str = "token") -> None:
        """Push one stream delta frame into SSE queue.

        Called by API route stream_callback for providers that support token streaming.
        """
        if not content or kind not in {"token", "thinking"}:
            return
        self._has_stream_deltas.setdefault(run_id, set()).add(kind)
        self._queue.put_nowait(
            _sse(
                {
                    "type": kind,
                    "run_id": run_id,
                    "content": content,
                }
            )
        )

    def _resolve_run_id(self, event_run_id: str) -> str:
        """Map harness-internal root run_id to the API-visible run_id.

        The harness generates its own UUID for the root agent; the frontend only
        knows the UUID we returned from POST /api/run.  We detect the root run_id
        on the first event and substitute the api_run_id so the frontend can
        route root events correctly.  Child run_ids are left untouched.
        """
        if self._harness_root_run_id is None:
            self._harness_root_run_id = event_run_id
        if self._api_run_id and event_run_id == self._harness_root_run_id:
            return self._api_run_id
        return event_run_id

    async def on_event(self, event: Event) -> None:
        if self._inner is not None:
            await self._inner.on_event(event)

        # Initialise run start timestamp from the first event's built-in ts field.
        if self._run_start_ts is None:
            self._run_start_ts = event.ts

        rid = self._resolve_run_id(event.run_id)

        if isinstance(event, TaskStartEvent):
            await self._queue.put(
                _sse(
                    {
                        "type": "task_context",
                        "run_id": rid,
                        "system": event.system_prompt,
                        "tool_names": [t.name for t in event.tools],
                    }
                )
            )

        elif isinstance(event, StepStartEvent):
            ts_ms = (event.ts - self._run_start_ts) * 1000
            await self._queue.put(
                _sse(
                    {
                        "type": "step_start",
                        "run_id": rid,
                        "step": event.step_id,
                        "ts_ms": round(ts_ms, 1),
                    }
                )
            )

        elif isinstance(event, ModelResponseEvent):
            if event.model:
                self._last_model[rid] = event.model
            streamed_kinds = self._has_stream_deltas.get(rid, set())
            had_thinking_deltas = "thinking" in streamed_kinds
            had_token_deltas = "token" in streamed_kinds

            if event.thinking and not had_thinking_deltas:
                await self._queue.put(
                    _sse(
                        {
                            "type": "thinking",
                            "run_id": rid,
                            "content": event.thinking,
                        }
                    )
                )
            if event.content and not had_token_deltas:
                await self._queue.put(
                    _sse(
                        {
                            "type": "token",
                            "run_id": rid,
                            "content": event.content,
                        }
                    )
                )
            if rid in self._has_stream_deltas:
                self._has_stream_deltas.pop(rid, None)
            for tc in event.tool_calls:
                await self._queue.put(
                    _sse(
                        {
                            "type": "tool_use",
                            "run_id": rid,
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.input,
                        }
                    )
                )

        elif isinstance(event, ToolResultEvent):
            await self._queue.put(
                _sse(
                    {
                        "type": "tool_result",
                        "run_id": rid,
                        "id": event.tool_call_id,
                        "name": event.tool_name,
                        "output": event.result,
                        "error": event.error,
                        "duration_ms": event.duration_ms,
                    }
                )
            )

        elif isinstance(event, SegmentBoundaryEvent):
            if event.reason == "compaction":
                await self._queue.put(
                    _sse(
                        {
                            "type": "compact",
                            "run_id": rid,
                            "before_msgs": event.before_msgs,
                            "after_msgs": event.after_msgs,
                            "before_tokens": event.before_tokens,
                            "after_tokens": event.after_tokens,
                        }
                    )
                )

        elif isinstance(event, StepEndEvent):
            delta = event.cumulative_cost_usd - self._last_cost.get(rid, 0.0)
            self._last_cost[rid] = event.cumulative_cost_usd
            await self._queue.put(
                _sse(
                    {
                        "type": "step_end",
                        "run_id": rid,
                        "step": event.step_id + 1,
                        "cost_usd": round(delta, 8),
                        "duration_ms": round(event.duration_ms, 1),
                        "input_tokens": event.input_tokens,
                        "output_tokens": event.output_tokens,
                        "model": self._last_model.get(rid, ""),
                    }
                )
            )

        elif isinstance(event, BeforeModelEvent):
            non_system = [m for m in event.messages if m.role != "system"]
            await self._queue.put(
                _sse(
                    {
                        "type": "step_context",
                        "run_id": rid,
                        "step": event.step_id,
                        "tool_names": [t.name for t in event.tools],
                        "message_count": len(non_system),
                    }
                )
            )

        elif isinstance(event, ProcessorTriggerEvent):
            await self._queue.put(
                _sse(
                    {
                        "type": "processor_trigger",
                        "run_id": rid,
                        "step": event.step_id,
                        "processor": event.processor,
                        "hook": event.hook,
                        "action": event.action,
                        "detail": event.detail,
                    }
                )
            )

        elif isinstance(event, SpawnSubAgentEvent):
            task_desc = ""
            sub_task = event.sub_task
            if sub_task is not None:
                task_desc = getattr(sub_task, "description", str(sub_task))
            await self._queue.put(
                _sse(
                    {
                        "type": "child_start",
                        "parent_run_id": rid,
                        "child_run_id": event.child_run_id,
                        "task": task_desc[:200],
                    }
                )
            )

    async def on_raw_event(self, event: Event) -> None:
        """Forward pre-processor raw event to inner tracer (e.g. HarnessJournal)."""
        if self._inner is not None and hasattr(self._inner, "on_raw_event"):
            await self._inner.on_raw_event(event)

    async def flush(self) -> None:
        if self._inner is not None:
            await self._inner.flush()
