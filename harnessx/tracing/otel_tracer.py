# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

from ..core.events import Event, StepEndEvent, TaskEndEvent


class OTelTracer:
    """
    OpenTelemetry tracer. Creates spans per step.
    Requires: pip install harnessx
    """

    def __init__(
        self,
        endpoint: str | None = None,
        langfuse_endpoint: str | None = None,
        export_jsonl: bool = False,
        service_name: str = "harnessx",
    ):
        self.endpoint = endpoint or langfuse_endpoint
        self.export_jsonl = export_jsonl
        self.service_name = service_name
        self._tracer = None
        self._root_span = None
        self._events: list[Event] = []
        self._init_otel()

    def _init_otel(self) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )

            provider = TracerProvider()
            if self.endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    exporter = OTLPSpanExporter(endpoint=self.endpoint)
                except ImportError:
                    exporter = ConsoleSpanExporter()
            else:
                exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(self.service_name)
        except ImportError:
            pass

    async def on_event(self, event: Event) -> None:
        self._events.append(event)
        if self._tracer is None:
            return
        try:
            if isinstance(event, StepEndEvent):
                with self._tracer.start_as_current_span(f"step_{event.step_id}") as span:
                    span.set_attribute("step_id", event.step_id)
                    span.set_attribute("tokens", event.cumulative_tokens)
                    span.set_attribute("cost_usd", event.cumulative_cost_usd)
            elif isinstance(event, TaskEndEvent):
                with self._tracer.start_as_current_span("task_end") as span:
                    span.set_attribute("exit_reason", event.exit_reason)
                    span.set_attribute("total_tokens", event.total_tokens)
                    span.set_attribute("total_cost_usd", event.total_cost_usd)
        except Exception:
            pass

    async def on_raw_event(self, event: Event) -> None:
        pass  # OTel tracing only needs post-processor events

    async def flush(self) -> None:
        pass

    async def export_session_jsonl(self, run_id: str, path: str) -> None:
        if not self.export_jsonl:
            return
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        def _clean(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(i) for i in obj]
            return obj

        with open(path, "w", encoding="utf-8") as f:
            for event in self._events:
                d = dataclasses.asdict(event)
                record = {
                    "run_id": run_id,
                    "step": event.step_id,
                    "event_type": event.type,
                    "messages": [],
                    "reward": 0.0,
                    **_clean(d),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
