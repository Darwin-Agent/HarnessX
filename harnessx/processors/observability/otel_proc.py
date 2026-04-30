# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ...logging import logger

from ...core.events import StepEndEvent, TaskEndEvent
from ...core.processor import MultiHookProcessor


class OTelProcessor(MultiHookProcessor):
    """
    Hooks: step_end + task_end
    Creates OTel Spans per step, recording token/cost/tool metrics.
    """

    _singleton_group = "otel"
    _order = 30

    def __init__(self, endpoint: str | None = None, service_name: str = "harnessx"):
        self.endpoint = endpoint
        self.service_name = service_name
        self._tracer = None
        self._root_span = None
        self._init_tracer()

    def _init_tracer(self) -> None:
        # Without an explicit endpoint, stay silent (no ConsoleSpanExporter).
        # Configure endpoint= to enable real OTel export.
        if not self.endpoint:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider = TracerProvider()
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=self.endpoint)))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(self.service_name)
        except ImportError:
            pass

    def _emit_span(self, name: str, attributes: dict) -> None:
        if self._tracer is None:
            return
        try:
            with self._tracer.start_as_current_span(name) as span:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
        except Exception as e:
            logger.debug(f"OTel span error: {e}")

    async def on_step_end(self, event: StepEndEvent):
        self._emit_span(
            f"step_{event.step_id}",
            {
                "step_id": event.step_id,
                "cumulative_tokens": event.cumulative_tokens,
                "cumulative_cost_usd": event.cumulative_cost_usd,
                "summary": event.step_summary[:100],
            },
        )
        yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._emit_span(
            "task_end",
            {
                "exit_reason": event.exit_reason,
                "total_steps": event.total_steps,
                "total_tokens": event.total_tokens,
                "total_cost_usd": event.total_cost_usd,
            },
        )
        yield event
