# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from harnessx import BaseTask
from harnessx.core.builder import HarnessBuilder
from harnessx.core.events import ModelResponseEvent, Usage
from harnessx.core.model_config import ModelConfig
from harnessx.tools.inmemory import InMemoryToolRegistry
from harnessx.tracing.null_tracer import NullTracer


class _EmptyThenAnswerProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.context_window = 32000

    async def complete(self, messages, tools, stream_callback=None):
        self.calls += 1
        content = "" if self.calls == 1 else "Non-empty answer"
        return ModelResponseEvent(
            run_id="provider",
            step_id=0,
            content=content,
            finish_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1),
            tool_calls=(),
        )

    def count_tokens(self, messages):
        return 0

    def annotate_trajectory(self, trajectory) -> None:
        return None


@pytest.mark.asyncio
async def test_first_step_empty_end_turn_retries_once_and_sets_diagnostics():
    cfg = HarnessBuilder().slot(tool_registry=InMemoryToolRegistry(), tracer=NullTracer()).build()
    harness = ModelConfig(main=_EmptyThenAnswerProvider()).agentic(cfg)

    result = await harness.run(BaseTask(description="smoke", max_steps=2))

    assert result.total_steps == 2
    assert (result.final_output or "").strip() == "Non-empty answer"
    slots = (result.task_end.state_snapshot or {}).get("slots") or {}
    assert slots.get("__model_empty_end_turn_seen", {}).get("content") is True
    assert slots.get("__empty_end_turn_recovered", {}).get("content") is True
