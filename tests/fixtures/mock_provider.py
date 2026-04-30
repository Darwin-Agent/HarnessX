# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from harnessx.core.events import (
    Message,
    ModelResponseEvent,
    ToolCall,
    ToolSchema,
    Usage,
)


class MockProvider:
    """
    Deterministic mock provider for testing.
    Returns scripted responses in sequence.
    """

    def __init__(self, responses: list[str | dict] | None = None):
        self._responses = list(responses or ["Test response."])
        self._call_count = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        stream_callback=None,
        **kwargs,
    ) -> ModelResponseEvent:
        idx = min(self._call_count, len(self._responses) - 1)
        response = self._responses[idx]
        self._call_count += 1

        if isinstance(response, str):
            return ModelResponseEvent(
                run_id="",
                step_id=0,
                content=response,
                tool_calls=(),
                finish_reason="end_turn",
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        elif isinstance(response, dict):
            tool_calls = []
            for tc in response.get("tool_calls", []):
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", "call_1"),
                        name=tc["name"],
                        input=tc.get("input", {}),
                    )
                )
            return ModelResponseEvent(
                run_id="",
                step_id=0,
                content=response.get("content", ""),
                tool_calls=tuple(tool_calls),
                finish_reason="tool_use" if tool_calls else "end_turn",
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        raise ValueError(f"Unknown response type: {type(response)}")

    def count_tokens(self, messages: list[Message]) -> int:
        return sum(len(m.content) // 4 for m in messages)

    def annotate_trajectory(self, trajectory: object) -> None:  # noqa: D401
        pass
