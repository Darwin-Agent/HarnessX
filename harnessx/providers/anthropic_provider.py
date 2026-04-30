# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from ..core.events import Message, ModelResponseEvent, ToolCall, ToolSchema, Usage
from ._utils import count_tokens, emit_stream_delta
from .agentic import AgenticMixin
from .base import BaseModelProvider

_log = logging.getLogger(__name__)

# Retry config for 429 rate-limit errors.
# Token-rate limits typically reset on a per-minute window; waiting 15s then
# doubling gives the API time to replenish without burning the whole minute.
_RL_MAX_RETRIES = 5
_RL_BACKOFF = (15.0, 30.0, 60.0, 120.0, 240.0)  # seconds per attempt


class AnthropicProvider(AgenticMixin, BaseModelProvider):
    """Direct Anthropic SDK provider. Supports Extended Thinking."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int | None = None,
        extended_thinking: bool = False,
        thinking_budget_tokens: int = 10_000,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        request_timeout: float | None = None,  # alias for timeout (LiteLLM compat)
        default_headers: dict | None = None,
        **kwargs,
    ):
        # Strip LiteLLM routing prefix "anthropic/" if present — the Anthropic
        # SDK only wants the bare model name (e.g. "claude-sonnet-4-6").
        self.model = model.removeprefix("anthropic/")
        self.max_tokens = max_tokens
        self.extended_thinking = extended_thinking
        self.thinking_budget_tokens = thinking_budget_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout or request_timeout
        self._default_headers = default_headers
        self.kwargs = kwargs

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        stream_callback: "Callable[[object], None] | None" = None,
    ) -> ModelResponseEvent:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        client_kwargs: dict = {}
        if self._api_key is not None:
            client_kwargs["api_key"] = self._api_key
        if self._base_url is not None:
            client_kwargs["base_url"] = self._base_url
        if self._timeout is not None:
            client_kwargs["timeout"] = self._timeout
        if self._default_headers:
            client_kwargs["default_headers"] = self._default_headers
        client = anthropic.AsyncAnthropic(**client_kwargs)

        # Split system messages and convert tool results to Anthropic format
        system_text = ""
        conv_messages = []
        for m in messages:
            if m.role == "system":
                system_text += m.content + "\n"
            elif m.role == "tool":
                # Anthropic requires tool results as role=user with tool_result content blocks.
                # Group consecutive tool results into a single user message.
                tool_block = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": m.content,
                }
                if (
                    conv_messages
                    and conv_messages[-1]["role"] == "user"
                    and isinstance(conv_messages[-1]["content"], list)
                ):
                    conv_messages[-1]["content"].append(tool_block)
                else:
                    conv_messages.append({"role": "user", "content": [tool_block]})
            else:
                # For assistant messages with tool calls, build the full content array
                # including tool_use blocks (required by Anthropic API).
                if m.role == "assistant" and (m.tool_calls or m.thinking_blocks):
                    content_blocks: list = []
                    # Thinking blocks MUST come first — Anthropic API requirement.
                    # They must be replayed verbatim (with signature) from the original response
                    # so the API can verify integrity. Missing/modified blocks cause API errors.
                    for tb in m.thinking_blocks:
                        content_blocks.append(tb)
                    if m.content:
                        content_blocks.append({"type": "text", "text": m.content})
                    for tc in m.tool_calls:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.input,
                            }
                        )
                    conv_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    conv_messages.append({"role": m.role, "content": m.content})

        anthropic_tools = []
        for t in tools:
            anthropic_tools.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
            )

        kwargs = dict(self.kwargs)
        if system_text:
            kwargs["system"] = system_text.strip()

        # Extended thinking: adds a reasoning step before the response.
        # Requires temperature=1 (Anthropic API constraint).
        # max_tokens must be > thinking_budget_tokens.
        resolved_max = self.max_tokens if self.max_tokens is not None else 65_536
        if self.extended_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
            kwargs["temperature"] = 1  # API requires temp=1 for thinking; override any caller-supplied value
            # Ensure max_tokens can hold thinking + response tokens
            effective_max = max(resolved_max, self.thinking_budget_tokens + 1024)
        else:
            effective_max = resolved_max

        # Streaming is required when:
        # 1. Caller wants token deltas (stream_callback set), or
        # 2. Extended thinking is enabled (Anthropic API requirement), or
        # 3. max_tokens is large enough that the SDK would reject a non-streaming
        #    request with "Streaming is required for operations that may take
        #    longer than 10 minutes" (threshold ~32K for current models).
        _STREAMING_THRESHOLD = 32_000
        use_streaming = stream_callback is not None or self.extended_thinking or effective_max > _STREAMING_THRESHOLD

        _last_exc: Exception | None = None
        for attempt in range(_RL_MAX_RETRIES + 1):
            try:
                if use_streaming:
                    # Streaming: deliver token/thinking deltas via callback and
                    # collect full response at end.
                    async with client.messages.stream(
                        model=self.model,
                        max_tokens=effective_max,
                        messages=conv_messages,
                        tools=anthropic_tools if anthropic_tools else anthropic.NOT_GIVEN,
                        **kwargs,
                    ) as stream:
                        if stream_callback is not None:
                            async for event in stream:
                                etype = getattr(event, "type", "") or ""
                                if etype != "content_block_delta":
                                    continue
                                delta = getattr(event, "delta", None)
                                if delta is None:
                                    continue
                                dtype = getattr(delta, "type", "") or ""
                                if dtype == "text_delta":
                                    emit_stream_delta(
                                        stream_callback,
                                        getattr(delta, "text", "") or "",
                                        kind="token",
                                    )
                                elif dtype == "thinking_delta":
                                    emit_stream_delta(
                                        stream_callback,
                                        getattr(delta, "thinking", "") or "",
                                        kind="thinking",
                                    )
                        response = await stream.get_final_message()
                else:
                    response = await client.messages.create(
                        model=self.model,
                        max_tokens=effective_max,
                        messages=conv_messages,
                        tools=anthropic_tools if anthropic_tools else anthropic.NOT_GIVEN,
                        **kwargs,
                    )
                break  # success
            except anthropic.RateLimitError as exc:
                _last_exc = exc
                if attempt >= _RL_MAX_RETRIES:
                    raise
                delay = _RL_BACKOFF[min(attempt, len(_RL_BACKOFF) - 1)]
                _log.warning(
                    "AnthropicProvider: 429 rate limit on attempt %d/%d, retrying in %.0fs — %s",
                    attempt + 1,
                    _RL_MAX_RETRIES,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        content_text = ""
        thinking_text = ""
        thinking_blocks: list[dict] = []
        tool_calls = []
        for block in response.content:
            if block.type == "thinking":
                thinking_text += block.thinking
                # Preserve full block including signature — required for multi-turn replay.
                thinking_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": block.signature,
                    }
                )
            elif block.type == "redacted_thinking":
                # Redacted blocks have no accessible text but must still be replayed
                # verbatim in subsequent turns so the API knows they existed.
                thinking_blocks.append(
                    {
                        "type": "redacted_thinking",
                        "data": block.data,
                    }
                )
            elif block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        input=block.input if isinstance(block.input, dict) else {},
                    )
                )

        finish_reason = "end_turn"
        if response.stop_reason == "tool_use":
            finish_reason = "tool_use"
        elif response.stop_reason == "max_tokens":
            finish_reason = "max_tokens"

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return ModelResponseEvent(
            run_id="",
            step_id=0,
            content=content_text,
            thinking=thinking_text,
            thinking_blocks=tuple(thinking_blocks),
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            usage=usage,
            model=self.model,
        )

    def count_tokens(self, messages: list[Message]) -> int:
        return count_tokens(messages)
