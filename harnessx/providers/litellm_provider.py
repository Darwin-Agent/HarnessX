# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json as _json
import logging
from typing import Callable

from ..core.events import Message, ModelResponseEvent, ToolSchema, Usage
from ._utils import (
    as_text_delta,
    count_tokens,
    emit_stream_delta,
    parse_tool_calls,
    to_openai_content,
    to_openai_tools,
)
from .agentic import AgenticMixin
from .base import BaseModelProvider

_log = logging.getLogger(__name__)

_RL_MAX_RETRIES = 5
_RL_BACKOFF = (15.0, 30.0, 60.0, 120.0, 240.0)


_ANTHROPIC_MODEL_PREFIXES = ("claude-", "anthropic/")


def _is_anthropic_model(model: str) -> bool:
    return any(model.lower().startswith(p) for p in _ANTHROPIC_MODEL_PREFIXES)


def _register_model_if_unknown(model: str) -> None:
    """Register model with litellm if it isn't in the model cost DB.

    Prevents noisy DEBUG warnings when using custom or internal model names
    (e.g. "openai/preset-models") that litellm doesn't recognise. Uses a
    large-but-safe context window default; cost is set to zero since
    self-hosted endpoints don't have per-token billing tracked here.
    """
    try:
        import litellm

        # Strip provider prefix (e.g. "openai/preset-models" -> "preset-models")
        bare = model.split("/", 1)[-1] if "/" in model else model
        try:
            litellm.get_model_info(bare)
        except Exception:
            litellm.register_model(
                {
                    bare: {
                        "max_tokens": 32768,
                        "max_input_tokens": 32768,
                        "max_output_tokens": 8192,
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                        "litellm_provider": "openai",
                        "mode": "chat",
                    }
                }
            )
            _log.debug("Registered unknown model '%s' with litellm defaults.", bare)
    except Exception:
        pass  # never block the provider init over a cosmetic registration failure


def _delta_get(delta: object, key: str):
    if isinstance(delta, dict):
        return delta.get(key)
    return getattr(delta, key, None)


class LiteLLMProvider(AgenticMixin, BaseModelProvider):
    """Unified LLM provider via litellm. Supports any model string.

    Anthropic models (claude-* / anthropic/*) are NOT supported here.
    Use AnthropicProvider instead — LiteLLM cannot reliably round-trip
    extended thinking blocks (with signatures) in multi-turn conversations,
    which breaks both API correctness and prefix cache hits.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        request_timeout: int | None = None,
        timeout: float = 300.0,
        extra_headers: dict[str, str] | None = None,
        **kwargs,
    ):
        if _is_anthropic_model(model):
            raise ValueError(
                f"LiteLLMProvider does not support Anthropic model '{model}'. "
                "Use AnthropicProvider instead: it correctly handles extended "
                "thinking blocks (signatures) required for multi-turn conversations "
                "and prefix cache hits."
            )
        self.model = model
        self.timeout = timeout
        # Store declared parameters on ``self`` explicitly so they survive
        # YAML round-trip via ``_serialize_processor`` (which skips the
        # ``**kwargs`` catch-all — that slot is a runtime forwarding channel,
        # not a schema field). Previously both values were promoted into
        # ``self.kwargs`` and then read back at call time; that worked for
        # direct construction but silently dropped on reload, because the
        # serializer cannot round-trip catch-all contents symmetrically.
        self.request_timeout = request_timeout
        # Normalize the ``headers`` alias at construction time so the
        # serialized form consistently uses ``extra_headers``.
        if extra_headers is None and "headers" in kwargs:
            extra_headers = kwargs.pop("headers")
        self.extra_headers = extra_headers
        # ``self.kwargs`` now holds only truly unknown forwarding extras.
        self.kwargs = kwargs
        _register_model_if_unknown(model)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        stream_callback: "Callable[[object], None] | None" = None,
    ) -> ModelResponseEvent:
        try:
            import litellm
        except ImportError:
            raise ImportError("Install litellm: pip install litellm")

        # System messages are passed as {"role": "system", ...} in the messages list.
        # This is the standard OpenAI-compatible format; litellm converts them to
        # the provider-appropriate format internally (e.g. Anthropic "system=" param).
        # This differs from AnthropicProvider, which extracts system messages manually
        # because the direct Anthropic SDK requires them in the separate system= kwarg.
        litellm_messages = []
        for m in messages:
            msg: dict = {"role": m.role, "content": to_openai_content(m.content)}
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.name:
                msg["name"] = m.name
            if m.tool_calls:
                # OpenAI tool-use format: assistant message must carry tool_calls
                # so the model can correlate role=tool results on the next turn.
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": _json.dumps(tc.input),
                        },
                    }
                    for tc in m.tool_calls
                ]
                msg["content"] = to_openai_content(m.content)  # null is valid here
            litellm_messages.append(msg)

        litellm_tools = to_openai_tools(tools)

        kwargs = dict(self.kwargs)
        # Fold explicitly-stored declared params back into the call-kwargs.
        # setdefault preserves any override someone stuffed into
        # ``self.kwargs`` directly (runtime customization pathway).
        if self.request_timeout is not None:
            kwargs.setdefault("request_timeout", self.request_timeout)
        if self.extra_headers:
            kwargs.setdefault("extra_headers", self.extra_headers)
        if litellm_tools:
            kwargs["tools"] = litellm_tools
        if "max_tokens" not in kwargs and self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens

        for attempt in range(_RL_MAX_RETRIES + 1):
            try:
                if stream_callback is not None:
                    # Streaming: deliver text deltas via callback, reconstruct response from chunks.
                    stream_response = await litellm.acompletion(
                        model=self.model,
                        messages=litellm_messages,
                        stream=True,
                        timeout=self.timeout,
                        **kwargs,
                    )
                    chunks = []
                    async for chunk in stream_response:
                        chunks.append(chunk)
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if not delta:
                            continue
                        content_delta = as_text_delta(_delta_get(delta, "content"))
                        if content_delta:
                            emit_stream_delta(stream_callback, content_delta, kind="token")

                        # Reasoning/thinking deltas: providers expose different
                        # field names; normalize them to `thinking`.
                        reasoning_delta = (
                            as_text_delta(_delta_get(delta, "reasoning_content"))
                            or as_text_delta(_delta_get(delta, "reasoning"))
                            or as_text_delta(_delta_get(delta, "thinking"))
                        )
                        if reasoning_delta:
                            emit_stream_delta(stream_callback, reasoning_delta, kind="thinking")
                    response = litellm.stream_chunk_builder(chunks, messages=litellm_messages)
                else:
                    response = await litellm.acompletion(
                        model=self.model,
                        messages=litellm_messages,
                        stream=False,
                        timeout=self.timeout,
                        **kwargs,
                    )
                break  # success
            except litellm.exceptions.RateLimitError as exc:
                if attempt >= _RL_MAX_RETRIES:
                    raise
                delay = _RL_BACKOFF[min(attempt, len(_RL_BACKOFF) - 1)]
                _log.warning(
                    "LiteLLMProvider: 429 rate limit on attempt %d/%d, retrying in %.0fs — %s",
                    attempt + 1,
                    _RL_MAX_RETRIES,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        finish_reason = choice.finish_reason or "end_turn"

        # reasoning_content is a first-class field on LiteLLM's Message (1.x+).
        # Providers that expose separate thinking (Qwen3/QwQ, DeepSeek-R1, etc.)
        # return it here; we store it in thinking and leave content as-is.
        thinking = getattr(msg, "reasoning_content", None) or ""

        tool_calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = parse_tool_calls(msg.tool_calls)
            finish_reason = "tool_use"

        usage_obj = response.usage if hasattr(response, "usage") else None
        usage = Usage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0,
        )

        return ModelResponseEvent(
            run_id="",
            step_id=0,
            content=content,
            thinking=thinking,
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            usage=usage,
            model=self.model,
        )

    @property
    def context_window(self) -> int:
        """Return input context window size; defaults to 64 000 if unknown."""
        try:
            import litellm

            info = litellm.get_model_info(self.model)
            return info.get("max_input_tokens") or info.get("max_tokens") or 64_000
        except Exception:
            return 64_000

    @property
    def max_output_tokens(self) -> int | None:
        """Return maximum output tokens for a single completion; None if unknown (let litellm decide)."""
        try:
            import litellm

            info = litellm.get_model_info(self.model)
            return info.get("max_output_tokens") or info.get("max_tokens") or None
        except Exception:
            return None

    def count_tokens(self, messages: list[Message]) -> int:
        return count_tokens(messages)
