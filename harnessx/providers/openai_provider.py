# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import logging
import time

from ..core.events import Message, ModelResponseEvent, ToolSchema, Usage
from ._utils import count_tokens, parse_tool_calls, to_openai_content, to_openai_tools
from .base import BaseModelProvider

logger = logging.getLogger(__name__)

_throttle_lock = None
_last_request_time = 0.0
_MIN_REQUEST_INTERVAL = 1.0


async def _throttle():
    global _throttle_lock, _last_request_time
    if _throttle_lock is None:
        _throttle_lock = asyncio.Lock()
    async with _throttle_lock:
        elapsed = time.monotonic() - _last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            await asyncio.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time = time.monotonic()


_ANTHROPIC_MODEL_PREFIXES = ("claude-", "anthropic/")


def _is_anthropic_model(model: str) -> bool:
    return any(model.lower().startswith(p) for p in _ANTHROPIC_MODEL_PREFIXES)


def _fix_tool_call_pairing(messages: list[dict]) -> list[dict]:
    """Ensure tool_call/tool_result pairing is valid for the OpenAI API.

    The OpenAI API has two strict requirements:
    1. After each assistant message with tool_calls, the IMMEDIATELY following
       messages must be tool-role messages for ALL tool_call_ids.
    2. Every tool-role message must reference a tool_call_id from a preceding
       assistant message.

    This function fixes violations that arise from compaction/token trimming.
    """
    # Pass 1: for each assistant+tool_calls, find which tool_call_ids have
    # immediate tool results. Build the set of valid (assistant_idx, tc_id) pairs.
    valid_tc_ids: set[str] = set()
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tc_ids = {tc["id"] for tc in msg["tool_calls"]}
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tid = messages[j].get("tool_call_id")
                if tid in tc_ids:
                    valid_tc_ids.add(tid)
                j += 1
            orphaned = tc_ids - valid_tc_ids
            if orphaned:
                logger.warning(
                    "_fix_tool_call_pairing pass1: msg[%d] has %d orphaned tool_calls: %s",
                    i,
                    len(orphaned),
                    orphaned,
                )
            i = j
        else:
            i += 1

    # Pass 2: rebuild, keeping only valid pairs and dropping orphans.
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            kept = [tc for tc in msg["tool_calls"] if tc["id"] in valid_tc_ids]
            if kept:
                fixed = dict(msg)
                fixed["tool_calls"] = kept
                result.append(fixed)
                kept_ids = {tc["id"] for tc in kept}
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    if messages[j].get("tool_call_id") in kept_ids:
                        result.append(messages[j])
                    j += 1
                i = j
            else:
                stripped = {k: v for k, v in msg.items() if k != "tool_calls"}
                if not stripped.get("content"):
                    stripped["content"] = ""
                result.append(stripped)
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    j += 1
                i = j
        elif msg.get("role") == "tool":
            if msg.get("tool_call_id") in valid_tc_ids:
                result.append(msg)
            else:
                logger.warning(
                    "_fix_tool_call_pairing pass2: dropping orphan tool result tc_id=%s",
                    msg.get("tool_call_id"),
                )
            i += 1
        else:
            result.append(msg)
            i += 1

    # Final verification: scan result for any remaining violations
    for idx, msg in enumerate(result):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        tc_ids = {tc["id"] for tc in msg["tool_calls"]}
        following_ids = set()
        for k in range(idx + 1, len(result)):
            if result[k].get("role") != "tool":
                break
            following_ids.add(result[k].get("tool_call_id"))
        missing = tc_ids - following_ids
        if missing:
            logger.error(
                "_fix_tool_call_pairing VERIFICATION FAILED at msg[%d]: "
                "tool_calls %s still missing results. Stripping them.",
                idx,
                missing,
            )
            msg["tool_calls"] = [tc for tc in msg["tool_calls"] if tc["id"] not in missing]
            if not msg["tool_calls"]:
                del msg["tool_calls"]
                if not msg.get("content"):
                    msg["content"] = ""

    return result


class OpenAIProvider(BaseModelProvider):
    """OpenAI SDK provider (Chat Completions API). Also supports compatible endpoints (Ollama, etc.).

    Anthropic models (claude-* / anthropic/*) are NOT supported here.
    Use AnthropicProvider instead — the OpenAI SDK cannot round-trip
    extended thinking blocks (with signatures) required by the Anthropic API
    for multi-turn correctness and prefix cache hits.

    Reasoning models (o-series, GPT-5.x Thinking) note:
      - Chat Completions API: reasoning tokens are fully internal; API does not
        return or accept them. Multi-turn history is correct as-is. The `thinking`
        field in ModelResponseEvent captures any exposed reasoning summary (best-effort).
      - Responses API: reasoning *items* should be passed back each turn for best
        quality. This provider uses Chat Completions only and therefore does not
        support Responses API reasoning round-trip. If you need that, implement a
        dedicated ResponsesAPIProvider.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ):
        if _is_anthropic_model(model):
            raise ValueError(
                f"OpenAIProvider does not support Anthropic model '{model}'. "
                "Use AnthropicProvider instead: it correctly handles extended "
                "thinking blocks (signatures) required for multi-turn conversations "
                "and prefix cache hits."
            )
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        # Accept common aliases used by different config paths.
        self.extra_headers = (
            extra_headers
            or kwargs.pop("extra_headers", None)
            or kwargs.pop("headers", None)
            or kwargs.pop("default_headers", None)
        )
        self.max_tokens = max_tokens
        self.stream = stream
        self.context_window = self._infer_context_window(model)
        self.kwargs = kwargs

    @staticmethod
    def _infer_context_window(model: str) -> int:
        m = model.lower()
        if m.startswith("gpt-5") or m.startswith("gpt-4o") or m.startswith("o"):
            return 128_000
        if m.startswith("gpt-4-turbo") or m.startswith("gpt-4-1"):
            return 128_000
        if m.startswith("gpt-4"):
            return 8_192
        if m.startswith("gpt-3.5"):
            return 16_384
        return 128_000

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        stream_callback=None,
    ) -> ModelResponseEvent:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        client_kwargs = {}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.extra_headers:
            client_kwargs["default_headers"] = self.extra_headers
        client_kwargs["max_retries"] = 0
        client = AsyncOpenAI(**client_kwargs)

        oai_messages = []
        for m in messages:
            content = to_openai_content(m.content)
            if m.role == "tool" and content is None:
                content = "(empty)"
            msg = {"role": m.role, "content": content}
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.name:
                msg["name"] = m.name
            if m.tool_calls:
                import json as _json

                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": _json.dumps(tc.input) if isinstance(tc.input, dict) else str(tc.input),
                        },
                    }
                    for tc in m.tool_calls
                ]
            oai_messages.append(msg)

        oai_messages = _fix_tool_call_pairing(oai_messages)

        oai_tools = to_openai_tools(tools)

        kwargs = dict(self.kwargs)
        if oai_tools:
            kwargs["tools"] = oai_tools

        global _last_request_time
        max_retries = 4
        base_delay = 3.0

        if self.stream:
            return await self._complete_streaming(client, oai_messages, kwargs, max_retries, base_delay)

        for attempt in range(1, max_retries + 1):
            try:
                await _throttle()
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=oai_messages,
                    max_tokens=self.max_tokens,
                    **kwargs,
                )
                break
            except Exception as e:
                status = getattr(e, "status_code", None) or getattr(e, "status", None)
                err_str = str(e).lower()
                is_retryable = status in (429, 500, 502, 503, 529) or "rate" in err_str
                # Retry on timeouts with longer backoff
                if "timeout" in err_str or "timed out" in err_str:
                    is_retryable = True
                    base_delay = max(base_delay, 30.0)
                # Content policy flags can be non-deterministic — retry once
                if status == 400 and ("flagged" in err_str or "content_policy" in err_str) and attempt == 1:
                    is_retryable = True
                if is_retryable and attempt < max_retries:
                    delay = min(base_delay * (2 ** (attempt - 1)) + (attempt * 0.5), 120.0)
                    logger.warning(
                        "API %s (attempt %d/%d), retrying in %.1fs", status or e, attempt, max_retries, delay
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        finish_reason = choice.finish_reason or "stop"

        # Capture reasoning content from o-series models.
        # OpenAI exposes this via message.reasoning (Responses API) or
        # message.reasoning_content (some SDK versions). We check both.
        thinking = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""

        tool_calls = []
        if msg.tool_calls:
            tool_calls = parse_tool_calls(msg.tool_calls)
            finish_reason = "tool_use"

        # Some models (e.g. Qwen3 thinking mode) return finish_reason="tool_calls"
        # but no actual tool_calls in the message.  Normalize to "stop" so
        # downstream processors (e.g. SelfVerifyProcessor) treat it as a done turn.
        if not tool_calls and finish_reason == "tool_calls":
            finish_reason = "stop"

        usage = Usage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
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

    async def _complete_streaming(self, client, oai_messages, kwargs, max_retries, base_delay):
        """Streaming path — required for providers that truncate non-streaming responses (e.g. Tongyi/QwQ-32B)."""
        import json as _json
        import re as _re

        for attempt in range(1, max_retries + 1):
            try:
                await _throttle()
                stream = await client.chat.completions.create(
                    model=self.model,
                    messages=oai_messages,
                    max_tokens=self.max_tokens,
                    stream=True,
                    **kwargs,
                )
                break
            except Exception as e:
                status = getattr(e, "status_code", None) or getattr(e, "status", None)
                err_str = str(e).lower()
                is_retryable = status in (429, 500, 502, 503, 529) or "rate" in err_str
                if status == 400 and ("flagged" in err_str or "content_policy" in err_str) and attempt == 1:
                    is_retryable = True
                if is_retryable and attempt < max_retries:
                    delay = min(base_delay * (2 ** (attempt - 1)) + (attempt * 0.5), 30.0)
                    logger.warning(
                        "API %s (attempt %d/%d), retrying in %.1fs", status or e, attempt, max_retries, delay
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        content_parts: list[str] = []
        finish_reason = "stop"
        input_tokens = 0
        output_tokens = 0
        # tool_calls accumulator: index -> {id, name, arguments_parts}
        tc_accum: dict[int, dict] = {}

        async for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            if delta.content:
                content_parts.append(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_accum:
                        tc_accum[idx] = {"id": "", "name": "", "arguments_parts": []}
                    if tc_delta.id:
                        tc_accum[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_accum[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_accum[idx]["arguments_parts"].append(tc_delta.function.arguments)

        full_content = "".join(content_parts)

        # Strip <think>...</think> blocks (QwQ-32B reasoning) → move to thinking field
        thinking = ""
        think_match = _re.search(r"<think>(.*?)</think>", full_content, _re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            full_content = _re.sub(r"<think>.*?</think>", "", full_content, flags=_re.DOTALL).strip()

        # Build tool calls from accumulated deltas
        from ..core.events import ToolCall

        tool_calls = []
        for idx in sorted(tc_accum):
            tc = tc_accum[idx]
            args_str = "".join(tc["arguments_parts"])
            try:
                args = _json.loads(args_str) if args_str else {}
            except _json.JSONDecodeError:
                args = {"raw": args_str}
            tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], input=args))

        if tool_calls:
            finish_reason = "tool_use"

        # Rough token estimate when API doesn't report usage
        if output_tokens == 0 and (full_content or thinking):
            output_tokens = max(1, len(full_content + thinking) // 4)

        return ModelResponseEvent(
            run_id="",
            step_id=0,
            content=full_content,
            thinking=thinking,
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            model=self.model,
        )

    def count_tokens(self, messages: list[Message]) -> int:
        return count_tokens(messages)
