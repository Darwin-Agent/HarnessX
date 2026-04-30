# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import uuid

from ..core.events import Message, ModelResponseEvent, ToolCall, ToolSchema, Usage
from ._utils import count_tokens
from .agentic import AgenticMixin
from .base import BaseModelProvider

_ANTHROPIC_MODEL_PREFIXES = ("claude-", "anthropic/")


def _is_anthropic_model(model: str) -> bool:
    return any(model.lower().startswith(p) for p in _ANTHROPIC_MODEL_PREFIXES)


class ResponsesAPIProvider(AgenticMixin, BaseModelProvider):
    """OpenAI Responses API provider.

    Differences from OpenAIProvider (Chat Completions):
    - Uses ``client.responses.create()`` instead of ``client.chat.completions.create()``
    - Reasoning items (type="reasoning") are returned in ``response.output`` and must
      be passed back verbatim on the next turn for multi-turn correctness.
      We store them in ``Message.thinking_blocks`` so the runloop can replay them.
    - Reasoning summary text (when available) is captured into ``ModelResponseEvent.thinking``.
    - Tool calls use the Responses API format (``item.call_id``, not ``item.id``).
    - System prompt is passed via the ``instructions`` parameter, not as a message.

    Recommended for: GPT-5.x Thinking, o1, o3, o4-mini and any future OpenAI
    reasoning models where reasoning item round-trip matters.

    Anthropic models are not supported — use AnthropicProvider instead.
    """

    def __init__(
        self,
        model: str = "o4-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,  # "low" | "medium" | "high"
        reasoning_summary: bool = False,  # request reasoning summary text
        max_output_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs,
    ):
        if _is_anthropic_model(model):
            raise ValueError(
                f"ResponsesAPIProvider does not support Anthropic model '{model}'. Use AnthropicProvider instead."
            )
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.max_output_tokens = max_output_tokens
        self._timeout = timeout
        self.kwargs = kwargs

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

        client_kwargs: dict = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        if self._timeout is not None:
            client_kwargs["timeout"] = self._timeout
        client = AsyncOpenAI(**client_kwargs)

        # ── Build input ────────────────────────────────────────────────────────
        # System messages become the ``instructions`` parameter.
        # Reasoning items (stored in Message.thinking_blocks) must be replayed
        # verbatim — they are inserted before the assistant text/tool blocks.
        # Tool results use the Responses API ``function_call_output`` type.
        instructions: str | None = None
        input_items: list = []

        for m in messages:
            if m.role == "system":
                instructions = (instructions or "") + m.content + "\n"
                continue

            if m.role == "tool":
                # Responses API tool result format
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": m.tool_call_id or "",
                        "output": m.content if isinstance(m.content, str) else json.dumps(m.content),
                    }
                )
                continue

            if m.role == "assistant":
                # Replay reasoning items first (required for multi-turn correctness)
                for tb in m.thinking_blocks:
                    input_items.append(tb)

                if m.tool_calls:
                    for tc in m.tool_calls:
                        input_items.append(
                            {
                                "type": "function_call",
                                "id": tc.id,
                                "call_id": tc.id,
                                "name": tc.name,
                                "arguments": json.dumps(tc.input),
                            }
                        )
                    if m.content:
                        input_items.append(
                            {
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": m.content}],
                            }
                        )
                else:
                    input_items.append(
                        {
                            "role": "assistant",
                            "content": m.content or "",
                        }
                    )
                continue

            # user messages
            input_items.append(
                {
                    "role": m.role,
                    "content": _to_responses_content(m.content),
                }
            )

        # ── Build tools ────────────────────────────────────────────────────────
        responses_tools = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            }
            for t in tools
        ]

        # ── Build kwargs ───────────────────────────────────────────────────────
        kwargs: dict = dict(self.kwargs)
        if instructions:
            kwargs["instructions"] = instructions.strip()
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if responses_tools:
            kwargs["tools"] = responses_tools

        # Reasoning configuration
        if self.reasoning_effort or self.reasoning_summary:
            reasoning_cfg: dict = {}
            if self.reasoning_effort:
                reasoning_cfg["effort"] = self.reasoning_effort
            if self.reasoning_summary:
                reasoning_cfg["generate_summary"] = "auto"
            kwargs["reasoning"] = reasoning_cfg

        # ── Call API ───────────────────────────────────────────────────────────
        response = await client.responses.create(
            model=self.model,
            input=input_items if input_items else "",
            **kwargs,
        )

        # ── Parse output ───────────────────────────────────────────────────────
        content_text = ""
        thinking_text = ""
        thinking_blocks: list[dict] = []
        tool_calls: list[ToolCall] = []
        finish_reason = "end_turn"

        for item in response.output:
            itype = getattr(item, "type", None)

            if itype == "reasoning":
                # Capture reasoning summary (text) for trajectory display
                summary = getattr(item, "summary", None) or []
                for s in summary:
                    thinking_text += getattr(s, "text", "")

                # Store the full reasoning item for next-turn replay.
                # We preserve the raw dict so it can be passed back verbatim.
                thinking_blocks.append(_reasoning_item_to_dict(item))

            elif itype == "message":
                for block in getattr(item, "content", None) or []:
                    btype = getattr(block, "type", None)
                    if btype == "output_text":
                        content_text += getattr(block, "text", "")

            elif itype == "function_call":
                try:
                    inp = json.loads(item.arguments) if item.arguments else {}
                except json.JSONDecodeError:
                    inp = {}
                # call_id is the correlation key for function_call_output
                tc_id = getattr(item, "call_id", None) or getattr(item, "id", None) or str(uuid.uuid4())
                tool_calls.append(ToolCall(id=tc_id, name=item.name, input=inp))
                finish_reason = "tool_use"

        if not tool_calls and getattr(response, "status", None) == "completed":
            finish_reason = "end_turn"

        # ── Usage ──────────────────────────────────────────────────────────────
        usage_obj = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(usage_obj, "input_tokens", 0) if usage_obj else 0,
            output_tokens=getattr(usage_obj, "output_tokens", 0) if usage_obj else 0,
            cache_read_tokens=(
                getattr(getattr(usage_obj, "input_tokens_details", None), "cached_tokens", 0) if usage_obj else 0
            ),
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


# ── Helpers ────────────────────────────────────────────────────────────────────


def _to_responses_content(content: "str | list") -> "str | list":
    """Convert internal content to Responses API input_text/input_image format."""
    if isinstance(content, str):
        return content
    result = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            result.append({"type": "input_text", "text": block.get("text", "")})
        elif btype == "image":
            src = block.get("source", {})
            if src.get("type") == "base64":
                result.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{src['media_type']};base64,{src['data']}",
                    }
                )
            elif src.get("type") == "url":
                result.append({"type": "input_image", "image_url": src["url"]})
    return result or content


def _reasoning_item_to_dict(item) -> dict:
    """Serialize a Responses API reasoning output item to a plain dict for replay."""
    d: dict = {"type": "reasoning", "id": getattr(item, "id", None)}

    encrypted = getattr(item, "encrypted_content", None)
    if encrypted:
        d["encrypted_content"] = encrypted

    summary = getattr(item, "summary", None) or []
    if summary:
        d["summary"] = [{"type": getattr(s, "type", "summary"), "text": getattr(s, "text", "")} for s in summary]

    return d
