# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import uuid
from typing import Any, Callable

from ..core.events import Message, ToolCall, ToolSchema


def to_openai_tools(tools: list[ToolSchema]) -> list[dict]:
    """Convert ToolSchema list to the OpenAI function-calling wire format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def parse_tool_calls(raw_tool_calls) -> list[ToolCall]:
    """Parse an OpenAI-format tool_calls list into ToolCall dataclasses."""
    result = []
    for tc in raw_tool_calls:
        try:
            inp = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            inp = {}
        result.append(
            ToolCall(
                id=tc.id or str(uuid.uuid4()),
                name=tc.function.name,
                input=inp,
            )
        )
    return result


def count_tokens(messages: list[Message]) -> int:
    from ..core.events import rough_token_count

    return rough_token_count(messages)


def to_openai_content(content: "str | list") -> "str | list | None":
    """Convert internal content (Anthropic format) to OpenAI/litellm wire format.

    Anthropic image block::
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
    becomes::
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    """
    if isinstance(content, str):
        return content or None
    result = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            result.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            src = block.get("source", {})
            if src.get("type") == "base64":
                url = f"data:{src['media_type']};base64,{src['data']}"
                result.append({"type": "image_url", "image_url": {"url": url}})
            elif src.get("type") == "url":
                result.append({"type": "image_url", "image_url": {"url": src["url"]}})
    return result or None


def as_text_delta(value: Any) -> str:
    """Best-effort conversion of provider delta payloads to text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                # OpenAI-style content list chunks may look like:
                # {"type":"text","text":"..."} or {"type":"output_text","text":"..."}
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""


def emit_stream_delta(
    stream_callback: "Callable[[object], None] | None",
    content: str,
    *,
    kind: str = "token",
) -> None:
    """Emit one stream delta to callback with backward compatibility.

    New callback payload shape:
      {"type": "token" | "thinking", "content": "..."}

    Legacy callbacks (CLI) accept plain string deltas only. We fall back to
    string mode for token deltas and ignore non-token kinds on legacy callbacks.
    """
    if stream_callback is None or not content:
        return
    if kind not in {"token", "thinking"}:
        return

    mode = getattr(stream_callback, "__harnessx_stream_mode__", None)
    if mode == "structured":
        try:
            stream_callback({"type": kind, "content": content})
        except Exception:
            return
        return
    if mode == "text":
        if kind != "token":
            return
        try:
            stream_callback(content)
        except Exception:
            return
        return

    try:
        stream_callback({"type": kind, "content": content})
        try:
            setattr(stream_callback, "__harnessx_stream_mode__", "structured")
        except Exception:
            pass
        return
    except Exception:
        try:
            setattr(stream_callback, "__harnessx_stream_mode__", "text")
        except Exception:
            pass
        if kind != "token":
            return
    try:
        stream_callback(content)
    except Exception:
        # Stream callback must be best-effort only; never fail the run for UI I/O.
        return
