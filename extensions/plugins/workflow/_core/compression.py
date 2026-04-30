# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harnessx.core.events import Message

_TOOL_RESULT_MAX = 300  # chars per tool result before truncation
_THINKING_STRIP = True


def compress_session(messages: "list[Message]") -> "list[Message]":
    """Return a shortened copy of *messages* suitable for the extractor sub-harness.

    Operations (in order):
    1. Strip thinking/thinking_blocks from all messages.
    2. Truncate tool result message content to ``_TOOL_RESULT_MAX`` chars.

    The result preserves role/content/tool_calls structure so the sub-harness
    model can still follow the conversation flow.  Thinking is omitted because
    it can be very long and is not needed for workflow summarisation.
    """
    from harnessx.core.events import Message

    compressed: list[Message] = []
    for msg in messages:
        if msg.role == "tool":
            content = msg.content
            if isinstance(content, str) and len(content) > _TOOL_RESULT_MAX:
                content = content[:_TOOL_RESULT_MAX] + f"\n…[truncated, {len(msg.content)} chars total]"
            compressed.append(
                Message(
                    role=msg.role,
                    content=content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                    tool_calls=msg.tool_calls,
                    thinking="",
                    thinking_blocks=(),
                )
            )
        elif msg.thinking or msg.thinking_blocks:
            # Strip thinking from assistant messages
            compressed.append(
                Message(
                    role=msg.role,
                    content=msg.content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                    tool_calls=msg.tool_calls,
                    thinking="",
                    thinking_blocks=(),
                )
            )
        else:
            compressed.append(msg)
    return compressed


def count_tool_calls(messages: "list[Message]") -> int:
    """Count tool calls made by the assistant across all messages."""
    total = 0
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            total += len(msg.tool_calls)
    return total
