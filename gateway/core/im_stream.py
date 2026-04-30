"""Per-request IM stream context + tool-progress processor.

Keeps the harness stream_callback lightweight (text tokens only) while
letting tool call events reach the Discord / Feishu / Telegram stream queue
via a ContextVar that is set by the dispatcher before each harness.run().
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING

from harnessx import MultiHookProcessor
from harnessx.core.events import ToolCallEvent, ToolResultEvent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Set by ChannelDispatcher._handle_event() before harness.run() so that
# IMProgressProcessor can push events into the active send_stream queue.
im_stream_q_var: ContextVar[asyncio.Queue | None] = ContextVar("im_stream_q", default=None)


class IMProgressProcessor(MultiHookProcessor):
    """Forwards tool start/end markers into the active IM stream queue.

    The Discord channel's send_stream() interprets these dicts to display
    a temporary "⚙️ tool_name…" overlay while a tool is running.
    Other channels simply ignore dict events.
    """

    async def on_before_tool(self, event: ToolCallEvent):
        q = im_stream_q_var.get(None)
        if q is not None:
            try:
                q.put_nowait({"type": "tool_start", "name": event.tool_name})
            except asyncio.QueueFull:
                pass
        yield event

    async def on_after_tool(self, event: ToolResultEvent):
        q = im_stream_q_var.get(None)
        if q is not None:
            try:
                q.put_nowait({"type": "tool_end", "name": event.tool_name})
            except asyncio.QueueFull:
                pass
        yield event
