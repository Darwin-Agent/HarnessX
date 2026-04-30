from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import AsyncIterator

from harnessx.core.events import BeforeModelEvent
from harnessx.core.processor import MultiHookProcessor

from ..base_channel import ConversationType


class IMUserContextProcessor(MultiHookProcessor):
    """
    Inject dynamic per-turn context into the last user message at on_before_model.

    Each turn re-injects (sender may differ in group chat):
    - Sender name / ID
    - Conversation type (DM / GROUP / TOPIC / COMMENT)
    - Group name, topic name
    - Whether @ triggered
    - Message timestamp
    """

    _singleton_group = "im_user_context"
    _order = 5

    @staticmethod
    def _sender_display(sender_name: str, sender_id: str) -> str:
        """
        Build a compact sender label like "alice#9f3a".
        Mirrors the qwenpaw-style sender decoration for group readability.
        """
        name = (sender_name or "").strip() or "unknown"
        sid = (sender_id or "").strip()
        suffix = sid[-4:] if len(sid) >= 4 else (sid or "????")
        return f"{name}#{suffix}"

    async def on_before_model(self, event: BeforeModelEvent) -> AsyncIterator[BeforeModelEvent]:
        from ..dispatch import _im_event_var

        msg_event = _im_event_var.get(None)
        if msg_event is None or not event.messages:
            yield event
            return

        # Contract-safe guard:
        # before_model processors may only modify the tail message when tail role is user.
        # If tail is assistant/tool, this processor must no-op.
        if event.messages[-1].role != "user":
            yield event
            return

        ctx = msg_event.conversation
        sender_id = (msg_event.sender_id or "").strip() or "unknown"
        sender_display = self._sender_display(msg_event.sender_name, sender_id)
        message_time = ""
        try:
            dt = datetime.fromtimestamp(float(msg_event.ts)).astimezone()
            message_time = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            message_time = ""
        lines = [
            "[Message Context]",
            f"- Sender: {sender_display}",
            f"- Sender ID: {sender_id}",
            f"- Conversation Type: {ctx.type.value.upper()}",
        ]
        if message_time:
            lines.append(f"- Message Time: {message_time}")

        match ctx.type:
            case ConversationType.GROUP:
                if ctx.group_name:
                    lines.append(f"- Group: {ctx.group_name}")
            case ConversationType.TOPIC:
                if ctx.group_name:
                    lines.append(f"- Group: {ctx.group_name}")
                if ctx.topic_name:
                    lines.append(f"- Topic: {ctx.topic_name}")
            case ConversationType.COMMENT:
                if ctx.post_id:
                    lines.append(f"- Comment Thread ID: {ctx.post_id}")

        if ctx.mentioned:
            lines.append("- Mention: user mentioned you directly")

        header = "\n".join(lines)
        messages = list(event.messages)

        # Tail is guaranteed user by the guard above; only mutate this message.
        m = messages[-1]
        old_content = m.content if isinstance(m.content, str) else str(m.content)
        new_content = f"{header}\n\n{old_content}"
        messages[-1] = dataclasses.replace(m, content=new_content)

        yield dataclasses.replace(event, messages=tuple(messages))
