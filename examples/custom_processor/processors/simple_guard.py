# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from typing import Iterable

from harnessx.core.events import Message, StepStartEvent
from harnessx.core.processor import MultiHookProcessor


class SimpleGuardProcessor(MultiHookProcessor):
    """Example custom processor with configurable keyword checks.

    Args:
        blocked_patterns: Case-insensitive patterns to detect in user text.
        mode: "warn" appends a warning; "off" disables behavior.
    """

    _singleton_group = "examples.simple_guard"
    _order = 15

    def __init__(
        self,
        blocked_patterns: Iterable[str] | None = None,
        mode: str = "warn",
    ) -> None:
        self._patterns = [p.lower() for p in (blocked_patterns or []) if str(p).strip()]
        self._mode = mode

    async def on_step_start(self, event: StepStartEvent):
        if self._mode != "warn" or not self._patterns:
            yield event
            return

        # In minimal harnesses, step_start may not have an assembled
        # `messages` context yet (only `raw_messages` is populated).
        source_messages = event.messages or event.raw_messages
        last_user = ""
        for msg in reversed(source_messages):
            if msg.role == "user":
                if isinstance(msg.content, str):
                    last_user = msg.content
                else:
                    last_user = str(msg.content)
                break

        text = last_user.lower()
        hit = next((p for p in self._patterns if p in text), None)
        if not hit:
            yield event
            return

        warning = "[SimpleGuard] Detected a risky instruction pattern. Proceed carefully and keep policy constraints."
        new_last_user = last_user + "\n\n" + warning
        new_messages = source_messages[:-1] + (Message(role="user", content=new_last_user),)
        new_count = event.token_count + len(warning.split())  # Rough token count for the added warning
        yield dataclasses.replace(
            event,
            messages=new_messages,
            token_count=new_count,
        )
