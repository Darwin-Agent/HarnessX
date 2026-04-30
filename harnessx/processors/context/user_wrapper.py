# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, AsyncIterator

from ...core.events import BeforeModelEvent, rough_token_count
from ...core.processor import MultiHookProcessor

if TYPE_CHECKING:
    from .strategies.user_wrapper.base import BaseUserPromptWrapper


class UserWrapperProcessor(MultiHookProcessor):
    """Wrap the final user turn before the model sees it.

    Args:
        user_wrapper: Strategy implementing
                      ``async wrap(message, task) -> Message``.
                      When ``None`` this processor is a no-op.
    """

    _singleton_group = "context.user_wrapper"
    _order = 5

    def __init__(self, user_wrapper: "BaseUserPromptWrapper | None" = None) -> None:
        self.user_wrapper = user_wrapper

    async def on_before_model(self, event: BeforeModelEvent) -> AsyncIterator[BeforeModelEvent]:
        if self.user_wrapper is None:
            yield event
            return
        messages = list(event.messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                messages[i] = await self.user_wrapper.wrap(messages[i], event.task)
                break
        yield dataclasses.replace(
            event,
            messages=tuple(messages),
            token_count=rough_token_count(messages),
        )
