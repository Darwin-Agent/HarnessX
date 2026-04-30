# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses

from ...core.events import (
    StepStartEvent,
    BoundaryHint,
    rough_token_count,
    trim_messages_to_budget,
)
from ...core.processor import MultiHookProcessor


class TokenBudgetProcessor(MultiHookProcessor):
    """
    Hooks: step_start
    If the assembled context exceeds ``ratio * event.token_budget``, drop
    messages (oldest non-system first) until it fits.

    This is a hard safety guard that operates directly on
    ``StepStartEvent.messages``.

    Args:
        ratio: Fraction of token_budget to use as the trim threshold (default 0.8).
    """

    _singleton_group = "token_budget"
    _order = 10

    def __init__(self, ratio: float = 0.8):
        self.ratio = ratio

    async def on_step_start(self, event: StepStartEvent):
        target = int(event.context_window * self.ratio)
        current_count = rough_token_count(list(event.messages))
        if current_count <= target:
            yield event
            return

        msg_list = list(event.messages)
        msgs = trim_messages_to_budget(msg_list, target)

        surviving_ids = {id(m) for m in msgs}
        kept_positions = [i for i, m in enumerate(msg_list) if id(m) in surviving_ids]
        raw_msgs = tuple(event.raw_messages[i] for i in kept_positions if i < len(event.raw_messages))

        new_count = rough_token_count(msgs)
        yield dataclasses.replace(
            event,
            messages=tuple(msgs),
            raw_messages=raw_msgs,
            token_count=new_count,
            boundary_hint=BoundaryHint(
                reason="token_budget",
                before_msgs=len(msg_list),
                after_msgs=len(msgs),
                before_tokens=current_count,
                after_tokens=new_count,
            ),
        )
