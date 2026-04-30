# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ...core.events import BeforeModelEvent
from ...core.processor import MultiHookProcessor
from ...core.runloop import BudgetExceededError
from ...logging import logger


class CostGuardProcessor(MultiHookProcessor):
    """
    Hooks: before_model
    Raises BudgetExceededError when cumulative cost >= max_usd.
    Logs a warning at warning_threshold * max_usd.

    Reads cost from ``BeforeModelEvent.cumulative_cost_usd`` — no external
    state injection needed.
    """

    _singleton_group = "cost_guard"
    _order = 10

    def __init__(self, max_usd: float = 1.0, warning_threshold: float = 0.8):
        self.max_usd = max_usd
        self.warning_threshold = warning_threshold

    async def on_before_model(self, event: BeforeModelEvent):
        cost = event.cumulative_cost_usd
        if cost >= self.max_usd:
            raise BudgetExceededError(f"Cost limit exceeded: ${cost:.4f} >= ${self.max_usd:.4f}")
        if cost >= self.max_usd * self.warning_threshold:
            logger.warning(f"Cost warning: ${cost:.4f} / ${self.max_usd:.4f} ({cost / self.max_usd * 100:.0f}%)")
        yield event
