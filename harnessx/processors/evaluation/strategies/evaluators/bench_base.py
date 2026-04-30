# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING
from .....core.events import EvalResult
from .....core.state import State

if TYPE_CHECKING:
    from .....core.harness import BaseTask


class BenchEvaluator:
    """Base class for benchmark evaluators. Subclass and implement evaluate()."""

    @abstractmethod
    async def evaluate(self, task: "BaseTask", state: State) -> EvalResult: ...
