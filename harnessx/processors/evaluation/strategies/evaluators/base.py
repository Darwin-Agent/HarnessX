# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from .....core.events import EvalResult
from .....core.state import State

if TYPE_CHECKING:
    from .....core.harness import BaseTask


@runtime_checkable
class BaseEvaluator(Protocol):
    async def evaluate(self, task: "BaseTask", state: State) -> EvalResult: ...
