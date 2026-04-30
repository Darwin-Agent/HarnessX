# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.events import Event


@runtime_checkable
class BaseTracer(Protocol):
    async def on_event(self, event: Event) -> None: ...
    async def on_raw_event(self, event: Event) -> None: ...
    async def flush(self) -> None: ...
