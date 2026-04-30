# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ..core.events import Event


class NullTracer:
    """No-op tracer. Discards all events. Use in benchmarks to reduce I/O overhead."""

    async def on_event(self, event: Event) -> None:
        pass

    async def on_raw_event(self, event: Event) -> None:
        pass

    async def flush(self) -> None:
        pass
