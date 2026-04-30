# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from harnessx.core.events import TaskEndEvent
from harnessx.plugins.dimensions.light_memory._core.types import PluginConfig
from harnessx.plugins.dimensions.light_memory.processors import (
    LightMemoryCaptureProcessor,
)


class TestLightMemoryCaptureCompat:
    @pytest.mark.asyncio
    async def test_light_memory_capture_ignores_failed_task_without_success_attr(
        self,
    ) -> None:
        proc = LightMemoryCaptureProcessor()
        proc.configure(
            PluginConfig(memory_root="/tmp/hx-light-memory-test", auto_capture=True),
            provider=None,
        )

        event = TaskEndEvent(
            run_id="r1",
            step_id=0,
            exit_reason="error",
            error="RuntimeError: boom",
            final_messages=(),
        )

        got = [e async for e in proc.on_task_end(event)]
        assert got == [event]
