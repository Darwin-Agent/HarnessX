# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from harnessx.core.builder import HarnessBuilder
from harnessx.bundles.context import make_context
from harnessx.bundles.tools import make_tools
from harnessx.bundles.execution import make_execution
from harnessx.bundles.control import make_control
from harnessx.processors.context.strategies.system_prompt.default import (
    DefaultSystemPromptBuilder,
)
from harnessx.processors.memory.strategies.sliding_window import SlidingWindowMemory
from harnessx.processors.memory.memory_retrieval import MemoryRetrievalProcessor
from harnessx.processors.memory.memory_extraction import MemoryExtractionProcessor
from harnessx.processors.observability.otel_proc import OTelProcessor


def build_assistant(
    *,
    memory_window: int = 20,
    compaction_threshold: int = 140_000,
    skill_loading: bool = False,
    working_dir: str | None = None,
    obs_otel: bool = True,
) -> HarnessBuilder:
    """Return a fully-configured personal assistant builder.

    Call ``.build()`` then combine with a ``ModelConfig`` via
    ``ModelConfig(main=provider).agentic(config)``.
    """
    memory = SlidingWindowMemory(n=memory_window)

    builder = (
        HarnessBuilder()
        | make_context(system_builder=DefaultSystemPromptBuilder())
        | make_tools(skill_loading=skill_loading)
        | make_execution()
        | make_control(
            include_reliability=True,
            include_budget=True,
            token_threshold=compaction_threshold,
        )
    )
    builder = builder.add(MemoryExtractionProcessor(memory=memory, threshold=compaction_threshold)).add(
        MemoryRetrievalProcessor(memory=memory, top_k=10)
    )
    if obs_otel:
        builder = builder.add(OTelProcessor())
    return builder
