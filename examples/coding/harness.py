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
from harnessx.processors.context.env_context_injector import EnvironmentContextInjector


def build_coding(
    *,
    memory_window: int = 40,
    compaction_threshold: int = 80_000,
    skill_loading: bool = True,
    working_dir: str | None = None,
) -> HarnessBuilder:
    """Return a fully-configured coding agent builder.

    Call ``.slot(workspace=...).build()`` then combine with a ``ModelConfig``
    via ``ModelConfig(main=provider).agentic(config)``.
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
    if working_dir:
        builder = builder.add(EnvironmentContextInjector(working_dir=working_dir))
    return builder
