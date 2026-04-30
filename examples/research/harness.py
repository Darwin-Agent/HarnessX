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
from harnessx.processors.evaluation.evaluation import EvaluationProcessor
from harnessx.processors.evaluation.strategies.evaluators.llm_judge import (
    LLMJudgeEvaluator,
)
from harnessx.processors.observability.otel_proc import OTelProcessor


def build_research(
    *,
    memory_window: int = 50,
    compaction_threshold: int = 120_000,
    max_cost_usd: float = 5.0,
    obs_otel: bool = True,
) -> HarnessBuilder:
    """Return a fully-configured deep-research builder.

    Call ``.build()`` then combine with a ``ModelConfig`` via
    ``ModelConfig(main=provider, evaluator=judge_provider).agentic(config)``.
    """
    memory = SlidingWindowMemory(n=memory_window)

    builder = (
        HarnessBuilder()
        | make_context(system_builder=DefaultSystemPromptBuilder())
        | make_tools()
        | make_execution()
        | make_control(
            include_reliability=False,
            include_budget=True,
            token_threshold=compaction_threshold,
            max_cost_usd=max_cost_usd,
        )
    )
    builder = (
        builder.add(MemoryExtractionProcessor(memory=memory, threshold=compaction_threshold))
        .add(MemoryRetrievalProcessor(memory=memory, top_k=10))
        .add(EvaluationProcessor(LLMJudgeEvaluator()))
    )
    if obs_otel:
        builder = builder.add(OTelProcessor())
    return builder
