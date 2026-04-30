# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from harnessx.core.builder import HarnessBuilder
from harnessx.core.harness import HarnessConfig
from harnessx.processors.control.token_budget import TokenBudgetProcessor
from harnessx.processors.evaluation.evaluation import EvaluationProcessor
from harnessx.plugins.dimensions.rl import RLControlPlugin
from harnessx.processors.context.system_prompt import SystemPromptProcessor
from harnessx.processors.context.strategies.system_prompt.null import (
    NullSystemPromptBuilder,
)
from harnessx.tools.inmemory import InMemoryToolRegistry
from harnessx.tracing.null_tracer import NullTracer

if TYPE_CHECKING:
    from harnessx.rl.config import RLConfigSpec
    from harnessx.rl.task import RLTask
    from harnessx.tools.base import Tool


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class _FixedSystemPromptBuilder:
    """System prompt builder that returns a fixed string.

    Used when the task or spec provides an explicit system_prompt string.
    """

    def __init__(self, text: str) -> None:
        self._text = text

    async def build(self, workspace: Any = None) -> str:
        return self._text


def _build_tool_registry(tools: list["Tool"]) -> InMemoryToolRegistry:
    """Build an InMemoryToolRegistry from a list of Tool instances."""
    registry = InMemoryToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _resolve_system_builder(spec: "RLConfigSpec", task: "RLTask") -> Any:
    """Resolve system prompt builder from task metadata or spec default."""
    task_system_prompt = (task.metadata or {}).get("system_prompt", "")
    system_text = task_system_prompt or spec.system_prompt
    return _FixedSystemPromptBuilder(system_text) if system_text else NullSystemPromptBuilder()


# ---------------------------------------------------------------------------
# build_rl_harness_config — framework-agnostic HarnessConfig factory
# ---------------------------------------------------------------------------


def build_rl_harness_config(
    spec: "RLConfigSpec",
    provider: Any,
    task: "RLTask",
) -> HarnessConfig:
    """
    Build a per-run HarnessConfig from an RLConfigSpec.  Framework-agnostic.

    Per-run means: evaluator_cls(task) instantiated fresh each call,
    closing over task.label.  Processors are also fresh per run so their
    internal state (fingerprints, counts) starts clean for each episode.

    System prompt priority:
        1. task.metadata["system_prompt"]  — sample-level override
        2. spec.system_prompt              — task-type default
        3. None                            — NullSystemPromptBuilder (empty)

    Composes via HarnessBuilder:
        - step_snapshots=False + NullTracer (no snapshots, no tracing overhead)
        - SystemPromptProcessor (fixed or null builder)
        - TokenBudgetProcessor(ratio=1.0) as hard safety window guard
        - RLControlPlugin (RLSignalCollectorProcessor + EpisodeMetricsProcessor)
        - EvaluationProcessor(evaluator)
        - extra_processors from spec (per hook, appended last)

    Args:
        spec:     RLConfigSpec (or subclass) with task configuration.
        provider: BaseModelProvider (e.g. SGLangProvider) — already configured.
        task:     RLTask built by spec.task_builder.build(sample).

    Returns:
        HarnessConfig ready for ModelConfig(main=provider).agentic(config).run(task).
    """
    evaluator = spec.evaluator_cls(task)
    system_builder = _resolve_system_builder(spec, task)

    builder = (
        HarnessBuilder()
        .slot(step_snapshots=False, tracer=NullTracer())
        .add(SystemPromptProcessor(system_builder))
        .add(TokenBudgetProcessor(ratio=1.0))
        .plugin(RLControlPlugin())
        .add(EvaluationProcessor(evaluator))
    )

    for hook, procs in (spec.extra_processors or {}).items():
        for proc in procs:
            builder = builder.add(proc, hook=hook)

    return builder.slot(
        tool_registry=_build_tool_registry(spec.tools),
    ).build()
