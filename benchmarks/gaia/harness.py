"""
GAIA benchmark harness factory.

Separates harness construction from the CLI runner so the same config
can be reused or tested independently.

Evaluation is NOT wired into the harness — callers run the benchmark
evaluator externally after ``harness.run()`` returns. The per-task
trajectory ``.md`` files (the meta-agent's inputs) carry pass/fail
correctness signals in frontmatter (``eval_passed`` / ``eval_score``
only) but never the dataset's expected answer text and never the
evaluator's textual reason, so the meta-agent reasons about
behaviour + correctness without seeing the known target string.

Usage::

    from benchmarks.gaia.harness import make_gaia_harness
    from benchmarks.gaia.evaluator import GAIAPipelineEvaluator
    from harnessx.core.model_config import ModelConfig

    harness_config = make_gaia_harness(logs_dir=logs_dir)
    model = ModelConfig(main=provider)
    harness = model.agentic(harness_config)

    pipeline_eval = GAIAPipelineEvaluator(judge_provider=judge_provider)
    result = await harness.run(task)
    eval_result = await pipeline_eval.evaluate_answer(
        result.final_output, task.final_answer,
    )
"""

from __future__ import annotations

import os
from pathlib import Path

from harnessx.core.builder import HarnessBuilder
from harnessx.core.harness import HarnessConfig
from harnessx.processors.observability.checkpoint import CheckpointProcessor
from harnessx.processors.context.system_prompt import SystemPromptProcessor
from harnessx.processors.context.user_wrapper import UserWrapperProcessor
from harnessx.processors.control.cost_guard import CostGuardProcessor
from harnessx.processors.control.loop_detection import LoopDetectionProcessor
from harnessx.processors.evaluation.evaluation import EvaluationProcessor
from harnessx.processors.observability.otel_proc import OTelProcessor
from harnessx.processors.control.token_budget import TokenBudgetProcessor
from harnessx.tools.builtin import build_gaia_tools
from harnessx.tracing.journal import HarnessJournal
from harnessx.workspace.workspace import Workspace

from .evaluator import GAIAPipelineEvaluator
from .defaults import (
    CHECKPOINT_EVERY_N,
    COST_GUARD_MAX_USD,
    LOOP_THRESHOLD,
    LOOP_WINDOW_SIZE,
    TOKEN_BUDGET_RATIO,
    GPT5_CHECKPOINT_EVERY_N,
    GPT5_COST_GUARD_MAX_USD,
    GPT5_LOOP_THRESHOLD,
    GPT5_LOOP_WINDOW_SIZE,
    GPT5_TOKEN_BUDGET_RATIO,
    QWEN35_9B_CHECKPOINT_EVERY_N,
    QWEN35_9B_COST_GUARD_MAX_USD,
    QWEN35_9B_LOOP_THRESHOLD,
    QWEN35_9B_LOOP_WINDOW_SIZE,
    QWEN35_9B_TOKEN_BUDGET_RATIO,
)

from harnessx.processors.context.strategies.system_prompt.default import DefaultSystemPromptBuilder

_system_builder = DefaultSystemPromptBuilder()


def _gaia_workspace() -> Workspace:
    """Unrestricted workspace so the agent can read task attachments in /tmp etc."""
    return Workspace(agent_id="gaia", root=Path("/tmp/harnessx-gaia"), mode=None)


def make_gaia_builder(
    logs_dir: Path | None = None,
    *,
    max_cost_usd: float = COST_GUARD_MAX_USD,
) -> HarnessBuilder:
    """Build a HarnessBuilder for GAIA — returns builder (not built config).

    Use this when you need to customize the builder further (e.g. MetaHarness).
    No evaluator is wired in; run the evaluator externally on
    ``result.final_output`` after ``harness.run()`` returns.
    """
    tracer_dir = str((logs_dir or Path("oh_runs")) / "gaia_runs")

    return (
        HarnessBuilder()
        .slot(
            tool_registry=build_gaia_tools(),
            tracer=HarnessJournal(
                base_dir=tracer_dir,
                export_jsonl=True,
            ),
            workspace=_gaia_workspace(),
        )
        .add(SystemPromptProcessor(_system_builder))
        .add(UserWrapperProcessor())
        .add(TokenBudgetProcessor(ratio=TOKEN_BUDGET_RATIO))
        .add(CostGuardProcessor(max_usd=max_cost_usd))
        .add(
            LoopDetectionProcessor(
                window_size=LOOP_WINDOW_SIZE,
                threshold=LOOP_THRESHOLD,
            )
        )
        .add(CheckpointProcessor(every_n=CHECKPOINT_EVERY_N))
        .add(OTelProcessor())
    )


def make_gaia_harness(
    logs_dir: Path | None = None,
    *,
    max_cost_usd: float = COST_GUARD_MAX_USD,
) -> HarnessConfig:
    """Build a HarnessConfig for the GAIA benchmark.

    Composes the DeepResearch processor stack:

    - ``SystemPromptProcessor``     — Jinja2 system prompt
    - ``TokenBudgetProcessor``      — keeps context within token limits
    - ``CostGuardProcessor``        — hard per-task cost ceiling
    - ``LoopDetectionProcessor``    — catches repeated search/fetch loops
    - ``CheckpointProcessor``       — periodic state snapshots
    - ``OTelProcessor``             — OpenTelemetry tracing

    Evaluation is externalised — see module docstring.
    """
    return make_gaia_builder(
        logs_dir=logs_dir,
        max_cost_usd=max_cost_usd,
    ).build()


# ── GAIA-specific system prompt (used by GPT-5 / QwQ-32B presets) ────────────

_GAIA_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "prompts", "gaia_agent.j2")
_GPT5_TEMPLATE_PATH = _GAIA_TEMPLATE_PATH


def _gaia_system_builder():
    if os.path.exists(_GAIA_TEMPLATE_PATH):
        from harnessx.processors.context.strategies.system_prompt.template import TemplateSystemPromptBuilder

        return TemplateSystemPromptBuilder(_GAIA_TEMPLATE_PATH)
    return _system_builder


# ── GPT-5 preset ─────────────────────────────────────────────────────────────


def make_gaia_builder_gpt5(
    logs_dir: Path | None = None,
    *,
    max_cost_usd: float = GPT5_COST_GUARD_MAX_USD,
    pipeline_evaluator: GAIAPipelineEvaluator | None = None,
) -> HarnessBuilder:
    """Build a HarnessBuilder for GAIA with GPT-5 tuned parameters.

    Uses extended tool set (with code_interpreter), higher step/cost/token
    budgets, and the GAIA-specific system prompt (``gaia_agent.j2``).

    Inline evaluation is **opt-in**: pass a ``pipeline_evaluator`` to wire an
    ``EvaluationProcessor`` into the stack. Leave it ``None`` (default) to
    keep trajectories free of success labels — required by runners that
    externalise evaluation (e.g. ``recipe/gaia_evolver``).
    """
    from harnessx.tools.builtin import build_gaia_tools_full

    tracer_dir = str((logs_dir or Path("oh_runs")) / "gaia_gpt5_runs")

    builder = (
        HarnessBuilder()
        .slot(
            tool_registry=build_gaia_tools_full(),
            tracer=HarnessJournal(
                base_dir=tracer_dir,
                export_jsonl=True,
            ),
            workspace=_gaia_workspace(),
        )
        .add(SystemPromptProcessor(_gaia_system_builder()))
        .add(UserWrapperProcessor())
        .add(TokenBudgetProcessor(ratio=GPT5_TOKEN_BUDGET_RATIO))
        .add(CostGuardProcessor(max_usd=max_cost_usd))
        .add(
            LoopDetectionProcessor(
                window_size=GPT5_LOOP_WINDOW_SIZE,
                threshold=GPT5_LOOP_THRESHOLD,
            )
        )
        .add(CheckpointProcessor(every_n=GPT5_CHECKPOINT_EVERY_N))
        .add(OTelProcessor())
    )
    if pipeline_evaluator is not None:
        builder = builder.add(EvaluationProcessor(pipeline_evaluator))
    return builder


def make_gaia_harness_gpt5(
    logs_dir: Path | None = None,
    *,
    max_cost_usd: float = GPT5_COST_GUARD_MAX_USD,
    pipeline_evaluator: GAIAPipelineEvaluator | None = None,
) -> HarnessConfig:
    """Build a HarnessConfig for GAIA with GPT-5 preset."""
    return make_gaia_builder_gpt5(
        logs_dir=logs_dir,
        max_cost_usd=max_cost_usd,
        pipeline_evaluator=pipeline_evaluator,
    ).build()


# ── Qwen3.5-9B preset ─────────────────────────────────────────────────────────


def make_gaia_builder_qwen35_9b(
    logs_dir: Path | None = None,
    *,
    max_cost_usd: float = QWEN35_9B_COST_GUARD_MAX_USD,
    pipeline_evaluator: GAIAPipelineEvaluator | None = None,
) -> HarnessBuilder:
    """HarnessBuilder for GAIA with Qwen3.5-9B tuned parameters.

    The deployment is expected to parse Qwen's tool-call format
    server-side (SGLang / vLLM with the appropriate parser flag),
    so the harness sees structured API tool calls and no recovery
    processor is needed. No code_interpreter — small Qwen models
    reject that tool name.
    """
    from harnessx.tools.builtin import build_gaia_tools_qw

    tracer_dir = str((logs_dir or Path("oh_runs")) / "gaia_qwen35_9b_runs")

    builder = (
        HarnessBuilder()
        .slot(
            tool_registry=build_gaia_tools_qw(),
            tracer=HarnessJournal(
                base_dir=tracer_dir,
                export_jsonl=True,
            ),
            workspace=_gaia_workspace(),
        )
        .add(SystemPromptProcessor(_gaia_system_builder()))
        .add(UserWrapperProcessor())
        .add(TokenBudgetProcessor(ratio=QWEN35_9B_TOKEN_BUDGET_RATIO))
        .add(CostGuardProcessor(max_usd=max_cost_usd))
        .add(
            LoopDetectionProcessor(
                window_size=QWEN35_9B_LOOP_WINDOW_SIZE,
                threshold=QWEN35_9B_LOOP_THRESHOLD,
            )
        )
        .add(CheckpointProcessor(every_n=QWEN35_9B_CHECKPOINT_EVERY_N))
        .add(OTelProcessor())
    )
    if pipeline_evaluator is not None:
        builder = builder.add(EvaluationProcessor(pipeline_evaluator))
    return builder


def make_gaia_harness_qwen35_9b(
    logs_dir: Path | None = None,
    *,
    max_cost_usd: float = QWEN35_9B_COST_GUARD_MAX_USD,
    pipeline_evaluator: GAIAPipelineEvaluator | None = None,
) -> HarnessConfig:
    """Build a HarnessConfig for GAIA with Qwen3.5-9B preset."""
    return make_gaia_builder_qwen35_9b(
        logs_dir=logs_dir,
        max_cost_usd=max_cost_usd,
        pipeline_evaluator=pipeline_evaluator,
    ).build()
