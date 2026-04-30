"""HarnessBuilder-based configuration for tau2-bench.

Uses HarnessX's HarnessBuilder to configure the LLM provider and
context assembly pipeline.

tau2 owns the simulation loop (user simulator, tool execution, evaluation).
The Harness provides the per-step infrastructure: context assembly via
SystemPromptProcessor + TokenBudgetProcessor and model calls via the
appropriate provider (AnthropicProvider or LiteLLMProvider).

Usage::

    from benchmarks.tau2.harness import make_tau2_harness

    # Anthropic format (via AnthropicProvider)
    config = make_tau2_harness(
        model="anthropic/claude-haiku-4-5-20251001",
        api_base=os.environ.get("ANTHROPIC_API_BASE"),
    )

    # OpenAI-compatible format (via LiteLLMProvider)
    config = make_tau2_harness(
        model="claude-haiku-4-5-20251001",
        api_base=os.environ.get("OPENAI_API_BASE"),
    )
"""

from __future__ import annotations

from typing import Any

from harnessx.core.builder import HarnessBuilder
from harnessx.core.harness import HarnessConfig
from harnessx.processors.context.system_prompt import SystemPromptProcessor
from harnessx.processors.control.token_budget import TokenBudgetProcessor
from harnessx.processors.context.strategies.system_prompt.null import (
    NullSystemPromptBuilder,
)
from harnessx.tracing.journal import HarnessJournal
from .policy_hint import PolicyHintProcessor

_ANTHROPIC_PREFIXES = ("anthropic/", "claude-")


def make_tau2_provider(
    model: str,
    api_base: str | None = None,
    api_key: str = "EMPTY",
    extra_headers: dict | None = None,
    extended_thinking: bool = False,
    thinking_budget_tokens: int = 8000,
    timeout: float | None = None,
) -> Any:
    """Build an HarnessX model provider from a model identifier.

    Args:
        model: Model identifier.
               Anthropic format: ``"anthropic/claude-haiku-4-5-20251001"``
               OpenAI format:    ``"claude-haiku-4-5-20251001"``
        api_base: Optional API endpoint override.
                  Anthropic format: set ``ANTHROPIC_API_BASE`` env var.
                  OpenAI format: set ``OPENAI_API_BASE`` env var.
        api_key: API key (default ``"EMPTY"`` for proxy setups).
        extra_headers: Extra HTTP headers.
        extended_thinking: Enable extended thinking (AnthropicProvider only).
        thinking_budget_tokens: Token budget for thinking (default 8000).

    Returns:
        AnthropicProvider or LiteLLMProvider instance.
    """
    is_anthropic = any(model.startswith(p) for p in _ANTHROPIC_PREFIXES)

    if is_anthropic:
        from harnessx.providers.anthropic_provider import AnthropicProvider

        sdk_model = model[len("anthropic/") :] if model.startswith("anthropic/") else model
        prov_kwargs: dict[str, Any] = {}
        if api_key and api_key != "EMPTY":
            prov_kwargs["api_key"] = api_key
        if api_base:
            prov_kwargs["base_url"] = api_base
        prov_kwargs["default_headers"] = extra_headers or {}
        if timeout is not None:
            prov_kwargs["timeout"] = timeout
        return AnthropicProvider(
            model=sdk_model,
            extended_thinking=extended_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            **prov_kwargs,
        )
    else:
        from harnessx.providers.litellm_provider import LiteLLMProvider

        prov_kwargs = {}
        if api_key and api_key != "EMPTY":
            prov_kwargs["api_key"] = api_key
        if api_base:
            prov_kwargs["api_base"] = api_base
        if extra_headers:
            prov_kwargs["extra_headers"] = extra_headers
        return LiteLLMProvider(model=model, **prov_kwargs)


def make_tau2_harness(
    model: str,
    api_base: str | None = None,
    api_key: str = "EMPTY",
    logs_dir: str = "runs",
    tool_registry: Any = None,
    extra_headers: dict | None = None,
    policy_hints: bool = False,
) -> HarnessConfig:
    """Build a HarnessConfig for tau2-bench using HarnessBuilder.

    tau2 manages the simulation loop (user simulator, tool execution,
    evaluation).  The Harness provides:

    - **SystemPromptProcessor** (NullSystemPromptBuilder) — tau2 owns the system prompt
    - **TokenBudgetProcessor** for hard context-window safety
    - **HarnessJournal** for writing JSONL logs to ``logs_dir/{run_id}/``

    Note: the model provider is built separately via ``make_tau2_provider``
    and passed to ``ModelConfig(main=provider).agentic(config)`` to create
    the runnable ``Harness``.

    No cross-session memory is used: ``AgentState.oh_state`` (the HarnessX
    ``State`` object) is kept alive across all ``Harness.run()`` resume calls
    and serves as the authoritative conversation history.

    Args:
        model: Model identifier (used only for provider-type detection).
        api_base: Optional API endpoint override.
        api_key: API key (default ``"EMPTY"`` for proxy setups).
        logs_dir: Directory for JSONL trace logs (default: ``"runs"``).
        tool_registry: Pre-built ``InMemoryToolRegistry`` with stub tools.
        extra_headers: Extra HTTP headers.
        policy_hints: If True, insert ``PolicyHintProcessor`` (order=2) into the
                      pipeline.

    Returns:
        HarnessConfig with context assembly processor and tracer (no model).
    """
    builder = (
        HarnessBuilder()
        .slot(tracer=HarnessJournal(base_dir=logs_dir, export_jsonl=True))
        .add(SystemPromptProcessor(NullSystemPromptBuilder()))
        .add(TokenBudgetProcessor())
    )
    if policy_hints:
        builder = builder.add(PolicyHintProcessor())
    if tool_registry is not None:
        builder = builder.slot(tool_registry=tool_registry)
    return builder.build()
