# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseModelProvider


# ── Error taxonomy ────────────────────────────────────────────────────────────


class ErrorClass(Enum):
    RATE_LIMIT = "rate_limit"  # 429 — retry with backoff, then fallback + cooldown
    SERVER_ERROR = "server_error"  # 500/503 — retry once, then fallback
    TIMEOUT = "timeout"  # network timeout — retry once, then fallback
    AUTH_ERROR = "auth_error"  # 401/403 — skip entire ProviderEntry, no retry
    NOT_FOUND = "not_found"  # 404 model not found — fallback, no retry
    CONTEXT_EXCEEDED = "context_exceeded"  # context too long — raise immediately, no fallback
    UNKNOWN = "unknown"  # unclassified — retry once, then fallback


@dataclass(frozen=True)
class _RetryPolicy:
    max_retries: int  # how many times to retry before signalling fallback
    fallback: bool  # True = try next model/provider after retries exhausted
    mark_entry_failed: bool = False  # True = skip all other models of this ProviderEntry


_RETRY_POLICY: dict[ErrorClass, _RetryPolicy] = {
    ErrorClass.RATE_LIMIT: _RetryPolicy(max_retries=5, fallback=True),
    ErrorClass.SERVER_ERROR: _RetryPolicy(max_retries=2, fallback=True),
    ErrorClass.TIMEOUT: _RetryPolicy(max_retries=2, fallback=True),
    ErrorClass.AUTH_ERROR: _RetryPolicy(max_retries=0, fallback=True, mark_entry_failed=True),
    ErrorClass.NOT_FOUND: _RetryPolicy(max_retries=0, fallback=True),
    ErrorClass.CONTEXT_EXCEEDED: _RetryPolicy(max_retries=0, fallback=False),
    ErrorClass.UNKNOWN: _RetryPolicy(max_retries=1, fallback=True),
}

# Backoff intervals per retry attempt (seconds).
_BACKOFF: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0, 120.0)


def classify_error(exc: Exception) -> ErrorClass:
    """Map a provider exception to an ErrorClass for retry/fallback decisions."""
    module = type(exc).__module__ or ""
    name = type(exc).__name__
    msg = str(exc).lower()

    # Anthropic SDK
    if "anthropic" in module:
        if name == "RateLimitError":
            return ErrorClass.RATE_LIMIT
        if name == "AuthenticationError":
            return ErrorClass.AUTH_ERROR
        if name == "APITimeoutError":
            return ErrorClass.TIMEOUT
        if name == "NotFoundError":
            return ErrorClass.NOT_FOUND
        if name == "InternalServerError":
            return ErrorClass.SERVER_ERROR
        if name == "BadRequestError" and ("context" in msg or "too long" in msg or "maximum" in msg):
            return ErrorClass.CONTEXT_EXCEEDED

    # LiteLLM
    if "litellm" in module:
        if name == "RateLimitError":
            return ErrorClass.RATE_LIMIT
        if name in ("AuthenticationError", "PermissionDeniedError"):
            return ErrorClass.AUTH_ERROR
        if "ContextWindow" in name or "context_window" in name.lower():
            return ErrorClass.CONTEXT_EXCEEDED
        if name in ("ServiceUnavailableError", "APIError") and any(code in msg for code in ("503", "500", "502")):
            return ErrorClass.SERVER_ERROR

    # OpenAI SDK
    if "openai" in module:
        if name == "RateLimitError":
            return ErrorClass.RATE_LIMIT
        if name == "AuthenticationError":
            return ErrorClass.AUTH_ERROR
        if name == "APITimeoutError":
            return ErrorClass.TIMEOUT
        if "context" in name.lower() or "maximum context" in msg:
            return ErrorClass.CONTEXT_EXCEEDED

    # httpx / asyncio timeouts
    if isinstance(exc, asyncio.TimeoutError):
        return ErrorClass.TIMEOUT
    if "timeout" in name.lower() or "timeout" in msg:
        return ErrorClass.TIMEOUT

    return ErrorClass.UNKNOWN


# ── Config dataclasses ────────────────────────────────────────────────────────


@dataclass
class ModelEntry:
    """Configuration for a single model endpoint within a ProviderEntry."""

    model: str
    extra_headers: dict[str, str] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    context_window: int | None = None  # override auto-detected value
    timeout: float = 300.0
    is_default: bool = False
    max_retries: int | None = None  # None = use ProviderEntry.max_retries
    # Reasoning controls for OpenAI/LiteLLM reasoning models.
    reasoning_effort: str | None = None
    reasoning_summary: bool | None = None

    @classmethod
    def from_dict(cls, d: dict) -> ModelEntry:
        d = dict(d)
        return cls(
            model=d.pop("model"),
            extra_headers=d.pop("extra_headers", None),
            temperature=d.pop("temperature", None),
            max_tokens=d.pop("max_tokens", None),
            context_window=d.pop("context_window", None),
            timeout=d.pop("timeout", 300.0),
            is_default=d.pop("is_default", d.pop("default", False)),
            max_retries=d.pop("max_retries", None),
            reasoning_effort=d.pop("reasoning_effort", None),
            reasoning_summary=(bool(d.pop("reasoning_summary")) if "reasoning_summary" in d else None),
        )

    def to_dict(self) -> dict:
        d: dict = {"model": self.model}
        if self.extra_headers:
            d["extra_headers"] = self.extra_headers
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        if self.context_window is not None:
            d["context_window"] = self.context_window
        if self.timeout != 300.0:
            d["timeout"] = self.timeout
        if self.is_default:
            d["is_default"] = True
        if self.max_retries is not None:
            d["max_retries"] = self.max_retries
        if self.reasoning_effort is not None:
            d["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_summary is not None:
            d["reasoning_summary"] = bool(self.reasoning_summary)
        return d


@dataclass
class ProviderEntry:
    """Configuration for one provider backend with one or more models.

    The ``type`` field selects the provider class:
      - ``"anthropic"``  → :class:`AnthropicProvider`
      - ``"openai"``     → :class:`OpenAIProvider`
      - ``"litellm"``    → :class:`LiteLLMProvider`
    """

    models: list[ModelEntry] = field(default_factory=list)
    type: str = "anthropic"
    api_key: str | None = None
    api_base: str | None = None
    default_headers: dict[str, str] | None = None
    max_retries: int = 5
    max_cooldown: float = 60.0  # seconds; caps all per-model cooldowns
    # Anthropic extended-thinking controls.
    extended_thinking: bool = False
    thinking_budget_tokens: int = 10_000
    # Optional provider-level reasoning effort fallback (LiteLLM/OpenAI).
    reasoning_effort: str | None = None
    reasoning_summary: bool | None = None

    @classmethod
    def from_dict(cls, d: dict) -> ProviderEntry:
        d = dict(d)
        raw_models = d.pop("models", [])
        models = [ModelEntry.from_dict(m) if isinstance(m, dict) else m for m in raw_models]
        raw_budget = d.pop("thinking_budget_tokens", 10_000)
        try:
            budget = int(raw_budget)
        except Exception:
            budget = 10_000
        # Shorthand: top-level "model" key with no "models" list
        if not models and "model" in d:
            models = [ModelEntry(model=d.pop("model"))]
        return cls(
            models=models,
            type=d.pop("type", "anthropic"),
            api_key=d.pop("api_key", None),
            api_base=d.pop("api_base", None),
            default_headers=d.pop("default_headers", None),
            max_retries=d.pop("max_retries", 5),
            max_cooldown=d.pop("max_cooldown", 60.0),
            extended_thinking=bool(d.pop("extended_thinking", False)),
            thinking_budget_tokens=budget,
            reasoning_effort=d.pop("reasoning_effort", None),
            reasoning_summary=(bool(d.pop("reasoning_summary")) if "reasoning_summary" in d else None),
        )

    def to_dict(self, include_credentials: bool = False) -> dict:
        d: dict = {
            "type": self.type,
            "models": [m.to_dict() for m in self.models],
        }
        if self.api_base:
            d["api_base"] = self.api_base
        if self.default_headers:
            d["default_headers"] = self.default_headers
        if self.max_retries != 5:
            d["max_retries"] = self.max_retries
        if self.max_cooldown != 60.0:
            d["max_cooldown"] = self.max_cooldown
        if self.extended_thinking:
            d["extended_thinking"] = True
        if self.thinking_budget_tokens != 10_000:
            d["thinking_budget_tokens"] = self.thinking_budget_tokens
        if self.reasoning_effort is not None:
            d["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_summary is not None:
            d["reasoning_summary"] = bool(self.reasoning_summary)
        if include_credentials and self.api_key:
            d["api_key"] = self.api_key
        return d

    def build_provider(self, model_entry: ModelEntry) -> BaseModelProvider:
        """Instantiate the right provider class for *model_entry*."""
        headers = {**(self.default_headers or {}), **(model_entry.extra_headers or {})}
        extra: dict = {}
        if model_entry.temperature is not None:
            extra["temperature"] = model_entry.temperature
        if model_entry.max_tokens is not None:
            extra["max_tokens"] = model_entry.max_tokens

        if self.type == "anthropic":
            from .anthropic_provider import AnthropicProvider

            return AnthropicProvider(
                model=model_entry.model,
                api_key=self.api_key,
                base_url=self.api_base,
                default_headers=headers or None,
                timeout=model_entry.timeout,
                extended_thinking=self.extended_thinking,
                thinking_budget_tokens=self.thinking_budget_tokens,
                **extra,
            )
        if self.type == "openai":
            from .openai_provider import OpenAIProvider

            if self.api_key:
                extra["api_key"] = self.api_key
            if self.api_base:
                extra["base_url"] = self.api_base
            if headers:
                extra["extra_headers"] = headers
            effort = model_entry.reasoning_effort or self.reasoning_effort
            if effort:
                extra["reasoning_effort"] = effort
            summary = (
                model_entry.reasoning_summary if model_entry.reasoning_summary is not None else self.reasoning_summary
            )
            if summary is not None:
                extra["reasoning_summary"] = bool(summary)
            return OpenAIProvider(
                model=model_entry.model,
                **extra,
            )

        # "litellm"
        from .litellm_provider import LiteLLMProvider

        if self.api_key:
            extra["api_key"] = self.api_key
        if self.api_base:
            extra["api_base"] = self.api_base
        if headers:
            extra["extra_headers"] = headers
        effort = model_entry.reasoning_effort or self.reasoning_effort
        if effort:
            extra["reasoning_effort"] = effort
        summary = model_entry.reasoning_summary if model_entry.reasoning_summary is not None else self.reasoning_summary
        if summary is not None:
            extra["reasoning_summary"] = bool(summary)
        return LiteLLMProvider(
            model=model_entry.model,
            timeout=model_entry.timeout,
            **extra,
        )
