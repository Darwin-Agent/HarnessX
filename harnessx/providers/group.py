# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Callable

from .agentic import AgenticMixin
from .base import BaseModelProvider
from .spec import (
    ErrorClass,
    ProviderEntry,
    _BACKOFF,
    _RETRY_POLICY,
    classify_error,
)

if TYPE_CHECKING:
    from ..core.events import Message, ModelResponseEvent, ToolSchema

_log = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────


class AllProvidersExhaustedError(Exception):
    """All models in the ProviderGroup have been tried and failed.

    ``tried_models`` lists the models that were attempted (in order).
    ``errors`` holds the corresponding exception for each attempt, allowing
    callers to inspect root causes.
    """

    def __init__(self, tried_models: list[str], errors: list[Exception]) -> None:
        self.tried_models = tried_models
        self.errors = errors
        detail = " → ".join(f"{m}({type(e).__name__})" for m, e in zip(tried_models, errors))
        super().__init__(f"All providers exhausted. Attempted: {detail or '(none)'}")


class _FallbackSignal(Exception):
    """Internal: current model failed, try the next one in the chain."""

    def __init__(
        self,
        error_class: ErrorClass,
        original: Exception,
        *,
        mark_entry_failed: bool = False,
    ) -> None:
        self.error_class = error_class
        self.original = original
        self.mark_entry_failed = mark_entry_failed


# ── Runtime wrappers ──────────────────────────────────────────────────────────


class _ModelRuntime:
    """Wraps one (provider, model) pair with cooldown tracking and retry logic."""

    def __init__(
        self,
        provider: BaseModelProvider,
        model_name: str,
        max_retries: int,
        max_cooldown: float,
        is_default: bool = False,
    ) -> None:
        self._provider = provider
        self.model_name = model_name
        self._max_retries = max_retries
        self._max_cooldown = max_cooldown
        self.is_default = is_default
        self._cooldown_until: float = 0.0

    def is_cooling(self) -> bool:
        return time.monotonic() < self._cooldown_until

    def set_cooldown(self, seconds: float) -> None:
        self._cooldown_until = time.monotonic() + min(seconds, self._max_cooldown)

    async def try_complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        stream_callback=None,
    ) -> ModelResponseEvent:
        """Attempt completion with per-error retry policy.

        Raises:
            _FallbackSignal: retries exhausted or non-retryable error — caller
                             should try the next model in the chain.
            Exception:       non-fallbackable error (e.g. ContextExceeded) that
                             should propagate immediately.
        """
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return await self._provider.complete(messages, tools, stream_callback=stream_callback)

            except Exception as exc:
                ec = classify_error(exc)
                policy = _RETRY_POLICY[ec]
                last_exc = exc

                if not policy.fallback:
                    raise  # e.g. CONTEXT_EXCEEDED — propagate, no fallback

                if policy.mark_entry_failed:
                    raise _FallbackSignal(ec, exc, mark_entry_failed=True)

                retries_allowed = min(self._max_retries, policy.max_retries)
                if attempt < retries_allowed:
                    delay = min(
                        _BACKOFF[min(attempt, len(_BACKOFF) - 1)],
                        self._max_cooldown,
                    )
                    _log.warning(
                        "Provider retry %d/%d model=%s (%s): %s",
                        attempt + 1,
                        retries_allowed,
                        self.model_name,
                        ec.value,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    continue

                # Retries exhausted
                if ec == ErrorClass.RATE_LIMIT:
                    self.set_cooldown(self._max_cooldown)
                raise _FallbackSignal(ec, last_exc)  # type: ignore[arg-type]

        raise _FallbackSignal(
            ErrorClass.UNKNOWN,
            last_exc or RuntimeError("unexpected retry loop exit"),
        )


class _ProviderEntryRuntime:
    """Groups model runtimes for one ProviderEntry; tracks auth-level failures."""

    def __init__(self, models: list[_ModelRuntime]) -> None:
        self._models = models
        self.entry_failed = False  # set True on AUTH_ERROR — skip all models

    def available_models(self) -> list[_ModelRuntime]:
        if self.entry_failed:
            return []
        return [m for m in self._models if not m.is_cooling()]


def _build_entry_runtime(pentry: ProviderEntry) -> _ProviderEntryRuntime:
    runtimes: list[_ModelRuntime] = []
    # Default models first, then others in declared order
    ordered = sorted(pentry.models, key=lambda m: (not m.is_default,))
    for mentry in ordered:
        provider = pentry.build_provider(mentry)
        model_max_retries = mentry.max_retries if mentry.max_retries is not None else pentry.max_retries
        runtimes.append(
            _ModelRuntime(
                provider=provider,
                model_name=mentry.model,
                max_retries=model_max_retries,
                max_cooldown=pentry.max_cooldown,
                is_default=mentry.is_default,
            )
        )
    return _ProviderEntryRuntime(runtimes)


# ── ProviderGroup ─────────────────────────────────────────────────────────────


class ProviderGroup(AgenticMixin, BaseModelProvider):
    """Composite provider that tries models in order, falling back on failure.

    Accepts three entry formats — mix freely::

        ProviderGroup([
            # 1. Full config (recommended for production)
            ProviderEntry(
                type="anthropic",
                api_key="sk-ant-...",
                models=[
                    ModelEntry("claude-sonnet-4-6", temperature=0.7, is_default=True),
                    ModelEntry("claude-haiku-4-5"),
                ],
            ),
            # 2. Dict shorthand (YAML-friendly)
            {"type": "openai", "models": [{"model": "gpt-4o", "default": True}]},
            # 3. Existing provider instance (compatibility)
            LiteLLMProvider("openrouter/meta-llama/llama-3.3-70b-instruct"),
        ])

    Args:
        entries:       Ordered fallback list. Each entry is a ProviderEntry,
                       a dict (passed to ProviderEntry.from_dict), or a bare
                       BaseModelProvider instance.
        max_retries:   Default retry limit per model (overridden by ModelEntry.max_retries).
        max_cooldown:  Maximum cooldown seconds after a rate-limit failure (default 60).
        on_fallback:   Optional callback invoked when a fallback occurs:
                       ``on_fallback(from_model, to_model, reason_str)``.
                       Use this to surface notices to interactive users.
    """

    def __init__(
        self,
        entries: list[ProviderEntry | BaseModelProvider | dict],
        *,
        max_retries: int = 5,
        max_cooldown: float = 60.0,
        on_fallback: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self._max_retries = max_retries
        self._max_cooldown = max_cooldown
        self._on_fallback = on_fallback
        self._entry_runtimes: list[_ProviderEntryRuntime] = []
        # Store serializable configs for YAML round-trip; None for bare provider instances
        self._pentry_configs: list[ProviderEntry | None] = []

        for entry in entries:
            if isinstance(entry, dict):
                pentry = ProviderEntry.from_dict(entry)
                self._entry_runtimes.append(_build_entry_runtime(pentry))
                self._pentry_configs.append(pentry)
            elif isinstance(entry, ProviderEntry):
                self._entry_runtimes.append(_build_entry_runtime(entry))
                self._pentry_configs.append(entry)
            else:
                # Bare BaseModelProvider instance
                model_name = getattr(entry, "model", type(entry).__name__)
                runtime = _ModelRuntime(
                    provider=entry,
                    model_name=model_name,
                    max_retries=max_retries,
                    max_cooldown=max_cooldown,
                    is_default=True,
                )
                self._entry_runtimes.append(_ProviderEntryRuntime([runtime]))
                self._pentry_configs.append(None)

    # ── BaseModelProvider interface ───────────────────────────────────────────

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        stream_callback=None,
    ) -> ModelResponseEvent:
        tried_models: list[str] = []
        aggregated_errors: list[Exception] = []

        for entry_rt in self._entry_runtimes:
            for model_rt in entry_rt.available_models():
                try:
                    result = await model_rt.try_complete(messages, tools, stream_callback=stream_callback)

                    if tried_models:
                        # Fallback occurred — annotate and notify
                        reason = aggregated_errors[-1].__class__.__name__ if aggregated_errors else "unknown"
                        _log.warning(
                            "Provider fallback: %s → %s (%s)",
                            " → ".join(tried_models),
                            result.model,
                            reason,
                        )
                        if self._on_fallback:
                            self._on_fallback(tried_models[0], result.model, reason)
                        result = dataclasses.replace(
                            result,
                            attempted_models=tuple(tried_models),
                        )
                    return result

                except _FallbackSignal as fs:
                    tried_models.append(model_rt.model_name)
                    aggregated_errors.append(fs.original)
                    _log.warning(
                        "Provider %s failed (%s), trying next",
                        model_rt.model_name,
                        fs.error_class.value,
                    )
                    if fs.mark_entry_failed:
                        entry_rt.entry_failed = True
                        break  # skip remaining models of this ProviderEntry
                    continue

                except Exception:
                    # Non-fallbackable (e.g. CONTEXT_EXCEEDED) — propagate immediately
                    raise

        raise AllProvidersExhaustedError(tried_models, aggregated_errors)

    def count_tokens(self, messages: list[Message]) -> int:
        """Estimate tokens using the first available (non-cooling) provider."""
        for entry_rt in self._entry_runtimes:
            for model_rt in entry_rt.available_models():
                return model_rt._provider.count_tokens(messages)
        # All cooling — fall back to first runtime regardless
        for entry_rt in self._entry_runtimes:
            if entry_rt._models:
                return entry_rt._models[0]._provider.count_tokens(messages)
        from ._utils import count_tokens

        return count_tokens(messages)

    @property
    def model(self) -> str:
        """Primary model name (first default, or first model overall)."""
        for entry_rt in self._entry_runtimes:
            for m in entry_rt._models:
                if m.is_default:
                    return m.model_name
            if entry_rt._models:
                return entry_rt._models[0].model_name
        return "unknown"

    def to_dict(self, include_credentials: bool = False) -> dict:
        """Serialize to a plain dict suitable for YAML round-trip.

        ``include_credentials=False`` (default) omits ``api_key`` fields.
        Bare provider instances (passed as ``BaseModelProvider`` objects rather
        than ``ProviderEntry``) are represented as ``{"_bare": true, "model": name}``.
        """
        entries: list[dict] = []
        for pentry, entry_rt in zip(self._pentry_configs, self._entry_runtimes):
            if pentry is None:
                # Bare provider instance — record model name only
                model_name = entry_rt._models[0].model_name if entry_rt._models else "unknown"
                entries.append({"_bare": True, "model": model_name})
            else:
                entries.append(pentry.to_dict(include_credentials=include_credentials))
        d: dict = {"entries": entries}
        if self._max_retries != 5:
            d["max_retries"] = self._max_retries
        if self._max_cooldown != 60.0:
            d["max_cooldown"] = self._max_cooldown
        return d
