# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from harnessx.providers.spec import (
    ErrorClass,
    ModelEntry,
    ProviderEntry,
    classify_error,
)
from harnessx.providers.group import (
    AllProvidersExhaustedError,
    ProviderGroup,
    _ModelRuntime,
    _ProviderEntryRuntime,
)
from harnessx.core.events import ModelResponseEvent, Usage
from harnessx.core.model_config import ModelConfig


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_response(model: str = "test-model") -> ModelResponseEvent:
    return ModelResponseEvent(
        run_id="r",
        step_id=0,
        content="ok",
        model=model,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _fake_provider(responses):
    """Build a mock provider that yields responses/exceptions in sequence."""
    calls = list(responses)
    call_index = [0]

    class FakeProvider:
        model = "fake-model"

        async def complete(self, messages, tools, stream_callback=None, **kwargs):
            i = call_index[0]
            call_index[0] += 1
            r = calls[min(i, len(calls) - 1)]
            if isinstance(r, Exception):
                raise r
            return r

        def count_tokens(self, messages):
            return 10

    return FakeProvider()


# Fake exceptions that classify_error recognises
class _AnthropicRateLimit(Exception):
    pass


_AnthropicRateLimit.__module__ = "anthropic"
_AnthropicRateLimit.__qualname__ = "RateLimitError"


# Override __class__.__name__ via a subclass trick
class _ARL(_AnthropicRateLimit):
    pass


_ARL.__name__ = "RateLimitError"
_ARL.__module__ = "anthropic"


class _AnthropicAuth(Exception):
    pass


_AnthropicAuth.__name__ = "AuthenticationError"
_AnthropicAuth.__module__ = "anthropic"


class _AnthropicContext(Exception):
    pass


_AnthropicContext.__name__ = "BadRequestError"
_AnthropicContext.__module__ = "anthropic"


class _AnthropicServer(Exception):
    pass


_AnthropicServer.__name__ = "InternalServerError"
_AnthropicServer.__module__ = "anthropic"


class _AnthropicTimeout(Exception):
    pass


_AnthropicTimeout.__name__ = "APITimeoutError"
_AnthropicTimeout.__module__ = "anthropic"


# ── classify_error ────────────────────────────────────────────────────────────


class TestProviderGroup:
    def test_classify_rate_limit(self):
        assert classify_error(_ARL()) == ErrorClass.RATE_LIMIT

    def test_classify_auth_error(self):
        assert classify_error(_AnthropicAuth()) == ErrorClass.AUTH_ERROR

    def test_classify_context_exceeded(self):
        exc = _AnthropicContext("maximum context length exceeded")
        assert classify_error(exc) == ErrorClass.CONTEXT_EXCEEDED

    def test_classify_server_error(self):
        assert classify_error(_AnthropicServer()) == ErrorClass.SERVER_ERROR

    def test_classify_timeout(self):
        assert classify_error(_AnthropicTimeout()) == ErrorClass.TIMEOUT

    def test_classify_asyncio_timeout(self):
        assert classify_error(asyncio.TimeoutError()) == ErrorClass.TIMEOUT

    def test_classify_unknown(self):
        assert classify_error(ValueError("something weird")) == ErrorClass.UNKNOWN

    # ── Basic fallback ────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_primary_provider_success(self):
        p = _fake_provider([_make_response("claude")])
        group = ProviderGroup([p])
        result = await group.complete([], [])
        assert result.model == "claude"
        assert result.attempted_models == ()

    @pytest.mark.asyncio
    async def test_fallback_to_second_provider(self):
        """First provider raises SERVER_ERROR; second should be tried."""
        p1 = _fake_provider([_AnthropicServer()])
        p2 = _fake_provider([_make_response("gpt-4o")])
        p2.model = "gpt-4o"

        with patch("asyncio.sleep"):
            group = ProviderGroup([p1, p2], max_retries=0)
            result = await group.complete([], [])

        assert result.model == "gpt-4o"
        assert result.attempted_models == ("fake-model",)

    @pytest.mark.asyncio
    async def test_all_providers_exhausted_raises(self):
        """When every provider fails, AllProvidersExhaustedError includes aggregated errors."""
        p1 = _fake_provider([_AnthropicServer()])
        p2 = _fake_provider([_AnthropicServer()])

        with patch("asyncio.sleep"):
            group = ProviderGroup([p1, p2], max_retries=0)
            with pytest.raises(AllProvidersExhaustedError) as exc_info:
                await group.complete([], [])

        err = exc_info.value
        assert len(err.tried_models) == 2
        assert len(err.errors) == 2
        assert all(isinstance(e, _AnthropicServer) for e in err.errors)

    # ── Context exceeded — no fallback ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_context_exceeded_propagates_immediately(self):
        """CONTEXT_EXCEEDED must raise immediately without trying fallback providers."""
        p1 = _fake_provider([_AnthropicContext("maximum context length exceeded")])
        p2 = _fake_provider([_make_response("fallback")])

        group = ProviderGroup([p1, p2], max_retries=0)
        with pytest.raises(_AnthropicContext):
            await group.complete([], [])

    # ── Auth error skips entire ProviderEntry ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_auth_error_skips_all_models_of_entry(self):
        """AUTH_ERROR on any model should mark the entire ProviderEntry as failed."""
        resp = _make_response("haiku")
        p_sonnet = _fake_provider([_AnthropicAuth()])
        p_haiku = _fake_provider([resp])  # same entry — should NOT be tried
        p_fallback = _fake_provider([_make_response("gpt-4o")])
        p_fallback.model = "gpt-4o"

        # Build two entries manually: first entry has two models, second is the fallback

        m_sonnet = _ModelRuntime(p_sonnet, "claude-sonnet", max_retries=0, max_cooldown=60)
        m_haiku = _ModelRuntime(p_haiku, "claude-haiku", max_retries=0, max_cooldown=60)
        entry1 = _ProviderEntryRuntime([m_sonnet, m_haiku])
        m_fallback = _ModelRuntime(p_fallback, "gpt-4o", max_retries=0, max_cooldown=60, is_default=True)
        entry2 = _ProviderEntryRuntime([m_fallback])

        group = ProviderGroup.__new__(ProviderGroup)
        group._max_retries = 0
        group._max_cooldown = 60.0
        group._on_fallback = None
        group._entry_runtimes = [entry1, entry2]

        result = await group.complete([], [])

        assert result.model == "gpt-4o"
        assert entry1.entry_failed is True
        # claude-haiku should NOT have been called
        assert p_haiku.model == "fake-model"

    # ── Rate limit: retry then fallback ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_rate_limit_retries_before_fallback(self):
        """Rate limit errors should be retried max_retries times before fallback."""
        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        # p1 always raises rate limit; p2 succeeds
        p1 = _fake_provider([_ARL()] * 10)
        p2 = _fake_provider([_make_response("fallback")])
        p2.model = "fallback"

        with patch("harnessx.providers.group.asyncio.sleep", new=fake_sleep):
            group = ProviderGroup([p1, p2], max_retries=3)
            result = await group.complete([], [])

        assert result.model == "fallback"
        # 3 retries means 3 sleep calls
        assert len(sleep_calls) == 3

    @pytest.mark.asyncio
    async def test_rate_limit_sets_cooldown(self):
        """After rate-limit retries exhausted, the model should be in cooldown."""
        p1 = _fake_provider([_ARL()] * 10)
        _p2 = _fake_provider([_make_response("fallback")])

        runtime = _ModelRuntime(p1, "p1", max_retries=0, max_cooldown=60.0)

        with patch("asyncio.sleep"):
            from harnessx.providers.group import _FallbackSignal

            with pytest.raises(_FallbackSignal):
                await runtime.try_complete([], [])

        assert runtime.is_cooling()

    # ── Cooldown: cooling model skipped ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_cooling_model_skipped(self):
        """A model in cooldown should not be tried; next model should be used."""
        p1 = _fake_provider([_make_response("p1")])
        p2 = _fake_provider([_make_response("p2")])
        p2.model = "p2"

        group = ProviderGroup([p1, p2])
        # Manually put p1's runtime into cooldown
        group._entry_runtimes[0]._models[0].set_cooldown(9999)

        result = await group.complete([], [])
        assert result.model == "p2"

    # ── on_fallback callback ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_on_fallback_callback_called(self):
        p1 = _fake_provider([_AnthropicServer()])
        p2 = _fake_provider([_make_response("gpt-4o")])
        p2.model = "gpt-4o"

        callback_args = []

        def on_fallback(from_m, to_m, reason):
            callback_args.append((from_m, to_m, reason))

        with patch("asyncio.sleep"):
            group = ProviderGroup([p1, p2], max_retries=0, on_fallback=on_fallback)
            await group.complete([], [])

        assert len(callback_args) == 1
        from_m, to_m, reason = callback_args[0]
        assert from_m == "fake-model"
        assert to_m == "gpt-4o"
        assert reason  # non-empty reason string

    @pytest.mark.asyncio
    async def test_no_callback_when_no_fallback(self):
        """Callback must NOT be called when primary provider succeeds."""
        p = _fake_provider([_make_response("primary")])
        called = []
        group = ProviderGroup([p], on_fallback=lambda *a: called.append(a))
        await group.complete([], [])
        assert called == []

    # ── attempted_models transparency ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_attempted_models_populated_on_fallback(self):
        p1 = _fake_provider([_AnthropicServer(), _AnthropicServer()])
        p1.model = "model-a"
        p2 = _fake_provider([_AnthropicServer()])
        p2.model = "model-b"
        p3 = _fake_provider([_make_response("model-c")])
        p3.model = "model-c"

        with patch("asyncio.sleep"):
            group = ProviderGroup([p1, p2, p3], max_retries=0)
            result = await group.complete([], [])

        assert result.model == "model-c"
        assert "model-a" in result.attempted_models
        assert "model-b" in result.attempted_models

    @pytest.mark.asyncio
    async def test_attempted_models_empty_on_success(self):
        p = _fake_provider([_make_response("primary")])
        group = ProviderGroup([p])
        result = await group.complete([], [])
        assert result.attempted_models == ()

    # ── ModelConfig + ProviderGroup ───────────────────────────────────────────────

    def test_model_config_accepts_provider_group(self):
        p1 = _fake_provider([_make_response("m1")])
        p2 = _fake_provider([_make_response("m2")])
        pg = ProviderGroup([p1, p2])
        model = ModelConfig(main=pg)
        assert model.main is pg
        assert isinstance(model.main, ProviderGroup)

    def test_model_config_single_provider_unchanged(self):
        p = _fake_provider([_make_response("m1")])
        model = ModelConfig(main=p)
        # Single provider stored as-is — no wrapping
        assert model.main is p
        assert not isinstance(model.main, ProviderGroup)

    # ── ProviderEntry / ModelEntry config ────────────────────────────────────────

    def test_model_entry_from_dict(self):
        me = ModelEntry.from_dict(
            {
                "model": "claude-sonnet-4-6",
                "temperature": 0.5,
                "is_default": True,
                "max_retries": 3,
            }
        )
        assert me.model == "claude-sonnet-4-6"
        assert me.temperature == 0.5
        assert me.is_default is True
        assert me.max_retries == 3

    def test_provider_entry_from_dict_shorthand(self):
        """Single-model shorthand: {"type": "anthropic", "model": "claude-sonnet-4-6"}"""
        pe = ProviderEntry.from_dict(
            {
                "type": "anthropic",
                "model": "claude-sonnet-4-6",
                "api_key": "sk-test",
            }
        )
        assert len(pe.models) == 1
        assert pe.models[0].model == "claude-sonnet-4-6"
        assert pe.api_key == "sk-test"

    def test_provider_entry_from_dict_multi_model(self):
        pe = ProviderEntry.from_dict(
            {
                "type": "anthropic",
                "models": [
                    {"model": "claude-sonnet-4-6", "default": True},
                    {"model": "claude-haiku-4-5"},
                ],
            }
        )
        assert len(pe.models) == 2
        assert pe.models[0].is_default is True
        assert pe.models[1].model == "claude-haiku-4-5"

    def test_default_model_ordered_first(self):
        """Default model should be tried first regardless of declaration order."""
        from harnessx.providers.group import _build_entry_runtime

        pe = ProviderEntry(
            type="anthropic",
            models=[
                ModelEntry("haiku", is_default=False),
                ModelEntry("sonnet", is_default=True),
            ],
        )
        # Mock build_provider to avoid real instantiation
        built = []

        _original_build = pe.build_provider

        def mock_build(m):
            built.append(m.model)
            p = _fake_provider([_make_response(m.model)])
            p.model = m.model
            return p

        pe.build_provider = mock_build
        entry_rt = _build_entry_runtime(pe)

        models = [r.model_name for r in entry_rt._models]
        assert models[0] == "sonnet"  # default first
        assert models[1] == "haiku"

    # ── ProviderGroup.model property ─────────────────────────────────────────────

    def test_provider_group_model_property(self):
        p = _fake_provider([_make_response("primary")])
        p.model = "claude-sonnet-4-6"
        group = ProviderGroup([p])
        assert group.model == "claude-sonnet-4-6"

    def test_provider_group_model_prefers_default(self):
        m1 = _ModelRuntime(_fake_provider([]), "haiku", 5, 60, is_default=False)
        m2 = _ModelRuntime(_fake_provider([]), "sonnet", 5, 60, is_default=True)
        group = ProviderGroup.__new__(ProviderGroup)
        group._entry_runtimes = [_ProviderEntryRuntime([m1, m2])]
        group._on_fallback = None
        assert group.model == "sonnet"
