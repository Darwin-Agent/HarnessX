# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations


from harnessx.providers.spec import ModelEntry, ProviderEntry
from harnessx.providers.group import ProviderGroup


# ── ModelEntry.to_dict ────────────────────────────────────────────────────────


def _fake_provider(model_name: str):
    class FakeProvider:
        model = model_name

        def count_tokens(self, messages):
            return 10

    return FakeProvider()


class TestProviderSerialization:
    def test_model_entry_to_dict_minimal(self):
        me = ModelEntry(model="claude-sonnet-4-6")
        d = me.to_dict()
        assert d == {"model": "claude-sonnet-4-6"}

    def test_model_entry_to_dict_full(self):
        me = ModelEntry(
            model="claude-sonnet-4-6",
            temperature=0.7,
            max_tokens=4096,
            context_window=200_000,
            timeout=120.0,
            is_default=True,
            max_retries=3,
            extra_headers={"X-Custom": "val"},
        )
        d = me.to_dict()
        assert d["model"] == "claude-sonnet-4-6"
        assert d["temperature"] == 0.7
        assert d["max_tokens"] == 4096
        assert d["context_window"] == 200_000
        assert d["timeout"] == 120.0
        assert d["is_default"] is True
        assert d["max_retries"] == 3
        assert d["extra_headers"] == {"X-Custom": "val"}

    def test_model_entry_to_dict_omits_defaults(self):
        """Fields with default values should be omitted from the output dict."""
        me = ModelEntry(model="gpt-4o", timeout=300.0, is_default=False)
        d = me.to_dict()
        assert "timeout" not in d
        assert "is_default" not in d
        assert "temperature" not in d
        assert "max_tokens" not in d

    def test_model_entry_round_trip(self):
        me = ModelEntry(
            model="claude-haiku-4-5",
            temperature=0.5,
            is_default=True,
            max_retries=2,
        )
        restored = ModelEntry.from_dict(me.to_dict())
        assert restored.model == me.model
        assert restored.temperature == me.temperature
        assert restored.is_default == me.is_default
        assert restored.max_retries == me.max_retries

    # ── ProviderEntry.to_dict ─────────────────────────────────────────────────────

    def test_provider_entry_to_dict_minimal(self):
        pe = ProviderEntry(
            type="anthropic",
            models=[ModelEntry(model="claude-sonnet-4-6")],
        )
        d = pe.to_dict()
        assert d["type"] == "anthropic"
        assert len(d["models"]) == 1
        assert d["models"][0]["model"] == "claude-sonnet-4-6"
        assert "api_key" not in d  # credentials omitted by default
        assert "api_base" not in d

    def test_provider_entry_to_dict_omits_credential(self):
        pe = ProviderEntry(
            type="anthropic",
            api_key="sk-ant-secret",
            models=[ModelEntry(model="claude-sonnet-4-6")],
        )
        d = pe.to_dict(include_credentials=False)
        assert "api_key" not in d

    def test_provider_entry_to_dict_includes_credential_when_asked(self):
        pe = ProviderEntry(
            type="anthropic",
            api_key="sk-ant-secret",
            models=[ModelEntry(model="claude-sonnet-4-6")],
        )
        d = pe.to_dict(include_credentials=True)
        assert d["api_key"] == "sk-ant-secret"

    def test_provider_entry_to_dict_omits_defaults(self):
        """max_retries=5 and max_cooldown=60.0 are defaults — should be omitted."""
        pe = ProviderEntry(
            type="anthropic",
            models=[ModelEntry(model="m")],
            max_retries=5,
            max_cooldown=60.0,
        )
        d = pe.to_dict()
        assert "max_retries" not in d
        assert "max_cooldown" not in d

    def test_provider_entry_to_dict_includes_non_defaults(self):
        pe = ProviderEntry(
            type="openai",
            api_base="https://custom.api.example.com/v1",
            models=[ModelEntry(model="gpt-4o")],
            max_retries=3,
            max_cooldown=30.0,
            default_headers={"X-Org": "acme"},
        )
        d = pe.to_dict()
        assert d["api_base"] == "https://custom.api.example.com/v1"
        assert d["max_retries"] == 3
        assert d["max_cooldown"] == 30.0
        assert d["default_headers"] == {"X-Org": "acme"}

    def test_provider_entry_round_trip(self):
        pe = ProviderEntry(
            type="anthropic",
            api_base="https://proxy.example.com",
            models=[
                ModelEntry(model="claude-sonnet-4-6", is_default=True, temperature=0.7),
                ModelEntry(model="claude-haiku-4-5"),
            ],
            max_retries=3,
        )
        restored = ProviderEntry.from_dict(pe.to_dict())
        assert restored.type == pe.type
        assert restored.api_base == pe.api_base
        assert len(restored.models) == 2
        assert restored.models[0].model == "claude-sonnet-4-6"
        assert restored.models[0].is_default is True
        assert restored.models[0].temperature == 0.7
        assert restored.models[1].model == "claude-haiku-4-5"
        assert restored.max_retries == 3

    # ── ProviderGroup.to_dict ─────────────────────────────────────────────────────

    def test_provider_group_to_dict_from_provider_entry(self):
        pe = ProviderEntry(
            type="anthropic",
            models=[
                ModelEntry(model="claude-sonnet-4-6", is_default=True),
                ModelEntry(model="claude-haiku-4-5"),
            ],
        )
        group = ProviderGroup.__new__(ProviderGroup)
        group._max_retries = 5
        group._max_cooldown = 60.0
        group._on_fallback = None
        # Simulate __init__ side-effects

        group._entry_runtimes = []
        group._pentry_configs = []

        # Use real __init__ path instead
        group = ProviderGroup([pe])
        d = group.to_dict()
        assert "entries" in d
        assert len(d["entries"]) == 1
        entry = d["entries"][0]
        assert entry["type"] == "anthropic"
        assert len(entry["models"]) == 2
        assert entry["models"][0]["model"] == "claude-sonnet-4-6"
        assert entry["models"][0]["is_default"] is True

    def test_provider_group_to_dict_omits_defaults(self):
        pe = ProviderEntry(type="anthropic", models=[ModelEntry(model="m")])
        group = ProviderGroup([pe])
        d = group.to_dict()
        assert "max_retries" not in d
        assert "max_cooldown" not in d

    def test_provider_group_to_dict_bare_provider(self):
        """Bare provider instance serializes with _bare=True."""
        p = _fake_provider("my-model")
        group = ProviderGroup([p])
        d = group.to_dict()
        assert d["entries"][0]["_bare"] is True
        assert d["entries"][0]["model"] == "my-model"

    def test_provider_group_to_dict_multi_entry(self):
        pe1 = ProviderEntry(
            type="anthropic",
            models=[ModelEntry(model="claude-sonnet-4-6", is_default=True)],
            api_base="https://proxy-a.example.com",
        )
        pe2 = ProviderEntry(
            type="openai",
            models=[ModelEntry(model="gpt-4o")],
        )
        group = ProviderGroup([pe1, pe2])
        d = group.to_dict()
        assert len(d["entries"]) == 2
        assert d["entries"][0]["type"] == "anthropic"
        assert d["entries"][0]["api_base"] == "https://proxy-a.example.com"
        assert d["entries"][1]["type"] == "openai"

    # ── ModelConfig _target_ dict round-trip ──────────────────────────────────────

    def test_model_config_provider_group_from_dict(self):
        """ModelConfig.from_dict() with v1 format instantiates a ProviderGroup."""
        from harnessx.core.model_config import ModelConfig

        # v1 format: top-level key is role name, value is _target_ spec
        model_cfg = {
            "main": {
                "_target_": "harnessx.providers.group.ProviderGroup",
                "entries": [
                    {
                        "type": "anthropic",
                        "models": [
                            {"model": "claude-sonnet-4-6", "is_default": True},
                            {"model": "claude-haiku-4-5"},
                        ],
                    }
                ],
            }
        }
        mc = ModelConfig.from_dict(model_cfg)
        assert isinstance(mc.main, ProviderGroup)
        assert mc.main.model == "claude-sonnet-4-6"

    def test_model_config_multi_entry_provider_group(self):
        """Multi-provider ProviderGroup correctly selects first entry as main."""
        from harnessx.core.model_config import ModelConfig

        model_cfg = {
            "main": {
                "_target_": "harnessx.providers.group.ProviderGroup",
                "entries": [
                    {
                        "type": "anthropic",
                        "models": [{"model": "claude-sonnet-4-6", "is_default": True}],
                    },
                    {"type": "openai", "models": [{"model": "gpt-4o"}]},
                ],
            }
        }
        mc = ModelConfig.from_dict(model_cfg)
        assert isinstance(mc.main, ProviderGroup)
        assert mc.main.model == "claude-sonnet-4-6"
