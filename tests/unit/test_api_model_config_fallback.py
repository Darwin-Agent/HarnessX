# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from harnessx.api.routes import run as run_route


class TestApiModelConfigFallback:
    def test_has_main_provider_spec_v1_and_v2(self) -> None:
        assert run_route._has_main_provider_spec({"main": {"_target_": "x"}}) is True
        assert (
            run_route._has_main_provider_spec(
                {
                    "schema_version": 2,
                    "models": [{"id": "m1"}],
                    "roles": {"main": {"default": "m1"}},
                }
            )
            is True
        )
        assert run_route._has_main_provider_spec({"judge": {"_target_": "x"}}) is False
        assert run_route._has_main_provider_spec({}) is False

    def test_resolve_model_config_uses_request_when_main_present(self, monkeypatch) -> None:
        # Valid minimal v1 payload using a built-in placeholder provider.
        provider_config = {
            "main": {
                "_target_": "harnessx.providers.unconfigured.UnConfiguredProvider",
                "slot": "main",
            }
        }

        sentinel = object()
        monkeypatch.setattr(run_route, "_load_default_model_config", lambda: sentinel)

        mc = run_route._resolve_model_config(provider_config)
        assert mc is not sentinel
        assert mc.main.__class__.__name__ == "UnConfiguredProvider"

    def test_resolve_model_config_falls_back_when_main_missing(self, monkeypatch) -> None:
        sentinel = object()
        monkeypatch.setattr(run_route, "_load_default_model_config", lambda: sentinel)

        mc = run_route._resolve_model_config({"judge": {"_target_": "x"}})
        assert mc is sentinel

    def test_resolve_model_config_falls_back_when_from_dict_raises(self, monkeypatch) -> None:
        sentinel = object()
        monkeypatch.setattr(run_route, "_load_default_model_config", lambda: sentinel)

        from harnessx.core import model_config as model_config_mod

        def _raise(_cfg):
            raise ValueError("ModelConfig requires a 'main' provider.")

        monkeypatch.setattr(model_config_mod.ModelConfig, "from_dict", staticmethod(_raise))

        mc = run_route._resolve_model_config({"main": {"_target_": "broken"}})
        assert mc is sentinel
