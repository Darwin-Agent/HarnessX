# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from harnessx.api.routes import run as run_route


class TestApiHarnessPayloadSanitize:
    def test_sanitize_harness_payload_drops_nested_model_config_specs(self) -> None:
        payload = {
            "processors": [
                {
                    "_target_": "pkg.SomeProcessor",
                    "name": "ok",
                    "model": {
                        "_target_": "harnessx.core.model_config.ModelConfig",
                    },
                    "nested": {
                        "x": 1,
                        "model_config": {
                            "models": [{"id": "m1"}],
                            "roles": {"main": {"default": "m1"}},
                        },
                    },
                }
            ],
            "plugins": None,
        }

        cleaned = run_route._sanitize_harness_config_payload(payload)
        p0 = cleaned["processors"][0]

        assert "model" not in p0
        assert "model_config" not in p0["nested"]
        assert p0["name"] == "ok"
        assert p0["nested"]["x"] == 1

    def test_sanitize_harness_payload_drops_local_cli_processor_targets(self) -> None:
        payload = {
            "processors": [
                {
                    "_target_": "harnessx.cli._chat.<locals>._CLIToolPrinter",
                    "_code_hash": "sha256:deadbeef",
                },
                {"_target_": "pkg.RealProcessor", "x": 1},
            ],
            "plugins": None,
        }

        cleaned = run_route._sanitize_harness_config_payload(payload)
        assert len(cleaned["processors"]) == 1
        assert cleaned["processors"][0]["_target_"] == "pkg.RealProcessor"
