# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from harnessx.core.builder import HarnessBuilder
from harnessx.core.harness import HarnessConfig
from harnessx.core.model_config import ModelConfig
from harnessx.plugins.builtins.slash_processor import SlashCommandProcessor
from harnessx.providers.unconfigured import UnConfiguredProvider


def _flatten_processors(cfg) -> list[object]:
    from harnessx.core.harness import _instantiate_runtime

    rt = _instantiate_runtime(cfg)
    out: list[object] = []
    for proc_list in rt.processors.values():
        out.extend(proc_list)
    return out


class TestBuilderMetadataAndRuntimeFields:
    def test_harness_config_ignores_metadata_keys(self) -> None:
        cfg = HarnessConfig(
            processors=[
                {
                    "_target_": "harnessx.plugins.builtins.slash_processor.SlashCommandProcessor",
                    "_code_hash": "sha256:deadbeefdeadbeef",
                }
            ]
        )

        assert any(isinstance(p, SlashCommandProcessor) for p in _flatten_processors(cfg))

    def test_exported_harness_config_does_not_embed_model_config(self) -> None:
        mc = ModelConfig(main=UnConfiguredProvider("main"))
        cfg = HarnessBuilder().add(SlashCommandProcessor(model_config=mc)).build()

        processors = [p for p in cfg.processors if isinstance(p, dict)]
        slash = next(p for p in processors if p.get("_target_", "").endswith(".SlashCommandProcessor"))

        assert "model_config" not in slash

    def test_serialize_processor_skips_local_classes(self) -> None:
        from harnessx.core.harness import _serialize_processor

        class _LocalProc:
            def __init__(self, x: int = 1):
                self.x = x

        assert _serialize_processor(_LocalProc()) is None
