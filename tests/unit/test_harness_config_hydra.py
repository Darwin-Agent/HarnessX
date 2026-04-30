# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations


from harnessx.core.config_schema import (
    NullTracerConfig,
    SandboxConfig,
    ToolRegistryConfig,
    TracerConfig,
    WorkspaceConfig,
)
from harnessx.core.harness import HarnessConfig, _instantiate_runtime, _serialize_plugin


class TestStructuredConfig:
    def test_structured_empty(self) -> None:
        from omegaconf import OmegaConf

        OmegaConf.structured(HarnessConfig())

    def test_structured_with_all_subconfigs(self) -> None:
        from omegaconf import OmegaConf

        config = HarnessConfig(
            tracer=TracerConfig(silent=True),
            workspace=WorkspaceConfig(agent_id="x"),
            tool_registry=ToolRegistryConfig(builtin=["Bash"]),
            sandbox_provider=SandboxConfig(),
        )
        yaml = OmegaConf.to_yaml(OmegaConf.structured(config))
        assert "silent: true" in yaml
        assert "agent_id: x" in yaml
        assert "Bash" in yaml

    def test_to_yaml_round_trip(self) -> None:
        config = HarnessConfig(
            tracer=TracerConfig(silent=True),
            workspace=WorkspaceConfig(agent_id="roundtrip"),
            tool_registry=ToolRegistryConfig(builtin=["Bash", "Read"]),
        )
        yaml = config.to_yaml()
        config2 = HarnessConfig.from_yaml(yaml)
        assert config2.tracer.silent is True
        assert config2.workspace.agent_id == "roundtrip"
        assert config2.tool_registry.builtin == ["Bash", "Read"]

    def test_to_yaml_with_processors(self) -> None:
        config = HarnessConfig(processors=[{"_target_": "foo.Bar", "k": 1}])
        yaml = config.to_yaml()
        assert "_target_: foo.Bar" in yaml
        config2 = HarnessConfig.from_yaml(yaml)
        assert config2.processors[0]["k"] == 1

    def test_to_yaml_with_plugins(self) -> None:
        config = HarnessConfig(plugins=[{"_target_": "my.module.MyPlugin", "window": 8}])
        yaml = config.to_yaml()
        config2 = HarnessConfig.from_yaml(yaml)
        assert config2.plugins[0]["window"] == 8

    def test_from_yaml_strips_model_section(self) -> None:
        yaml = "model:\n  _target_: foo.Bar\ntracer:\n  silent: true\n"
        config = HarnessConfig.from_yaml(yaml)
        assert config.tracer.silent is True
        assert not hasattr(config, "model")

    def test_null_tracer_round_trip(self) -> None:
        config = HarnessConfig(tracer=NullTracerConfig())
        yaml = config.to_yaml()
        assert "null_tracer.NullTracer" in yaml
        config2 = HarnessConfig.from_yaml(yaml)
        assert config2.tracer._target_ == "harnessx.tracing.null_tracer.NullTracer"

    def test_null_tracer_instantiates_correctly(self) -> None:
        from harnessx.tracing.null_tracer import NullTracer

        config = HarnessConfig(tracer=NullTracerConfig())
        rt = _instantiate_runtime(config)
        assert isinstance(rt.tracer, NullTracer)

    def test_config_store_registration(self) -> None:
        from harnessx.core.config_store import register_harnessx_configs

        register_harnessx_configs()
        register_harnessx_configs()

    def test_builder_produces_serializable_config(self) -> None:
        from omegaconf import OmegaConf
        from harnessx.core.builder import HarnessBuilder
        from harnessx.bundles import context

        config = (HarnessBuilder() | context).build()
        OmegaConf.structured(config)

    def test_serialize_plugin_with_kwargs(self) -> None:
        import dataclasses

        @dataclasses.dataclass
        class FakePlugin:
            window: int = 8

        FakePlugin.__module__ = "my.fake_module"
        FakePlugin.__qualname__ = "FakePlugin"

        plugin = FakePlugin(window=16)
        d = _serialize_plugin(plugin)
        assert d is not None
        assert d["_target_"].endswith("FakePlugin")
        assert d["window"] == 16

    def test_instantiate_runtime_restores_plugins(self) -> None:
        from harnessx.plugins.builtins.slash_processor import SlashCommandProcessor

        target = "harnessx.plugins.builtins.slash_processor.SlashCommandProcessor"
        config = HarnessConfig(plugins=[{"_target_": target}])
        rt = _instantiate_runtime(config)
        assert len(rt.plugins) == 1
        assert isinstance(rt.plugins[0], SlashCommandProcessor)
