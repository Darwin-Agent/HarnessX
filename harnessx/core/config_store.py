# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations


def register_harnessx_configs() -> None:
    """Register all HarnessX structured configs into the Hydra ConfigStore.

    Call once at CLI / API entry points. Safe to call multiple times (idempotent).
    Not called automatically to avoid side-effects in non-Hydra applications.
    """
    from hydra.core.config_store import ConfigStore

    from .config_schema import (
        NullTracerConfig,
        PluginConfig,
        SandboxConfig,
        ToolRegistryConfig,
        TracerConfig,
        WorkspaceConfig,
    )
    from .harness import HarnessConfig

    cs = ConfigStore.instance()
    cs.store(name="journal", node=TracerConfig, group="harness/tracer")
    cs.store(name="null", node=NullTracerConfig, group="harness/tracer")
    cs.store(name="default", node=WorkspaceConfig, group="harness/workspace")
    cs.store(name="default", node=ToolRegistryConfig, group="harness/tool_registry")
    cs.store(name="local", node=SandboxConfig, group="harness/sandbox")
    cs.store(name="base", node=PluginConfig, group="harness/plugin")
    cs.store(name="base", node=HarnessConfig, group="harness")
