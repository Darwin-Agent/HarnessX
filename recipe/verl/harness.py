# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Build HarnessConfig for verl backend — delegates to harnessx.rl.builder."""

from __future__ import annotations

from typing import Any

from harnessx.core.harness import HarnessConfig
from harnessx.rl.builder import build_rl_harness_config
from harnessx.rl.task import RLTask
from recipe.verl.spec import VerlConfigSpec


def make_verl_harness(
    spec: VerlConfigSpec, provider: Any, task: RLTask, behavior: HarnessConfig | None = None
) -> HarnessConfig:
    """Build a per-run HarnessConfig from spec, provider, and task."""
    config = build_rl_harness_config(spec, provider, task)
    if behavior is None:
        return config
    from recipe.coevolution.behavior import validate_behavior
    from harnessx.processors.context.system_prompt import SystemPromptProcessor
    from recipe.verl.processor.early_commit import EarlyCommitProcessor

    validate_behavior(behavior)
    runtime = [
        processor
        for processor in config._rt_procs
        if not isinstance(processor, (SystemPromptProcessor, EarlyCommitProcessor))
    ]
    config.processors = list(behavior.processors)
    config._rt_procs = runtime
    config.init_workspace = False
    return config
