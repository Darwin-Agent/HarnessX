# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from harnessx.rl.builder import build_rl_harness_config
from harnessx.core.harness import HarnessConfig

if TYPE_CHECKING:
    from recipe.slime.spec import SlimeConfigSpec
    from harnessx.rl.task import RLTask


def make_slime_harness(
    spec: "SlimeConfigSpec",
    provider: Any,
    task: "RLTask",
) -> HarnessConfig:
    """
    Build a per-run HarnessConfig for Slime RL training.

    Delegates to build_rl_harness_config() (framework-agnostic core),
    which composes via RLControlPlugin + HarnessBuilder pattern.

    Per-run means: evaluator is freshly instantiated for each Sample,
    closing over task.label.  All processors are also fresh so their
    internal state (fingerprints, counts) starts clean each episode.

    Args:
        spec:     SlimeConfigSpec from HARNESS_CONFIGS registry.
        provider: SGLangProvider configured with formatter_factory and
                  inter_turn_formatter_factory (wired in harness_rollout.py).
        task:     RLTask built by spec.task_builder.build(sample).

    Returns:
        HarnessConfig with step_snapshots=False and standard RL processor pipeline.
    """
    return build_rl_harness_config(spec, provider, task)
