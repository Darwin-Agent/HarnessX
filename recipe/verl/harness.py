# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Build HarnessConfig for verl backend — delegates to harnessx.rl.builder."""
from __future__ import annotations

from typing import Any

from harnessx.core.harness import HarnessConfig
from harnessx.rl.builder import build_rl_harness_config
from harnessx.rl.task import RLTask
from recipe.verl.spec import VerlConfigSpec


def make_verl_harness(spec: VerlConfigSpec, provider: Any, task: RLTask) -> HarnessConfig:
    """Build a per-run HarnessConfig from spec, provider, and task."""
    return build_rl_harness_config(spec, provider, task)
