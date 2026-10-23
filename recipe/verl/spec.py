# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""VerlConfigSpec — task specification for verl RL backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from harnessx.rl.config import RLConfigSpec


@dataclass
class VerlConfigSpec(RLConfigSpec):
    """Extends RLConfigSpec with verl-specific fields.

    Mirrors SlimeConfigSpec but adapted for verl's data format:
    - No formatter_factory (verl handles tokenization via apply_chat_template)
    - extra_reward_fn returns {"score_delta": float, ...}
    - processor_factory controls the per-run processor pipeline
    """

    extra_reward_fn: Callable[..., dict] | None = None
    reward_shaping: bool = True
    processor_factory: Callable | None = None
    tool_call_format: str = "qwen3_coder"
    tool_result_role: str = "user"
