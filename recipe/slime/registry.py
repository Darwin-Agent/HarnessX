# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING

from recipe.slime.spec import SlimeConfigSpec
from recipe.slime.math.builder import MathTaskBuilder
from recipe.slime.math.evaluator import MathBoxedEvaluator
from recipe.slime.math.rewards import RetoolCompatPRM, math_format_reward
from recipe.slime.math.tools import code_interpreter_tool
from recipe.slime.math.formatter import (
    make_retool_formatter,
    make_retool_inter_turn_formatter,
)
from harnessx.rl.task import NullPRM, EnhancedToolSuccessPRM

if TYPE_CHECKING:
    from slime.utils.types import Sample


_MATH_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use Python tools to solve mathematical "
    "problems. When you need to perform calculations, use the code_interpreter "
    "tool to execute code and get results. "
    "Give the final answer in \\boxed{} notation."
)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

HARNESS_CONFIGS: dict[str, SlimeConfigSpec] = {
    # Phase 1: Math RL — dapo-math-17k training + AIME evaluation
    # reward_shaping=False (default): pure terminal ±1, no PRM or format bonus
    # reward_shaping=True: RetoolCompatPRM + math_format_reward (retool-compat shaping)
    "math": SlimeConfigSpec(
        task_builder=MathTaskBuilder(),
        tools=[code_interpreter_tool],
        evaluator_cls=MathBoxedEvaluator,
        prm=NullPRM(),  # default: pure terminal ±1
        reward_weights={"terminal": 1.0},
        system_prompt=_MATH_SYSTEM_PROMPT,
        max_steps=5,
        task_type="math",
        formatter_factory=make_retool_formatter,
        inter_turn_formatter_factory=make_retool_inter_turn_formatter,
        # reward_shaping=False (default) → prm=NullPRM, extra_reward_fn=None
    ),
    # Math with reward shaping enabled (retool-compat PRM + format bonus)
    "math_shaped": SlimeConfigSpec(
        task_builder=MathTaskBuilder(),
        tools=[code_interpreter_tool],
        evaluator_cls=MathBoxedEvaluator,
        prm=RetoolCompatPRM(),
        reward_weights={"terminal": 1.0},
        system_prompt=_MATH_SYSTEM_PROMPT,
        max_steps=5,
        task_type="math",
        formatter_factory=make_retool_formatter,
        inter_turn_formatter_factory=make_retool_inter_turn_formatter,
        extra_reward_fn=math_format_reward,
        reward_shaping=True,
    ),
    # Math with dense per-step tool reward shaping
    "math_dense": SlimeConfigSpec(
        task_builder=MathTaskBuilder(),
        tools=[code_interpreter_tool],
        evaluator_cls=MathBoxedEvaluator,
        prm=EnhancedToolSuccessPRM(
            success_bonus=0.05,
            error_penalty=0.10,
            loop_penalty=0.20,
        ),
        reward_weights={"terminal": 1.0, "tool_success": 0.05, "tool_error": -0.10},
        system_prompt=_MATH_SYSTEM_PROMPT,
        max_steps=5,
        task_type="math",
        formatter_factory=make_retool_formatter,
        inter_turn_formatter_factory=make_retool_inter_turn_formatter,
        extra_reward_fn=math_format_reward,
        reward_shaping=True,
    ),
}


# ---------------------------------------------------------------------------
# load_harness_config
# ---------------------------------------------------------------------------


def load_harness_config(sample: "Sample") -> SlimeConfigSpec:
    """Resolve the SlimeConfigSpec for a given Sample.

    Routing logic:
    1. sample.metadata["task_type"] if present
    2. "math" as default (covers dapo-math-17k and AIME eval)
    """
    metadata = getattr(sample, "metadata", None) or {}
    task_type = metadata.get("task_type", "math")
    spec = HARNESS_CONFIGS.get(task_type)
    if spec is None:
        spec = HARNESS_CONFIGS["math"]
    return spec
