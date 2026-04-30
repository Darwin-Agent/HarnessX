# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harnessx.rl.config import RLConfigSpec


@dataclass
class SlimeConfigSpec(RLConfigSpec):
    """
    Slime-specific extension of RLConfigSpec.

    Adds tokenizer-level formatting (for SGLang incremental tokenization)
    and extra_reward_fn (for Slime's reward_func hook).

    Inherited fields from RLConfigSpec:
        task_builder, tools, evaluator_cls, prm, system_prompt,
        max_steps, extra_processors, task_type, reward_weights

    Added fields:
        formatter_factory:
            (tokenizer) -> (messages, tools) -> list[int]
            Full message formatter for step 0.  None = apply_chat_template.
        inter_turn_formatter_factory:
            (tokenizer) -> (new_messages) -> list[int]
            Incremental inter-turn formatter for step t+1.  None = native tail.
        extra_reward_fn:
            (sample, eval_result, traj) -> dict | None
            Task-specific extra reward fields merged into reward_func() output.
            Must return a dict; "score_delta" key is added to the final score.
            All other keys are logged to wandb as reward decomposition fields.
    """

    formatter_factory: Any = None  # (tokenizer) -> (messages, tools) -> list[int]
    inter_turn_formatter_factory: Any = None  # (tokenizer) -> (new_messages) -> list[int]
    extra_reward_fn: Any = None  # (sample, eval_result, traj) -> dict | None
    reward_shaping: bool = False  # False = pure terminal ±1 (NullPRM, no extra_reward_fn)
    # True  = enable PRM shaping + extra_reward_fn
