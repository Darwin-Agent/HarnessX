# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harnessx.tools.base import Tool
    from harnessx.rl.task import ProcessRewardModel, RLEvaluator, TaskBuilder


@dataclass
class RLConfigSpec:
    """
    Framework-agnostic RL task configuration.

    Contains only harnessx-native types — zero framework-specific dependencies.
    Subclassed by framework adapters (SlimeConfigSpec, etc.) to add
    framework-specific fields.

    Fields:
        task_builder:     TaskBuilder — converts a training sample → RLTask.
                          The only place that knows about the upstream data format.
        tools:            list[Tool] — Tool instances registered in the tool registry.
                          Pass actual Tool objects; no global name lookup needed.
        evaluator_cls:    type[RLEvaluator] — freshly instantiated per run (closes over task.label).
        prm:              ProcessRewardModel — per-step reward model (NullPRM, EnhancedToolSuccessPRM, …).
        system_prompt:    str — default system prompt. task.metadata["system_prompt"] overrides this.
        max_steps:        int — max harness steps per episode (propagated to RLTask.max_steps).
        extra_processors: dict[hook, list[Processor]] — merged into standard RL processor bundle.
        task_type:        str — for logging and routing (e.g. "math", "code").
        reward_weights:   dict[str, float] — reserved for aggregate_reward() (not yet used).
    """

    task_builder: "TaskBuilder"
    tools: "list[Tool]"
    evaluator_cls: "type[RLEvaluator]"
    prm: "ProcessRewardModel"
    system_prompt: str = ""
    max_steps: int = 16
    extra_processors: dict[str, list] = field(default_factory=dict)
    task_type: str = ""
    reward_weights: dict[str, float] = field(default_factory=lambda: {"terminal": 1.0})
