# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Research task registration for verl backend."""

from __future__ import annotations

from recipe.verl.spec import VerlConfigSpec
from recipe.verl.research.builder import ResearchTaskBuilder
from recipe.verl.research.config import cfg
from recipe.verl.research.evaluator import ResearchAnswerEvaluator
from recipe.verl.research.prompt import SYSTEM_PROMPT as RESEARCH_SYSTEM_PROMPT
from recipe.verl.research.rewards import research_extra_reward_fn
from recipe.verl.research.tools import RESEARCH_TOOLS
from harnessx.rl.task import NullPRM
from harnessx.processors.context.system_prompt import SystemPromptProcessor
from recipe.verl.research.skill_tool import RLSkillRuntimeProcessor
from recipe.verl.processor.early_commit import EarlyCommitProcessor
from recipe.verl.processor.evaluation import VerlEvaluationProcessor


def research_processors(system_builder, evaluator):
    return [
        SystemPromptProcessor(system_builder),
        RLSkillRuntimeProcessor(skills_dir=cfg.skills_dir),
        EarlyCommitProcessor(max_steps=cfg.max_steps, token_budget_ratio=cfg.early_commit_ratio),
        VerlEvaluationProcessor(evaluator),
    ]


RESEARCH_CONFIGS: dict[str, VerlConfigSpec] = {
    "research": VerlConfigSpec(
        task_builder=ResearchTaskBuilder(),
        tools=RESEARCH_TOOLS,
        evaluator_cls=ResearchAnswerEvaluator,
        prm=NullPRM(),
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        max_steps=cfg.max_steps,
        task_type="research",
        extra_reward_fn=research_extra_reward_fn,
        reward_shaping=True,
        processor_factory=research_processors,
        tool_call_format=cfg.tool_call_format,
        tool_result_role=cfg.tool_result_role,
    ),
}
