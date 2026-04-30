# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...base import HarnessPlugin
from .llm_strategy import LLMMetaStrategy
from .processors import MetaPlanProcessor, MetaReflectProcessor

if TYPE_CHECKING:
    from ....meta.strategy import MetaStrategy
    from ....providers.base import BaseModelProvider

logger = logging.getLogger(__name__)


class LightMetaPlugin(HarnessPlugin):
    """LLM-based adaptive harness with skill-based learning.

    Adds plan (on_task_start) and reflect (on_task_end) capabilities.
    Skills are read from and written to ``skills_dir``.

    Args:
        skills_dir: Directory for skill files (SKILL.md).  All skills
            (task-skills and meta-skills) live here.
        planning_provider: Model provider for plan/reflect inner agents.
            Falls back to the main harness provider if not set.
        strategy: Custom MetaStrategy implementation.  Defaults to
            LLMMetaStrategy.
        reflect_enabled: Whether to run reflect after each task.
            Set to False for plan-only mode.
    """

    name = "meta.light"
    version = "0.1.0"
    description = (
        "LLM-based adaptive harness: plan reads meta-skills to adapt "
        "tools/processors; reflect writes skills from trajectories."
    )

    def __init__(
        self,
        skills_dir: str | Path,
        planning_provider: "BaseModelProvider | None" = None,
        strategy: "MetaStrategy | None" = None,
        reflect_enabled: bool = True,
    ) -> None:
        super().__init__()
        self._skills_dir = Path(skills_dir).resolve()
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        self._planning_provider = planning_provider
        self._reflect_enabled = reflect_enabled

        self._strategy: MetaStrategy = strategy or LLMMetaStrategy(
            skills_dir=self._skills_dir,
        )

        self._plan_proc = MetaPlanProcessor(
            strategy=self._strategy,
            skills_dir=self._skills_dir,
        )
        self._reflect_proc = MetaReflectProcessor(
            strategy=self._strategy,
            skills_dir=self._skills_dir,
            enabled=reflect_enabled,
        )
        self.processors = [self._plan_proc, self._reflect_proc]

    # Cost tracking (populated by processors after each run)
    @property
    def last_plan_cost_usd(self) -> float:
        return self._plan_proc.last_cost_usd

    @property
    def last_reflect_cost_usd(self) -> float:
        return self._reflect_proc.last_cost_usd

    def setup(self, config: Any) -> None:
        """Wire provider to strategy and propagate skills_dir to SystemPromptProcessor."""
        # 1. Bind provider
        provider = self._planning_provider
        if provider is None:
            try:
                provider = config.model_config.main
            except AttributeError:
                logger.warning(
                    "LightMetaPlugin: no planning_provider and cannot "
                    "extract from config.model_config — plan/reflect "
                    "will fail if strategy needs a provider"
                )

        if provider is not None and isinstance(self._strategy, LLMMetaStrategy):
            self._strategy.bind_provider(provider)

        # 2. Propagate skills_dir to SkillRuntimePlugin so both the system-prompt
        #    listing and per-step injection pick up the plugin's skills dir.
        self._propagate_skills_dir(config)

    def _propagate_skills_dir(self, config: Any) -> None:
        """Append plugin's skills_dir to SkillRuntimePlugin (additive only)."""
        from ....plugins.dimensions.skill_runtime import SkillRuntimePlugin
        from ....processors.tools.skill_loader import ProgressiveSkillLoader

        # Primary: delegate to SkillRuntimePlugin which owns both the listing and loader.
        for plugin in getattr(config, "plugins", []) or []:
            if isinstance(plugin, SkillRuntimePlugin):
                plugin.append_extra_skills_dir(self._skills_dir)
                logger.info(
                    "LightMetaPlugin: appended skills_dir=%s to SkillRuntimePlugin",
                    self._skills_dir,
                )
                return

        # Fallback: bare ProgressiveSkillLoader with no SkillRuntimePlugin present.
        processors = getattr(config, "processors", None) or []
        seen: set[int] = set()
        for p in processors:
            if id(p) in seen:
                continue
            seen.add(id(p))
            if isinstance(p, ProgressiveSkillLoader):
                existing = getattr(p, "_extra_skills_dirs", None)
                if existing is not None and self._skills_dir not in existing:
                    existing.append(self._skills_dir)
                    logger.info(
                        "LightMetaPlugin: appended skills_dir=%s to ProgressiveSkillLoader",
                        self._skills_dir,
                    )
