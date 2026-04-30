# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ..core.builder import HarnessBuilder
from .context import make_window_mgmt
from .reliability import reliability
from ..processors.context.env_context_injector import EnvironmentContextInjector
from ..processors.tools.skill_loader import ProgressiveSkillLoader


def make_coding(
    # EnvironmentContextInjector
    working_dir: str | None = None,
    timeout_seconds: int | None = None,
    constraints: dict[str, str] | None = None,
    # ProgressiveSkillLoader
    skills_dir: str | None = None,
    skill_top_k: int = 5,
    # CompactionProcessor
    token_threshold: int = 80_000,
    message_threshold: int = 100,
    # ToolFailureGuard
    max_tool_failures: int = 3,
    # Feature flags
    include_skill_loader: bool = True,
) -> HarnessBuilder:
    """Return a customised coding harness bundle.

    All parameters are optional — the defaults work for most coding tasks.

    Args:
        include_skill_loader: Include ProgressiveSkillLoader (default True).
                              Set False when no skills are available (e.g. TB2).
    """
    env_builder = HarnessBuilder().add(
        EnvironmentContextInjector(
            working_dir=working_dir,
            timeout_seconds=timeout_seconds,
            constraints=constraints or {},
        )
    )
    if include_skill_loader:
        env_builder = env_builder.add(
            ProgressiveSkillLoader(
                skills_dir=skills_dir,
                top_k=skill_top_k,
            )
        )
    ctx_layer = make_window_mgmt(
        token_threshold=token_threshold,
        message_threshold=message_threshold,
        max_tool_failures=max_tool_failures,
        skill_tool_names=["skill", "run_skill"],
    )
    return reliability | env_builder | ctx_layer


coding: HarnessBuilder = make_coding()
"""Coding agent harness bundle with default parameters.

Plug into any ``HarnessBuilder`` via ``| coding``.
"""
