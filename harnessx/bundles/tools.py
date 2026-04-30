# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ..core.builder import HarnessBuilder
from ..processors.tools.skill_loader import ProgressiveSkillLoader
from ..processors.tools.tool_whitelist import ToolWhitelistProcessor
from ..processors.tools.model_schema_adapter import ModelSpecificSchemaAdapter


def make_tools(
    # ProgressiveSkillLoader
    skill_loading: bool = False,
    skills_dir: str | None = None,
    skill_top_k: int = 5,
    # ToolWhitelistProcessor
    whitelist: list[str] | None = None,
    dangerous_tools: list[str] | None = None,
    require_approval: bool = False,
    # ModelSpecificSchemaAdapter
    model_schema_target: str | None = None,
) -> HarnessBuilder:
    """Return a tools capability bundle.

    Args:
        skill_loading:       Include ``ProgressiveSkillLoader`` (default False).
        skills_dir:          Directory from which skills are loaded.  Only used
                             when ``skill_loading`` is True.
        skill_top_k:         Maximum number of skills injected per turn.
        whitelist:           Explicit list of allowed tool names.  When provided,
                             all other tools are blocked unless ``allow_all=True``.
        dangerous_tools:     Tool names that require approval before execution.
                             Defaults to ``["Bash", "Write"]``.
        require_approval:    When True, all tool calls require explicit user approval.
        model_schema_target: Target model identifier for schema adaptation
                             (e.g. ``"gpt-5.4"``).  When provided,
                             ``ModelSpecificSchemaAdapter`` is added.
    """
    builder = HarnessBuilder()
    if skill_loading:
        builder = builder.add(
            ProgressiveSkillLoader(
                skills_dir=skills_dir,
                top_k=skill_top_k,
            )
        )
    if whitelist is not None or dangerous_tools is not None or require_approval:
        builder = builder.add(
            ToolWhitelistProcessor(
                allowed_tools=whitelist,
                dangerous_tools=dangerous_tools,
                allow_all=(not whitelist and not require_approval),
            )
        )
    if model_schema_target is not None:
        builder = builder.add(ModelSpecificSchemaAdapter(model=model_schema_target))
    return builder
