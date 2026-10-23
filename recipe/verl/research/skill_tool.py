# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""RL-compatible skill loading: load_skill tool + system prompt injection.

Unlike the existing ProgressiveSkillLoader (which auto-injects full content
per step based on keyword matching), the RL version is model-driven:
  - RLSkillRuntimeProcessor injects an <available_skills> listing (names +
    short descriptions) into the system prompt on task_start.
  - The model must explicitly call load_skill to get full SKILL.md content.

This teaches the model to decide WHEN skills are needed — a learnable behavior
for RL training — rather than relying on an oracle keyword matcher.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import AsyncIterator

from harnessx.core.events import TaskStartEvent
from harnessx.core.processor import MultiHookProcessor
from harnessx.tools.base import Tool
from harnessx.workspace.skill_index import SkillIndex


# ---------------------------------------------------------------------------
# load_skill Tool
# ---------------------------------------------------------------------------


def make_load_skill_tool(
    skills_dir: str | Path,
    enabled_skills: list[str] | None = None,
) -> Tool:
    """Factory: creates a load_skill Tool bound to a skills directory.

    Args:
        skills_dir:      Path to the skills root (e.g. extensions/skills/).
                         Each subdirectory with a SKILL.md is a skill.
        enabled_skills:  Optional whitelist of skill names. None = all skills.

    Returns:
        A Tool instance that can be registered in InMemoryToolRegistry.
    """
    _skills_dir = Path(skills_dir)
    _enabled = set(enabled_skills) if enabled_skills is not None else None
    _index: SkillIndex | None = None

    def _get_index() -> SkillIndex:
        nonlocal _index
        if _index is None:
            _index = SkillIndex(_skills_dir)
        return _index

    async def load_skill(skill_name: str) -> str:
        idx = _get_index()
        meta = idx.get_meta(skill_name)
        if meta is None:
            available = [s.name for s in idx.list_skills()]
            if _enabled is not None:
                available = [n for n in available if n in _enabled]
            return f"Error: skill '{skill_name}' not found. Available skills: {available}"
        if _enabled is not None and skill_name not in _enabled:
            return f"Error: skill '{skill_name}' is not enabled."
        try:
            content = meta.path.read_text(encoding="utf-8", errors="replace")
            return content
        except OSError as e:
            return f"Error reading skill '{skill_name}': {e}"

    return Tool(
        name="load_skill",
        description=(
            "Load a skill's full instructions by name. Use this when you need "
            "detailed step-by-step instructions for a specific skill listed in "
            "<available_skills>. Returns the complete SKILL.md content."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to load (must match a name from <available_skills>)",
                }
            },
            "required": ["skill_name"],
        },
        fn=load_skill,
        tags=["skill"],
    )


# ---------------------------------------------------------------------------
# RLSkillRuntimeProcessor
# ---------------------------------------------------------------------------


class RLSkillRuntimeProcessor(MultiHookProcessor):
    """Injects <available_skills> listing into system prompt for RL training.

    On task_start, appends an XML block listing all available skills (name +
    short description) to the system prompt. The model can then call the
    load_skill tool to get the full instructions for any skill it needs.

    Args:
        skills_dir:       Path to the skills root directory.
        enabled_skills:   Optional whitelist. None = all skills visible.
        max_skills_shown: Cap on number of skills in the listing.
    """

    _order = 15

    def __init__(
        self,
        skills_dir: str | Path,
        enabled_skills: list[str] | None = None,
        max_skills_shown: int = 10,
    ) -> None:
        self._skills_dir = Path(skills_dir)
        self._enabled_skills = enabled_skills
        self._max_skills_shown = max_skills_shown

    async def on_task_start(self, event: TaskStartEvent) -> AsyncIterator[TaskStartEvent]:
        block = self._build_skills_block()
        if block:
            prompt = (event.system_prompt or "").rstrip() + "\n\n" + block
            yield dataclasses.replace(event, system_prompt=prompt)
        else:
            yield event

    def _build_skills_block(self) -> str:
        if not self._skills_dir.is_dir():
            return ""
        idx = SkillIndex(self._skills_dir)
        skills = idx.list_skills()
        if self._enabled_skills is not None:
            skills = [s for s in skills if s.name in self._enabled_skills]
        skills = skills[: self._max_skills_shown]
        if not skills:
            return ""

        lines = ["## Available Skills", "", "<available_skills>"]
        for s in skills:
            lines.append("  <skill>")
            lines.append(f"    <name>{s.name}</name>")
            desc = (s.description or "")[:200]
            lines.append(f"    <description>{desc}</description>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        lines.append("")
        lines.append(
            "To use a skill, call the `load_skill` tool with the skill name "
            "to get detailed instructions, then follow them."
        )
        return "\n".join(lines)
