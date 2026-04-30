# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter

from harnessx.api.models import SkillInfo
from harnessx.home import agent_home
from harnessx.workspace.skill_index import SkillIndex

router = APIRouter()

# Built-in skills shipped with the repo (source of truth for initial copy)
_BUILTIN_SKILLS = Path(__file__).parents[3] / "extensions" / "skills"


def _ensure_home_skills(home_skills: Path) -> None:
    """Copy built-in skills to AGENT_HOME/skills/ if not already present.

    Called lazily on the first API request so that the UI shows skills even
    before any harness.run() has triggered WorkspaceInitializer.
    Idempotent: skips skill dirs that already exist.
    """
    if not _BUILTIN_SKILLS.is_dir():
        return
    home_skills.mkdir(parents=True, exist_ok=True)
    for skill_dir in sorted(_BUILTIN_SKILLS.iterdir()):
        if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
            continue
        dst = home_skills / skill_dir.name
        if not dst.exists():
            shutil.copytree(skill_dir, dst)


@router.get("/skills", response_model=list[SkillInfo])
async def get_skills():
    """Return skills from AGENT_HOME/skills/ + plugin skill dirs.

    On the first call, built-in skills are lazily copied to AGENT_HOME/skills/
    so the UI shows them even before any harness run.  The index is rebuilt on
    each request so newly added/removed skills are reflected without restart.
    Plugin skills from AGENT_HOME/plugins/*/skills/*/ are included so the
    Settings → Skills page lists them for enable/disable toggling.
    """
    from harnessx.workspace.skill_index import collect_plugin_skill_dirs

    home = agent_home()
    home_skills = home / "skills"
    _ensure_home_skills(home_skills)
    plugin_dirs = collect_plugin_skill_dirs(home)
    return [
        SkillInfo(name=s.name, description=s.description)
        for s in SkillIndex(home_skills, extra_dirs=plugin_dirs).list_skills()
    ]
