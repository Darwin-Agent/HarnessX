# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workspace import Workspace


class WorkspaceInitializer:
    """
    Prepares a workspace for a new agent run by copying:

    1. ``workspace/default_prompts/*.md`` → workspace root (AGENTS.md, TOOLS.md)

    2. ``extensions/skills/<name>/`` → ``AGENT_HOME/skills/<name>/``
       Full skill directories (SKILL.md + scripts + resource files) copied to
       the shared AGENT_HOME so all agents can read them.  When
       ``workspace.home`` is not set, falls back to ``workspace.root/skills/``.

    Design principles:
    - **Idempotent**: skips files / skill dirs that already exist
    - **Shared**: skills go to AGENT_HOME/skills/, visible to all agents
    - **Concurrent-safe**: multiple agents can call initialize() — the
      ``if not exists`` guard prevents duplicate copies

    Args:
        prompts_root: Path to the default prompt templates directory.
                      Defaults to ``harnessx/workspace/default_prompts/``.
        skills_root:  Path to the built-in skills directory.
                      Defaults to ``extensions/skills/`` at repo root.
    """

    def __init__(self, prompts_root: Path | None = None, skills_root: Path | None = None):
        self.prompts_root = prompts_root or (Path(__file__).parent / "default_prompts")
        self.skills_root = skills_root or (Path(__file__).parents[2] / "extensions" / "skills")

    async def initialize(
        self,
        workspace: "Workspace",
        template: str = "default",
        copy_skills: bool = True,
    ) -> None:
        """
        Initialize the workspace with default files and skills.

        Args:
            workspace: The Workspace to initialize.
            template: Ignored — kept for API compatibility.
            copy_skills: If True, copy built-in skills to the shared skills dir.
        """
        ws_root = workspace.root
        ws_root.mkdir(parents=True, exist_ok=True)

        # 1. Copy default workspace files (AGENTS.md, TOOLS.md)
        await self._copy_template(ws_root)

        # 2. Copy skills to AGENT_HOME/skills/ (shared) or workspace.root/skills/ (fallback)
        if copy_skills:
            home = getattr(workspace, "home", None)
            if home is not None:
                skills_dst = Path(home) / "skills"
            else:
                skills_dst = ws_root / "skills"
            await self._copy_skills(skills_dst)

    async def _copy_template(self, ws_root: Path) -> None:
        """Copy workspace markdown files (AGENTS.md, TOOLS.md) to workspace root.

        Idempotent: skips files that already exist.
        """
        for src in self.prompts_root.glob("*.md"):
            dst = ws_root / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    async def _copy_skills(self, skills_dst: Path) -> None:
        """Copy full skill directories to the target skills directory.

        Idempotent: skips skill dirs that already exist.  Safe for concurrent
        calls from multiple agents — the ``if not exists`` guard ensures each
        skill is copied at most once.
        """
        skills_src = self.skills_root
        if not skills_src.exists():
            return

        skills_dst.mkdir(parents=True, exist_ok=True)

        for skill_dir in sorted(skills_src.iterdir()):
            if not skill_dir.is_dir():
                continue
            if not (skill_dir / "SKILL.md").exists():
                continue
            skill_dst_dir = skills_dst / skill_dir.name
            if not skill_dst_dir.exists():
                shutil.copytree(skill_dir, skill_dst_dir)

    def available_templates(self) -> list[str]:
        """List available workspace template names."""
        if not self.prompts_root.exists():
            return []
        return sorted(d.name for d in self.prompts_root.iterdir() if d.is_dir())
