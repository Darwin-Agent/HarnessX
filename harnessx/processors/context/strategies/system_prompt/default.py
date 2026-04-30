# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .....workspace.workspace import Workspace


_DEFAULT_SOUL = """\
# System

You are a capable AI assistant with access to tools and specialized skills.

When approaching tasks:
- Think step by step before acting
- Use skills when relevant — read their `SKILL.md` for full instructions
- Verify your work before declaring done
- Be concise and direct
"""


class DefaultSystemPromptBuilder:
    """Builds the system prompt from workspace config files.

    Reads SOUL.md, AGENTS.md, USER.md, TOOLS.md, CRON.md, MEMORY.md verbatim.
    No placeholder substitution — dynamic content belongs in step_start processors.

    Args:
        max_skills_shown: Skills listed in the system prompt (default 5).
        spawn_depth:      Current sub-agent depth (None = top-level, no spawn section).
        max_spawn_depth:  Depth at which the agent becomes a leaf worker.
    """

    def __init__(
        self,
        max_skills_shown: int = 10,
        spawn_depth: int | None = None,
        max_spawn_depth: int = 3,
        enabled_skills: list[str] | None = None,
        extra_skills_dirs: list[str | Path] | None = None,
        persona_root: "str | Path | None" = None,
    ):
        self.max_skills_shown = max_skills_shown
        self.spawn_depth = spawn_depth
        self.max_spawn_depth = max_spawn_depth
        self.enabled_skills = enabled_skills  # None = all; [] = none
        # Additional skill root dirs appended to the default resolution.
        # Never replaces AGENT_HOME/skills or workspace/skills — only adds.
        # Stored as str (not Path) so HarnessConfig.to_yaml's processor
        # serializer can round-trip them — PyYAML can't represent PosixPath.
        self.extra_skills_dirs: list[str] = [str(p) for p in (extra_skills_dirs or [])]
        # Separate read-only "persona" root — holds SOUL.md / AGENTS.md / skills/
        # when the agent's writable workspace.root is decoupled from its identity
        # source tree (e.g. the meta-agent shipping its persona inside the package).
        # When set: _read_ws_file falls back here if workspace.root has no override,
        # and persona_root/"skills" is appended to extra_skills_dirs. Stored as
        # str for the same YAML-representation reason as extra_skills_dirs.
        self.persona_root: str | None = str(persona_root) if persona_root else None
        if self.persona_root is not None:
            persona_skills = str(Path(self.persona_root) / "skills")
            if persona_skills not in self.extra_skills_dirs:
                self.extra_skills_dirs.append(persona_skills)

    async def build(self, workspace: "Workspace | None" = None) -> str:
        parts: list[str] = []
        parts.append(self._read_ws_file(workspace, "SOUL.md") or _DEFAULT_SOUL)

        skills_section = self._build_skills_section(workspace)
        if skills_section:
            parts.append(skills_section)

        ctx_block = self._build_workspace_context(workspace)
        if ctx_block:
            parts.append(ctx_block)

        spawn_section = self._build_spawn_section()
        if spawn_section:
            parts.append(spawn_section)

        return "\n\n".join(p.strip() for p in parts if p.strip())

    def _read_ws_file(self, workspace: "Workspace | None", filename: str) -> str:
        # Workspace override wins so per-project AGENTS.md / SOUL.md still
        # shadow the persona defaults.
        if workspace is not None:
            try:
                path = workspace.root / filename
                if path.exists():
                    return path.read_text(encoding="utf-8")
            except Exception:
                pass
        if self.persona_root is not None:
            try:
                path = Path(self.persona_root) / filename
                if path.exists():
                    return path.read_text(encoding="utf-8")
            except Exception:
                pass
        return ""

    def _resolve_skills_dir(self, workspace: "Workspace | None"):
        """Resolve the skills directory.

        Priority:
        1. ``AGENT_HOME/skills/`` (``workspace.home / "skills"``) — shared
           across all agents under the same AGENT_HOME.
        2. ``workspace.root / "skills"`` — per-workspace fallback.
        """
        if workspace is None:
            return None
        home = getattr(workspace, "home", None)
        if home is not None:
            from pathlib import Path

            home_skills = Path(home) / "skills"
            if home_skills.is_dir():
                return home_skills
        if workspace.root is not None:
            skills_dir = workspace.root / "skills"
            if skills_dir.exists():
                return skills_dir
        return None

    def _build_skills_section(self, workspace: "Workspace | None") -> str:
        try:
            from .....workspace.skill_index import (
                SkillIndex,
                collect_plugin_skill_dirs,
                expand_skill_roots,
            )

            skills_dir = self._resolve_skills_dir(workspace)
            # Preserve prior behavior: no skills section unless workspace
            # resolves a skills_dir OR the caller provided extra_skills_dirs.
            if skills_dir is None and not self.extra_skills_dirs:
                return ""
            home = getattr(workspace, "home", None) if workspace else None
            plugin_dirs = collect_plugin_skill_dirs(home)
            # Additive: expand extra_skills_dirs (plugin-provided roots) into
            # individual skill dirs, appended to whatever plugin_dirs already has.
            extra_dirs = plugin_dirs + expand_skill_roots(self.extra_skills_dirs)
            # Sentinel primary keeps SkillIndex from falling back to
            # extensions/skills when the workspace has no skills_dir.
            primary = skills_dir if skills_dir is not None else Path("/__no_primary_skills__")
            idx = SkillIndex(primary, extra_dirs=extra_dirs)
            all_skills = idx.list_skills()
            # Filter by enabled_skills when set (None = all, CLI default)
            if self.enabled_skills is not None:
                all_skills = [s for s in all_skills if s.name in self.enabled_skills]
            shown = all_skills[: self.max_skills_shown]
            if shown:
                # Build XML block from filtered list
                lines = ["<available_skills>"]
                for s in shown:
                    lines.append("  <skill>")
                    lines.append(f"    <name>{s.name}</name>")
                    lines.append(f"    <description>{idx._short_desc(s.description)}</description>")
                    lines.append(f"    <location>{s.path}</location>")
                    lines.append("  </skill>")
                lines.append("</available_skills>")
                body = "\n".join(lines)
                total = len(all_skills)
                where = skills_dir if skills_dir is not None else ", ".join(str(p) for p in self.extra_skills_dirs)
                footer = (
                    f"\n> Showing {len(shown)} of {total} skills. "
                    f"Skills directory: `{where}` — use Glob or Bash to find more."
                    if total > self.max_skills_shown
                    else f"\n> Skills directory: `{where}`"
                )
                return f"## Available Skills\n\n{body}{footer}"
        except Exception:
            pass
        return ""

    def _build_workspace_context(self, workspace: "Workspace | None") -> str:
        sections = []
        for filename in ("AGENTS.md", "USER.md", "TOOLS.md", "CRON.md", "MEMORY.md"):
            content = self._read_ws_file(workspace, filename)
            if content and content.strip():
                sections.append(f"### {filename}\n\n{content.strip()}")
        if not sections:
            return ""
        return "## Workspace\n\n" + "\n\n---\n\n".join(sections)

    def _build_spawn_section(self) -> str:
        if self.spawn_depth is None:
            return ""
        if self.spawn_depth >= self.max_spawn_depth:
            return (
                "## Sub-Agent Spawning\n\n"
                "You are a **leaf worker** and cannot spawn further sub-agents. "
                "Focus on completing your assigned task directly."
            )
        return (
            "## Sub-Agent Spawning\n\n"
            "You can delegate tasks to sub-agents using the `spawn_subagent` tool.\n\n"
            "- **`wait=true`** (default): blocks until the sub-agent finishes.\n"
            "- **`wait=false`**: fire-and-forget. Result arrives as a user message: "
            "`[Subagent label=<label>] Task completed: <output>`.\n\n"
            "You can override the child's `model`, `system_prompt`, `tools`, "
            "`max_steps`, and `max_cost_usd`. Empty/zero values inherit from this agent."
        )
