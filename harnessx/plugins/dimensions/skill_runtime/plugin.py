# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

from ....core.processor import MultiHookProcessor
from ....processors.tools.skill_loader import ProgressiveSkillLoader
from ....workspace.skill_index import (
    SkillIndex,
    SkillMeta,
    collect_plugin_skill_dirs,
    expand_skill_roots,
)
from ...base import HarnessPlugin

if TYPE_CHECKING:
    from ....core.events import TaskStartEvent


class SkillSystemPromptProcessor(MultiHookProcessor):
    """Injects <available_skills> block into the system prompt on task_start."""

    _order = 10  # before SkillRuntimeProcessor (11) and ProgressiveSkillLoader (12)
    _singleton_group = "_skill_sysprompt"
    __hx_runtime_only__ = True

    def __init__(self, plugin: "SkillRuntimePlugin") -> None:
        self._plugin = plugin

    async def on_task_start(self, event: "TaskStartEvent") -> AsyncIterator:
        # Skip if agent has no tools — skills require tools to execute.
        if not event.tools:
            yield event
            return
        block = self._plugin._build_available_skills_section(event.workspace)
        if block and "<available_skills>" not in (event.system_prompt or ""):
            prompt = (event.system_prompt or "").rstrip() + "\n\n" + block
            yield dataclasses.replace(event, system_prompt=prompt)
        else:
            yield event


class SkillRuntimeProcessor(MultiHookProcessor):
    """Thin task_start hook: detect skill-dir changes and bust loader caches."""

    _order = 11  # runs just before ProgressiveSkillLoader (_order=12)
    _singleton_group = "_skill_runtime"
    __hx_runtime_only__ = True

    def __init__(self, plugin: "SkillRuntimePlugin") -> None:
        self._plugin = plugin

    async def on_task_start(self, event: "TaskStartEvent") -> AsyncIterator:
        sig = self._plugin._dir_signature()
        if sig != self._plugin._dir_sig:
            self._plugin._dir_sig = sig
            self._plugin._loader.clear_caches()
        yield event


class SkillRuntimePlugin(HarnessPlugin):
    """HarnessX built-in skill runtime plugin.

    Owns the full skills pipeline:
    - SkillSystemPromptProcessor: injects <available_skills> listing into system prompt.
    - SkillRuntimeProcessor: detects skill-directory changes on task_start and
      invalidates loader caches so new/removed SKILL.md files are picked up
      without restarting the harness.
    - ProgressiveSkillLoader: injects full skill content per step.
    """

    name = "_builtin_skill_runtime"
    version = "0.1.0"
    description = "Built-in skill runtime management"

    def __init__(
        self,
        enabled_skills: list[str] | None = None,
        auto_inject: bool = True,
        max_skills_shown: int = 10,
        extra_skills_dirs: list[str | Path] | None = None,
    ) -> None:
        super().__init__()
        # None → all skills enabled; [] → all skills disabled (inject nothing)
        self._enabled_skills: list[str] | None = list(enabled_skills) if enabled_skills is not None else None
        self._auto_inject = bool(auto_inject)
        self._max_skills_shown = max_skills_shown
        self._extra_skills_dirs: list[Path] = [Path(p) for p in (extra_skills_dirs or [])]
        self._loader = ProgressiveSkillLoader(
            enabled_skills=self._enabled_skills,
            extra_skills_dirs=list(self._extra_skills_dirs),
        )
        # Keep _loader as a runtime instance — it holds plugin state (enabled_skills,
        # dir signatures) that is managed via set_enabled_skills().  Serializing it
        # would produce a new instance on restart and lose that state.
        self._loader.__hx_runtime_only__ = True
        self._dir_sig: str = ""
        # processors: empty when auto_inject is off so nothing is wired in
        self.processors = (
            [
                SkillSystemPromptProcessor(self),
                SkillRuntimeProcessor(self),
                self._loader,
            ]
            if self._auto_inject
            else []
        )

    # ── Directory-change detection ────────────────────────────────────────────

    def _skill_dirs(self) -> tuple[Path | None, list[Path]]:
        """Return (primary_skills_dir, plugin_skill_dirs) from AGENT_HOME."""
        try:
            from ....home import agent_home

            home = agent_home()
            primary = home / "skills"
            plugin_dirs = collect_plugin_skill_dirs(home)
            return (primary if primary.is_dir() else None), plugin_dirs
        except Exception:
            return None, []

    def _dir_signature(self) -> str:
        """Cheap mtime-based fingerprint of all skill directories."""
        primary, extra = self._skill_dirs()
        dirs = ([primary] if primary else []) + extra
        if not dirs:
            return ""
        try:
            parts: list[str] = []
            for d in sorted(set(dirs)):
                if not d.is_dir():
                    continue
                # Include subdirectory mtimes so adding/removing a skill dir
                # (which is typically a subdirectory) busts the cache.
                mtime = max(
                    (e.stat().st_mtime for e in d.iterdir()),
                    default=0.0,
                )
                parts.append(f"{d}:{mtime:.3f}")
            return "|".join(parts)
        except Exception:
            return ""

    # ── System prompt section builder ─────────────────────────────────────────

    def _build_available_skills_section(self, workspace: object) -> str:
        """Build <available_skills> XML block for the system prompt."""
        try:
            skills_dir = self._resolve_skills_dir(workspace)
            if skills_dir is None and not self._extra_skills_dirs:
                return ""
            home = getattr(workspace, "home", None) if workspace else None
            plugin_dirs = collect_plugin_skill_dirs(home)
            extra_dirs = plugin_dirs + expand_skill_roots(self._extra_skills_dirs)
            primary = skills_dir if skills_dir is not None else Path("/__no_primary_skills__")
            idx = SkillIndex(primary, extra_dirs=extra_dirs)
            all_skills = idx.list_skills()
            all_skills = [s for s in all_skills if not s.name.startswith("meta-")]
            if self._enabled_skills is not None:
                all_skills = [s for s in all_skills if s.name in self._enabled_skills]
            shown = all_skills[: self._max_skills_shown]
            if not shown:
                return ""
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
            where = skills_dir if skills_dir is not None else ", ".join(str(p) for p in self._extra_skills_dirs)
            footer = (
                f"\n> Showing {len(shown)} of {total} skills. "
                f"Skills directory: `{where}` — use Glob or Bash to find more."
                if total > self._max_skills_shown
                else f"\n> Skills directory: `{where}`"
            )
            usage = (
                "\nTo use a skill: `Read` its `<location>` path to get full instructions, "
                "then follow them — typically via `Bash`."
            )
            return f"## Available Skills\n\n{body}{footer}{usage}"
        except Exception:
            return ""

    def _resolve_skills_dir(self, workspace: object) -> Path | None:
        if workspace is None:
            return None
        home = getattr(workspace, "home", None)
        if home is not None:
            home_skills = Path(home) / "skills"
            if home_skills.is_dir():
                return home_skills
        root = getattr(workspace, "root", None)
        if root is not None:
            skills_dir = Path(root) / "skills"
            if skills_dir.exists():
                return skills_dir
        return None

    # ── Public API (used by CLI startup, Lab UI API routes, LightMetaPlugin) ──

    def list_skills(self) -> list[SkillMeta]:
        """Return metadata for all skills in AGENT_HOME/skills/ and plugin dirs."""
        primary, extra = self._skill_dirs()
        if primary is None and not extra:
            return []
        sentinel = primary or Path("/__no_primary_skills__")
        try:
            return SkillIndex(sentinel, extra_dirs=extra).list_skills()
        except Exception:
            return []

    def set_enabled_skills(self, skills: list[str] | None) -> None:
        """Update which skills are injected.  None = all; [] = none."""
        self._enabled_skills = list(skills) if skills is not None else None
        self._loader.enabled_skills = self._enabled_skills

    def append_extra_skills_dir(self, path: Path) -> None:
        """Additive: append an extra skills dir (used by LightMetaPlugin)."""
        path = Path(path)
        if path not in self._extra_skills_dirs:
            self._extra_skills_dirs.append(path)
            if path not in self._loader._extra_skills_dirs:
                self._loader._extra_skills_dirs.append(path)
            self._loader.clear_caches()

    async def warmup_summary(self) -> dict[str, Any]:
        """Scan skill dirs and return summary counts for startup display."""
        skills = self.list_skills()
        total = len(skills)
        enabled = len(self._enabled_skills) if self._enabled_skills is not None else total
        return {"skills": total, "enabled": enabled}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self, config: object) -> None:
        """Copy built-in skills from extensions/skills/ to AGENT_HOME/skills/ on startup."""
        import shutil

        try:
            from ....home import agent_home
            from ....workspace.initializer import WorkspaceInitializer

            skills_src = WorkspaceInitializer().skills_root
            if not skills_src.exists():
                return
            skills_dst = agent_home() / "skills"
            skills_dst.mkdir(parents=True, exist_ok=True)
            for skill_dir in sorted(skills_src.iterdir()):
                if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
                    continue
                dst = skills_dst / skill_dir.name
                if not dst.exists():
                    shutil.copytree(skill_dir, dst)
        except Exception:
            pass

    async def stop(self) -> None:
        self._loader.clear_caches()
