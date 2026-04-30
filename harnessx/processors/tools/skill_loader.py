# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path

from ...core.events import (
    BeforeModelEvent,
    Message,
    StepStartEvent,
    TaskEndEvent,
    TaskStartEvent,
    _extract_text,
)
from ...core.processor import MultiHookProcessor
from ...workspace.skill_index import (
    SkillIndex,
    SkillMeta,
    collect_plugin_skill_dirs,
    expand_skill_roots,
)

logger = logging.getLogger(__name__)

# Module-level compiled patterns — avoids re-compiling on every step_start call
_WORD_RE = re.compile(r"\w+")
_SKILLS_BLOCK_RE = re.compile(r"\n*<available_skills>.*?</available_skills>", re.DOTALL)


def _last_user_text(messages: tuple) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return _extract_text(m.content)
    return ""


def _score_skill(skill: SkillMeta, query: str) -> int:
    """Keyword overlap score between skill metadata and query text."""
    query_lower = query.lower()
    name_lower = skill.name.lower()
    desc_lower = skill.description.lower()
    score = 0
    for word in _WORD_RE.findall(query_lower):
        if len(word) >= 4 and (word in name_lower or word in desc_lower):
            score += 1
    return score


class ProgressiveSkillLoader(MultiHookProcessor):
    """Inject full SKILL.md content for skills that match the current query.

    On each step, keyword-matches the latest user message against skill
    descriptions.  Matching skills have their full content injected as a
    user message so the model can act on them without a separate Read call.

    The static ``<available_skills>`` listing is injected by
    ``SkillSystemPromptProcessor`` (part of ``SkillRuntimePlugin``) on
    task start and is NOT duplicated here.

    Args:
        skills_dir:    Path to the skills directory.  When ``None``, reads
                       ``StepStartEvent.workspace.root / "skills"``.
        top_k:         Maximum number of skills to inject per step (default 5).
        min_score:     Minimum keyword match score (default 1).
        replace_block: When ``True``, remove the ``<available_skills>`` XML
                       block from the system prompt during ``on_task_start``.
                       Use when full-content injection makes the static listing
                       redundant.
    """

    _singleton_group = "progressive_skill_loader"
    _order = 12

    def __init__(
        self,
        skills_dir: str | Path | None = None,
        top_k: int = 5,
        min_score: int = 1,
        replace_block: bool = False,
        enabled_skills: list[str] | None = None,
        extra_skills_dirs: list[str | Path] | None = None,
    ) -> None:
        self._skills_dir = Path(skills_dir) if skills_dir else None
        self.top_k = top_k
        self.min_score = min_score
        self.replace_block = replace_block
        self.enabled_skills = enabled_skills  # None = all skills enabled
        # Extra skill root dirs appended to the default resolution — additive,
        # never replaces skills_dir / AGENT_HOME/skills / workspace/skills.
        self._extra_skills_dirs: list[Path] = [Path(p) for p in (extra_skills_dirs or [])]
        self._indexes: dict[tuple, SkillIndex] = {}
        self._content_cache: dict[Path, str] = {}
        self._cached_workspace = None
        self._cached_raw_messages: tuple = ()

    def _get_index(self, skills_dir: Path | None, extra_dirs: list[Path] | None = None) -> SkillIndex:
        # Cache key includes extra_dirs so a change in plugin set busts the cache.
        key = (skills_dir, tuple(extra_dirs or ()))
        if key not in self._indexes:
            self._indexes[key] = SkillIndex(skills_dir, extra_dirs=extra_dirs)
        return self._indexes[key]

    def _resolve_skills_dir(self, workspace: object) -> Path | None:
        """Resolve the skills directory.

        Priority:
        1. Explicit ``skills_dir`` passed at construction time.
        2. ``AGENT_HOME/skills/`` (``workspace.home / "skills"``) — shared
           across all agents under the same AGENT_HOME.
        3. ``workspace.root / "skills"`` — per-workspace fallback.
        """
        if self._skills_dir:
            return self._skills_dir
        if workspace is not None:
            # Prefer AGENT_HOME/skills/ for cross-agent sharing
            home = getattr(workspace, "home", None)
            if home is not None:
                home_skills = Path(home) / "skills"
                if home_skills.is_dir():
                    return home_skills
            # Fallback to per-workspace skills/
            if hasattr(workspace, "root"):
                return Path(workspace.root) / "skills"
        return None

    def _read_skill(self, path: Path) -> str:
        if path not in self._content_cache:
            try:
                self._content_cache[path] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                self._content_cache[path] = f"(could not read {path})"
        return self._content_cache[path]

    def _build_full_block(self, skills: list[SkillMeta], query: str) -> str:
        scored = sorted(
            [(s, _score_skill(s, query)) for s in skills],
            key=lambda x: x[1],
            reverse=True,
        )
        selected = [s for s, score in scored if score >= self.min_score][: self.top_k]
        if not selected:
            return ""
        parts = ["\n\n## Relevant Skills"]
        for s in selected:
            parts.append(f"\n### {s.name}\n{self._read_skill(s.path)}")
        return "\n".join(parts)

    async def on_task_start(self, event: TaskStartEvent):
        # Only active when replace_block=True: strip the default <available_skills>
        # XML block so the model doesn't see a static listing alongside dynamic content.
        if not self.replace_block:
            yield event
            return
        prompt = _SKILLS_BLOCK_RE.sub("", event.system_prompt)
        if prompt != event.system_prompt:
            yield dataclasses.replace(event, system_prompt=prompt)
        else:
            yield event

    async def on_step_start(self, event: StepStartEvent):
        """Cache workspace and raw_messages for use in on_before_model."""
        self._cached_workspace = event.workspace
        self._cached_raw_messages = event.raw_messages
        yield event

    async def on_before_model(self, event: BeforeModelEvent):
        """Inject full content of query-matched skills as a user message."""
        workspace = self._cached_workspace
        skills_dir = self._resolve_skills_dir(workspace)
        # Preserve prior behavior: skip when nothing is configured anywhere.
        if not skills_dir and not self._extra_skills_dirs:
            logger.info("SkillLoader: no skills_dir resolved, skipping")
            yield event
            return
        home = getattr(workspace, "home", None) if workspace else None
        plugin_dirs = collect_plugin_skill_dirs(home)
        # Additive: expand extra_skills_dirs (plugin-provided roots) into
        # individual skill dirs, appended to whatever plugin_dirs already has.
        extra_dirs = plugin_dirs + expand_skill_roots(self._extra_skills_dirs)
        # Sentinel avoids SkillIndex's fallback to extensions/skills.
        primary = skills_dir if skills_dir else Path("/__no_primary_skills__")
        all_skills = self._get_index(primary, extra_dirs=extra_dirs).list_skills()
        # meta-skills are for the plan agent only, not the task agent
        all_skills = [s for s in all_skills if not s.name.startswith("meta-")]
        # Filter by enabled_skills when set (None = all enabled, e.g. CLI default)
        if self.enabled_skills is not None:
            skills = [s for s in all_skills if s.name in self.enabled_skills]
        else:
            skills = all_skills
        if not skills:
            logger.info(
                "SkillLoader: no skills found in %s (enabled=%s)",
                skills_dir,
                self.enabled_skills,
            )
            yield event
            return

        logger.info(
            "SkillLoader: %d/%d skills available in %s",
            len(skills),
            len(all_skills),
            skills_dir,
        )

        # Only inject at the start of a new user turn, not mid-task after tool calls.
        # When the last raw message is tool/assistant, the model is continuing an
        # in-progress chain — injecting skills docs here confuses it into responding
        # to the docs instead of completing the task.
        raw = self._cached_raw_messages
        if raw and raw[-1].role != "user":
            logger.info(
                "SkillLoader: skipping (last raw_message role=%s, not user)",
                raw[-1].role,
            )
            yield event
            return

        query = _last_user_text(event.messages)
        block = self._build_full_block(skills, query)
        if block:
            matched = [
                s.name
                for s, sc in sorted(
                    [(s, _score_skill(s, query)) for s in skills],
                    key=lambda x: x[1],
                    reverse=True,
                )
                if sc >= self.min_score
            ][: self.top_k]
            logger.info(
                "SkillLoader: injected %d skill(s) %s for query=%r",
                len(matched),
                matched,
                query[:80],
            )
            yield dataclasses.replace(
                event,
                messages=event.messages + (Message(role="user", content=block.strip()),),
            )
        else:
            logger.info(
                "SkillLoader: no skills matched query=%r (min_score=%d)",
                query[:80],
                self.min_score,
            )
            yield event

    def clear_caches(self) -> None:
        """Flush content and index caches.  Called by SkillRuntimePlugin on dir change."""
        self._content_cache.clear()
        self._indexes.clear()
        logger.info("SkillLoader: caches cleared")

    async def on_task_end(self, event: TaskEndEvent):
        yield event
