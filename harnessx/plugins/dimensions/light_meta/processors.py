# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from ....core.processor import MultiHookProcessor

if TYPE_CHECKING:
    from ....core.events import TaskEndEvent, TaskStartEvent
    from ....meta.strategy import MetaStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plan processor
# ---------------------------------------------------------------------------


class MetaPlanProcessor(MultiHookProcessor):
    """Read meta-skills at task start and adapt tools + system prompt.

    - If no meta-skills exist: pass-through (zero overhead)
    - If meta-skills exist: call strategy.plan() to get adaptations,
      then apply tool changes, processor changes, and inject skill catalog
    """

    _singleton_group = "meta.plan"
    _order = 0  # runs first

    def __init__(self, strategy: "MetaStrategy", skills_dir: Path) -> None:
        self._strategy = strategy
        self._skills_dir = skills_dir
        self.last_cost_usd: float = 0.0
        self._baseline_snapshot: dict | None = None  # lazy init on first task

    # ── Snapshot / Restore helpers ─────────────────────────────────────

    def _rt_procs(self) -> "dict | None":
        """Return the live runtime processor dict, or None if not bound."""
        rt = getattr(self, "_harness_runtime", None)
        if rt is not None:
            return rt.processors
        # Fallback: _harness_runtime not bound yet (tests / direct construction)
        return None

    def _snapshot_processors(self) -> dict:
        """Capture baseline processor lists + their mutable attrs (shallow copy)."""
        procs_dict = self._rt_procs() or {}
        snapshot: dict = {
            "lists": {k: list(v) for k, v in procs_dict.items()},
            "attrs": {},
        }
        for procs in procs_dict.values():
            for p in procs:
                group = getattr(p, "_singleton_group", None)
                if group and group not in snapshot["attrs"]:
                    snapshot["attrs"][group] = {k: v for k, v in p.__dict__.items() if not k.startswith("_")}
        return snapshot

    def _restore_baseline(self) -> None:
        """Restore processor lists and attrs to baseline snapshot."""
        if self._baseline_snapshot is None:
            return
        rt = getattr(self, "_harness_runtime", None)
        if rt is None:
            return
        snap = self._baseline_snapshot
        # Restore hook lists (re-adds any removed processors)
        rt.processors.clear()
        rt.processors.update({k: list(v) for k, v in snap["lists"].items()})
        # Restore processor attrs (undoes parameter changes)
        for procs in rt.processors.values():
            for p in procs:
                group = getattr(p, "_singleton_group", None)
                if group and group in snap["attrs"]:
                    for attr, val in snap["attrs"][group].items():
                        setattr(p, attr, val)

    # ── Main hook ───────────────────────────────────────────────────────

    async def on_task_start(
        self,
        event: "TaskStartEvent",
    ) -> AsyncIterator["TaskStartEvent"]:
        # Snapshot baseline on first call; restore on subsequent calls
        # so each task starts from the original processor configuration.
        if getattr(self, "_harness_runtime", None) is not None:
            if self._baseline_snapshot is None:
                self._baseline_snapshot = self._snapshot_processors()
            else:
                self._restore_baseline()

        from ....workspace.skill_index import SkillIndex

        idx = SkillIndex(self._skills_dir)
        all_skills = idx.list_skills()
        meta_skills_raw = [s for s in all_skills if s.name.startswith("meta-")]
        task_skills = [s for s in all_skills if not s.name.startswith("meta-")]

        if not meta_skills_raw:
            logger.info("[Plan] No meta-skills found — using baseline config")
            if task_skills:
                logger.info("[Plan] %d task-skills available (injected by SystemPromptProcessor):", len(task_skills))
                for s in task_skills:
                    logger.info("[Plan]   task-skill: %s", s.name)
            yield event
            return

        # Parse meta-skill frontmatter for strategy
        meta_skills: list[dict[str, str]] = []
        for ms in meta_skills_raw:
            fm: dict[str, str] = {
                "name": ms.name,
                "description": ms.description,
                "path": str(ms.path),
            }
            try:
                text = ms.path.read_text(encoding="utf-8")
                for line in text.splitlines()[1:]:
                    if line.strip() == "---":
                        break
                    if ":" in line:
                        k, _, v = line.partition(":")
                        fm[k.strip()] = v.strip().strip('"').strip("'")
            except Exception:
                pass
            meta_skills.append(fm)

        # Get current tool names
        available_tools = [t.name for t in event.tools]

        # Run plan
        task_desc = event.task_description
        adaptation = await self._strategy.plan(task_desc, available_tools, meta_skills)

        # Track cost
        if hasattr(self._strategy, "last_plan_cost_usd"):
            self.last_cost_usd = self._strategy.last_plan_cost_usd

        # ── Apply tool removals ──────────────────────────────────────────
        # Only filter the per-task event.tools; do NOT mutate the shared
        # tool_registry so subsequent tasks still start with the full baseline.
        if adaptation.tool_removals:
            rm_set = set(adaptation.tool_removals)
            new_tools = tuple(t for t in event.tools if t.name not in rm_set)
            event = dataclasses.replace(event, tools=new_tools)
            logger.info("[Plan] Applied tool removals: %s", adaptation.tool_removals)

        # ── Apply tool additions ─────────────────────────────────────────
        # Add tools to the per-task event only; do NOT mutate the shared
        # tool_registry so subsequent tasks still start with the full baseline.
        if adaptation.tool_additions:
            from ....tools.builtin import (
                bash_tool,
                read_tool,
                write_tool,
                edit_tool,
                glob_tool,
                grep_tool,
                web_search_tool,
                web_fetch_tool,
                browser_tool,
            )

            _TOOL_MAP = {
                "Bash": bash_tool,
                "Read": read_tool,
                "Write": write_tool,
                "Edit": edit_tool,
                "Glob": glob_tool,
                "Grep": grep_tool,
                "web_search": web_search_tool,
                "web_fetch": web_fetch_tool,
                "browser": browser_tool,
            }
            existing = {t.name for t in event.tools}
            added = []
            added_schemas = []
            for name in adaptation.tool_additions:
                tool = _TOOL_MAP.get(name)
                if tool and tool.name not in existing:
                    added_schemas.append(tool.to_schema())
                    added.append(name)
                    existing.add(tool.name)
            if added:
                event = dataclasses.replace(event, tools=event.tools + tuple(added_schemas))
                logger.info("[Plan] Applied tool additions: %s", added)

        # ── Apply processor removals ─────────────────────────────────────
        rt_procs = self._rt_procs()
        if adaptation.processor_removals and rt_procs is not None:
            rm_groups = set(adaptation.processor_removals)
            for hook in list(rt_procs.keys()):
                rt_procs[hook] = [p for p in rt_procs[hook] if getattr(p, "_singleton_group", None) not in rm_groups]
            logger.info("[Plan] Applied processor removals: %s", adaptation.processor_removals)

        # ── Apply processor config ──────────────────────────────────────
        if adaptation.processor_config and rt_procs is not None:
            all_procs: dict = {}
            for procs in rt_procs.values():
                for p in procs:
                    group = getattr(p, "_singleton_group", None)
                    if group and group not in all_procs:
                        all_procs[group] = p
            for group, params in adaptation.processor_config.items():
                proc = all_procs.get(group)
                if proc and isinstance(params, dict):
                    for attr, val in params.items():
                        if hasattr(proc, attr):
                            setattr(proc, attr, val)
                    logger.info("[Plan] Applied processor config: %s -> %s", group, params)
                else:
                    logger.warning("[Plan] Processor %r not found, skipping config", group)

        # Skill catalog injection is handled by SystemPromptProcessor
        # (skills_dir propagated via LightMetaPlugin.setup)
        if task_skills:
            logger.info("[Plan] %d task-skills available (injected by SystemPromptProcessor):", len(task_skills))
            for s in task_skills:
                logger.info("[Plan]   task-skill: %s — %s", s.name, s.description[:80])

        logger.info("[Plan] Adaptation complete. Final tool count: %d", len(event.tools))

        yield event


# ---------------------------------------------------------------------------
# Reflect processor
# ---------------------------------------------------------------------------


class MetaReflectProcessor(MultiHookProcessor):
    """Analyze trajectory at task end and write skills via strategy.reflect()."""

    _singleton_group = "meta.reflect"
    _order = 99  # runs last, after evaluation

    def __init__(
        self,
        strategy: "MetaStrategy",
        skills_dir: Path,
        enabled: bool = True,
    ) -> None:
        self._strategy = strategy
        self._skills_dir = skills_dir
        self._enabled = enabled
        self.last_cost_usd: float = 0.0

    async def on_task_end(
        self,
        event: "TaskEndEvent",
    ) -> AsyncIterator["TaskEndEvent"]:
        if not self._enabled:
            yield event
            return

        eval_result = getattr(event, "eval_result", None)
        eval_passed = eval_result.passed if eval_result else False
        eval_reason = eval_result.reason if eval_result else ""
        task_desc = getattr(event, "task_description", "")

        # Build full trajectory text (all 5 sections)
        trajectory_text = self._build_trajectory_text(event)

        # Run reflect
        try:
            await self._strategy.reflect(
                task_description=task_desc,
                trajectory_text=trajectory_text,
                eval_passed=eval_passed,
                eval_reason=eval_reason,
            )
            if hasattr(self._strategy, "last_reflect_cost_usd"):
                self.last_cost_usd = self._strategy.last_reflect_cost_usd
        except Exception:
            logger.exception("[Reflect] Reflect processor failed")

        yield event

    def _build_trajectory_text(self, event: "TaskEndEvent") -> str:
        """Build trajectory summary from TaskEndEvent.

        Note: TaskEndEvent has limited info compared to full HarnessResult.
        For richer trajectories (with step details), use the recipe-level
        _build_trajectory_text() that takes HarnessResult + config.
        """
        lines: list[str] = ["# Trajectory Summary\n"]
        lines.append("## Result\n")
        lines.append(f"- exit_reason: {getattr(event, 'exit_reason', '?')}")
        lines.append(f"- total_steps: {getattr(event, 'total_steps', '?')}")
        final_out = getattr(event, "final_output", "") or ""
        if final_out:
            lines.append(f"- final_output: {final_out[:500]}")
        er = getattr(event, "eval_result", None)
        if er:
            lines.append(f"- eval_passed: {er.passed}")
            lines.append(f"- eval_score: {er.score}")
            lines.append(f"- eval_reason: {er.reason}")
        return "\n".join(lines)
