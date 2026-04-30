# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from harnessx.core.processor import MultiHookProcessor
from harnessx.plugins.base import HarnessPlugin
from harnessx.processors._sp_utils import sp_append

from ._core.compression import compress_session, count_tool_calls
from ._core.engine import WorkflowEngine
from ._core.state import WorkflowPluginState
from ._core.tool import build_workflow_tools

if TYPE_CHECKING:
    from harnessx.core.events import TaskStartEvent

_logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _resolve_workflow_dir(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        from harnessx.home import agent_home

        return str(agent_home() / "workflows")
    except Exception:
        return os.path.expanduser("~/.harnessx/workflows")


# ── WorkflowRecallProcessor ────────────────────────────────────────────────────

_RECALL_TAG = "workflow-recall"
_RECALL_HEADER = """\
## Stored Workflows (Procedural Memory)

The following workflows were previously learned from similar tasks.
If the current task matches one of these, call `flow_exec(name=..., params=...)` to execute it directly.
"""


class WorkflowRecallProcessor(MultiHookProcessor):
    """Inject matching stored workflow candidates into the system prompt."""

    _order = 6  # before WorkflowInternalizationProcessor (8) and WorkflowGuidanceProcessor (7)

    def __init__(self, workflow_dir: str) -> None:
        self._workflow_dir = workflow_dir

    async def on_task_start(self, event: "TaskStartEvent") -> AsyncIterator["TaskStartEvent"]:
        candidates = self._find_candidates(event.task_description)
        if candidates:
            section = self._format_section(candidates)
            yield dataclasses.replace(
                event,
                system_prompt=sp_append(event.system_prompt, section),
            )
        else:
            yield event

    def _find_candidates(self, task_description: str) -> list[dict]:
        """Scan workflow_dir for YAMLs that match task_description."""
        wf_path = Path(self._workflow_dir)
        if not wf_path.exists():
            return []

        query_words = set(_tokenize(task_description))
        candidates: list[tuple[int, dict]] = []

        for yaml_file in wf_path.glob("*.yaml"):
            try:
                import yaml  # type: ignore[import]

                data = yaml.safe_load(yaml_file.read_text())
                if not isinstance(data, dict):
                    continue
                score = _score_match(data, query_words)
                if score > 0:
                    candidates.append(
                        (
                            score,
                            {
                                "name": data.get("name", yaml_file.stem),
                                "description": data.get("description", ""),
                                "tags": data.get("tags") or [],
                                "trigger_patterns": data.get("trigger_patterns") or [],
                                "params": data.get("params") or [],
                            },
                        )
                    )
            except Exception:
                continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in candidates[:3]]  # top 3

    def _format_section(self, candidates: list[dict]) -> str:
        lines = [f"\n\n<{_RECALL_TAG}>", _RECALL_HEADER.strip()]
        for c in candidates:
            params_str = ""
            if c["params"]:
                param_names = [p.get("name", "?") for p in c["params"]]
                params_str = f" (params: {', '.join(param_names)})"
            lines.append(f"\n### `{c['name']}`{params_str}")
            if c["description"]:
                lines.append(c["description"])
            if c["trigger_patterns"]:
                lines.append("Triggers: " + ", ".join(f'"{p}"' for p in c["trigger_patterns"]))
        lines.append(f"</{_RECALL_TAG}>")
        return "\n".join(lines)


def _tokenize(text: str) -> list[str]:
    import re

    return [w.lower() for w in re.split(r"[\s\-_/.,;:!?]+", text) if len(w) > 2]


def _score_match(data: dict, query_words: set[str]) -> int:
    """Score how well a workflow matches a set of query words."""
    score = 0
    name_words = set(_tokenize(data.get("name", "")))
    desc_words = set(_tokenize(data.get("description", "")))
    tag_words = set(_tokenize(" ".join(data.get("tags") or [])))
    pattern_words = set(_tokenize(" ".join(data.get("trigger_patterns") or [])))

    score += 3 * len(query_words & name_words)
    score += 2 * len(query_words & tag_words)
    score += 2 * len(query_words & pattern_words)
    score += 1 * len(query_words & desc_words)
    return score


# ── WorkflowInternalizationProcessor ──────────────────────────────────────────


class WorkflowInternalizationProcessor(MultiHookProcessor):
    """At task_start, check if the previous task should be internalized.

    State (per plugin instance, also persisted to disk):
    - task_start_idx: message index where the current task begins
    - internalized_idxs: task_start_idx values already internalized

    Logic (runs at the START of task N+1, looks back at task N):
    1. Determine message range for task N: [prev_start, curr_start)
    2. Count tool calls in that range — if < complexity_threshold, skip
    3. If prev_start already in internalized_idxs, skip
    4. If judge_model set: call lightweight judge to confirm completion
    5. Fire-and-forget sub-harness: compress session → extractor
    6. Mark prev_start as internalized
    """

    _order = 8

    def __init__(
        self,
        plugin_state: WorkflowPluginState,
        workflow_dir: str,
        judge_model: str | None,
        extractor_model: str,
        complexity_threshold: int,
    ) -> None:
        self._state = plugin_state
        self._workflow_dir = workflow_dir
        self._judge_model = judge_model
        self._extractor_model = extractor_model
        self._complexity_threshold = complexity_threshold

    async def on_task_start(self, event: "TaskStartEvent") -> AsyncIterator["TaskStartEvent"]:
        yield event  # always pass through immediately — internalization is background

        messages = list(event.state.messages) if event.state else []
        curr_start = max(0, len(messages) - 1)  # index of new user message (last one)

        prev_start = self._state.task_start_idx
        self._state.task_start_idx = curr_start

        # Nothing to internalize on first task
        if prev_start >= curr_start:
            return

        segment = messages[prev_start:curr_start]
        if not segment:
            return

        # Complexity gate
        tool_call_count = count_tool_calls(segment)
        if tool_call_count < self._complexity_threshold:
            _logger.debug(
                "workflow: skip internalization (tool calls %d < threshold %d)",
                tool_call_count,
                self._complexity_threshold,
            )
            return

        # Deduplication gate
        if self._state.is_internalized(prev_start):
            _logger.debug(
                "workflow: skip internalization (already internalized idx=%d)",
                prev_start,
            )
            return

        # Fire background internalization (don't await — non-blocking)
        task_desc = event.task_description or ""
        asyncio.create_task(
            self._internalize(prev_start, segment, task_desc),
            name=f"workflow-internalize-{prev_start}",
        )

    async def _internalize(
        self,
        prev_start: int,
        segment: list,
        task_desc: str,
    ) -> None:
        """Background: judge completion, compress, then run extractor."""
        try:
            # Step 1: judge completion
            if self._judge_model:
                from ._core.judge import judge_task_complete

                completed = await judge_task_complete(segment, task_desc, self._judge_model)
                if not completed:
                    _logger.debug("workflow: judge said not complete — skip internalization")
                    return

            # Step 2: compress session
            compressed = compress_session(segment)

            # Step 3: run extractor sub-harness
            from ._core.extractor import spawn_extractor

            done_future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
            await spawn_extractor(
                messages=compressed,
                workflow_dir=self._workflow_dir,
                extractor_model=self._extractor_model,
                on_done=done_future,
            )

            # Step 4: mark as internalized only if extractor succeeded
            if done_future.done() and done_future.result():
                self._state.mark_internalized(prev_start)
                _logger.info("workflow: internalized segment starting at msg idx=%d", prev_start)
            else:
                _logger.debug("workflow: extractor reported failure for idx=%d", prev_start)

        except Exception as exc:
            _logger.warning("workflow: internalization error: %s", exc, exc_info=True)


# ── Guidance processor ─────────────────────────────────────────────────────────

_GUIDANCE = """
## Workflow Tools

**`flow`** — declare and execute an ad-hoc multi-step shell pipeline.
Use for novel data-collection tasks where you want labelled outputs and optional
approval gates.  Steps can reference prior outputs as `$step_id`.

**`flow_exec`** — execute a *stored* workflow by name.
Use when a matching workflow appears in the recall section above.  Pass params
to fill in variable parts (`app_name`, `namespace`, etc.).  This replays a
previously learned procedure directly, saving multi-turn reasoning tokens.

**`flow_resume`** — resume a paused workflow after an approval gate.

### flow step fields
| Field      | Purpose                                              |
|------------|------------------------------------------------------|
| `id`       | Step name; output accessible as `$id`               |
| `shell`    | Shell command; supports `$ref` interpolation         |
| `condition`| Skip when falsy (`$prev.success`, `$score > 0.7`)   |
| `approval` | Pause and require human review before this step      |

### $ref syntax
| Expression        | Value                     |
|-------------------|---------------------------|
| `$step_id`        | Full stdout of that step  |
| `$step_id.success`| "true" or "false"         |
"""

_GUIDANCE_TAG = "workflow-guidance"


class WorkflowGuidanceProcessor(MultiHookProcessor):
    """Inject workflow usage guidance into the system prompt once per task."""

    _order = 7

    async def on_task_start(self, event: "TaskStartEvent") -> AsyncIterator["TaskStartEvent"]:
        section = f"\n\n<{_GUIDANCE_TAG}>\n{_GUIDANCE.strip()}\n</{_GUIDANCE_TAG}>\n"
        yield dataclasses.replace(event, system_prompt=sp_append(event.system_prompt, section))


# ── WorkflowPlugin ─────────────────────────────────────────────────────────────


class WorkflowPlugin(HarnessPlugin):
    """Procedural memory plugin: learns and replays complex task workflows.

    At the end of a complex multi-turn task, the plugin internalises the
    procedure as a YAML workflow file.  On future similar tasks, it injects
    matching workflow candidates into the system prompt so the agent can call
    ``flow_exec`` to replay the procedure directly.

    Args:
        judge_model:          Small model for completion judgment.  Set to a
                              lightweight model (e.g. haiku) for low cost.
                              If None, judgment step is skipped (all complex
                              tasks are internalized without completion check).
        extractor_model:      Model for the sub-harness that writes YAMLs.
                              Defaults to the system ANTHROPIC_DEFAULT_MODEL env
                              var, then falls back to a warning + no extraction.
        workflow_dir:         Directory where YAML workflows are stored.
                              Defaults to ~/.harnessx/workflows/.
        complexity_threshold: Minimum number of tool calls in a task segment
                              before internalization is triggered (default 10).
        guidance:             Whether to inject workflow tool usage guidance into
                              the system prompt (default True).
        recall:               Whether to inject matching workflows into the system
                              prompt at task start (default True).
        internalize:          Whether to run the internalization processor
                              (default True).  Set False to use stored workflows
                              without learning new ones.
    """

    name = "workflow"
    version = "0.1.0"
    description = (
        "Procedural memory: internalise complex tasks as reusable workflow YAMLs, "
        "recall matching procedures at task start, execute via flow_exec."
    )

    def __init__(
        self,
        judge_model: str | None = None,
        extractor_model: str | None = None,
        workflow_dir: str | None = None,
        complexity_threshold: int = 10,
        guidance: bool = True,
        recall: bool = True,
        internalize: bool = True,
    ) -> None:
        super().__init__()

        self._workflow_dir = _resolve_workflow_dir(workflow_dir)

        # Warn if extractor_model not configured
        _extractor_model = extractor_model or os.environ.get("ANTHROPIC_DEFAULT_MODEL", "")
        if internalize and not _extractor_model:
            warnings.warn(
                "WorkflowPlugin: extractor_model not set and ANTHROPIC_DEFAULT_MODEL env var "
                "is empty.  Workflow internalization will be disabled.  Set extractor_model= "
                "or ANTHROPIC_DEFAULT_MODEL to enable learning.",
                stacklevel=2,
            )
            internalize = False

        if judge_model is None:
            _logger.debug(
                "WorkflowPlugin: judge_model not set — completion judgment disabled; "
                "all complex tasks will be internalized unconditionally."
            )

        self._engine = WorkflowEngine()
        self.tools = build_workflow_tools(self._engine, self._workflow_dir)

        self.processors = []

        # Recall: inject matching workflows into system prompt
        if recall:
            self.processors.append(WorkflowRecallProcessor(self._workflow_dir))

        # Guidance: inject tool usage docs
        if guidance:
            self.processors.append(WorkflowGuidanceProcessor())

        # Internalization: learn from completed tasks
        if internalize:
            plugin_state = WorkflowPluginState(self._workflow_dir)
            self.processors.append(
                WorkflowInternalizationProcessor(
                    plugin_state=plugin_state,
                    workflow_dir=self._workflow_dir,
                    judge_model=judge_model,
                    extractor_model=_extractor_model,
                    complexity_threshold=complexity_threshold,
                )
            )
