# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import copy
import json
from contextvars import ContextVar
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from ..logging import logger

if TYPE_CHECKING:
    from ..core.harness import HarnessConfig
    from ..core.model_config import ModelConfig
    from ..core.state import State
    from .base import Tool

# ---------------------------------------------------------------------------
# ContextVar — set by RunLoop before each tool call
# Keys: run_id, step_id, spawn_depth, state, tracer, model_config,
#       harness_config, child_harness_config
# ---------------------------------------------------------------------------

_spawn_ctx: ContextVar[dict] = ContextVar("spawn_ctx", default={})

SPAWN_TOOL_NAME = "spawn_subagent"

_MAX_SPAWN_DEPTH = 3


# ---------------------------------------------------------------------------
# Module-level tool function — reads parent configs from RunLoop context
# ---------------------------------------------------------------------------


async def spawn_subagent(
    task: str,
    model: str = "",
    system_prompt: str = "",
    tools: list = [],  # noqa: B006 — JSON schema default
    max_steps: int = 0,
    max_cost_usd: float = 0.0,
    wait: bool = True,
    label: str = "",
    share_workspace: bool = False,
) -> str:
    """Spawn a sub-agent to handle a delegated task.

    Args:
        task:            The sub-task description (becomes the child's first user message).
        model:           Model name override (empty = inherit parent's model).
        system_prompt:   System prompt for the child (empty = use default worker prompt).
        tools:           Allowed tool names (empty list = inherit all parent tools).
        max_steps:       Step budget override (0 = inherit parent's default).
        max_cost_usd:    Cost budget override (0.0 = no extra limit).
        wait:            True = block until child finishes and return its output.
                         False = fire-and-forget; result arrives as a user message.
        label:           Human-readable label for tracking async children.
        share_workspace: True = child shares parent workspace root (can read parent files).
                         False (default) = child gets an isolated subdir under parent
                         workspace; skills are still shared via the same AGENT_HOME.
    """
    ctx = _spawn_ctx.get()
    parent_model_config: "ModelConfig | None" = ctx.get("model_config")
    parent_harness_config: "HarnessConfig | None" = ctx.get("harness_config")
    preset_child_config: "HarnessConfig | None" = ctx.get("child_harness_config")
    parent_run_id: str = ctx.get("run_id", "")
    parent_state: "State | None" = ctx.get("state")
    current_depth: int = ctx.get("spawn_depth", 0)
    runtime_tracer: Any = ctx.get("tracer")

    if parent_model_config is None or parent_harness_config is None:
        return "Error: spawn context not available (tool called outside a RunLoop)."

    if current_depth >= _MAX_SPAWN_DEPTH:
        return (
            f"Cannot spawn sub-agent: maximum nesting depth ({_MAX_SPAWN_DEPTH}) reached. "
            "Complete this task without further delegation."
        )

    from ..core.harness import BaseTask
    from ..core.events import make_run_id

    child_run_id = make_run_id()

    overrides: dict = {
        "model": model,
        "system_prompt": system_prompt,
        "tools": tools,
    }
    child_depth = current_depth + 1

    # ── Model override ────────────────────────────────────────────────────────
    if overrides.get("model"):
        new_model = overrides["model"]
        try:
            new_main = copy.copy(parent_model_config.main)
            new_main.model = new_model
        except Exception:
            from ..providers.litellm_provider import LiteLLMProvider

            new_main = LiteLLMProvider(model=new_model)
        child_model_config = parent_model_config.copy(main=new_main)
    else:
        child_model_config = parent_model_config

    # ── HarnessConfig inheritance ─────────────────────────────────────────────
    # If the caller (e.g. gateway) pre-built a stripped child config, use it as
    # the base so IM-specific processors and workspace files don't leak in.
    if preset_child_config is not None:
        child_harness_config = _apply_child_overrides(
            preset_child_config,
            overrides,
            child_depth,
            _MAX_SPAWN_DEPTH,
            runtime_tracer,
            parent_run_id,
            child_run_id=child_run_id,
            share_workspace=share_workspace,
            parent_harness_config=parent_harness_config,
        )
    else:
        child_harness_config = _default_child_config(
            parent_harness_config,
            overrides,
            child_depth,
            _MAX_SPAWN_DEPTH,
            runtime_tracer,
            parent_run_id,
        )
        # Workspace: carve out an isolated subdir under the parent workspace.
        # share_workspace=True keeps the parent root so the child can read parent files.
        parent_ws = _resolve_workspace(getattr(parent_harness_config, "workspace", None))
        if parent_ws is not None and not share_workspace:
            try:
                child_harness_config = child_harness_config.copy(
                    workspace=parent_ws.child(child_run_id), init_workspace=False
                )
            except Exception:
                pass

    subtask = BaseTask(
        description=task,
        max_steps=max_steps if max_steps > 0 else 50,
        max_cost_usd=max_cost_usd if max_cost_usd > 0 else None,
        spawn_depth=child_depth,
    )

    effective_label = label or child_run_id[:8]

    child_harness = child_model_config.agentic(child_harness_config)

    # Notify tracer so the frontend knows a child agent is starting.
    if runtime_tracer is not None:
        try:
            from ..core.events import SpawnSubAgentEvent

            await runtime_tracer.on_event(
                SpawnSubAgentEvent(
                    run_id=parent_run_id,
                    step_id=ctx.get("step_id", 0),
                    child_run_id=child_run_id,
                    sub_task=subtask,
                )
            )
        except Exception:
            pass

    if wait:
        result = await child_harness.run(subtask, parent_run_id=parent_run_id)
        logger.debug(
            "Subagent {} (label={}) completed sync, run_id={}",
            effective_label,
            effective_label,
            result.run_id,
        )
        return result.final_output or "(no output)"

    # ── Asynchronous: fire-and-forget ─────────────────────────────────────────
    if parent_state is not None:
        from ..core.state import PendingSubagent

        parent_state.pending_subagents[effective_label] = PendingSubagent(
            label=effective_label,
            task=task,
            model=model,
            system_prompt=system_prompt,
            tools=list(tools),
        )

    async def _run_child() -> None:
        try:
            result = await child_harness.run(subtask, parent_run_id=parent_run_id)
            logger.debug(
                "Async subagent {} completed, run_id={}",
                effective_label,
                result.run_id,
            )
            if parent_state is not None:
                from ..core.events import Message

                parent_state.add_message(
                    Message(
                        role="user",
                        content=(
                            f"[Subagent label={effective_label} run_id={result.run_id}] "
                            f"Task completed:\n{result.final_output or '(no output)'}"
                        ),
                    )
                )
                parent_state.pending_subagents.pop(effective_label, None)
        except Exception as exc:
            logger.error("Async subagent {} failed: {}", effective_label, exc)
            if parent_state is not None:
                from ..core.events import Message

                parent_state.add_message(
                    Message(
                        role="user",
                        content=f"[Subagent label={effective_label}] Task failed: {exc}",
                    )
                )
                parent_state.pending_subagents.pop(effective_label, None)

    asyncio.create_task(_run_child())
    return json.dumps({"status": "accepted", "label": effective_label})


# ---------------------------------------------------------------------------
# Module-level tool object — importable, serializable into ToolRegistryConfig
# ---------------------------------------------------------------------------

spawn_subagent_tool: "Tool" = None  # type: ignore[assignment]  # set below after _make_spawn_tool


def _make_spawn_tool() -> "Tool":
    from .base import Tool

    return Tool(
        name=SPAWN_TOOL_NAME,
        description=(
            "Spawn a sub-agent to handle a delegated task.\n\n"
            "Use wait=true (default) to block until the sub-agent completes and get its output "
            "directly as the tool result — ideal for sequential tasks where you need the result.\n\n"
            "Use wait=false for fire-and-forget parallel work. The result will arrive as a user "
            "message with the format: [Subagent label=<label>] Task completed: <output>. "
            "Track spawned children by label and only give your final answer after all expected "
            "completion messages have arrived.\n\n"
            "You can override the sub-agent's model, system prompt, and available tools. "
            "Empty values inherit from the parent agent."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The sub-task for the agent to complete.",
                },
                "model": {
                    "type": "string",
                    "description": "Model name override (empty = inherit parent model).",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "System prompt for the child agent (empty = inherit parent's).",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Registered tool names the sub-agent is allowed to use "
                        "(e.g. 'Bash', 'Read', 'Write', 'Glob', 'Grep'). "
                        "Leave empty to inherit all parent tools. "
                        "IMPORTANT: these are tool names, NOT skill names. "
                        "Skills listed in <available_skills> (docx, pdf, xlsx, …) are "
                        "filesystem scripts executed via Bash — they are never valid tool names here."
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Step budget override (0 = default 50).",
                },
                "max_cost_usd": {
                    "type": "number",
                    "description": "Cost budget in USD (0.0 = no extra limit).",
                },
                "wait": {
                    "type": "boolean",
                    "description": "True = sync (block for result). False = async fire-and-forget.",
                },
                "label": {
                    "type": "string",
                    "description": "Label for tracking async sub-agents (auto-generated if empty).",
                },
                "share_workspace": {
                    "type": "boolean",
                    "description": (
                        "True = child shares parent workspace root (can read/write parent files). "
                        "False (default) = child gets an isolated subdirectory under the parent "
                        "workspace; skills are still shared via AGENT_HOME."
                    ),
                },
            },
            "required": ["task"],
        },
        fn=spawn_subagent,
        tags=[],
        execution_target="local",
    )


spawn_subagent_tool = _make_spawn_tool()


# ---------------------------------------------------------------------------
# Factory: build_spawn_fn (used by meta_harness workers)
# ---------------------------------------------------------------------------


def build_spawn_fn(
    model_config: "ModelConfig",
    parent_harness_config: "HarnessConfig",
    child_config_fn: Callable | None = None,
    max_depth: int = _MAX_SPAWN_DEPTH,
) -> Callable[..., Coroutine]:
    """Return an async callable that spawns a child agent.

    This factory is used by the meta-harness worker system
    (``trajectory_digester``) to create a spawn function with a custom
    ``child_config_fn`` that controls how the child's HarnessConfig is
    built (e.g. restricting tools, injecting a static system prompt).

    Parameters
    ----------
    model_config:
        The ModelConfig to use for the child agent.
    parent_harness_config:
        The parent's HarnessConfig (passed to child_config_fn).
    child_config_fn:
        Optional ``(parent_config, overrides, depth, max_depth,
        tracer, run_id) -> HarnessConfig`` factory.  Falls back to
        :func:`_default_child_config` when ``None``.
    max_depth:
        Maximum spawn nesting depth.
    """
    effective_cfg_fn = child_config_fn or _default_child_config

    async def _spawn(
        task: str,
        model: str = "",
        system_prompt: str = "",
        tools: list = [],  # noqa: B006
        max_steps: int = 0,
        max_cost_usd: float = 0.0,
        wait: bool = True,
        label: str = "",
    ) -> str:
        from ..core.harness import BaseTask

        overrides = {
            "model": model,
            "system_prompt": system_prompt,
            "tools": tools,
        }
        child_depth = 1  # workers are always depth-1 children

        child_config = effective_cfg_fn(
            parent_harness_config,
            overrides,
            child_depth,
            max_depth,
            None,  # runtime_tracer
            "",  # parent_run_id
        )

        # Model override
        if model:
            try:
                new_main = copy.copy(model_config.main)
                new_main.model = model
            except Exception:
                from ..providers.litellm_provider import LiteLLMProvider

                new_main = LiteLLMProvider(model=model)
            child_mc = model_config.copy(main=new_main)
        else:
            child_mc = model_config

        harness = child_mc.agentic(child_config)
        child_task = BaseTask(
            description=task,
            max_steps=max_steps or 50,
            max_cost_usd=max_cost_usd or 5.0,
        )

        try:
            result = await harness.run(child_task)
            return result.final_output or "(child agent produced no output)"
        except Exception as exc:
            return f"Error in child agent: {type(exc).__name__}: {exc}"

    return _spawn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_workspace(wc: Any) -> Any:
    """Convert WorkspaceConfig → Workspace so callers can use .child()."""
    from ..core.config_schema import WorkspaceConfig
    from ..workspace.workspace import Workspace
    from pathlib import Path as _Path

    if wc is None or isinstance(wc, Workspace):
        return wc
    if isinstance(wc, WorkspaceConfig):
        if wc.root is not None:
            return Workspace(
                root=_Path(wc.root),
                agent_id=wc.agent_id,
                project=wc.project,
                mode=wc.mode,
                home=_Path(wc.home) if wc.home is not None else None,
                parent_id=wc.parent_id,
            )
        if wc.home is not None:
            return Workspace(
                agent_id=wc.agent_id,
                project=wc.project,
                mode=wc.mode,
                home=_Path(wc.home),
                parent_id=wc.parent_id,
            )
    return None


def _default_child_config(
    parent_harness_config: "HarnessConfig",
    overrides: dict,
    child_depth: int,
    max_depth: int,
    runtime_tracer: Any = None,
    parent_run_id: str = "",
) -> "HarnessConfig":
    """Build child HarnessConfig inheriting parent behavior pipeline with overrides applied."""
    child_config = parent_harness_config
    if child_config.init_workspace:
        child_config = child_config.copy(init_workspace=False)

    is_leaf = child_depth >= max_depth

    # ── Tool restriction ──────────────────────────────────────────────────────
    from ..tools.inmemory import InMemoryToolRegistry
    from ..core.config_schema import ToolRegistryConfig as _TRC
    from ..core.harness import _build_tool_registry_from_config

    existing_cfg = child_config.tool_registry
    if isinstance(existing_cfg, _TRC):
        existing = _build_tool_registry_from_config(existing_cfg)
    elif existing_cfg is not None:
        existing = existing_cfg
    else:
        existing = InMemoryToolRegistry()

    if overrides.get("tools"):
        allowed = set(overrides["tools"])
        if is_leaf:
            # Leaf agents must not receive spawn_subagent even if explicitly requested.
            allowed.discard(SPAWN_TOOL_NAME)
        else:
            allowed.add(SPAWN_TOOL_NAME)
        new_registry = InMemoryToolRegistry()
        for tool_obj in existing._tools.values():
            if tool_obj.name in allowed:
                new_registry.register(tool_obj)
        child_config = child_config.copy(tool_registry=new_registry)
    elif is_leaf:
        # No explicit tool filter but this is a leaf — strip spawn_subagent.
        new_registry = InMemoryToolRegistry()
        for tool_obj in existing._tools.values():
            if tool_obj.name != SPAWN_TOOL_NAME:
                new_registry.register(tool_obj)
        child_config = child_config.copy(tool_registry=new_registry)

    # ── Tracer: nest child under parent run dir ───────────────────────────────
    import os as _os
    from ..tracing.journal import HarnessJournal as _HJ
    from ..core.config_schema import TracerConfig as _TC

    effective_tracer = runtime_tracer or parent_harness_config.tracer
    if isinstance(effective_tracer, _HJ) and parent_run_id:
        _parent_session = effective_tracer.session_id or parent_run_id
        child_base_dir = _os.path.join(effective_tracer.base_dir, _parent_session, "subagents")
        child_tracer: Any = _TC(
            base_dir=child_base_dir,
            export_jsonl=effective_tracer.export_jsonl,
            silent=effective_tracer.silent,
        )
    elif isinstance(effective_tracer, _HJ):
        child_tracer = _TC(
            base_dir=effective_tracer.base_dir,
            export_jsonl=effective_tracer.export_jsonl,
            silent=effective_tracer.silent,
        )
    else:
        child_tracer = effective_tracer
    child_config = child_config.copy(tracer=child_tracer)

    # ── Processor patching ────────────────────────────────────────────────────
    child_config = _patch_processors_for_child(
        child_config,
        system_prompt_override=overrides.get("system_prompt", ""),
        child_depth=child_depth,
        max_depth=max_depth,
    )

    return child_config


def _patch_processors_for_child(
    config: "HarnessConfig",
    system_prompt_override: str,
    child_depth: int,
    max_depth: int,
) -> "HarnessConfig":
    from ..processors.context.system_prompt import SystemPromptProcessor
    from ..processors.context.strategies.system_prompt.default import DefaultSystemPromptBuilder
    from ..core.builder import _instantiate

    all_procs: list = list(config.processors or []) + list(getattr(config, "_rt_procs", None) or [])
    new_procs: list = []
    for p in all_procs:
        if isinstance(p, dict) and "_target_" in p:
            try:
                inst: Any = _instantiate(p)
            except Exception:
                new_procs.append(p)
                continue
        else:
            inst = p

        if isinstance(inst, SystemPromptProcessor):
            if system_prompt_override:
                new_builder: Any = _StaticSystemPromptBuilder(system_prompt_override)
            elif isinstance(inst.system_builder, DefaultSystemPromptBuilder):
                new_builder = DefaultSystemPromptBuilder(
                    spawn_depth=child_depth,
                    max_spawn_depth=max_depth,
                    persona_root=inst.system_builder.persona_root,
                    extra_skills_dirs=inst.system_builder.extra_skills_dirs,
                )
            else:
                new_builder = inst.system_builder
            new_procs.append(SystemPromptProcessor(new_builder))
        else:
            new_procs.append(p)

    return config.copy(processors=new_procs, _rt_procs=[])


class _StaticSystemPromptBuilder:
    def __init__(self, text: str) -> None:
        self._text = text

    async def build(self, workspace: Any = None) -> str:
        return self._text


def _apply_child_overrides(
    base_config: "HarnessConfig",
    overrides: dict,
    child_depth: int,
    max_depth: int,
    runtime_tracer: Any,
    parent_run_id: str,
    child_run_id: str = "",
    share_workspace: bool = False,
    parent_harness_config: "HarnessConfig | None" = None,
) -> "HarnessConfig":
    """Apply overrides onto a pre-built child base config (e.g. from gateway).

    Unlike _default_child_config, this does NOT inherit from the parent pipeline —
    the caller is responsible for providing a stripped base. We only handle:
    - tool restriction / leaf-node spawn_subagent removal
    - system_prompt override
    - workspace: isolated subdir under parent (or shared if share_workspace=True)
    - tracer nesting
    """
    child_config = base_config
    if child_config.init_workspace:
        child_config = child_config.copy(init_workspace=False)

    is_leaf = child_depth >= max_depth

    # ── Tool restriction ──────────────────────────────────────────────────────
    from ..tools.inmemory import InMemoryToolRegistry
    from ..core.config_schema import ToolRegistryConfig as _TRC
    from ..core.harness import _build_tool_registry_from_config

    # Use base_config's tool registry as the allowed set — the caller (e.g. gateway)
    # is responsible for providing a curated registry without IM-specific tools.
    # Fall back to parent registry only when base_config has no registry at all.
    base_tool_cfg = base_config.tool_registry
    fallback_cfg = parent_harness_config.tool_registry if parent_harness_config else None
    source_cfg = base_tool_cfg if base_tool_cfg is not None else fallback_cfg
    if isinstance(source_cfg, _TRC):
        existing = _build_tool_registry_from_config(source_cfg)
    elif source_cfg is not None:
        existing = source_cfg
    else:
        existing = InMemoryToolRegistry()

    if overrides.get("tools"):
        allowed = set(overrides["tools"])
        if is_leaf:
            allowed.discard(SPAWN_TOOL_NAME)
        else:
            allowed.add(SPAWN_TOOL_NAME)
        new_registry = InMemoryToolRegistry()
        for tool_obj in existing._tools.values():
            if tool_obj.name in allowed:
                new_registry.register(tool_obj)
        child_config = child_config.copy(tool_registry=new_registry)
    elif is_leaf:
        new_registry = InMemoryToolRegistry()
        for tool_obj in existing._tools.values():
            if tool_obj.name != SPAWN_TOOL_NAME:
                new_registry.register(tool_obj)
        child_config = child_config.copy(tool_registry=new_registry)
    else:
        # Non-leaf, no explicit filter — inherit full tool set from parent.
        child_config = child_config.copy(tool_registry=existing)

    # ── Workspace ────────────────────────────────────────────────────────────
    # share_workspace=True  → reuse parent root (child can read parent files)
    # share_workspace=False → carve an isolated subdir: parent_root/agents/{child_run_id}/
    #                         home is inherited so skills are shared
    parent_ws = _resolve_workspace(parent_harness_config.workspace if parent_harness_config else None)
    if share_workspace:
        child_config = child_config.copy(workspace=parent_ws)
    elif parent_ws is not None:
        try:
            child_ws = parent_ws.child(child_run_id or "subagent")
            child_config = child_config.copy(workspace=child_ws, init_workspace=False)
        except Exception:
            child_config = child_config.copy(workspace=None)
    else:
        child_config = child_config.copy(workspace=None)

    # ── System prompt override ────────────────────────────────────────────────
    child_config = _patch_processors_for_child(
        child_config,
        system_prompt_override=overrides.get("system_prompt", ""),
        child_depth=child_depth,
        max_depth=max_depth,
    )

    # ── Tracer nesting ────────────────────────────────────────────────────────
    import os as _os
    from ..tracing.journal import HarnessJournal as _HJ
    from ..core.config_schema import TracerConfig as _TC

    effective_tracer = runtime_tracer or (parent_harness_config.tracer if parent_harness_config else None)
    if isinstance(effective_tracer, _HJ) and parent_run_id:
        _parent_session = effective_tracer.session_id or parent_run_id
        child_base_dir = _os.path.join(effective_tracer.base_dir, _parent_session, "subagents")
        child_tracer: Any = _TC(
            base_dir=child_base_dir,
            export_jsonl=effective_tracer.export_jsonl,
            silent=effective_tracer.silent,
        )
        child_config = child_config.copy(tracer=child_tracer)

    return child_config
