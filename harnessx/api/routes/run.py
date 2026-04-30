# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from harnessx.api.models import RunRequest, RunResponse
from harnessx.api.sse_tracer import _sse

router = APIRouter()

_DROP = object()

# run_id → SSE queue; cleared after stream ends.
_runs: dict[str, asyncio.Queue] = {}
# run_id → background execution task; cleared when _execute_run finishes.
_run_tasks: dict[str, asyncio.Task] = {}


async def shutdown_active_runs() -> None:
    """Best-effort shutdown hook: cancel all active run tasks and await exit."""
    tasks = [t for t in list(_run_tasks.values()) if t is not None and not t.done()]
    for task in tasks:
        try:
            task.cancel()
        except Exception:
            pass
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@router.post("/run", response_model=RunResponse)
async def start_run(req: RunRequest):
    """Start a harness run. Returns run_id + session_id; stream output via /api/run/{id}/stream."""
    run_id = str(uuid.uuid4())
    session_id = req.session_id or str(uuid.uuid4())

    queue: asyncio.Queue = asyncio.Queue()
    _runs[run_id] = queue
    _run_tasks[run_id] = asyncio.create_task(_execute_run(run_id, session_id, req, queue))
    return RunResponse(run_id=run_id, session_id=session_id)


@router.get("/run/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE stream for a run. Events: token | step_end | done | error."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="run not found")
    queue = _runs[run_id]
    return StreamingResponse(
        _sse_generator(run_id, queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/run/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Cancel an active run (frontend stop / user terminate)."""
    task = _run_tasks.get(run_id)
    if task is None:
        raise HTTPException(status_code=404, detail="run not found")
    if not task.done():
        task.cancel()
    return {"ok": True, "run_id": run_id}


async def _sse_generator(run_id: str, queue: asyncio.Queue) -> AsyncIterator[str]:
    try:
        while True:
            line = await queue.get()
            yield line
            if line.startswith("data:") and ('"type": "done"' in line or '"type": "error"' in line):
                break
    finally:
        _runs.pop(run_id, None)


async def _execute_run(run_id: str, session_id: str, req: RunRequest, queue: asyncio.Queue) -> None:
    """Build the harness from the descriptor + slot config, run the task, stream events."""
    from harnessx.api.sse_tracer import SSETracer
    from harnessx.core.harness import BaseTask

    harness = None
    try:
        harness_config, model_config = await asyncio.get_event_loop().run_in_executor(
            None, _build_config, req, session_id
        )

        harness = model_config.agentic(harness_config)
        # SSETracer wraps the journal created by _instantiate_runtime; injected
        # per-run via tracer_override so HarnessConfig stays serialisable.
        sse_tracer = SSETracer(queue=queue, inner=harness._rt.tracer, api_run_id=run_id)
        task = BaseTask(
            description=req.task,
            success_criteria=req.success_criteria or None,
            max_steps=req.max_steps,
            token_budget=req.token_budget,
        )

        def _stream_cb(delta: object) -> None:
            # Root-run deltas: providers may send either plain string tokens
            # or structured payloads {"type": "token|thinking", "content": "..."}.
            kind = "token"
            content = ""
            if isinstance(delta, str):
                content = delta
            elif isinstance(delta, dict):
                raw_kind = delta.get("type") or delta.get("kind")
                raw_content = delta.get("content") or delta.get("delta")
                if isinstance(raw_kind, str) and raw_kind in {"token", "thinking"}:
                    kind = raw_kind
                if isinstance(raw_content, str):
                    content = raw_content
            if content:
                sse_tracer.emit_stream_delta(run_id, content, kind=kind)

        # harness.run() auto-resumes from disk when session_id + workspace + HarnessJournal
        # are all configured. No manual wake() needed here.
        result = await harness.run(
            task,
            session_id=session_id,
            stream_callback=_stream_cb,
            tracer_override=sse_tracer,
        )

        passed = result.eval_result.passed if result.eval_result else None
        await queue.put(
            _sse(
                {
                    "type": "done",
                    "exit_reason": result.exit_reason,
                    "steps": result.total_steps,
                    "total_cost": result.total_cost_usd,
                    "total_input_tokens": result.total_input_tokens,
                    "total_output_tokens": result.total_output_tokens,
                    "passed": passed,
                    # Include error message when run_loop caught an exception internally
                    # so the frontend can display it prominently instead of silently
                    # showing "Done · error".
                    "error": result.error or "",
                }
            )
        )
    except Exception as exc:
        msg = str(exc)
        if "ModelConfig requires a 'main' provider" in msg:
            import traceback as _tb

            frames = _tb.extract_tb(exc.__traceback__)
            tail = " > ".join(f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}" for f in frames[-4:]) or "-"
            diag = _provider_config_diag(req.provider_config)
            msg = (
                f"{msg} "
                f"(diag: {diag}; stack_tail: {tail}) "
                "If provider_config already contains main, "
                "the backend process is likely stale; restart Lab in the same Python env."
            )
        await queue.put(_sse({"type": "error", "message": msg}))
    except asyncio.CancelledError:
        # Explicit cancel endpoint / process interrupt.
        await queue.put(
            _sse(
                {
                    "type": "done",
                    "exit_reason": "interrupted",
                    "steps": 0,
                    "total_cost": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "passed": None,
                    "error": "",
                }
            )
        )
    finally:
        if harness is not None:
            try:
                await harness.cleanup()
            except Exception:
                pass
        _run_tasks.pop(run_id, None)
        # NOTE:
        # Do not clear _runs here.
        #
        # The frontend opens SSE in a second request after POST /run returns.
        # If we pop here (especially on fast error paths), GET /stream can race
        # and return 404 before the client connects, which surfaces as
        # "SSE connection lost".
        #
        # Cleanup is handled by _sse_generator() once the terminal event
        # (done/error) has been consumed or the stream is closed.
        pass


def _build_config(req: RunRequest, session_id: str):
    """Build (HarnessConfig, ModelConfig) synchronously (called in executor)."""
    from harnessx.core.harness import HarnessConfig as _HC
    from harnessx.tools.builtin import build_default_tools
    from harnessx.tools.inmemory import InMemoryToolRegistry
    from harnessx.home import agent_home
    from harnessx.workspace.workspace import Workspace

    # ── Workspace (AGENT_HOME-derived) ────────────────────────────────────────
    # Use "home" mode so the agent can access shared AGENT_HOME resources
    # (memory, skills, config) that live outside the per-session workspace root.
    # The jail is still enforced — paths outside AGENT_HOME are rejected.
    workspace = Workspace(
        agent_id=req.agent_id,
        project=req.project,
        home=agent_home(),
        mode="home",
    )

    sc = req.slot_config
    _persist_slot_config(sc)

    # ── ModelConfig — built separately, combined via model_config.agentic() ──
    try:
        model_config = _resolve_model_config(req.provider_config)
    except Exception as exc:
        # Last-resort guard: never fail an API run solely due to malformed
        # client provider_config. Fall back to persisted/env defaults.
        if "ModelConfig requires a 'main' provider" in str(exc):
            model_config = _load_default_model_config()
        else:
            raise

    # ── Tool registry ─────────────────────────────────────────────────────────
    registry = build_default_tools()
    if sc.enabled_tools is not None:
        enabled = set(sc.enabled_tools)
        filtered = InMemoryToolRegistry()
        for t in registry.list():
            if t.name in enabled:
                filtered.register(t)
        registry = filtered

    # ── Sandbox ───────────────────────────────────────────────────────────────
    if sc.sandbox_type == "remote" and sc.sandbox_url:
        raise NotImplementedError(
            "Remote sandbox is not yet supported via the API. "
            "Set sandbox_type='local' or implement a custom SandboxProvider."
        )

    # ── Auto-load enabled plugins from AGENT_HOME/plugins/ ───────────────────
    # Plugins in ~/.harnessx/plugins/ are shared across all agents.  Any plugin
    # not in the disabled list is automatically injected into every run so that
    # Lab UI enable/disable actually takes effect.
    _auto_plugin_specs = _collect_auto_plugins()
    sanitized_hcfg = _sanitize_harness_config_payload(req.harness_config.model_dump())

    # Merge auto-discovered plugins with any explicitly listed in the descriptor;
    # explicit descriptor plugins (more specific) take priority via deduplication.
    existing_plugins = sanitized_hcfg.get("plugins") or []
    sanitized_hcfg["plugins"] = _auto_plugin_specs + existing_plugins
    harness_config = _HC(
        processors=sanitized_hcfg.get("processors") or [],
        plugins=sanitized_hcfg.get("plugins") or [],
    )

    # ── Mount MCP runtime plugin (task_start hot-reload) ─────────────────────
    from harnessx.plugins.dimensions.mcp_runtime import McpRuntimePlugin

    harness_config = _mount_plugin(
        harness_config,
        McpRuntimePlugin(
            mcp_config=req.harness_config.mcp_config,
            base_dir=workspace.root,
            ensure_primary=True,
        ),
    )

    # ── Mount Skill runtime plugin (dir-change detection + progressive inject) ─
    from harnessx.plugins.dimensions.skill_runtime import SkillRuntimePlugin

    harness_config = _mount_plugin(
        harness_config,
        SkillRuntimePlugin(enabled_skills=sc.enabled_skills),
    )

    from harnessx.core.config_schema import TracerConfig

    harness_config = harness_config.copy(
        tool_registry=registry,
        tracer=TracerConfig(
            session_id=session_id,
            base_dir=str(workspace.root / "sessions"),
            silent=os.environ.get("HARNESSX_LAB_SILENT") == "1",
        ),
        workspace=workspace,
    )

    # spawn_subagent is included in build_default_tools() — no explicit registration needed.

    return harness_config, model_config


def _mount_plugin(harness_config, plugin):
    """Mount an already-instantiated plugin into both processors and plugins."""
    procs = list(harness_config.processors or [])
    procs.extend(list(getattr(plugin, "processors", []) or []))
    plugins = list(getattr(harness_config, "plugins", []) or [])
    plugins.append(plugin)
    return harness_config.copy(processors=procs, plugins=plugins)


def _has_main_provider_spec(provider_config: dict[str, Any]) -> bool:
    """Return True when provider_config includes a usable main model slot."""
    if not provider_config:
        return False
    if "main" in provider_config:
        return True
    roles = provider_config.get("roles")
    if isinstance(roles, dict):
        main_role = roles.get("main")
        if isinstance(main_role, dict) and bool(main_role.get("default")):
            return True
    return False


def _provider_config_diag(provider_config: dict[str, Any]) -> str:
    """Return a sanitized one-line shape summary for debugging API payload issues."""
    if not isinstance(provider_config, dict):
        return f"type={type(provider_config).__name__}"

    keys = sorted(provider_config.keys())
    main = provider_config.get("main")
    main_type = type(main).__name__ if main is not None else "None"
    main_target = ""
    main_model = ""
    if isinstance(main, dict):
        main_target = str(main.get("_target_", ""))
        main_model = str(main.get("model", ""))

    roles = provider_config.get("roles")
    role_keys: list[str] = []
    if isinstance(roles, dict):
        role_keys = sorted(str(k) for k in roles.keys())

    return (
        f"keys={keys}, has_main={('main' in provider_config)}, "
        f"main_type={main_type}, main_target={main_target or '-'}, "
        f"main_model={main_model or '-'}, roles={role_keys}"
    )


def _should_drop_nested_model_spec(key: str | None, value: dict[str, Any]) -> bool:
    """Whether a dict in harness_config should be stripped before instantiate()."""
    target = value.get("_target_")
    if isinstance(target, str) and target.endswith(".ModelConfig"):
        return True
    # MCP runtime is injected at runtime from mcp_servers.json via McpRuntimePlugin;
    # persisted runtime task_start processors carry non-serializable loader/plugin refs.
    if isinstance(target, str) and target.endswith("McpRuntimeTaskStartProcessor"):
        return True
    if isinstance(target, str) and ("<locals>" in target or target.startswith("harnessx.cli._chat.")):
        # Local helper classes from CLI runtime are not importable from YAML.
        return True

    # Legacy/accidental payloads sometimes embed full model config under
    # processor kwargs. ModelConfig is supplied separately via provider_config.
    if key in {"model", "model_config", "provider_config"}:
        if ("models" in value and "roles" in value) or ("main" in value):
            return True

    return False


def _sanitize_harness_node(value: Any, key: str | None = None) -> Any:
    """Recursively remove nested model-config fragments from harness payload."""
    if isinstance(value, dict):
        if _should_drop_nested_model_spec(key, value):
            return _DROP
        out: dict[str, Any] = {}
        for k, v in value.items():
            sv = _sanitize_harness_node(v, k)
            if sv is _DROP:
                continue
            out[k] = sv
        return out
    if isinstance(value, list):
        out_list: list[Any] = []
        for item in value:
            sv = _sanitize_harness_node(item, None)
            if sv is _DROP:
                continue
            out_list.append(sv)
        return out_list
    return value


def _sanitize_harness_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Sanitize incoming harness_config payload before constructing HarnessConfig."""
    cleaned = _sanitize_harness_node(payload, None)
    if isinstance(cleaned, dict):
        return cleaned
    return {"processors": [], "plugins": None}


def _parse_env_extra_headers() -> dict[str, str]:
    raw = os.environ.get("EXTRA_HEADERS", "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        import json

        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            k, _, v = part.partition(":")
            out[k.strip()] = v.strip()
    return out


def _load_persisted_model_config():
    """Load ModelConfig from AGENT_HOME/model_config.yaml when available.

    Also checks ~/.harnessx/model_config.yaml as a compatibility fallback when
    HARNESSX_HOME points elsewhere.
    """
    from harnessx.core.model_config import ModelConfig
    from harnessx.home import agent_home
    from pathlib import Path

    candidates = [
        agent_home() / "model_config.yaml",
        Path.home() / ".harnessx" / "model_config.yaml",
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            return ModelConfig.from_yaml_file(path)
        except Exception:
            continue
    return None


def _load_default_model_config():
    """Best-effort default model resolution for API runs (same priority as CLI)."""
    from harnessx.core.model_config import ModelConfig

    if (persisted := _load_persisted_model_config()) is not None:
        return persisted

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    anthropic_model = os.environ.get("ANTHROPIC_DEFAULT_MAIN_MODEL")
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_DEFAULT_MAIN_MODEL")
    litellm_key = os.environ.get("LITELLM_API_KEY")
    litellm_model = os.environ.get("LITELLM_DEFAULT_MAIN_MODEL")
    env_headers = _parse_env_extra_headers()
    timeout = os.environ.get("HARNESSX_REQUEST_TIMEOUT")

    if anthropic_key or anthropic_model:
        from harnessx.providers.anthropic_provider import AnthropicProvider

        model_name = anthropic_model or "claude-sonnet-4-6"
        kw: dict[str, Any] = {}
        if anthropic_key:
            kw["api_key"] = anthropic_key
        base_url = os.environ.get("ANTHROPIC_API_BASE") or os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            kw["base_url"] = base_url
        if env_headers:
            kw["default_headers"] = env_headers
        if timeout:
            kw["timeout"] = float(timeout)
        return ModelConfig(main=AnthropicProvider(model_name, **kw))

    if openai_key or openai_model:
        from harnessx.providers.openai_provider import OpenAIProvider

        model_name = openai_model or "gpt-4o"
        kw: dict[str, Any] = {}
        if openai_key:
            kw["api_key"] = openai_key
        api_base = os.environ.get("OPENAI_API_BASE")
        if api_base:
            kw["base_url"] = api_base
        if env_headers:
            kw["extra_headers"] = env_headers
        if timeout:
            kw["timeout"] = int(timeout)
        return ModelConfig(main=OpenAIProvider(model_name, **kw))

    if litellm_key or litellm_model:
        from harnessx.providers.litellm_provider import LiteLLMProvider

        model_name = litellm_model or "claude-sonnet-4-6"
        kw: dict[str, Any] = {}
        if litellm_key:
            kw["api_key"] = litellm_key
        api_base = os.environ.get("LITELLM_API_BASE")
        if api_base:
            kw["api_base"] = api_base
        if env_headers:
            kw["extra_headers"] = env_headers
        if timeout:
            kw["request_timeout"] = int(timeout)
        return ModelConfig(main=LiteLLMProvider(model_name, **kw))

    # Last-resort fallback for a cold environment (same as CLI fallback).
    from harnessx.providers.anthropic_provider import AnthropicProvider

    return ModelConfig(main=AnthropicProvider("claude-sonnet-4-6"))


def _collect_auto_plugins() -> list[Any]:
    """Return path strings for enabled plugins from AGENT_HOME/plugins/ and scan_dirs.

    Reads ``plugins_state.json`` for the disabled list and user-configured
    scan directories, then calls :func:`discover_plugins` with those settings.
    Each discovered plugin is represented as its filesystem path so
    ``HarnessBuilder.plugin()`` can load and deduplicate it normally.
    """
    try:
        import json
        from pathlib import Path
        from harnessx.plugins.discovery import discover_plugins

        from harnessx.home import agent_home

        candidates = [
            agent_home() / "plugins_state.json",
            Path.home() / ".harnessx" / "plugins_state.json",
        ]
        state: dict = {}
        seen: set[Path] = set()
        for state_file in candidates:
            if state_file in seen:
                continue
            seen.add(state_file)
            if not state_file.exists():
                continue
            try:
                parsed = json.loads(state_file.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    state = parsed
                    break
            except Exception:
                continue

        disabled: set[str] = set(state.get("disabled", []))
        scan_dirs = [Path(p) for p in state.get("scan_dirs", []) if Path(p).is_dir()]

        plugins = discover_plugins(
            extra_paths=scan_dirs if scan_dirs else None,
            disabled=disabled,
            include_claude_plugins=False,  # Claude Code plugins not auto-loaded in runs
        )
        specs: list[Any] = []
        for p in plugins:
            root = getattr(p, "_plugin_root", None)
            if root:
                specs.append(str(root))
        return specs
    except Exception:
        return []


def _slot_config_candidates() -> list[Path]:
    """AGENT_HOME-first + ~/.harnessx compatibility paths for slot config."""
    from harnessx.home import agent_home

    primary = agent_home() / "slot_config.json"
    compat = Path.home() / ".harnessx" / "slot_config.json"
    if compat == primary:
        return [primary]
    return [primary, compat]


def _persist_slot_config(slot_config: Any) -> None:
    """Persist latest slot config so CLI can mirror Lab runtime toggles."""
    try:
        data: dict[str, Any]
        if hasattr(slot_config, "model_dump"):
            data = dict(slot_config.model_dump())
        elif isinstance(slot_config, dict):
            data = dict(slot_config)
        else:
            return

        path = _slot_config_candidates()[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "enabled_tools": data.get("enabled_tools"),
                    "enabled_skills": data.get("enabled_skills"),
                    "sandbox_type": data.get("sandbox_type"),
                    "sandbox_url": data.get("sandbox_url"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        # Best-effort persistence only; never block a run.
        return


def _resolve_model_config(provider_config: dict[str, Any]):
    """Resolve model config from request payload, with default fallback.

    UI state can transiently send malformed or partial provider_config
    (for example during startup hydration). In that case we fall back to
    server defaults instead of surfacing a hard "missing main" error.
    """
    from harnessx.core.model_config import ModelConfig

    if _has_main_provider_spec(provider_config):
        try:
            return ModelConfig.from_dict(provider_config)
        except Exception:
            # Graceful fallback for malformed UI payloads.
            pass
    return _load_default_model_config()
