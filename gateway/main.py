from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn
import yaml

from harnessx.logging import configure_logging

logger = logging.getLogger("gateway")

# ── Runtime paths ─────────────────────────────────────────────────────────────

_GW_RUN_DIR = Path("/tmp/hx-gateway")
_PID_FILE = _GW_RUN_DIR / "gateway.pid"
_LOG_FILE = _GW_RUN_DIR / "gateway.log"


# ── Config loading ─────────────────────────────────────────────────────────────


def _load_config(path: Path | None = None) -> dict:
    cfg_path = path or (Path.home() / ".harnessx" / "gateway.yaml")
    if not cfg_path.exists():
        logger.warning("Config file not found at %s — no channels will be loaded", cfg_path)
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_model_config() -> Any:
    """Load ModelConfig from model_config.yaml, mirroring CLI/Lab UI priority.

    Priority: AGENT_HOME/model_config.yaml → ~/.harnessx/model_config.yaml → env HARNESSX_MODEL → gpt-4o
    """
    from harnessx.core.model_config import ModelConfig
    from harnessx.providers.litellm_provider import LiteLLMProvider

    try:
        from harnessx.home import agent_home

        candidates = [agent_home() / "model_config.yaml", Path.home() / ".harnessx" / "model_config.yaml"]
    except Exception:
        candidates = [Path.home() / ".harnessx" / "model_config.yaml"]

    seen: set[Path] = set()
    for p in candidates:
        if p in seen or not p.exists():
            seen.add(p)
            continue
        seen.add(p)
        try:
            mc = ModelConfig.from_yaml_file(p)
            logger.info("Loaded model config from %s", p)
            return mc
        except Exception as e:
            logger.warning("Failed to load model_config.yaml from %s: %s", p, e)

    fallback = os.environ.get("HARNESSX_MODEL", "gpt-4o")
    logger.info("No model_config.yaml found, falling back to %s", fallback)
    return ModelConfig(main=LiteLLMProvider(fallback))


# ── Harness building ───────────────────────────────────────────────────────────

_GW_PROMPT_TEMPLATES = Path(__file__).parent / "core" / "prompt_templates"


def _build_harness(
    channel_cfg: dict,
    default_cfg: dict,
    model_config: Any,
    agent_id: str = "gateway",
    channel_name: str | None = None,
) -> Any:
    """Build a Harness instance from config.

    Base pipeline: same as CLI _load_default() (full-capability agent: reliability
    guards, window management, skill loader, LightMemoryPlugin, plugin discovery).
    IM processors are added on top: IMSystemProcessor + IMUserContextProcessor.

    Workspace initialization:
    - harnessx default_prompts (AGENTS.md, TOOLS.md, USER.md) are copied first
    - gateway/core/prompt_templates/ overlays on top (IM-specific AGENTS.md, USER.md)
    - All copies are idempotent (skip if file already exists)

    IM sessions are isolated from CLI sessions via the im-workspaces/ base:
        agent_home/im-workspaces/{agent_id}/                     ← file workspace + prompts
        agent_home/im-workspaces/{agent_id}/{channel}/sessions/  ← sessions & traces
    """
    from .core.processors.im_system import IMSystemProcessor
    from .core.processors.im_user_context import IMUserContextProcessor
    from .core.im_stream import IMProgressProcessor
    from .core.processors.im_tool_guard import IMToolGuardProcessor
    from .core.processors.im_secret_redact import IMSecretRedactProcessor

    def _strip_env_context_injector(processors: list | tuple | None) -> list:
        from harnessx.processors.context.env_context_injector import (
            EnvironmentContextInjector,
        )

        filtered: list = []
        for p in list(processors or []):
            if isinstance(p, EnvironmentContextInjector):
                continue
            if isinstance(p, dict):
                target = str(p.get("_target_", "")).strip()
                if target == "harnessx.processors.context.env_context_injector.EnvironmentContextInjector" or (
                    target.endswith(".EnvironmentContextInjector") and "env_context_injector" in target
                ):
                    continue
            filtered.append(p)
        return filtered

    try:
        from harnessx.home import agent_home

        home = agent_home()
    except Exception:
        home = Path.home() / ".harnessx"
    agent_root = home / "im-workspaces" / agent_id

    # HarnessConfig path override (load external YAML)
    harness_config = None
    harness_cfg_path = channel_cfg.get("harness_config") or default_cfg.get("harness_config")
    if harness_cfg_path:
        try:
            p = Path(harness_cfg_path).expanduser()
            with open(p) as f:
                raw = yaml.safe_load(f)
            import yaml as _yaml
            from harnessx.core.harness import HarnessConfig as _HC

            harness_config = _HC.from_yaml(_yaml.dump(raw))
        except Exception as e:
            logger.warning("Failed to load harness_config from %s: %s", harness_cfg_path, e)

    if harness_config is None:
        # Directory layout:
        #   agent_root/                     ← shared workspace root + prompt templates + harness_config.yaml
        #   agent_root/{channel}/sessions/  ← per-channel run artifacts (sessions/traces)
        channel_dir = agent_root / (channel_name or "default")
        sessions_dir = channel_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Prompt templates live at the agent level — shared across all channels.
        _copy_prompt_templates(agent_root)

        # Start from the same full-capability pipeline as the CLI default agent
        # (reliability guards, window mgmt, skill loader, LightMemoryPlugin, plugins),
        # then prepend the IM-specific processors so they run first.
        from harnessx.cli import _load_default as _cli_load_default
        from harnessx.core.config_schema import TracerConfig

        base = _cli_load_default()
        im_procs = [
            IMToolGuardProcessor(),
            IMSecretRedactProcessor(),
            IMSystemProcessor(),
            IMUserContextProcessor(),
            IMProgressProcessor(),
        ]
        base_procs = _strip_env_context_injector(list(base.processors or []))
        harness_config = base.copy(processors=im_procs + base_procs)

        # Wire in the default tool registry (same as CLI — filesystem + web + browser),
        # then add the IM-only im_send_file tool so agents can push files back to users.
        from harnessx.tools.builtin import build_default_tools
        from .core.im_send_file import im_send_file as _im_send_file_tool

        _tool_registry = build_default_tools()
        _tool_registry.register(_im_send_file_tool)
        harness_config = harness_config.copy(tool_registry=_tool_registry)

        harness_config = harness_config.copy(tracer=TracerConfig(base_dir=str(sessions_dir), silent=True))

        # Persist harness_config.yaml at the agent level (shared, idempotent).
        cfg_yaml_path = agent_root / "harness_config.yaml"
        if not cfg_yaml_path.exists():
            try:
                harness_config.to_yaml_file(cfg_yaml_path)
                logger.debug("Wrote harness_config.yaml → %s", cfg_yaml_path)
            except Exception as e:
                logger.warning("Failed to write harness_config.yaml: %s", e)

    # Gateway should not inject EnvironmentContextInjector.
    harness_config = harness_config.copy(processors=_strip_env_context_injector(list(harness_config.processors or [])))

    # Gateway workspace is always derived from AGENT_HOME, not gateway.yaml.
    try:
        from harnessx.workspace.workspace import Workspace

        harness_config = harness_config.copy(
            workspace=Workspace(root=agent_root, agent_id=agent_id, home=home, mode="home")
        )
        logger.info("[%s/%s] workspace: %s", agent_id, channel_name or "default", agent_root)
    except Exception as e:
        logger.warning("[%s/%s] workspace setup failed: %s", agent_id, channel_name or "default", e)

    harness = model_config.agentic(harness_config)

    # Build a stripped child config for sub-agents: CLI base pipeline only,
    # no IM-specific processors, no workspace files, no IM-only tools.
    try:
        from harnessx.cli import _load_default as _cli_load_default_child
        from harnessx.tools.builtin import build_default_tools as _build_child_tools
        from harnessx.processors.context.system_prompt import SystemPromptProcessor
        from harnessx.processors.context.strategies.system_prompt.default import DefaultSystemPromptBuilder

        _child_persona_root = str(_GW_PROMPT_TEMPLATES / "subagents")

        child_base = _cli_load_default_child()
        child_base_procs = _strip_env_context_injector(list(child_base.processors or []))
        # Inject persona_root into the DefaultSystemPromptBuilder so sub-agents
        # always read subagents/AGENTS.md (file convention, task guidance) even
        # when their workspace is empty (init_workspace=False).
        child_base_procs = [
            SystemPromptProcessor(
                DefaultSystemPromptBuilder(
                    max_skills_shown=p.system_builder.max_skills_shown,
                    enabled_skills=p.system_builder.enabled_skills,
                    extra_skills_dirs=p.system_builder.extra_skills_dirs,
                    persona_root=_child_persona_root,
                )
            )
            if isinstance(p, SystemPromptProcessor) and isinstance(p.system_builder, DefaultSystemPromptBuilder)
            else p
            for p in child_base_procs
        ]
        child_config = child_base.copy(
            processors=child_base_procs,
            tool_registry=_build_child_tools(),  # standard tools only, no im_send_file
            workspace=None,
            init_workspace=False,
        )
        harness.child_harness_config = child_config
    except Exception as e:
        logger.warning("[%s/%s] child_harness_config build failed: %s", agent_id, channel_name or "default", e)

    return harness


def _copy_prompt_templates(ws_root: Path) -> None:
    """Copy prompt templates into the IM workspace (idempotent).

    Priority (highest wins):
    1. gateway/core/prompt_templates/*.md  — IM-specific overrides
    2. harnessx/workspace/default_prompts/*.md  — base defaults (TOOLS.md etc.)

    Files that already exist in the workspace are never overwritten.
    """
    import shutil
    from harnessx.workspace.initializer import WorkspaceInitializer

    ws_root.mkdir(parents=True, exist_ok=True)

    # Gateway IM templates first (higher priority — written before defaults so
    # the idempotent guard in the defaults pass keeps them untouched).
    if _GW_PROMPT_TEMPLATES.exists():
        for src in sorted(_GW_PROMPT_TEMPLATES.glob("*.md")):
            dst = ws_root / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                logger.debug("prompt template → %s", dst)

    # harnessx default_prompts second (fills in TOOLS.md etc. not in gateway templates).
    default_prompts = WorkspaceInitializer().prompts_root
    if default_prompts.exists():
        for src in sorted(default_prompts.glob("*.md")):
            dst = ws_root / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                logger.debug("default template → %s", dst)


# ── Channel instantiation ──────────────────────────────────────────────────────


def _instantiate_channels(config: dict, dispatcher: Any) -> list[Any]:
    from .channels import get_channel_class, _auto_discover

    _auto_discover()

    channels_cfg = config.get("channels", {})
    channels = []
    for name, ch_cfg in channels_cfg.items():
        if not ch_cfg.get("enabled", False):
            continue
        # channel_type holds the platform key (e.g. "discord"); falls back to name for
        # the common case where name == platform type (e.g. name="telegram")
        platform_type = ch_cfg.get("channel_type", name)
        cls = get_channel_class(platform_type)
        if cls is None:
            logger.warning(
                "Channel '%s' (type '%s') not found in registry (missing dependency?)",
                name,
                platform_type,
            )
            continue
        ch = cls(config=ch_cfg, dispatcher=dispatcher)
        ch.name = name  # override class-level name so routing uses the instance name
        dispatcher.register(ch)
        channels.append(ch)
        logger.info("Registered channel: %s (type: %s)", name, platform_type)
    return channels


# ── Async server loop ──────────────────────────────────────────────────────────


async def _run(config_path: Path | None = None, host: str = "0.0.0.0", port: int = 8080) -> None:
    config = _load_config(config_path)
    gateway_cfg = config.get("gateway", {})

    from .core.session_store import SessionStore
    from .core.dispatch import ChannelDispatcher
    from .core.cron import CronManager
    from .server import app, set_dispatcher, set_config_path, set_cron_manager

    effective_config_path = config_path or (Path.home() / ".harnessx" / "gateway.yaml")
    set_config_path(effective_config_path)

    store = SessionStore()

    default_cfg = config.get("default", {})
    gw_agent_id = gateway_cfg.get("agent_id", "gateway")
    model_config = _load_model_config()

    channel_harnesses: dict[str, Any] = {}
    for name, ch_cfg in config.get("channels", {}).items():
        if not ch_cfg.get("enabled", False):
            continue
        agent_id = ch_cfg.get("agent_id", gw_agent_id)
        channel_harnesses[name] = _build_harness(
            ch_cfg, default_cfg, model_config, agent_id=agent_id, channel_name=name
        )

    # default_harness is used by the web UI chat and as a fallback
    default_harness = _build_harness({}, default_cfg, model_config, agent_id=gw_agent_id)

    dispatcher = ChannelDispatcher(
        default_harness=default_harness,
        session_store=store,
        channel_harnesses=channel_harnesses,
    )
    dispatcher.set_config_path(effective_config_path)
    set_dispatcher(dispatcher)

    # Inject per-channel workspace media dir into channel configs before instantiation.
    # Files received from users land in agent_root/{channel}/media/ alongside sessions/.
    try:
        from harnessx.home import agent_home as _agent_home_fn

        _gw_home = _agent_home_fn()
    except Exception:
        _gw_home = Path.home() / ".harnessx"
    for _ch_name, _ch_cfg in config.get("channels", {}).items():
        if not _ch_cfg.get("enabled", False):
            continue
        _aid = _ch_cfg.get("agent_id", gw_agent_id)
        _media_dir = _gw_home / "im-workspaces" / _aid / _ch_name / "media"
        _media_dir.mkdir(parents=True, exist_ok=True)
        _ch_cfg["_workspace_media_dir"] = str(_media_dir)

    _instantiate_channels(config, dispatcher)

    if not dispatcher.channels:
        logger.warning("No channels enabled. Add channels to ~/.harnessx/gateway.yaml")

    stop_event = asyncio.Event()
    _sig_count = 0

    def _signal_handler(sig: signal.Signals) -> None:
        nonlocal _sig_count
        _sig_count += 1
        if _sig_count == 1:
            logger.info("Received %s — initiating graceful shutdown…", sig.name)
            stop_event.set()
        else:
            logger.warning("Forced exit.")
            import os

            os._exit(1)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler, sig)

    await dispatcher.start_all()

    # CronManager — lives at agent_root (shared across channels)
    try:
        from harnessx.home import agent_home as _agent_home

        _home = _agent_home()
    except Exception:
        _home = Path.home() / ".harnessx"
    agent_root = _home / "im-workspaces" / gw_agent_id
    cron_manager = CronManager(dispatcher=dispatcher, agent_root=agent_root)
    set_cron_manager(cron_manager)
    heartbeat_cfg = config.get("heartbeat")
    await cron_manager.start(heartbeat_cfg=heartbeat_cfg)

    effective_host = gateway_cfg.get("host", host)
    effective_port = gateway_cfg.get("port", port)

    uv_config = uvicorn.Config(
        app=app,
        host=effective_host,
        port=effective_port,
        log_level="warning",
        access_log=False,
    )
    uv_server = uvicorn.Server(uv_config)
    uv_task = asyncio.create_task(uv_server.serve(), name="uvicorn")

    logger.info(
        "Gateway running on http://%s:%d — %d channel(s) active",
        effective_host,
        effective_port,
        len(dispatcher.channels),
    )

    await stop_event.wait()
    logger.info("Shutting down…")

    uv_server.should_exit = True
    try:
        await asyncio.wait_for(uv_task, timeout=5.0)
    except (asyncio.TimeoutError, Exception):
        uv_task.cancel()

    try:
        await cron_manager.stop()
    except Exception:
        pass

    try:
        await asyncio.wait_for(dispatcher.stop_all(timeout=10.0), timeout=12.0)
    except (asyncio.TimeoutError, Exception):
        pass

    logger.info("Shutdown complete.")


# ── Process management helpers ─────────────────────────────────────────────────


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _read_pid() -> int | None:
    try:
        return int(_PID_FILE.read_text().strip())
    except Exception:
        return None


def _self_exe() -> list[str]:
    """Return argv prefix to re-invoke this CLI (installed script or direct python)."""
    exe = shutil.which("hx-gateway")
    if exe:
        return [exe]
    return [sys.executable, str(Path(__file__))]


# ── CLI sub-commands ───────────────────────────────────────────────────────────


def _cmd_start(config: Path | None, host: str, port: int, log_level: str) -> None:
    _GW_RUN_DIR.mkdir(parents=True, exist_ok=True)

    pid = _read_pid()
    if pid and _is_running(pid):
        print(f"Gateway is already running  [PID {pid}]")
        print(f"  Console: http://localhost:{port}/console/")
        print(f"  Logs:    {_LOG_FILE}")
        return

    cmd = _self_exe() + ["_serve", "--host", host, "--port", str(port), "--log-level", log_level]
    if config:
        cmd += ["-c", str(config)]

    log_fh = open(_LOG_FILE, "a")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    _PID_FILE.write_text(str(proc.pid))

    # Wait briefly to catch immediate startup failures
    time.sleep(1.5)
    if proc.poll() is not None:
        print(f"Gateway failed to start — check logs: {_LOG_FILE}")
        sys.exit(1)

    print(f"Gateway started  [PID {proc.pid}]")
    print(f"  Logs:    {_LOG_FILE}  (hx-gateway logs)")
    print(f"  Console: http://localhost:{port}/console/")
    print("  Stop:    hx-gateway stop")
    print("  Restart: hx-gateway restart")


def _cmd_stop(quiet: bool = False) -> None:
    pid = _read_pid()
    if pid is None:
        if not quiet:
            print("Gateway is not running (no PID file)")
        return

    if not _is_running(pid):
        _PID_FILE.unlink(missing_ok=True)
        if not quiet:
            print("Gateway is not running (stale PID file removed)")
        return

    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        time.sleep(0.5)
        if not _is_running(pid):
            break
    else:
        # Still running after 10s — force kill
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    _PID_FILE.unlink(missing_ok=True)
    if not quiet:
        print(f"Gateway stopped  [PID {pid}]")


def _cmd_logs(n: int, follow: bool) -> None:
    if not _LOG_FILE.exists():
        print(f"No log file at {_LOG_FILE}. Is the gateway running?")
        return
    if follow:
        subprocess.run(["tail", "-f", "-n", str(n), str(_LOG_FILE)])
    else:
        lines = _LOG_FILE.read_text(errors="replace").splitlines()
        print("\n".join(lines[-n:]))


def _cmd_status() -> None:
    pid = _read_pid()
    if pid is None:
        print("stopped")
        return
    if _is_running(pid):
        print(f"running  [PID {pid}]")
        print(f"  Logs:    {_LOG_FILE}")
    else:
        print("stopped  (stale PID file removed)")
        _PID_FILE.unlink(missing_ok=True)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="hx-gateway",
        description="HarnessX IM Gateway",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="command")

    # ── start ─────────────────────────────────────────────────────────────────
    p_start = sub.add_parser("start", help="Start gateway in background")
    p_start.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to gateway.yaml (default: ~/.harnessx/gateway.yaml)",
    )
    p_start.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_start.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    p_start.add_argument(
        "--log-level", default="INFO", metavar="LEVEL", help="Log level: DEBUG|INFO|WARNING (default: INFO)"
    )

    # ── stop ──────────────────────────────────────────────────────────────────
    sub.add_parser("stop", help="Stop the background gateway")

    # ── restart ───────────────────────────────────────────────────────────────
    p_restart = sub.add_parser("restart", help="Restart the gateway")
    p_restart.add_argument("-c", "--config", type=Path, default=None, metavar="PATH")
    p_restart.add_argument("--host", default="0.0.0.0")
    p_restart.add_argument("--port", type=int, default=8080)
    p_restart.add_argument("--log-level", default="INFO", metavar="LEVEL")

    # ── logs ──────────────────────────────────────────────────────────────────
    p_logs = sub.add_parser("logs", help="Show gateway logs")
    p_logs.add_argument("-n", type=int, default=50, metavar="N", help="Show last N lines (default: 50)")
    p_logs.add_argument("-f", "--follow", action="store_true", help="Follow log output (like tail -f)")

    # ── status ────────────────────────────────────────────────────────────────
    sub.add_parser("status", help="Show gateway process status")

    # ── _serve (internal, launched by start) ──────────────────────────────────
    p_serve = sub.add_parser("_serve", help=argparse.SUPPRESS)
    p_serve.add_argument("-c", "--config", type=Path, default=None)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    if args.cmd == "_serve":
        configure_logging(level=args.log_level, compact=False)
        asyncio.run(_run(config_path=args.config, host=args.host, port=args.port))

    elif args.cmd == "start":
        _cmd_start(args.config, args.host, args.port, args.log_level)

    elif args.cmd == "stop":
        _cmd_stop()

    elif args.cmd == "restart":
        _cmd_stop(quiet=True)
        _cmd_start(args.config, args.host, args.port, args.log_level)

    elif args.cmd == "logs":
        _cmd_logs(args.n, args.follow)

    elif args.cmd == "status":
        _cmd_status()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
