# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import contextmanager


# ── Banner ────────────────────────────────────────────────────────────────────

_LOGO_SMALL = (
    " _  _    _    ___  _  _  ___  ___  ___ __  __\n"
    "| || |  /_\\  | _ \\| \\| || __|/ __|/ __|\\ \\/ /\n"
    "| __ | / _ \\ |   /| .` || _| \\__ \\\\__ \\ >  < \n"
    "|_||_|/_/ \\_\\|_|_\\|_|\\_||___||___/|___//_/\\_\\"
)


def _print_banner(mode: str, extra: str = "") -> None:
    """Print the HarnessX ASCII banner to stderr.

    Skipped when stderr is not a TTY or the terminal is too narrow.
    """
    if not sys.stderr.isatty():
        return
    try:
        import os

        cols = os.get_terminal_size(sys.stderr.fileno()).columns
    except OSError:
        cols = 80

    version = _get_version().split()[-1]

    CYN = "\033[36m"
    BLD = "\033[1m"
    DIM = "\033[2m"
    NC = "\033[0m"

    w = sys.stderr.write

    if cols < 64:
        # Narrow terminal: single-line fallback
        w(f"\n{BLD}HarnessX{NC} {DIM}{version}  {mode}{NC}\n\n")
        return

    w("\n")
    for line in _LOGO_SMALL.splitlines():
        w(f"{CYN}{line}{NC}\n")
    w("\n")

    meta = f"{DIM}{version}  ·  {mode}{NC}"
    if extra:
        meta = f"{DIM}{version}  ·  {mode}  ·  {extra}{NC}"
    w(f"{meta}\n\n")


@contextmanager
def _sigterm_as_keyboard_interrupt():
    """Translate SIGTERM into KeyboardInterrupt for best-effort graceful teardown."""
    import signal

    if not hasattr(signal, "SIGTERM"):
        yield
        return

    old_handler = signal.getsignal(signal.SIGTERM)

    def _handler(signum, frame):
        raise KeyboardInterrupt()

    installed = False
    try:
        signal.signal(signal.SIGTERM, _handler)
        installed = True
    except Exception:
        installed = False

    try:
        yield
    finally:
        if installed:
            try:
                signal.signal(signal.SIGTERM, old_handler)
            except Exception:
                pass


# ── Argument parser ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hx",
        description="HarnessX agent harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=_get_version())

    # Shared options live on the top-level parser only.
    # main() normalises argv so flags always appear before the subcommand.
    _add_shared(p)

    sub = p.add_subparsers(dest="command")

    lab_p = sub.add_parser("lab", help="Start Harness Lab UI (requires pip install 'harnessx')")
    lab_p.add_argument(
        "--port",
        type=int,
        default=7861,
        metavar="PORT",
        help="Port to listen on (default: 7861)",
    )
    lab_p.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: skip static serving (Vite handles it)",
    )
    lab_p.add_argument("--open", action="store_true", help="Open browser automatically after startup")
    lab_p.add_argument("--verbose", action="store_true", help="Print per-event trace logs to console")

    # plugin — manage HarnessX plugins
    plugin_p = sub.add_parser("plugin", help="Manage HarnessX plugins")
    plugin_sub = plugin_p.add_subparsers(dest="plugin_command")

    plugin_sub.add_parser("list", help="List discovered plugins")

    convert_p = plugin_sub.add_parser("convert", help="Convert a Claude Code plugin to HarnessX format")
    convert_p.add_argument(
        "src",
        metavar="SRC_DIR",
        help="Source directory containing a Claude Code plugin.json",
    )
    convert_p.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="DST_DIR",
        help="Output directory (default: {src}_oh)",
    )

    add_p = plugin_sub.add_parser("add", help="Install a plugin to ~/.harnessx/plugins/")
    add_p.add_argument(
        "src",
        metavar="SRC",
        help="Plugin directory, Python class path, or dotted import path",
    )
    install_p = plugin_sub.add_parser(
        "install",
        help="Alias of 'plugin add' (install a plugin to ~/.harnessx/plugins/)",
    )
    install_p.add_argument(
        "src",
        metavar="SRC",
        help="Plugin directory, plugin name from 'plugin list', or dotted import path",
    )

    remove_p = plugin_sub.add_parser("remove", help="Uninstall a plugin from ~/.harnessx/plugins/")
    remove_p.add_argument("name", metavar="NAME", help="Plugin directory name (as shown by 'plugin list')")
    remove_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    uninstall_p = sub.add_parser("uninstall", help="Completely remove HarnessX from this system")
    uninstall_p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip all confirmation prompts and remove everything",
    )

    return p


def _add_shared(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-p",
        "--print",
        dest="print_mode",
        action="store_true",
        help="Non-interactive: print response and exit (pipe-friendly)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode for built-in default harness (enables SelfVerifyProcessor)",
    )
    p.add_argument(
        "--router",
        nargs="?",
        const="",
        default=None,
        metavar="PARAMS",
        help="Enable model router: classify query complexity and "
        "route between small/main models. "
        "Optional comma-separated key=value params: "
        "confidence_threshold (0.0-1.0, default 0.7), "
        "router_token_budget (int, default 512). "
        "Example: --router confidence_threshold=0.8,router_token_budget=1024. "
        "Requires 'small' role in model_config.yaml.",
    )
    p.add_argument("--max-steps", type=int, default=30, metavar="N")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show structured trace logs (default: silent)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Show debug-level logs (processor decisions, routing details)",
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="RUN_ID",
        help="Resume a previous chat session by run_id (printed at the end of each chat session)",
    )


def _get_version() -> str:
    try:
        from harnessx import __version__

        return f"harnessx {__version__}"
    except Exception:
        return "harnessx (unknown version)"


# ── Config assembly ───────────────────────────────────────────────────────────


def _parse_env_extra_headers() -> "dict[str, str]":
    """Parse EXTRA_HEADERS env var → dict.

    Supports two formats::

        KEY:VALUE,KEY2:VALUE2        (comma-separated)
        {"KEY": "VALUE", ...}        (JSON object)
    """
    import os as _os2
    import json as _json

    raw = _os2.environ.get("EXTRA_HEADERS", "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            return _json.loads(raw)
        except Exception:
            pass
    result: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            k, _, v = part.partition(":")
            result[k.strip()] = v.strip()
    return result


def _load_persisted_model_config():
    """Load ModelConfig from AGENT_HOME/model_config.yaml with compat fallback."""
    from harnessx.core.model_config import ModelConfig

    for path in _state_candidates("model_config.yaml"):
        if not path.exists():
            continue
        try:
            return ModelConfig.from_yaml_file(path)
        except Exception:
            continue
    return None


def _state_candidates(filename: str):
    """Return AGENT_HOME-first, ~/.harnessx-compatible candidate paths."""
    from pathlib import Path
    from harnessx.home import agent_home

    primary = agent_home() / filename
    compat = Path.home() / ".harnessx" / filename
    if compat == primary:
        return [primary]
    return [primary, compat]


def _load_agent_harness_config():
    """Load AGENT_HOME agent-shared harness_config.yaml for default CLI agent.

    Returns HarnessConfig when a valid workspace config file exists, else None.
    """
    from harnessx.core.harness import HarnessConfig
    from harnessx.home import agent_harness_config_path, default_agent_id

    cfg_path = agent_harness_config_path(default_agent_id())
    if not cfg_path.exists():
        return None

    try:
        return HarnessConfig.from_yaml_file(cfg_path)
    except Exception:
        return None


def _load_plugins_state() -> tuple[set[str], list["Path"]]:  # noqa: F821
    """Load plugin disabled list + scan dirs from plugins_state.json."""
    import json
    from pathlib import Path

    state: dict = {}
    for path in _state_candidates("plugins_state.json"):
        if not path.exists():
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(state, dict):
                break
        except Exception:
            continue

    disabled: set[str] = set(state.get("disabled", []))
    scan_dirs = [Path(p) for p in state.get("scan_dirs", []) if Path(p).is_dir()]
    return disabled, scan_dirs


def _load_persisted_slot_config() -> dict:
    """Load latest slot_config persisted by Lab runs (best-effort)."""
    import json

    for path in _state_candidates("slot_config.json"):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def _apply_enabled_skills(harness_config, enabled_skills: list[str]) -> None:
    """Apply enabled_skills filtering to SkillRuntimePlugin."""
    from harnessx.plugins.dimensions.skill_runtime import SkillRuntimePlugin
    from harnessx.processors.tools.skill_loader import ProgressiveSkillLoader

    skills_arg = enabled_skills if enabled_skills else None
    # Primary: delegate to SkillRuntimePlugin.
    for plugin in getattr(harness_config, "plugins", []):
        if isinstance(plugin, SkillRuntimePlugin):
            plugin.set_enabled_skills(skills_arg)
            return

    # Fallback: bare ProgressiveSkillLoader without the plugin (custom configs).
    for proc in harness_config.processors or []:
        if isinstance(proc, ProgressiveSkillLoader):
            proc.enabled_skills = skills_arg


def _mount_plugin(harness_config, plugin):
    """Mount an already-instantiated plugin into both processors and plugins."""
    procs = list(harness_config.processors or [])
    procs.extend(list(getattr(plugin, "processors", []) or []))
    plugins = list(getattr(harness_config, "plugins", []) or [])
    plugins.append(plugin)
    return harness_config.copy(processors=procs, plugins=plugins)


def _build_harness(args: argparse.Namespace):
    """Build the CLI HarnessConfig (behavior pipeline, no model).

    Also initializes logging for the CLI session. Always call this before
    ``_build_model()`` so logging is configured first.
    """

    verbose = getattr(args, "verbose", False)
    debug = getattr(args, "debug", False)

    from harnessx.logging import configure_logging

    if debug:
        configure_logging(level="DEBUG")
        verbose = True
    else:
        configure_logging(level="INFO" if verbose else "WARNING")

    if not verbose:
        import logging
        import warnings

        logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
        logging.getLogger("litellm").setLevel(logging.CRITICAL)
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            import litellm

            litellm.suppress_debug_info = True
            litellm.set_verbose = False
        except ImportError:
            pass

    # CLI always uses the agent_home harness_config.yaml or built-in defaults.
    # To use a custom harness programmatically:
    #   harness = HarnessConfig.from_yaml(open("harness_config.yaml").read())
    #   model   = ModelConfig.from_yaml_file("model_config.yaml")
    #   agent   = model.agentic(harness)
    loaded = _load_agent_harness_config()
    if loaded is not None:
        harness_config = loaded
        from harnessx.plugins.dimensions.mcp_runtime import McpRuntimePlugin

        harness_config = _mount_plugin(harness_config, McpRuntimePlugin(ensure_primary=True))
        from harnessx.plugins.dimensions.skill_runtime import SkillRuntimePlugin

        _slot_y = _load_persisted_slot_config()
        _es_y = _slot_y.get("enabled_skills")
        harness_config = _mount_plugin(
            harness_config,
            SkillRuntimePlugin(
                enabled_skills=[str(s) for s in _es_y] if isinstance(_es_y, list) else None,
                auto_inject=_slot_y.get("skill_auto_inject", True) is not False,
            ),
        )
    else:
        harness_config = _load_default(strict=bool(getattr(args, "strict", False)))

    from harnessx.tools.builtin import build_default_tools

    registry = build_default_tools()

    slot_cfg = _load_persisted_slot_config()
    enabled_tools = slot_cfg.get("enabled_tools")
    if isinstance(enabled_tools, list):
        from harnessx.tools.inmemory import InMemoryToolRegistry

        from harnessx.tools.spawn_subagent import SPAWN_TOOL_NAME as _STN

        enabled = set(str(name) for name in enabled_tools)
        enabled.add(_STN)  # spawn_subagent is always available
        filtered = InMemoryToolRegistry()
        tools = registry.list() if hasattr(registry, "list") else list(getattr(registry, "_tools", {}).values())
        for t in tools:
            if t.name in enabled:
                filtered.register(t)
        registry = filtered

    from harnessx.core.harness import _runtime_registry_to_config

    harness_config = harness_config.copy(tool_registry=_runtime_registry_to_config(registry))

    enabled_skills = slot_cfg.get("enabled_skills")
    if isinstance(enabled_skills, list):
        _apply_enabled_skills(harness_config, [str(s) for s in enabled_skills])

    # --router: enable and tune the ModelRouterProcessor already in the pipeline.
    router_raw = getattr(args, "router", None)
    if router_raw is not None:
        from harnessx.processors.multi_model.model_router import ModelRouterProcessor
        from harnessx.core.builder import _instantiate

        router_kwargs: dict = {}
        for pair in (router_raw or "").split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, _, val = pair.partition("=")
            key = key.strip()
            val = val.strip()
            if key in ("confidence_threshold",):
                router_kwargs[key] = float(val)
            elif key in ("router_token_budget", "max_router_steps"):
                router_kwargs[key] = int(val)
            elif key in ("router_key", "simple_key", "complex_key", "slot_key"):
                router_kwargs[key] = val
            else:
                print(f"WARNING: unknown --router param '{key}', ignored", file=sys.stderr)

        new_procs = []
        for p in harness_config.processors or []:
            inst = _instantiate(p) if isinstance(p, dict) else p
            if isinstance(inst, ModelRouterProcessor):
                inst.enabled = True
                for k, v in router_kwargs.items():
                    setattr(inst, k, v)
                new_procs.append(inst)
            else:
                new_procs.append(p)
        harness_config = harness_config.copy(processors=new_procs)

    if not verbose:
        from harnessx.core.config_schema import TracerConfig

        harness_config = harness_config.copy(tracer=TracerConfig(silent=True))

    if harness_config.workspace is None:
        import os
        from pathlib import Path
        from harnessx.workspace.workspace import Workspace
        from harnessx.home import agent_home, default_agent_id, default_project

        _legacy_ws = os.environ.get("HARNESSX_WORKSPACE")
        if _legacy_ws:
            workspace = Workspace(
                agent_id=default_agent_id(),
                root=Path(_legacy_ws).expanduser().resolve(),
                mode=None,
            )
        else:
            workspace = Workspace(
                agent_id=default_agent_id(),
                project=default_project(),
                home=agent_home(),
                mode=None,
            )
        from harnessx.core.harness import _runtime_workspace_to_config

        harness_config = harness_config.copy(workspace=_runtime_workspace_to_config(workspace))

    return harness_config


def _build_model(args: argparse.Namespace):
    """Build the CLI ModelConfig.

    Priority:
      1. AGENT_HOME/model_config.yaml
      2. ANTHROPIC_API_KEY / ANTHROPIC_DEFAULT_MAIN_MODEL  → AnthropicProvider
      3. OPENAI_API_KEY / OPENAI_DEFAULT_MAIN_MODEL        → LiteLLMProvider
      4. LITELLM_API_KEY / LITELLM_DEFAULT_MAIN_MODEL      → LiteLLMProvider
      5. Fallback: AnthropicProvider claude-sonnet-4-6
    """
    import os as _env

    if (persisted := _load_persisted_model_config()) is not None:
        return persisted

    from harnessx.core.model_config import ModelConfig

    _anthropic_key = _env.environ.get("ANTHROPIC_API_KEY")
    _anthropic_model = _env.environ.get("ANTHROPIC_DEFAULT_MAIN_MODEL")
    _openai_key = _env.environ.get("OPENAI_API_KEY")
    _openai_model = _env.environ.get("OPENAI_DEFAULT_MAIN_MODEL")
    _litellm_key = _env.environ.get("LITELLM_API_KEY")
    _litellm_model = _env.environ.get("LITELLM_DEFAULT_MAIN_MODEL")
    _env_headers = _parse_env_extra_headers()
    _timeout = _env.environ.get("HARNESSX_REQUEST_TIMEOUT")

    if _anthropic_key or _anthropic_model:
        from harnessx.providers.anthropic_provider import AnthropicProvider

        _sdk_model = _anthropic_model or "claude-sonnet-4-6"
        _ap_kw: dict = {}
        if _anthropic_key:
            _ap_kw["api_key"] = _anthropic_key
        _base_url = _env.environ.get("ANTHROPIC_API_BASE") or _env.environ.get("ANTHROPIC_BASE_URL")
        if _base_url:
            _ap_kw["base_url"] = _base_url
        if _env_headers:
            _ap_kw["default_headers"] = _env_headers
        if _timeout:
            _ap_kw["timeout"] = float(_timeout)
        model_config = ModelConfig(main=AnthropicProvider(_sdk_model, **_ap_kw))

    elif _openai_key or _openai_model:
        from harnessx.providers.litellm_provider import LiteLLMProvider

        _base_model = _openai_model or "gpt-4o"
        _kw: dict = {}
        if _openai_key:
            _kw["api_key"] = _openai_key
        _api_base = _env.environ.get("OPENAI_API_BASE")
        if _api_base:
            _kw["api_base"] = _api_base
        if _env_headers:
            _kw["extra_headers"] = _env_headers
        if _timeout:
            _kw["request_timeout"] = int(_timeout)
        model_config = ModelConfig(main=LiteLLMProvider(_base_model, **_kw))

    elif _litellm_key or _litellm_model:
        from harnessx.providers.litellm_provider import LiteLLMProvider

        _base_model = _litellm_model or "claude-sonnet-4-6"
        _kw = {}
        if _litellm_key:
            _kw["api_key"] = _litellm_key
        _api_base = _env.environ.get("LITELLM_API_BASE")
        if _api_base:
            _kw["api_base"] = _api_base
        if _env_headers:
            _kw["extra_headers"] = _env_headers
        if _timeout:
            _kw["request_timeout"] = int(_timeout)
        model_config = ModelConfig(main=LiteLLMProvider(_base_model, **_kw))

    else:
        from harnessx.providers.anthropic_provider import AnthropicProvider

        model_config = ModelConfig(main=AnthropicProvider("claude-sonnet-4-6"))

    # First-run auto-save so credentials survive across terminal sessions.
    _has_real_creds = bool(
        _anthropic_key or _anthropic_model or _openai_key or _openai_model or _litellm_key or _litellm_model
    )
    if _has_real_creds:
        from harnessx.home import agent_home as _agent_home

        _mc_path = _agent_home() / "model_config.yaml"
        if not _mc_path.exists():
            try:
                _mc_path.parent.mkdir(parents=True, exist_ok=True)
                _mc_path.write_text(model_config.to_yaml(), encoding="utf-8")
                if sys.stderr.isatty():
                    DIM = "\033[2m"
                    NC = "\033[0m"
                    sys.stderr.write(f"{DIM}  Config saved: {_mc_path}{NC}\n")
            except Exception:
                pass

    return model_config


def _build_agent(args: argparse.Namespace):
    """Build the CLI agent — a fully wired, ready-to-run ``Harness`` instance.

    This is the single entry point that combines ``_build_harness()`` +
    ``_build_model()`` and wires the spawn-subagent tool so delegated
    multi-agent work is available immediately.

    Use ``_build_harness()`` / ``_build_model()`` directly only when you need
    the components separately (e.g. interactive chat, which recreates sessions).
    """
    harness_config = _build_harness(args)
    model_config = _build_model(args)
    return model_config.agentic(harness_config)


def _load_default(*, strict: bool = False):
    """Build the CLI universal agent HarnessConfig using the HarnessBuilder pipeline.

    Full-capability general-purpose agent: coding, writing, research (via skills),
    planning, sub-agents, tools, light-memory, context compaction, reliability guards,
    and observability. Maintained by HarnessX as the official default experience.

    For specialised variants, see the examples/ directory — load them with
    ``-d examples/<name>/harness_config.yaml`` or explore them in Harness Lab.
    """
    from harnessx.bundles.context import make_context, make_window_mgmt
    from harnessx.bundles.reliability import make_reliability
    from harnessx.core.builder import HarnessBuilder
    from harnessx.processors.context.env_context_injector import (
        EnvironmentContextInjector,
    )
    from harnessx.plugins.dimensions.light_memory import LightMemoryPlugin

    from harnessx.plugins.dimensions.mcp_runtime import McpRuntimePlugin
    from harnessx.plugins.dimensions.skill_runtime import SkillRuntimePlugin

    from harnessx.processors.multi_model.model_router import ModelRouterProcessor

    builder = (
        (
            HarnessBuilder()
            | make_context().add(EnvironmentContextInjector())
            | make_reliability(self_verify=strict)
            | make_window_mgmt(token_threshold=140_000)
        )
        .add(ModelRouterProcessor(enabled=False))
        .plugin(LightMemoryPlugin(auto_capture=False))
    )

    # Auto-load installed plugins from AGENT_HOME/plugins/, honoring
    # Lab UI plugin-state toggles (disabled list + extra scan dirs).
    try:
        from harnessx.plugins.discovery import discover_plugins

        disabled, scan_dirs = _load_plugins_state()
        for p in discover_plugins(
            extra_paths=scan_dirs if scan_dirs else None,
            include_claude_plugins=False,
            disabled=disabled,
        ):
            builder = builder.plugin(p)
    except Exception:
        pass  # plugin discovery is best-effort; don't break CLI startup

    builder = builder.plugin(McpRuntimePlugin(ensure_primary=True))
    builder = builder.plugin(SkillRuntimePlugin())
    return builder.build()


# ── Run modes ─────────────────────────────────────────────────────────────────


async def _run_once(harness, task_desc: str, max_steps: int) -> None:
    from harnessx import BaseTask

    _install_loop_exception_filter(asyncio.get_running_loop())

    try:
        result = await harness.run(BaseTask(description=task_desc, max_steps=max_steps))
    finally:
        await _cleanup_harness(harness)

    if result.final_output:
        print(result.final_output)

    print(
        f"\n[steps={result.total_steps}"
        f"  prompt={result.total_input_tokens}"
        f"  complete={result.total_output_tokens}"
        f"  cost=${result.total_cost_usd:.4f}"
        f"  exit={result.exit_reason}]",
        file=sys.stderr,
    )


def _check_api_key(model_config) -> bool:
    """Warn on stderr if no API key is configured for the main provider.

    Returns:
        True when any main-provider API key source is configured, else False.
    """
    has_key = _has_api_key(model_config)
    if not has_key and sys.stderr.isatty():
        YLW = "\033[33m"
        BLD = "\033[1m"
        NC = "\033[0m"
        sys.stderr.write(
            f"\n{BLD}{YLW}  ⚠  No API key detected.{NC}\n"
            f"{YLW}     Model calls will fail without a valid key.{NC}\n"
            f"{YLW}     Configure one provider via environment variables:{NC}\n"
            f"{YLW}       Anthropic: ANTHROPIC_API_KEY (required), ANTHROPIC_DEFAULT_MAIN_MODEL (optional), ANTHROPIC_API_BASE (optional){NC}\n"
            f"{YLW}       OpenAI:    OPENAI_API_KEY (required), OPENAI_DEFAULT_MAIN_MODEL (optional), OPENAI_API_BASE (optional){NC}\n"
            f"{YLW}       LiteLLM:   LITELLM_API_KEY (required), LITELLM_DEFAULT_MAIN_MODEL (optional), LITELLM_API_BASE (optional){NC}\n"
            f"{YLW}     Or launch Harness Lab and configure it under Settings → Model:{NC}\n"
            f"{YLW}       harnessx lab{NC}\n\n"
        )
    return has_key


def _has_api_key(model_config) -> bool:
    """Return True when any API key source is configured for the main provider."""
    import os

    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LITELLM_API_KEY")
        or getattr(model_config.main, "_api_key", None)  # AnthropicProvider
        or getattr(model_config.main, "api_key", None)  # LiteLLMProvider / others
    )


def _detect_memory_backend(harness_config) -> str:
    """Best-effort memory backend label for CLI startup hints."""
    try:
        if any(
            type(p).__module__.startswith("harnessx.plugins.dimensions.light_memory")
            for p in getattr(harness_config, "plugins", []) or []
        ):
            return "light_memory"
    except Exception:
        pass

    try:
        if any(
            type(p).__module__.startswith("harnessx.plugins.dimensions.light_memory")
            for p in (harness_config.processors or [])
            if not isinstance(p, dict)
        ):
            return "light_memory"
    except Exception:
        pass

    return "none"


async def _cleanup_harness(harness) -> None:
    """Best-effort harness-scoped teardown."""
    import inspect

    if harness is None:
        return
    cleanup = getattr(harness, "cleanup", None)
    if not callable(cleanup):
        return
    try:
        result = cleanup()
        if inspect.isawaitable(result):
            await result
    except Exception:
        pass


def _is_cancel_scope_error(exc: BaseException) -> bool:
    """Return True for known anyio/TaskGroup cancel-scope shutdown noise."""
    if "cancel scope" in str(exc).lower():
        return True
    subs = getattr(exc, "exceptions", None)
    if subs:
        return any(_is_cancel_scope_error(e) for e in subs)
    return False


def _install_loop_exception_filter(loop) -> None:
    """Suppress known benign cancel-scope shutdown errors on this event loop."""
    orig = loop.get_exception_handler()

    def _handler(lp, ctx):
        exc = ctx.get("exception")
        if exc is not None and _is_cancel_scope_error(exc):
            return
        if orig is not None:
            orig(lp, ctx)
        else:
            lp.default_exception_handler(ctx)

    loop.set_exception_handler(_handler)


def _format_model_label(model_config, has_api_key: bool) -> str:
    """Format current model label for CLI status UI."""
    if not has_api_key:
        return "none"
    _main = model_config.main
    _mname = getattr(_main, "model", None) or getattr(_main, "_model", "") or "unknown"
    _pname = type(_main).__name__.replace("Provider", "") or "Provider"
    return f"{_pname}/{_mname}"


# ── Interactive prompt (prompt_toolkit) ──────────────────────────────────────

_SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show available slash commands",
    "/new": "Start a new session (clears history)",
    "/compact": "Compact the current context window",
    "/agent": "Switch to a different agent  /agent <id>",
    "/project": "Switch to a different project  /project <name>",
    "/home": "Show AGENT_HOME info",
    "/quit": "Exit  (also /exit, /q)",
}


def _build_prompt_session(ctrl_c_exit_fn=None):
    """Build a prompt_toolkit PromptSession with history, autocomplete, and key bindings.

    Args:
        ctrl_c_exit_fn: Optional callable() → bool.  Called on each Ctrl+C on empty
            buffer.  When it returns True the session exits; when False a warning is
            printed inline via run_in_terminal and the prompt stays live.

    Returns None if prompt_toolkit is unavailable or stdin is not a TTY (fallback to plain input()).
    """
    if not sys.stdin.isatty():
        return None
    try:
        from pathlib import Path
        from prompt_toolkit import PromptSession
        from prompt_toolkit.application import run_in_terminal
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
    except ImportError:
        return None

    # ── History file ──────────────────────────────────────────────────────────
    try:
        from harnessx.home import agent_home

        hist_path = Path(agent_home()) / "chat_history"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(hist_path))
    except Exception:
        history = None  # type: ignore[assignment]

    # ── Slash command completer ───────────────────────────────────────────────
    class _SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            # Only complete at the start of the buffer (no leading whitespace)
            if not text.lstrip(" ").startswith("/") or "\n" in text:
                return
            typed = text.lstrip(" ")
            for cmd, desc in _SLASH_COMMANDS.items():
                if cmd.startswith(typed):
                    yield Completion(
                        cmd,
                        start_position=-len(typed),
                        display_meta=desc,
                    )

    # ── Key bindings ──────────────────────────────────────────────────────────
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        """Enter always submits (even in multiline mode)."""
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")  # Alt+Enter / Meta+Enter
    @kb.add("c-j")  # Ctrl+J (common terminal newline binding)
    def _insert_newline(event):
        """Insert a literal newline (multi-line input)."""
        event.current_buffer.insert_text("\n")

    @kb.add("c-c")
    def _clear_or_exit(event):
        """Ctrl+C with text → clear input; Ctrl+C on empty → warn (first) or exit (second)."""
        buf = event.app.current_buffer
        if buf.text:
            buf.reset()
        elif ctrl_c_exit_fn is not None and ctrl_c_exit_fn():
            # Second press within window — exit the prompt session
            event.app.exit(exception=KeyboardInterrupt)
        else:
            # First press — show warning inline without leaving the prompt
            def _warn() -> None:
                sys.stderr.write("(Press Ctrl+C again to exit)\n")
                sys.stderr.flush()

            run_in_terminal(_warn)

    return PromptSession(
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        completer=_SlashCompleter(),
        complete_while_typing=True,
        key_bindings=kb,
        multiline=True,
    )


# ── Tool call display ─────────────────────────────────────────────────────────


def _fmt_tool_start(name: str, tool_input: dict) -> str:
    """One-line summary shown when a tool call begins."""
    CYN = "\033[36m"
    DIM = "\033[2m"
    NC = "\033[0m"
    # Pick the most useful single argument for each tool
    arg = ""
    n = name.lower()
    if n in ("bash",):
        arg = str(tool_input.get("command", ""))
    elif n in ("read", "write", "edit", "glob"):
        arg = str(tool_input.get("file_path", tool_input.get("path", tool_input.get("pattern", ""))))
    elif n in ("grep",):
        arg = str(tool_input.get("pattern", ""))
        path = str(tool_input.get("path", ""))
        if path:
            arg = f"{arg}  {path}"
    elif n in ("websearch",):
        arg = str(tool_input.get("query", ""))
    elif n in ("webfetch",):
        arg = str(tool_input.get("url", ""))
    else:
        # Generic: show first key=value pair
        for k, v in tool_input.items():
            arg = f"{k}={str(v)[:60]}"
            break
    arg_short = arg[:100] + ("…" if len(arg) > 100 else "")
    return f"\n{CYN}→ {name}{NC}  {DIM}{arg_short}{NC}\n"


def _fmt_tool_result(name: str, result: str, error: str | None, duration_ms: float) -> str:
    """One-line summary shown after a tool call completes."""
    DIM = "\033[2m"
    GRN = "\033[32m"
    RED = "\033[31m"
    NC = "\033[0m"
    ms = f"{duration_ms:.0f}ms" if duration_ms else ""
    if error:
        err_short = error[:80].replace("\n", " ") + ("…" if len(error) > 80 else "")
        return f"  {RED}✗{NC}  {DIM}{err_short}  {ms}{NC}\n"
    lines = result.count("\n") + 1 if result.strip() else 0
    summary = f"{lines} lines" if lines > 1 else (result.strip()[:60] if result.strip() else "done")
    return f"  {GRN}✓{NC}  {DIM}{summary}  {ms}{NC}\n"


async def _chat(
    harness_config,
    model_config,
    max_steps: int,
    resume_run_id: str | None = None,
    verbose: bool = False,
    initial_task: str | None = None,
) -> None:
    import signal
    import time
    import json as _json
    import uuid as _uuid

    from harnessx import BaseTask
    from harnessx.tracing.journal import HarnessJournal
    from harnessx.plugins import plugin_registry
    from harnessx.plugins.builtins.session import SessionPlugin
    from harnessx.plugins.builtins.agent_ctx import AgentContextPlugin
    from harnessx.plugins.builtins.slash_processor import SlashCommandProcessor

    _install_loop_exception_filter(asyncio.get_running_loop())

    _print_banner("interactive", "/help  /new  /compact  /agent  /project  /home  /model  /quit")
    _has_key_cfg = _check_api_key(model_config)

    # Show current model + config hints (TTY only)
    if sys.stderr.isatty():
        DIM = "\033[2m"
        NC = "\033[0m"
        _mem_backend = _detect_memory_backend(harness_config)

        sys.stderr.write(
            f"{DIM}  status bar: model shown in bottom toolbar\n"
            f"  memory backend: {_mem_backend}\n"
            f"  routes: multi-model · ProviderGroup · judge/compact roles → harnessx lab{NC}\n\n"
        )

    # Register built-in plugins so /help shows all commands (idempotent)
    plugin_registry.register(SessionPlugin())
    plugin_registry.register(AgentContextPlugin())

    # Wire SlashCommandProcessor + CLIToolPrinter into the harness.
    # SlashCommandProcessor goes at PRE phase ("*") to intercept before model.
    # CLIToolPrinter goes on before_tool / after_tool to display real-time summaries.
    from harnessx.core.processor import MultiHookProcessor as _MHP
    from harnessx.core.events import ToolCallEvent, ToolResultEvent

    # ── Streaming state (shared between stream_cb and CLIToolPrinter) ─────────
    _at_line_start = [True]  # True = stdout is at start of a new line
    _streamed = [False]  # True = at least one delta was printed this turn

    def _stream_cb(delta: str) -> None:
        sys.stdout.write(delta)
        sys.stdout.flush()
        _at_line_start[0] = delta.endswith("\n")
        _streamed[0] = True

    def _ensure_line_start() -> None:
        """Ensure stdout is at the start of a new line before printing tool output."""
        if not _at_line_start[0]:
            sys.stdout.write("\n")
            sys.stdout.flush()
            _at_line_start[0] = True

    class _CLIToolPrinter(_MHP):
        async def on_before_tool(self, event: ToolCallEvent):  # type: ignore[override]
            _ensure_line_start()
            sys.stdout.write(_fmt_tool_start(event.tool_name, event.tool_input))
            sys.stdout.flush()
            _at_line_start[0] = True
            yield event

        async def on_after_tool(self, event: ToolResultEvent):  # type: ignore[override]
            sys.stdout.write(_fmt_tool_result(event.tool_name, event.result, event.error, event.duration_ms))
            sys.stdout.flush()
            _at_line_start[0] = True
            yield event

    _tool_printer = _CLIToolPrinter()
    # Prepend SlashCommandProcessor so slash commands are handled first,
    # then existing processors, then _tool_printer for tool event output.
    chat_procs = (
        [SlashCommandProcessor(model_config=model_config)] + list(harness_config.processors or []) + [_tool_printer]
    )
    harness_config = harness_config.copy(processors=chat_procs)

    # Each chat session gets a stable session_id so turns are linked on disk
    # and `--resume` can restore them via HarnessJournal.wake().
    session_id = resume_run_id or str(_uuid.uuid4())

    def _make_harness(sid: str):
        from harnessx.core.config_schema import TracerConfig

        cfg = harness_config.copy(tracer=TracerConfig(session_id=sid, silent=not verbose))
        return model_config.agentic(cfg)

    harness = _make_harness(session_id)

    async def _print_mcp_startup_summary_once() -> None:
        """If MCP tools are available, print a one-time startup summary."""
        from harnessx.plugins.dimensions.mcp_runtime import McpRuntimePlugin

        plugin = next((p for p in harness.config.plugins if isinstance(p, McpRuntimePlugin)), None)
        if plugin is None:
            return
        try:
            summary = await plugin.warmup_summary()
        except asyncio.CancelledError:
            # MCP warmup is best-effort; do not block chat startup.
            # Uncancel the task: catching CancelledError without re-raising leaves the
            # asyncio task's cancellation counter incremented (Python 3.12+), which
            # causes subsequent awaits in the chat loop to be immediately cancelled.
            _t = asyncio.current_task()
            if _t is not None and callable(getattr(_t, "uncancel", None)):
                try:
                    while _t.cancelling():
                        _t.uncancel()
                except Exception:
                    pass
            return
        except Exception:
            return

        servers = int(summary.get("servers", 0))
        connected = int(summary.get("connected_servers", servers))
        tools = int(summary.get("tools", 0))
        if servers <= 0:
            return

        DIM = "\033[2m"
        NC = "\033[0m"
        if connected < servers:
            msg = f"MCP: {connected}/{servers} server(s) connected, {tools} tool(s)"
        else:
            msg = f"MCP: {servers} server(s), {tools} tool(s)"
        if sys.stderr.isatty():
            sys.stderr.write(f"{DIM}  {msg}{NC}\n")
        else:
            sys.stderr.write(f"{msg}\n")

    await _print_mcp_startup_summary_once()

    async def _print_skill_startup_summary_once() -> None:
        from harnessx.plugins.dimensions.skill_runtime import SkillRuntimePlugin

        plugin = next(
            (p for p in harness.config.plugins if isinstance(p, SkillRuntimePlugin)),
            None,
        )
        if plugin is None:
            return
        try:
            summary = await plugin.warmup_summary()
        except Exception:
            return

        total = int(summary.get("skills", 0))
        if total <= 0:
            return
        enabled = int(summary.get("enabled", total))
        msg = f"Skills: {total} available" + (f", {enabled} enabled" if enabled < total else "")
        DIM = "\033[2m"
        NC = "\033[0m"
        if sys.stderr.isatty():
            sys.stderr.write(f"{DIM}  {msg}{NC}\n")
        else:
            sys.stderr.write(f"{msg}\n")
        sys.stderr.flush()

    await _print_skill_startup_summary_once()

    async def _replace_harness(sid: str):
        nonlocal harness
        await _cleanup_harness(harness)
        harness = _make_harness(sid)

    _resume_seed_state = None
    _resume_seed_tokens: tuple[int, int] = (0, 0)
    _resume_seed_steps: int = 0

    if resume_run_id:
        # Verify the session exists before claiming we're resuming it.
        _resume_ws = harness._rt.workspace
        if _resume_ws is not None:
            _sess_idx = _resume_ws.root / "sessions" / f"{resume_run_id}.json"
            if not _sess_idx.exists():
                print(
                    f"Session not found: {resume_run_id}\n"
                    f"  (looked in: {_resume_ws.root / 'sessions'})\n"
                    f"Start a new session without --resume, or check the session id.",
                    file=sys.stderr,
                )
                await _cleanup_harness(harness)
                return
        print(f"Resuming session {resume_run_id} …", file=sys.stderr)
        # Seed turn stats baseline from resumed cumulative counters so the first
        # printed "+in/+out/steps" reports only this turn's delta, not full history.
        try:
            if harness._rt.workspace is not None:
                _wake_root = str(harness._rt.workspace.root)
            else:
                from pathlib import Path as _Path

                _tracer = getattr(getattr(harness, "_rt", None), "tracer", None)
                _base_dir = getattr(_tracer, "base_dir", "sessions")
                _wake_root = str(_Path(_base_dir).resolve().parent)
            _resume_seed_state = HarnessJournal.wake(resume_run_id, _wake_root)
            _resume_seed_tokens = (
                int(getattr(_resume_seed_state, "cumulative_input_tokens", 0)),
                int(getattr(_resume_seed_state, "cumulative_output_tokens", 0)),
            )
            _resume_seed_steps = int(getattr(_resume_seed_state, "step", 0))
        except (FileNotFoundError, KeyError, ValueError):
            _resume_seed_state = None

    # ── Ctrl+C state shared with prompt key binding ──────────────────────────
    # First Ctrl+C on empty buffer → inline warning (stays at prompt).
    # Second Ctrl+C within 1.5 s → exit.
    _ctrlc_ts = [0.0]

    def _should_exit_on_ctrlc() -> bool:
        now = time.monotonic()
        if now - _ctrlc_ts[0] < 1.5:
            return True
        _ctrlc_ts[0] = now
        return False

    # ── Prompt session (prompt_toolkit or fallback) ───────────────────────────
    _pt = _build_prompt_session(ctrl_c_exit_fn=_should_exit_on_ctrlc)
    _model_label = _format_model_label(model_config, _has_key_cfg)

    async def _read_input() -> str:
        """Read one turn from the user.  Returns stripped text."""
        # Blank line separator between model output and next prompt — but only
        # when there is actual output above (stdout not already at line start).
        # Suppressed after Ctrl+C warnings, empty turns, etc.
        if not _at_line_start[0]:
            sys.stdout.write("\n")
            sys.stdout.flush()
        if _pt is not None:
            try:
                from prompt_toolkit.formatted_text import HTML

                toolbar = HTML(
                    f" <b>session</b>:{session_id[:8]}…  "
                    f"<b>model</b>:{_model_label} "
                    f" <style bg='ansigray'> Alt+Enter = newline </style>"
                )
            except Exception:
                toolbar = None
            return await _pt.prompt_async("> ", bottom_toolbar=toolbar)
        # Fallback: plain input (no history, no completion)
        return input("> ")

    # ── Active task ref for SIGINT abort ─────────────────────────────────────
    _active_task: list[asyncio.Task | None] = [None]  # mutable ref for signal handler
    _user_aborted_turn: list[bool] = [False]

    loop = asyncio.get_event_loop()

    def _abort_handler(signum, frame):
        """SIGINT during a run: cancel the active task (single press = abort only)."""
        t = _active_task[0]
        if t and not t.done():
            _user_aborted_turn[0] = True
            loop.call_soon_threadsafe(t.cancel)

    def _clear_current_task_cancel_state() -> None:
        """Clear pending cancellation on current task after user-abort handling."""
        task = asyncio.current_task()
        if task is None:
            return
        uncancel = getattr(task, "uncancel", None)
        cancelling = getattr(task, "cancelling", None)
        if not callable(uncancel) or not callable(cancelling):
            return
        try:
            while task.cancelling():
                task.uncancel()
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _print_turn_stats(
        turn_steps: int,
        turn_in: int,
        ctx_in: int,
        turn_out: int,
        ctx_out: int,
        exit_reason: str,
    ) -> None:
        DIM = "\033[2m"
        NC = "\033[0m"
        if sys.stderr.isatty():
            sys.stderr.write(
                f"{DIM}[steps={turn_steps}"
                f"  +in={turn_in}  ctx={ctx_in}"
                f"  +out={turn_out}  out={ctx_out}"
                f"  exit={exit_reason}]{NC}\n"
            )
        else:
            sys.stderr.write(
                f"[steps={turn_steps}"
                f"  prompt={turn_in}  ctx={ctx_in}"
                f"  complete={turn_out}  last_out={ctx_out}"
                f"  exit={exit_reason}]\n"
            )

    prev_tokens: tuple[int, int] = _resume_seed_tokens
    prev_steps: int = _resume_seed_steps

    # True once a real model response completes without error.
    # Used to suppress the resume hint and clean up disk for empty sessions.
    _has_meaningful_turn = False

    # In-memory state carried between turns so the journal continues the same
    # JSONL segment instead of starting a new run_id on every harness.run().
    # Reset to None whenever a new harness / session is created.
    _current_state: "object | None" = _resume_seed_state

    # ── Run initial task provided on the command line ─────────────────────────
    if initial_task:
        _streamed[0] = False
        orig = signal.getsignal(signal.SIGINT)
        _active_task[0] = asyncio.current_task()
        _user_aborted_turn[0] = False
        signal.signal(signal.SIGINT, _abort_handler)
        try:
            result = await harness.run(
                BaseTask(description=initial_task, max_steps=max_steps),
                session_id=session_id,
                stream_callback=_stream_cb,
                _resume_state=_current_state,
            )
        except asyncio.CancelledError:
            if not _user_aborted_turn[0]:
                raise
            _clear_current_task_cancel_state()
            _ensure_line_start()
            sys.stderr.write("[interrupted]\n")
            sys.stderr.flush()
            result = None
        finally:
            _active_task[0] = None
            _user_aborted_turn[0] = False
            signal.signal(signal.SIGINT, orig)

        if result is not None:
            _ensure_line_start()
            cur_in = result.resume_state.cumulative_input_tokens
            cur_out = result.resume_state.cumulative_output_tokens
            turn_in = cur_in - prev_tokens[0]
            turn_out = cur_out - prev_tokens[1]
            turn_steps = result.total_steps - prev_steps
            prev_tokens = (cur_in, cur_out)
            prev_steps = result.total_steps
            if not _streamed[0] and result.final_output:
                print(result.final_output)
            _streamed[0] = False
            _print_turn_stats(
                turn_steps,
                turn_in,
                result.last_step_input_tokens,
                turn_out,
                result.last_step_output_tokens,
                result.exit_reason,
            )
            if result.exit_reason != "error":
                _has_meaningful_turn = True
                _current_state = result.resume_state

    # ── Main interactive loop ─────────────────────────────────────────────────
    while True:
        # ── Read input ────────────────────────────────────────────────────────
        try:
            raw = await _read_input()
        except KeyboardInterrupt:
            if _pt is None:
                # No prompt_toolkit: implement double-Ctrl+C manually (warning on first,
                # exit on second within 1.5 s) since the key binding path is unavailable.
                now = time.monotonic()
                if now - _ctrlc_ts[0] < 1.5:
                    sys.stderr.write("\nBye.\n")
                    break
                _ctrlc_ts[0] = now
                sys.stderr.write("\n(Press Ctrl+C again to exit)\n")
                sys.stderr.flush()
                continue
            # prompt_toolkit path: key binding showed the inline warning on first press;
            # KeyboardInterrupt reaching here means second press → exit.
            sys.stderr.write("\nBye.\n")
            break
        except EOFError:
            sys.stderr.write("\nBye.\n")
            break

        raw = raw.strip()
        if not raw:
            continue

        # ── Run turn (slash commands intercepted by SlashCommandProcessor) ────
        _streamed[0] = False
        orig_sigint = signal.getsignal(signal.SIGINT)
        _active_task[0] = asyncio.current_task()
        _user_aborted_turn[0] = False
        signal.signal(signal.SIGINT, _abort_handler)
        try:
            result = await harness.run(
                BaseTask(description=raw, max_steps=max_steps),
                session_id=session_id,
                stream_callback=_stream_cb,
                _resume_state=_current_state,
            )
        except asyncio.CancelledError:
            if not _user_aborted_turn[0]:
                raise
            _clear_current_task_cancel_state()
            _ensure_line_start()
            sys.stderr.write("[interrupted]\n")
            sys.stderr.flush()
            _streamed[0] = False
            continue
        except Exception as _exc:
            _ensure_line_start()
            RED = "\033[31m"
            BLD = "\033[1m"
            DIM = "\033[2m"
            NC = "\033[0m"
            _ename = type(_exc).__name__
            _emsg = str(_exc).splitlines()[0][:200]  # first line, truncated
            if verbose:
                import traceback as _tb

                sys.stderr.write(f"\n{RED}")
                _tb.print_exc(file=sys.stderr)
                sys.stderr.write(NC)
            else:
                # Friendly hint for common failure modes
                if "Auth" in _ename or "401" in _emsg or "Unauthorized" in _emsg:
                    _hint = "Check your API key (ANTHROPIC_API_KEY / OPENAI_API_KEY / LITELLM_API_KEY)"
                elif "Connection" in _ename or "connect" in _emsg.lower():
                    _hint = "Check network connectivity and API base URL"
                elif "omegaconf" in type(_exc).__module__:
                    _hint = "Config serialization error — please report this"
                else:
                    _hint = "Run with -v for a full traceback"
                sys.stderr.write(f"\n{BLD}{RED}  Error: {_ename}{NC}\n{RED}  {_emsg}{NC}\n{DIM}  {_hint}{NC}\n\n")
            _streamed[0] = False
            continue
        finally:
            _active_task[0] = None
            _user_aborted_turn[0] = False
            signal.signal(signal.SIGINT, orig_sigint)

        _ensure_line_start()
        er = result.exit_reason

        # Capture in-memory state for next turn (keeps same run_id / JSONL segment).
        # Overridden to None below when a new harness/session is created.
        _current_state = result.resume_state

        # ── Slash command results ─────────────────────────────────────────────
        if er == "slash:quit":
            break

        if er == "slash:new":
            session_id = result.final_output or str(_uuid.uuid4())
            await _replace_harness(session_id)
            _current_state = None
            prev_tokens = (0, 0)
            prev_steps = 0
            _streamed[0] = False
            _has_meaningful_turn = False
            continue

        if er == "slash:switch":
            try:
                data = _json.loads(result.final_output or "{}")
                session_id = data.get("session_id") or str(_uuid.uuid4())
                from harnessx.home import agent_home
                from harnessx.workspace.workspace import Workspace

                new_ws = Workspace(
                    agent_id=data["agent_id"],
                    project=data["project"],
                    home=agent_home(),
                )
                from harnessx.core.harness import _runtime_workspace_to_config

                harness_config = harness_config.copy(workspace=_runtime_workspace_to_config(new_ws))
                await _replace_harness(session_id)
                _current_state = None
                prev_tokens = (0, 0)
                prev_steps = 0
                _has_meaningful_turn = False
            except Exception as exc:
                print(f"  /agent|/project switch failed: {exc}", file=sys.stderr)
            _streamed[0] = False
            continue

        if er == "slash:model_use":
            try:
                data = _json.loads(result.final_output or "{}")
                role_name = data.get("role", "main")
                model_ref = data.get("model_ref", "")
                if not model_ref:
                    raise ValueError("model_ref is empty")

                # Build a new provider for the requested model_ref.
                # Check the persisted yaml registry first; fall back to direct instantiation.
                _candidates = _state_candidates("model_config.yaml")
                _yaml_path = next((p for p in _candidates if p.exists()), _candidates[0])
                new_provider = None
                if _yaml_path.exists():
                    try:
                        import yaml as _yaml

                        _yd = _yaml.safe_load(_yaml_path.read_text(encoding="utf-8"))
                        if _yd.get("schema_version") == 2:
                            _specs = {
                                m["id"]: {
                                    k: v for k, v in m.items() if not k.startswith("_") and k not in ("id", "provider")
                                }
                                for m in _yd.get("models", [])
                            }
                            if model_ref in _specs:
                                from harnessx.core.model_config import (
                                    _instantiate_provider,
                                )

                                new_provider = _instantiate_provider(_specs[model_ref])
                    except Exception:
                        pass

                if new_provider is None:
                    # Fall back: create provider from model name directly
                    from harnessx.providers.litellm_provider import (
                        LiteLLMProvider,
                        _is_anthropic_model,
                    )

                    if _is_anthropic_model(model_ref):
                        from harnessx.providers.anthropic_provider import (
                            AnthropicProvider,
                        )

                        new_provider = AnthropicProvider(model_ref)
                    else:
                        new_provider = LiteLLMProvider(model_ref)

                model_config = model_config.copy(**{role_name: new_provider})
                await _replace_harness(session_id)
                _current_state = None
                _has_key_cfg = _has_api_key(model_config)
                _model_label = _format_model_label(model_config, _has_key_cfg)
                DIM = "\033[2m"
                NC = "\033[0m"
                _pname = type(model_config.main).__name__.replace("Provider", "")
                if sys.stderr.isatty():
                    sys.stderr.write(
                        f"{DIM}  Switched {role_name} → {_pname}/{getattr(model_config.main, 'model', '')}"
                        f"  (session only — use /model default to persist){NC}\n"
                    )
            except Exception as _exc:
                print(f"  /model use failed: {_exc}", file=sys.stderr)
            _streamed[0] = False
            continue

        if er and er.startswith("slash:"):
            # slash:info, slash:compact_done, slash:unknown — processor already
            # printed any output; just continue the loop
            _streamed[0] = False
            continue

        # ── Run error (exit_reason == "error") ───────────────────────────────
        if er == "error":
            _ensure_line_start()
            RED = "\033[31m"
            BLD = "\033[1m"
            DIM = "\033[2m"
            NC = "\033[0m"
            _emsg = getattr(result, "error", "") or result.final_output or "unknown error"
            _emsg_first = _emsg.splitlines()[0][:300]
            # Infer a helpful hint from the error text
            _emsg_low = _emsg.lower()
            if "api_key" in _emsg_low or "auth" in _emsg_low or "401" in _emsg_low or "unauthorized" in _emsg_low:
                _hint = "No valid API key — set one of:\n    ANTHROPIC_API_KEY / OPENAI_API_KEY / LITELLM_API_KEY"
            elif "connect" in _emsg_low or "network" in _emsg_low or "timeout" in _emsg_low:
                _hint = "Check network connectivity and API base URL (ANTHROPIC_API_BASE etc.)"
            elif "model" in _emsg_low and ("not found" in _emsg_low or "does not exist" in _emsg_low):
                _hint = "Model not found — check ANTHROPIC_DEFAULT_MAIN_MODEL or /model list"
            else:
                _hint = "Run with -v for full traceback"
            sys.stderr.write(f"\n{BLD}{RED}  Error{NC}  {_emsg_first}\n{DIM}  {_hint}{NC}\n\n")
            _streamed[0] = False
            continue

        # ── Normal turn: display output ───────────────────────────────────────
        _has_meaningful_turn = True
        cur_in = result.resume_state.cumulative_input_tokens
        cur_out = result.resume_state.cumulative_output_tokens
        turn_in = cur_in - prev_tokens[0]
        turn_out = cur_out - prev_tokens[1]
        turn_steps = result.total_steps - prev_steps
        prev_tokens = (cur_in, cur_out)
        prev_steps = result.total_steps

        # If streaming was active, output was already printed token-by-token.
        # Only fall back to final_output when streaming did not fire at all
        # (e.g. the response was entirely tool_use with no text content).
        if not _streamed[0] and result.final_output:
            print(result.final_output)
        _streamed[0] = False

        _print_turn_stats(
            turn_steps,
            turn_in,
            result.last_step_input_tokens,
            turn_out,
            result.last_step_output_tokens,
            result.exit_reason,
        )

    # ── Exit: show session resume hint only if there is content worth resuming ──
    # A resumed session always has pre-existing content — never clean it up.
    _session_has_content = _has_meaningful_turn or bool(resume_run_id)
    if _session_has_content:
        DIM = "\033[2m"
        NC = "\033[0m"
        if sys.stderr.isatty():
            sys.stderr.write(f"\n{DIM}Session  : {session_id}\nTo resume: harnessx --resume {session_id}{NC}\n")
        else:
            sys.stderr.write(f"\nSession: {session_id}\nTo resume: harnessx --resume {session_id}\n")
    else:
        # Nothing meaningful happened — delete empty session artifacts from disk
        # so they don't accumulate across trivial exits (Ctrl+C, API key errors, etc.)
        import shutil as _shutil

        try:
            _ws = harness._rt.workspace
            if _ws is not None:
                _sessions_dir = _ws.root / "sessions"
                _sess_subdir = _sessions_dir / session_id
                _sess_idx = _sessions_dir / f"{session_id}.json"
                if _sess_subdir.exists():
                    _shutil.rmtree(_sess_subdir, ignore_errors=True)
                if _sess_idx.exists():
                    _sess_idx.unlink(missing_ok=True)
        except Exception:
            pass

    await _cleanup_harness(harness)


def _plugin(args) -> None:
    """Handle ``harnessx plugin`` subcommands."""
    plugin_cmd = getattr(args, "plugin_command", None)

    if plugin_cmd == "list":
        from pathlib import Path
        from harnessx.plugins.discovery import discover_plugins
        from harnessx.home import agent_home as _agent_home

        agent_plugins = (_agent_home() / "plugins").resolve()
        claude_plugins = (Path.home() / ".claude" / "plugins").resolve()

        def _classify_source(plugin_obj) -> tuple[str, str, str]:
            """Return (status, source, path) for display."""
            root = getattr(plugin_obj, "_plugin_root", None)
            if not root:
                return "external", "python", "-"

            p = Path(root).resolve()
            p_str = str(p)
            if p.is_relative_to(agent_plugins):
                return "installed", "harnessx", p_str
            if p.is_relative_to(claude_plugins):
                return "external", "claude", p_str
            return "external", "path", p_str

        plugins = discover_plugins()
        if not plugins:
            print("No plugins found.", file=sys.stderr)
            print(
                "Install plugins to ~/.harnessx/plugins/ or project .harnessx/plugins/",
                file=sys.stderr,
            )
            return
        print(f"Found {len(plugins)} plugin(s):", file=sys.stderr)
        for p in plugins:
            status, source, path = _classify_source(p)
            print(
                f"  {p.name} v{p.version}  [{status}] source={source}  —  {p.description}",
                file=sys.stderr,
            )
            if path != "-":
                print(f"    path: {path}", file=sys.stderr)
            if status != "installed":
                print(
                    f"    to install into ~/.harnessx/plugins/: hx plugin install {p.name}",
                    file=sys.stderr,
                )

    elif plugin_cmd == "convert":
        from pathlib import Path
        from harnessx.plugins.convert import convert_claude_plugin

        src = Path(args.src)
        dst = Path(args.output) if getattr(args, "output", None) else None
        try:
            result = convert_claude_plugin(src, dst)
            print(f"Converted plugin written to: {result}", file=sys.stderr)
            print(
                f"\nNext steps:\n"
                f"  1. Review {result}/plugin.json and fill in processor targets\n"
                f"  2. Implement processors in {result}/processors/\n"
                f"  3. Register: builder.plugin('{result}')",
                file=sys.stderr,
            )
        except (FileNotFoundError, FileExistsError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif plugin_cmd in ("add", "install"):
        from pathlib import Path
        from harnessx.plugins.discovery import discover_plugins
        from harnessx.plugins.loader import load_plugin
        from harnessx.home import agent_home as _agent_home
        import shutil

        src = (args.src or "").strip()
        install_dir = _agent_home() / "plugins"
        install_dir.mkdir(parents=True, exist_ok=True)

        resolved_dir: Path | None = None
        # 1) explicit local path
        src_path = Path(src).expanduser()
        if src and ("/" in src or "\\" in src or src.startswith(".") or src_path.exists()):
            if src_path.is_dir():
                resolved_dir = src_path.resolve()
            elif src_path.exists():
                print(f"Error: not a plugin directory: {src_path}", file=sys.stderr)
                sys.exit(1)

        # 2) plugin name from `plugin list` (e.g. Claude external plugin)
        if resolved_dir is None and src and "/" not in src and "\\" not in src:
            matches = [p for p in discover_plugins() if (p.name or "") == src]
            if len(matches) == 1:
                root = getattr(matches[0], "_plugin_root", None)
                if root and Path(root).is_dir():
                    resolved_dir = Path(root).resolve()
            elif len(matches) > 1:
                print(
                    f"Error: multiple plugins named '{src}' were found. Use a full directory path instead.",
                    file=sys.stderr,
                )
                sys.exit(1)

        load_src = str(resolved_dir) if resolved_dir is not None else src
        try:
            plugin = load_plugin(load_src)
        except Exception as e:
            print(f"Error loading plugin from {load_src!r}: {e}", file=sys.stderr)
            sys.exit(1)

        if resolved_dir is not None:
            dir_name = (plugin.name or resolved_dir.name).strip() or resolved_dir.name
            dst_dir = install_dir / dir_name
            if dst_dir.exists():
                print(f"Plugin already installed at: {dst_dir}", file=sys.stderr)
                sys.exit(1)
            shutil.copytree(resolved_dir, dst_dir)
            print(f"Installed '{plugin.name}' → {dst_dir}", file=sys.stderr)
        else:
            print(
                f"Plugin loaded from Python path ({plugin.name}). "
                "Python-path plugins don't need add/install — just register them in code.",
                file=sys.stderr,
            )

    elif plugin_cmd == "remove":
        from pathlib import Path
        from harnessx.home import agent_home as _agent_home
        import shutil

        name = args.name
        install_dir = _agent_home() / "plugins"
        target = install_dir / name

        if not target.exists():
            # Try matching by plugin.name field across all installed dirs
            from harnessx.plugins.loader import load_plugin

            matched = None
            for candidate in install_dir.iterdir() if install_dir.is_dir() else []:
                if not candidate.is_dir():
                    continue
                try:
                    p = load_plugin(candidate)
                    if p.name == name:
                        matched = candidate
                        break
                except Exception:
                    continue
            if matched is None:
                print(f"Plugin '{name}' not found in {install_dir}", file=sys.stderr)
                sys.exit(1)
            target = matched

        confirmed = getattr(args, "yes", False)
        if not confirmed:
            resp = input(f"Remove plugin '{target.name}' from {target}? [y/N] ").strip().lower()
            if resp not in ("y", "yes"):
                print("Aborted.", file=sys.stderr)
                return

        shutil.rmtree(target)
        print(f"Removed '{target.name}' from {install_dir}", file=sys.stderr)

    else:
        print("Usage: harnessx plugin {list|convert|add|install|remove}", file=sys.stderr)
        print("       harnessx plugin --help", file=sys.stderr)


def _uninstall(args) -> None:
    import os
    import shutil
    from pathlib import Path

    yes: bool = getattr(args, "yes", False)

    RED = "\033[0;31m" if sys.stdout.isatty() else ""
    GREEN = "\033[0;32m" if sys.stdout.isatty() else ""
    YELLOW = "\033[1;33m" if sys.stdout.isatty() else ""
    BLUE = "\033[0;34m" if sys.stdout.isatty() else ""
    BOLD = "\033[1m" if sys.stdout.isatty() else ""
    DIM = "\033[2m" if sys.stdout.isatty() else ""
    NC = "\033[0m" if sys.stdout.isatty() else ""

    def info(msg: str) -> None:
        print(f"{BLUE}▸{NC} {msg}")

    def success(msg: str) -> None:
        print(f"{GREEN}✓{NC} {msg}")

    def warn(msg: str) -> None:
        print(f"{YELLOW}!{NC} {msg}")

    def ask(prompt: str) -> bool:
        if yes:
            return True
        try:
            answer = input(f"{prompt} [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return answer in ("y", "yes")

    install_dir = os.environ.get("HARNESSX_INSTALL_DIR", str(Path.home() / ".local" / "share" / "harnessx"))
    bin_dir = str(Path.home() / ".local" / "bin")
    data_dir = str(Path.home() / ".harnessx")

    print()
    print(f"{BOLD}══════════════════════════════════{NC}")
    print(f"{RED}  HarnessX Uninstaller{NC}")
    print(f"{BOLD}══════════════════════════════════{NC}")
    print()
    print(f"  {DIM}This will remove HarnessX from your system.{NC}")
    print(f"  {DIM}You will be asked before each step.{NC}")
    print()

    if not ask("Proceed with HarnessX uninstall?"):
        info("Uninstall cancelled")
        return

    # Step 1: binaries
    print(f"\n{BOLD}Step 1/4  Remove CLI binaries{NC}")
    bin_names = ["hx", "harnessx", "hx-gateway", "harnessx-gateway"]
    found_bins = [os.path.join(bin_dir, b) for b in bin_names if os.path.lexists(os.path.join(bin_dir, b))]
    if found_bins:
        info("Will remove: " + "  ".join(found_bins))
        if ask("Remove CLI binaries?"):
            for b in found_bins:
                os.unlink(b)
            success("Binaries removed")
        else:
            warn("Skipped")
    else:
        info(f"No binaries found in {bin_dir} — skipping")

    # Step 2: install directory
    print(f"\n{BOLD}Step 2/4  Remove install directory{NC}")
    if os.path.isdir(install_dir):
        try:
            size = shutil.disk_usage(install_dir)
            size_mb = size.used // (1024 * 1024)
            size_str = f"{size_mb} MB"
        except OSError:
            size_str = "?"
        info(f"Install directory: {install_dir}  ({size_str})")
        if ask(f"Delete {install_dir} (source code, venv, built assets)?"):
            shutil.rmtree(install_dir)
            success("Install directory removed")
        else:
            warn(f"Skipped — install directory kept at {install_dir}")
    else:
        info(f"Install directory {install_dir} not found — skipping")

    # Step 3: workspace / config data
    print(f"\n{BOLD}Step 3/4  Remove workspace and config data{NC}")
    if os.path.isdir(data_dir):
        try:
            size = shutil.disk_usage(data_dir)
            size_mb = size.used // (1024 * 1024)
            size_str = f"{size_mb} MB"
        except OSError:
            size_str = "?"
        warn(f"This contains your agent workspace sessions, logs, and local config ({size_str}).")
        if ask(f"Delete {data_dir} (workspace sessions, logs, config)?"):
            shutil.rmtree(data_dir)
            success("Data directory removed")
        else:
            warn(f"Skipped — workspace data kept at {data_dir}")
    else:
        info(f"Data directory {data_dir} not found — skipping")

    # Step 4: shell profile
    print(f"\n{BOLD}Step 4/4  Clean shell profile{NC}")
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        rc_file = str(Path.home() / ".zshrc")
    elif (Path.home() / ".bash_profile").exists():
        rc_file = str(Path.home() / ".bash_profile")
    else:
        rc_file = str(Path.home() / ".bashrc")

    if os.path.isfile(rc_file):
        with open(rc_file) as f:
            content = f.read()
        if "# HarnessX" in content or "HARNESSX_WORKSPACE" in content:
            info(f"Will remove HarnessX PATH and HARNESSX_WORKSPACE entries from {rc_file}")
            if ask(f"Clean {rc_file}?"):
                lines = content.splitlines(keepends=True)
                cleaned = [
                    line
                    for line in lines
                    if "# HarnessX" not in line
                    and "HARNESSX_WORKSPACE" not in line
                    and f'export PATH="{bin_dir}:$PATH"' not in line
                ]
                with open(rc_file, "w") as f:
                    f.writelines(cleaned)
                success("Shell profile cleaned")
                warn(f"Restart your shell or run: source {rc_file}")
            else:
                warn(f"Skipped — remove these lines manually from {rc_file}:")
                warn("  # HarnessX")
                warn(f'  export PATH="{bin_dir}:$PATH"')
                warn("  export HARNESSX_WORKSPACE=...")
        else:
            info(f"No HarnessX entries found in {rc_file} — skipping")
    else:
        info(f"Shell profile {rc_file} not found — skipping")

    print()
    print(f"{BOLD}══════════════════════════════════{NC}")
    print(f"{GREEN}  Uninstall complete{NC}")
    print(f"{BOLD}══════════════════════════════════{NC}")
    print()
    info("HarnessX has been removed from your system.")
    print()


def _lab(args) -> None:
    port = getattr(args, "port", 7861)
    dev = getattr(args, "dev", False)
    auto_open = getattr(args, "open", False)
    verbose = getattr(args, "verbose", False)
    if not verbose:
        import os as _os

        _os.environ.setdefault("HARNESSX_LAB_SILENT", "1")

    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: Harness Lab requires extra dependencies.\nInstall with: pip install 'harnessx'",
            file=sys.stderr,
        )
        sys.exit(1)

    from harnessx.api.app import create_app

    app = create_app(serve_static=not dev)
    url = f"http://localhost:{port}"
    _print_banner("lab", url)
    if dev:
        DIM = "\033[2m"
        NC = "\033[0m"
        sys.stderr.write(f"  {DIM}dev mode — run `npm run dev` in frontend/{NC}\n\n")

    if auto_open:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import asyncio

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)

    async def _serve() -> None:
        _install_loop_exception_filter(asyncio.get_running_loop())
        await server.serve()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

_SUBCOMMANDS = {"lab", "plugin", "uninstall"}


def _anyio_run(coro) -> None:
    """Run a coroutine with anyio's asyncio backend for proper cancel-scope support.

    Bare asyncio.run() doesn't initialise anyio's backend state, so anyio cancel
    scopes created by MCP's streamablehttp_client can't exit cleanly when the
    connection is closed from a different asyncio task context, causing
    'Attempted to exit cancel scope in a different task' RuntimeErrors that
    propagate as CancelledErrors into prompt_toolkit and crash the CLI.
    """
    try:
        import anyio

        async def _wrapped():
            await coro

        anyio.run(_wrapped, backend="asyncio")
    except ImportError:
        asyncio.run(coro)


def main() -> None:
    try:
        from harnessx.core.config_store import register_harnessx_configs

        register_harnessx_configs()
    except Exception:
        pass

    # Normalise argv so that shared flags (-m, -H, …) always appear before the
    # subcommand token, where the top-level parser can consume them.
    #
    # Default behaviour (no subcommand): interactive mode.
    #
    # Case 1 — bare task string (no subcommand):
    #   harnessx -m model "do X"
    #   → interactive mode, "do X" run as the first turn
    #
    # Case 2 — explicit subcommand:
    #   harnessx run "do X"   → one-shot
    #   harnessx lab          → Harness Lab UI
    argv = sys.argv[1:]
    _FLAGS_WITH_VALUE = {
        "-m",
        "--model",
        "--max-steps",
        "-H",
        "--resume",
    }

    flags: list[str] = []  # collected flag tokens (flag + value pairs)
    rest: list[str] = []  # everything after the first non-flag token
    initial_task: str | None = None  # task string to run as first interactive turn

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in _FLAGS_WITH_VALUE and i + 1 < len(argv):
            flags += [arg, argv[i + 1]]
            i += 2
        elif arg == "--router":
            # --router takes an optional value (nargs="?"):
            #   --router key=val,key=val  → consume the next token as params
            #   --router -p "prompt"      → no params, don't consume -p
            #   --router "prompt"         → no params, don't consume prompt
            flags.append(arg)
            i += 1
            if i < len(argv) and "=" in argv[i]:
                flags.append(argv[i])
                i += 1
        elif arg.startswith("-"):
            flags.append(arg)
            i += 1
        else:
            rest = argv[i:]
            break

    if rest:
        first = rest[0]
        if first not in _SUBCOMMANDS:
            # bare task string → interactive mode with initial task
            initial_task = rest[0]
            argv = flags
        else:
            # explicit subcommand (run, lab, plugin) — ensure flags come before it
            argv = flags + rest
    else:
        argv = flags

    parser = _build_parser()
    args = parser.parse_args(argv)

    command = args.command

    if command == "lab":
        with _sigterm_as_keyboard_interrupt():
            _lab(args)
        return

    if command == "plugin":
        _plugin(args)
        return

    if command == "uninstall":
        _uninstall(args)
        return

    max_steps = getattr(args, "max_steps", 30)

    if getattr(args, "print_mode", False):
        # -p / --print: non-interactive, print response and exit
        if not initial_task:
            sys.stderr.write("error: -p/--print requires a prompt argument\n")
            sys.exit(1)
        try:
            with _sigterm_as_keyboard_interrupt():
                _anyio_run(_run_once(_build_agent(args), initial_task, max_steps))
        except KeyboardInterrupt:
            pass
    else:
        # Default: interactive chat — needs components separately for per-session rebuild
        harness_config = _build_harness(args)
        model_config = _build_model(args)
        _verbose = getattr(args, "verbose", False) or getattr(args, "debug", False)
        try:
            with _sigterm_as_keyboard_interrupt():
                _anyio_run(
                    _chat(
                        harness_config,
                        model_config,
                        max_steps,
                        initial_task=initial_task,
                        resume_run_id=getattr(args, "resume", None),
                        verbose=_verbose,
                    )
                )
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
