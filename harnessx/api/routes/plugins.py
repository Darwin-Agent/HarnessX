# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(tags=["plugins"])

# Persistence: tracks enabled state, scan directories, and individually-imported plugin paths
_STATE_FILE = Path.home() / ".harnessx" / "plugins_state.json"


# ── Models ────────────────────────────────────────────────────────────────────


class PluginInfo(BaseModel):
    id: str
    name: str
    description: str
    version: str
    path: str
    enabled: bool
    tool_count: int
    skill_count: int
    mcp_count: int


class PluginPatch(BaseModel):
    enabled: bool | None = None


class PluginImportRequest(BaseModel):
    path: str


class ScanDirRequest(BaseModel):
    path: str


class ScanDirsResponse(BaseModel):
    scan_dirs: list[str]


# ── Persistence helpers ───────────────────────────────────────────────────────


def _load_state() -> dict[str, Any]:
    """Return {disabled: list[str], scan_dirs: list[str], plugin_paths: list[str]}."""
    if not _STATE_FILE.exists():
        return {"disabled": [], "scan_dirs": [], "plugin_paths": []}
    try:
        state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        # Migrate legacy extra_paths → scan_dirs (pre-release format)
        if "extra_paths" in state and "scan_dirs" not in state:
            state["scan_dirs"] = state.pop("extra_paths")
        state.setdefault("disabled", [])
        state.setdefault("scan_dirs", [])
        state.setdefault("plugin_paths", [])
        return state
    except Exception:
        return {"disabled": [], "scan_dirs": [], "plugin_paths": []}


def _save_state(state: dict[str, Any]) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _plugin_id(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name or ""))


def _plugin_to_info(plugin: Any, disabled_names: set[str]) -> PluginInfo:
    # Prefer _plugin_root set by load_from_directory() — the actual filesystem path
    path = ""
    if hasattr(plugin, "_plugin_root") and plugin._plugin_root:
        path = str(plugin._plugin_root)
    else:
        # Fallback: Python source file (less accurate, only for code-defined plugins)
        import inspect

        try:
            src = inspect.getfile(type(plugin))
            path = str(Path(src).parent)
        except Exception:
            pass

    return PluginInfo(
        id=_plugin_id(plugin.name or path),
        name=plugin.name or "unnamed",
        description=plugin.description or "",
        version=getattr(plugin, "version", "0.1.0"),
        path=path,
        enabled=(plugin.name not in disabled_names),
        tool_count=len(getattr(plugin, "tools", []) or []),
        skill_count=len(getattr(plugin, "skill_dirs", []) or []),
        mcp_count=len(getattr(plugin, "mcp_servers", []) or []),
    )


def _load_all_plugins(state: dict[str, Any]) -> list[Any]:
    """Load plugins from scan_dirs (multi-plugin parent dirs) + plugin_paths (individual dirs)."""
    from ...plugins.discovery import discover_plugins
    from ...plugins.loader import load_from_directory

    scan_dirs = [Path(p) for p in state.get("scan_dirs", []) if Path(p).is_dir()]
    plugin_paths = [Path(p) for p in state.get("plugin_paths", []) if Path(p).is_dir()]

    try:
        plugins = discover_plugins(extra_paths=scan_dirs if scan_dirs else None)
    except Exception:
        plugins = []

    # Load individually-imported plugins that aren't already found via scan dirs
    known_names = {p.name for p in plugins}
    for pp in plugin_paths:
        try:
            p = load_from_directory(pp)
            if p.name not in known_names:
                plugins.append(p)
                known_names.add(p.name)
        except Exception:
            pass

    return plugins


# ── Plugin routes ─────────────────────────────────────────────────────────────
# IMPORTANT: fixed-path routes (scan-dirs, import) must come BEFORE the
# dynamic /{plugin_id} routes so FastAPI doesn't shadow them.


@router.get("/plugins", response_model=list[PluginInfo])
async def list_plugins() -> Any:
    """Discover all available plugins and return their info with enabled state."""
    state = _load_state()
    disabled = set(state.get("disabled", []))
    plugins = _load_all_plugins(state)
    return [_plugin_to_info(p, disabled) for p in plugins]


@router.get("/plugins/scan-dirs", response_model=ScanDirsResponse)
async def list_scan_dirs() -> Any:
    """List all plugin scan directories (parent directories that are auto-scanned).

    Always includes the built-in ``AGENT_HOME/plugins/`` directory so the UI
    shows where auto-discovered plugins come from.
    """
    from ...home import agent_home

    builtin = str(agent_home() / "plugins")
    state = _load_state()
    user_dirs = state.get("scan_dirs", [])
    # Prepend built-in dir; deduplicate in case the user added it manually.
    all_dirs = [builtin] + [d for d in user_dirs if d != builtin]
    return ScanDirsResponse(scan_dirs=all_dirs)


@router.post("/plugins/scan-dirs", response_model=ScanDirsResponse)
async def add_scan_dir(req: ScanDirRequest) -> Any:
    """Add a directory to the plugin scan path."""
    from ...home import agent_home

    expanded = os.path.expanduser(req.path)
    dir_path = Path(expanded).resolve()

    if not dir_path.is_dir():
        raise HTTPException(400, f"Not a directory: {req.path}")

    state = _load_state()
    scan_dirs: list[str] = state.get("scan_dirs", [])
    str_path = str(dir_path)
    if str_path not in scan_dirs:
        scan_dirs.append(str_path)
        state["scan_dirs"] = scan_dirs
        _save_state(state)

    # Return with built-in dir prepended (same as list_scan_dirs).
    builtin = str(agent_home() / "plugins")
    user_dirs = state.get("scan_dirs", [])
    all_dirs = [builtin] + [d for d in user_dirs if d != builtin]
    return ScanDirsResponse(scan_dirs=all_dirs)


@router.delete("/plugins/scan-dirs", response_model=ScanDirsResponse)
async def remove_scan_dir(
    path: str = Query(..., description="The scan directory path to remove"),
) -> Any:
    """Remove a directory from the plugin scan path."""
    from ...home import agent_home

    expanded = os.path.expanduser(path)
    str_path = str(Path(expanded).resolve())

    # Prevent removal of the built-in AGENT_HOME/plugins/ directory.
    builtin = str(agent_home() / "plugins")
    if str_path == builtin:
        raise HTTPException(400, "Cannot remove the built-in plugins directory")

    state = _load_state()
    state["scan_dirs"] = [p for p in state.get("scan_dirs", []) if p != str_path]
    _save_state(state)

    # Return with built-in dir prepended (same as list_scan_dirs).
    user_dirs = state.get("scan_dirs", [])
    all_dirs = [builtin] + [d for d in user_dirs if d != builtin]
    return ScanDirsResponse(scan_dirs=all_dirs)


@router.post("/plugins/import", response_model=PluginInfo, status_code=201)
async def import_plugin(req: PluginImportRequest) -> Any:
    """Copy a plugin directory into AGENT_HOME/plugins/ so it is auto-discovered by runs."""
    from ...home import agent_home
    from ...plugins.loader import load_from_directory

    expanded = os.path.expanduser(req.path)
    plugin_path = Path(expanded).resolve()

    if not plugin_path.is_dir():
        raise HTTPException(400, f"Not a directory: {req.path}")

    has_plugin_json = (plugin_path / "plugin.json").exists()
    has_claude_json = (plugin_path / ".claude-plugin" / "plugin.json").exists()
    if not has_plugin_json and not has_claude_json:
        raise HTTPException(
            400,
            "Directory does not contain plugin.json or .claude-plugin/plugin.json",
        )

    # Load first to validate and get the canonical plugin name.
    try:
        plugin = load_from_directory(plugin_path)
    except Exception as exc:
        raise HTTPException(422, f"Failed to load plugin: {exc}")

    # Copy into AGENT_HOME/plugins/<name>/ so discover_plugins picks it up automatically.
    plugin_name = plugin.name or plugin_path.name
    dest = agent_home() / "plugins" / plugin_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(plugin_path, dest)

    # Reload from the installed location to reflect the real path in the response.
    try:
        plugin = load_from_directory(dest)
    except Exception as exc:
        raise HTTPException(422, f"Failed to load installed plugin: {exc}")

    state = _load_state()
    disabled = set(state.get("disabled", []))
    return _plugin_to_info(plugin, disabled)


@router.patch("/plugins/{plugin_id}", response_model=PluginInfo)
async def patch_plugin(plugin_id: str, patch: PluginPatch) -> Any:
    """Toggle plugin enabled state."""
    state = _load_state()
    disabled: set[str] = set(state.get("disabled", []))

    plugins = _load_all_plugins(state)
    target = next((p for p in plugins if _plugin_id(p.name or "") == plugin_id), None)
    if target is None:
        raise HTTPException(404, f"Plugin not found: {plugin_id}")

    if patch.enabled is not None:
        if patch.enabled:
            disabled.discard(target.name)
        else:
            disabled.add(target.name)
        state["disabled"] = list(disabled)
        _save_state(state)

    return _plugin_to_info(target, disabled)


@router.delete("/plugins/{plugin_id}", status_code=204)
async def remove_plugin(plugin_id: str) -> None:
    """Remove a plugin: deletes the directory if it lives under AGENT_HOME/plugins/."""
    from ...home import agent_home

    state = _load_state()

    plugins = _load_all_plugins(state)
    target = next((p for p in plugins if _plugin_id(p.name or "") == plugin_id), None)
    if target is None:
        raise HTTPException(404, f"Plugin not found: {plugin_id}")

    # Delete the installed copy under AGENT_HOME/plugins/ when present.
    if hasattr(target, "_plugin_root") and target._plugin_root:
        root = Path(target._plugin_root)
        agent_plugins_dir = agent_home() / "plugins"
        if root.is_relative_to(agent_plugins_dir) and root.is_dir():
            shutil.rmtree(root)
        # Also clean up any legacy plugin_paths entry for this path.
        root_str = str(root)
        state["plugin_paths"] = [p for p in state.get("plugin_paths", []) if p != root_str]

    # Remove from disabled list (cleanup).
    disabled: set[str] = set(state.get("disabled", []))
    disabled.discard(target.name)
    state["disabled"] = list(disabled)
    _save_state(state)
