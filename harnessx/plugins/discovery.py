# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import HarnessPlugin

# Claude Code plugin index (contains installPath for each installed plugin)
CLAUDE_INSTALLED_PLUGINS_JSON = Path.home() / ".claude" / "plugins" / "installed_plugins.json"


def discover_plugins(
    workspace_root: Path | str | None = None,
    extra_paths: list[Path | str] | None = None,
    include_claude_plugins: bool = True,
    disabled: set[str] | None = None,
) -> list["HarnessPlugin"]:
    """Scan standard directories and return enabled plugins, deduplicated by name.

    Search order (first-found wins for same name):

    1. ``AGENT_HOME/plugins/`` — shared store, highest priority.
    2. ``extra_paths`` — user-configured scan directories.
    3. ``{workspace_root}/.harnessx/plugins/`` — project-level overrides.
    4. Claude Code installed plugins.

    Args:
        workspace_root:         Also scans ``{workspace_root}/.harnessx/plugins/``
                                (lowest priority).
        extra_paths:            Additional directories to scan (after AGENT_HOME).
        include_claude_plugins: Include plugins from Claude Code install index.
        disabled:               Set of plugin names to skip.  When ``None`` all
                                discovered plugins are returned.
    """
    from harnessx.home import agent_home as _agent_home

    plugins: list[HarnessPlugin] = []
    seen: dict[str, Path] = {}  # name → first-found path (for shadowing warnings)
    _disabled = disabled or set()

    def _try_load(candidate: Path) -> None:
        from .loader import load_from_directory

        try:
            plugin = load_from_directory(candidate)
        except Exception as exc:
            warnings.warn(
                f"Failed to load plugin from {candidate}: {exc}",
                stacklevel=3,
            )
            return

        name = plugin.name
        if not name:
            return

        if name in seen:
            warnings.warn(
                f"Plugin '{name}' at {candidate} is shadowed by the earlier installation at {seen[name]}; skipping.",
                stacklevel=3,
            )
            return

        seen[name] = candidate

        if name in _disabled:
            return  # registered as seen but not returned

        plugins.append(plugin)

    def _scan_dir(search_dir: Path) -> None:
        if not search_dir.is_dir():
            return
        for candidate in sorted(search_dir.iterdir()):
            if not candidate.is_dir():
                continue
            if (candidate / "plugin.json").exists() or (candidate / ".claude-plugin" / "plugin.json").exists():
                _try_load(candidate)

    # Priority 1 — AGENT_HOME shared plugin store (highest priority)
    _scan_dir(_agent_home() / "plugins")

    # Priority 2 — user-configured scan directories
    for p in extra_paths or []:
        _scan_dir(Path(p))

    # Priority 3 — workspace-local plugins (lowest priority)
    if workspace_root is not None:
        _scan_dir(Path(workspace_root) / ".harnessx" / "plugins")

    # Priority 4 — Claude Code installed plugins
    if include_claude_plugins:
        for candidate in _claude_installed_plugin_dirs():
            if candidate.is_dir():
                _try_load(candidate)

    return plugins


def discover_claude_plugins() -> list["HarnessPlugin"]:
    """Return plugins installed via ``claude plugin install``."""
    return discover_plugins(include_claude_plugins=True, workspace_root=None, extra_paths=None)


def _claude_installed_plugin_dirs() -> list[Path]:
    """Read install paths from the Claude Code installed_plugins.json index."""
    index_path = CLAUDE_INSTALLED_PLUGINS_JSON
    if not index_path.exists():
        return []

    try:
        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    dirs: list[Path] = []
    plugins_data = data.get("plugins", {})
    for plugin_key, installs in plugins_data.items():
        if not isinstance(installs, list):
            continue
        for entry in installs:
            install_path = entry.get("installPath")
            if install_path:
                dirs.append(Path(install_path))

    return dirs
