# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any

from .base import HarnessPlugin


def load_plugin(source: "str | Path | type | HarnessPlugin") -> HarnessPlugin:
    """Load a plugin from various source types.

    Args:
        source: One of:
            - ``HarnessPlugin`` instance → returned as-is
            - ``type`` subclassing ``HarnessPlugin`` → instantiated
            - ``Path`` or path string pointing to a plugin directory
              (HarnessX ``plugin.json`` at root, or Claude Code
              ``.claude-plugin/plugin.json`` subdirectory)
            - Dotted Python import path string (``"pkg.module.PluginClass"``)

    Returns:
        A ``HarnessPlugin`` instance ready to be registered.
    """
    if isinstance(source, HarnessPlugin):
        return source

    if isinstance(source, type) and issubclass(source, HarnessPlugin):
        return source()

    if isinstance(source, Path) or (
        isinstance(source, str) and ("/" in source or "\\" in source or source.startswith("."))
    ):
        return load_from_directory(Path(source))

    if isinstance(source, str):
        return load_from_dotted_path(source)

    raise TypeError(
        f"Cannot load plugin from {source!r}. "
        "Expected a HarnessPlugin instance/class, a filesystem path, "
        "or a dotted Python import path."
    )


def find_manifest_path(directory: Path) -> tuple[Path, bool]:
    """Locate the plugin.json in a plugin directory.

    Supports two layouts:
    - HarnessX:   ``{dir}/plugin.json``
    - Claude Code:   ``{dir}/.claude-plugin/plugin.json``

    Returns:
        (manifest_path, is_claude_code_format)

    Raises:
        FileNotFoundError if neither location has a manifest.
    """
    # HarnessX format (our converted plugins)
    oh_path = directory / "plugin.json"
    if oh_path.exists():
        return oh_path, False

    # Claude Code native format
    cc_path = directory / ".claude-plugin" / "plugin.json"
    if cc_path.exists():
        return cc_path, True

    raise FileNotFoundError(
        f"No plugin manifest found in {directory}. "
        "Expected either 'plugin.json' (HarnessX) or "
        "'.claude-plugin/plugin.json' (Claude Code)."
    )


def load_from_directory(directory: Path) -> HarnessPlugin:
    """Load a plugin from a directory.

    Supports both HarnessX (``plugin.json``) and Claude Code
    (``.claude-plugin/plugin.json``) directory layouts.  Additional
    capabilities are auto-discovered from the directory tree:

    - ``commands/*.md``      → ``plugin.commands`` (YAML frontmatter parsed)
    - ``skills/*/SKILL.md``  → ``plugin.skill_dirs``
    - ``hooks/hooks.json``   → ``plugin.lifecycle_hooks``
    - manifest ``mcpServers`` field → ``plugin.mcp_servers``
    """
    directory = directory.resolve()
    manifest_path, is_claude_code = find_manifest_path(directory)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if is_claude_code:
        # Convert the Claude Code manifest to an intermediate unified form
        manifest = _normalize_claude_code_manifest(manifest, directory)

    plugin = load_from_manifest(manifest, base_dir=directory)

    # Auto-discover additional capabilities from directory tree
    plugin.skill_dirs = _find_skill_dirs(directory)
    plugin.lifecycle_hooks = _load_hooks_json(directory)
    # Store the resolved plugin root so HarnessBuilder can pass it to ShellHookProcessor
    plugin._plugin_root = directory  # type: ignore[attr-defined]

    return plugin


def load_from_manifest(manifest: dict[str, Any], base_dir: Path | None = None) -> HarnessPlugin:
    """Instantiate a ``DynamicPlugin`` from a parsed manifest dict.

    Handles both pure Claude Code plugin.json (only standard fields) and
    HarnessX extended manifests (with processors/tools/slash_commands).
    Also reads ``mcpServers`` (Claude Code native field).
    """
    plugin = _DynamicPlugin()
    plugin.name = manifest.get("name", "")
    plugin.version = manifest.get("version", "0.1.0")
    plugin.description = manifest.get("description", "")

    plugin._setup_script = manifest.get("setup")
    plugin._stop_script = manifest.get("stop")

    plugin.commands = []
    for entry in manifest.get("commands", []):
        cmd = dict(entry)
        if base_dir and "prompt" in cmd and isinstance(cmd["prompt"], str) and cmd["prompt"].startswith("./"):
            prompt_path = base_dir / cmd["prompt"]
            if prompt_path.exists():
                cmd["prompt"] = prompt_path.read_text(encoding="utf-8")
        plugin.commands.append(cmd)

    plugin.mcp_servers = _normalise_mcp_servers(manifest.get("mcpServers", {}))

    plugin.processors = []
    for spec in manifest.get("processors", []):
        spec = dict(spec)
        if "_note" in spec:
            continue
        target = spec.pop("target", None) or spec.pop("_target_", None)
        if not target:
            raise ValueError(f"Plugin processor spec missing 'target': {spec!r}")
        try:
            plugin.processors.append(_instantiate(target, spec))
        except Exception as exc:
            plugin_name = manifest.get("name", "<unnamed>")
            raise ImportError(f"Plugin '{plugin_name}': failed to load processor '{target}': {exc}") from exc

    plugin.tools = []
    for spec in manifest.get("tools", []):
        spec = dict(spec)
        if "_note" in spec:
            continue
        target = spec.pop("target", None) or spec.pop("_target_", None)
        if not target:
            raise ValueError(f"Plugin tool spec missing 'target': {spec!r}")
        mod_path, attr = target.rsplit(".", 1)
        plugin.tools.append(getattr(importlib.import_module(mod_path), attr))

    plugin.slash_commands = {
        entry["command"]: entry.get("slot") for entry in manifest.get("slash_commands", []) if entry.get("command")
    }

    return plugin


def load_from_dotted_path(dotted: str) -> HarnessPlugin:
    """Load a plugin class or instance from a dotted Python import path."""
    try:
        mod_path, attr = dotted.rsplit(".", 1)
        mod = importlib.import_module(mod_path)
        obj = getattr(mod, attr)
    except (ValueError, ImportError, AttributeError):
        mod = importlib.import_module(dotted)
        obj = getattr(mod, "plugin", None)
        if obj is None:
            for name in dir(mod):
                candidate = getattr(mod, name)
                if (
                    isinstance(candidate, type)
                    and issubclass(candidate, HarnessPlugin)
                    and candidate is not HarnessPlugin
                ):
                    obj = candidate
                    break
        if obj is None:
            raise ImportError(
                f"No HarnessPlugin found in module {dotted!r}. Define a HarnessPlugin subclass or a 'plugin' attribute."
            )

    if isinstance(obj, HarnessPlugin):
        return obj
    if isinstance(obj, type) and issubclass(obj, HarnessPlugin):
        return obj()
    raise TypeError(f"{dotted!r} resolved to {obj!r}, which is not a HarnessPlugin class or instance.")


def _normalize_claude_code_manifest(manifest: dict, plugin_dir: Path) -> dict:
    """Convert a Claude Code .claude-plugin/plugin.json to unified form.

    Scans the ``commands/`` directory for ``.md`` files and builds the
    ``commands`` list by parsing each file's YAML frontmatter.
    """
    result = dict(manifest)
    commands_dir = plugin_dir / "commands"
    commands = []
    if commands_dir.is_dir():
        commands = [_md_file_to_command(f) for f in sorted(commands_dir.glob("*.md"))]
    result["commands"] = commands
    return result


def parse_command_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a Claude Code command .md file.

    Returns:
        (frontmatter_dict, body_text)  — body is the content after the
        closing ``---`` delimiter.  If no frontmatter is present, returns
        ``({}, text)``.
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    fm_text = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")

    try:
        # Minimal YAML parser: handle simple key: value and key: [list] forms
        fm = _parse_simple_yaml(fm_text)
    except Exception:
        fm = {}

    return fm, body


def _md_file_to_command(md_file: Path) -> dict:
    """Parse a ``commands/*.md`` file into a command dict."""
    fm, body = parse_command_frontmatter(md_file.read_text(encoding="utf-8"))
    cmd: dict[str, Any] = {
        "name": md_file.stem,
        "description": fm.get("description", md_file.stem),
        "prompt": body.strip(),
    }
    if "allowed-tools" in fm:
        cmd["allowed_tools"] = fm["allowed-tools"]
    if "argument-hint" in fm:
        cmd["argument_hint"] = fm["argument-hint"]
    if fm.get("hide-from-slash-command-tool"):
        cmd["hidden"] = True
    return cmd


def _parse_simple_yaml(text: str) -> dict:
    """Parse simple YAML subset used in Claude Code command frontmatter.

    Handles:
      - ``key: scalar``
      - ``key: "quoted string"``
      - ``key: [item1, item2, ...]``  (inline array)
    """
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip()
        value: Any = raw_value.strip()
        # Strip surrounding quotes
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        # Inline array: [a, b, c] or ["a", "b"]
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1]
            value = [item.strip().strip('"').strip("'") for item in re.split(r",\s*", inner) if item.strip()]
        result[key] = value
    return result


def _find_skill_dirs(plugin_dir: Path) -> list[Path]:
    """Scan ``{plugin_dir}/skills/`` for subdirectories containing SKILL.md."""
    skills_root = plugin_dir / "skills"
    if not skills_root.is_dir():
        return []
    found = []
    for candidate in sorted(skills_root.iterdir()):
        if candidate.is_dir() and (candidate / "SKILL.md").exists():
            found.append(candidate)
    return found


def _load_hooks_json(plugin_dir: Path) -> dict:
    """Load ``{plugin_dir}/hooks/hooks.json`` and return the hooks dict.

    Returns Claude Code event-keyed dict:
    ``{"Stop": [...], "PreToolUse": [...], "PostToolUse": [...]}``
    """
    hooks_path = plugin_dir / "hooks" / "hooks.json"
    if not hooks_path.exists():
        return {}
    try:
        with open(hooks_path, encoding="utf-8") as f:
            data = json.load(f)
        # Claude Code format: {"hooks": {"Stop": [...], ...}}
        # or flat: {"Stop": [...], ...}
        return data.get("hooks", data)
    except Exception:
        return {}


def _normalise_mcp_servers(raw: Any) -> list[dict]:
    """Normalise the ``mcpServers`` field to a flat list of server spec dicts.

    Accepts Claude Code dict-of-dicts format or an already-flat list.
    Each returned dict has at minimum: ``name``, ``transport``.
    """
    if not raw:
        return []

    # Already a list (HarnessX extended format)
    if isinstance(raw, list):
        result = []
        for item in raw:
            if isinstance(item, dict):
                spec = dict(item)
                if "transport" not in spec:
                    spec["transport"] = "http" if spec.get("url") else "stdio"
                result.append(spec)
        return result

    # Dict-of-dicts (Claude Code native): {"sqlite": {"command": "..."}, ...}
    if isinstance(raw, dict):
        result = []
        for name, cfg in raw.items():
            if not isinstance(cfg, dict):
                continue
            spec = dict(cfg)
            spec["name"] = name
            if "transport" not in spec:
                spec["transport"] = "http" if spec.get("url") else "stdio"
            result.append(spec)
        return result

    return []


class _DynamicPlugin(HarnessPlugin):
    """Plugin instance built dynamically from a manifest dict."""

    def __init__(self) -> None:
        self._setup_script: str | None = None
        self._stop_script: str | None = None
        self.processors = []
        self.tools = []
        self.slash_commands = {}
        self.commands = []
        self.skill_dirs = []
        self.mcp_servers = []
        self.lifecycle_hooks = {}

    def setup(self, config: Any) -> None:
        if self._setup_script:
            import os
            import subprocess
            import shlex

            env = {**os.environ}
            if getattr(self, "_plugin_root", None):
                env["CLAUDE_PLUGIN_ROOT"] = str(self._plugin_root)
            subprocess.run(shlex.split(self._setup_script), check=False, env=env)

    def stop(self) -> None:
        if self._stop_script:
            import os
            import subprocess
            import shlex

            env = {**os.environ}
            if getattr(self, "_plugin_root", None):
                env["CLAUDE_PLUGIN_ROOT"] = str(self._plugin_root)
            subprocess.run(shlex.split(self._stop_script), check=False, env=env)


def _instantiate(target: str, kwargs: dict) -> Any:
    """Import and instantiate a class from a dotted target path with kwargs."""
    mod_path, class_name = target.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)
