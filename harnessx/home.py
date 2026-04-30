# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import re
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_HOME = Path.home() / ".harnessx"
_DEFAULT_AGENT = "hxagent"
_DEFAULT_PROJECT = "hxproject"  # fallback only; runtime default is CWD-derived


# ── Core resolvers ────────────────────────────────────────────────────────────


def agent_home() -> Path:
    """Return the AGENT_HOME root, creating it if necessary.

    Reads ``HARNESSX_HOME`` env var; falls back to ``~/.harnessx``.
    """
    root = Path(os.environ.get("HARNESSX_HOME", str(_DEFAULT_HOME))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_agent_id() -> str:
    """Return the default agent_id from env or 'hxagent'."""
    return os.environ.get("HARNESSX_AGENT", _DEFAULT_AGENT).strip() or _DEFAULT_AGENT


def _cwd_project() -> str:
    """Derive a safe project name from the current working directory.

    Example: ``/root/projects/harnessx`` → ``root-projects-harnessx``
    """
    cwd = str(Path.cwd())
    # Strip leading path separators
    raw = cwd.lstrip("/\\")
    # Replace path separators with hyphens
    raw = raw.replace("/", "-").replace("\\", "-")
    # Replace any char that is not a letter, digit, hyphen, underscore, or dot
    sanitized = re.sub(r"[^a-zA-Z0-9\-_.]", "-", raw)
    # Collapse consecutive hyphens; strip leading/trailing hyphens
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    return sanitized or _DEFAULT_PROJECT


def default_project() -> str:
    """Return the default project from env, or derive from CWD.

    When ``HARNESSX_PROJECT`` is set it takes priority.  Otherwise the project
    name is derived from the current working directory, e.g.
    ``/root/projects/harnessx`` → ``root-projects-harnessx``.
    """
    env = os.environ.get("HARNESSX_PROJECT", "").strip()
    if env:
        return env
    return _cwd_project()


# ── Directory helpers ─────────────────────────────────────────────────────────


def plugins_dir() -> Path:
    """Shared plugin scan directory: ``AGENT_HOME/plugins/``."""
    d = agent_home() / "plugins"
    d.mkdir(parents=True, exist_ok=True)
    return d


def skills_dir() -> Path:
    """Shared skills directory: ``AGENT_HOME/skills/``."""
    d = agent_home() / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def agent_configs_dir() -> Path:
    """Per-agent harness config directory: ``AGENT_HOME/configs/``."""
    d = agent_home() / "configs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def agent_config_path(agent_id: str) -> Path:
    """Path to a named agent's harness YAML: ``AGENT_HOME/configs/{agent_id}.yaml``."""
    return agent_configs_dir() / f"{agent_id}.yaml"


def agent_workspace_root(
    agent_id: str | None = None,
    project: str | None = None,
    workspace_base: str = "workspaces",
) -> Path:
    """Return (and create) the workspace root for an agent/project pair.

    Layout: ``AGENT_HOME/{workspace_base}/{agent_id}/{project}/``

    ``workspace_base`` defaults to ``"workspaces"`` (CLI / Lab UI convention).
    Pass ``"im-workspaces"`` to isolate IM gateway sessions from CLI sessions.

    Args:
        agent_id: Agent identifier.  Defaults to ``HARNESSX_AGENT`` env /
                  ``"default"``.
        project:  Project name.  Defaults to ``HARNESSX_PROJECT`` env /
                  ``"default"``.
        workspace_base: Top-level subdirectory under AGENT_HOME.
    """
    aid = (agent_id or default_agent_id()).strip() or _DEFAULT_AGENT
    proj = (project or default_project()).strip() or _DEFAULT_PROJECT
    _validate_name(aid, "agent_id")
    _validate_name(proj, "project")
    d = agent_home() / workspace_base / aid / proj
    d.mkdir(parents=True, exist_ok=True)
    return d


def agent_harness_config_path(
    agent_id: str | None = None,
    workspace_base: str = "workspaces",
) -> Path:
    """Return per-agent shared harness config path.

    Layout: ``AGENT_HOME/{workspace_base}/{agent_id}/harness_config.yaml``

    This is intentionally agent-level (not project-level) so Lab UI and CLI
    share one default harness behavior for the current agent.
    """
    aid = (agent_id or default_agent_id()).strip() or _DEFAULT_AGENT
    _validate_name(aid, "agent_id")
    root = agent_home() / workspace_base / aid
    root.mkdir(parents=True, exist_ok=True)
    return root / "harness_config.yaml"


def agent_memory_dir(agent_id: str | None = None) -> Path:
    """Cross-project memory directory for an agent.

    Layout: ``AGENT_HOME/workspaces/{agent_id}/memory/``

    Memory is agent-level (not project-level) so the agent retains long-term
    context regardless of which project it is currently working on.
    """
    aid = (agent_id or default_agent_id()).strip() or _DEFAULT_AGENT
    d = agent_home() / "workspaces" / aid / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_agents() -> list[str]:
    """Return agent ids that have a workspace directory under AGENT_HOME."""
    ws_root = agent_home() / "workspaces"
    if not ws_root.exists():
        return []
    return sorted(p.name for p in ws_root.iterdir() if p.is_dir() and not p.name.startswith("."))


def list_projects(agent_id: str | None = None) -> list[str]:
    """Return project names under a given agent's workspace."""
    aid = (agent_id or default_agent_id()).strip() or _DEFAULT_AGENT
    ws = agent_home() / "workspaces" / aid
    if not ws.exists():
        return []
    return sorted(p.name for p in ws.iterdir() if p.is_dir() and p.name != "memory" and not p.name.startswith("."))


# ── Validation ────────────────────────────────────────────────────────────────

_SAFE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _validate_name(name: str, field: str) -> None:
    """Reject names that would cause path traversal or filesystem issues."""
    if not name:
        raise ValueError(f"{field} must not be empty")
    if name in (".", ".."):
        raise ValueError(f"{field} must not be '.' or '..'")
    invalid = set(name) - _SAFE_CHARS
    if invalid:
        raise ValueError(
            f"{field} {name!r} contains invalid characters: {sorted(invalid)}. "
            "Only letters, digits, hyphens, underscores, and dots are allowed."
        )
