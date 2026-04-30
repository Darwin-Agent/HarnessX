# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..sandbox.base import Mount


# ── Name validation ───────────────────────────────────────────────────────────

_SAFE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _validate_component(name: str, field_name: str) -> None:
    """Reject names that would cause path traversal or filesystem issues."""
    if not name or name in (".", ".."):
        raise ValueError(f"{field_name} must not be empty or '.' / '..'")
    invalid = set(name) - _SAFE_CHARS
    if invalid:
        raise ValueError(
            f"{field_name} {name!r} contains invalid characters: {sorted(invalid)}. "
            "Only letters, digits, hyphens, underscores, and dots are allowed."
        )


class WorkspaceEscapeError(Exception):
    """Raised when a path escapes the workspace root in isolated mode."""

    pass


class WorkspaceWriteError(Exception):
    """Raised when a write is attempted in readonly mode."""

    pass


@dataclass
class Workspace:
    """
    Defines the file system boundary for an Agent's tool execution.

    All builtin tools (Bash, Read, Write, Edit, Glob, Grep) route path
    operations through workspace.resolve(), preventing path traversal outside root.

    Minimal construction with AGENT_HOME auto-derivation::

        ws = Workspace(agent_id="alice", project="coding", home=agent_home())
        # ws.root == ~/.harnessx/workspaces/alice/coding/

    Explicit root (legacy / testing)::

        ws = Workspace(agent_id="alice", root=Path("/custom/path"))
    """

    agent_id: str
    root: Path | None = None
    """Workspace root directory.  If *None* and *home* is set, auto-derived as
    ``home / "workspaces" / agent_id / project``."""
    project: str = "default"
    """Project name used for auto-deriving the workspace root from AGENT_HOME."""
    home: Path | None = None
    """AGENT_HOME root (e.g. ``~/.harnessx``).

    When set alongside ``root=None``, the root is auto-derived from the
    standard AGENT_HOME layout.  Also enables ``mode="home"`` path-jail mode
    and hints the sandbox to mount the entire AGENT_HOME so the agent has
    access to memory, plugins, and skills across projects.
    """
    parent_id: str | None = None
    mode: str = "isolated"  # "isolated" | "shared" | "home" | "readonly" | None
    extra_mounts: list["Mount"] = field(default_factory=list)
    """Additional volume mounts to inject into the sandbox alongside root.

    The SandboxProvider mounts ``workspace.root → /workspace`` automatically.
    When ``home`` is set, SandboxProvider should also mount
    ``workspace.home → /agent_home``.
    Use extra_mounts for any further directories.
    """

    def __post_init__(self) -> None:
        if self.home is not None:
            self.home = Path(self.home).expanduser().resolve()
            self.home.mkdir(parents=True, exist_ok=True)

        if self.root is None:
            if self.home is None:
                raise ValueError(
                    "Workspace requires either 'root' or 'home' to be set. "
                    "Provide an explicit root path or set home to enable auto-derivation."
                )
            _validate_component(self.agent_id, "agent_id")
            _validate_component(self.project, "project")
            self.root = self.home / "workspaces" / self.agent_id / self.project

        self.root = Path(self.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str) -> Path:
        """
        Resolve a path relative to workspace root.
        Prevents path traversal outside workspace in isolated mode.
        Raises WorkspaceEscapeError if path escapes root in isolated mode.
        Raises WorkspaceWriteError for write operations in readonly mode.
        """
        target = (self.root / path).resolve()
        if self.mode == "isolated":
            try:
                target.relative_to(self.root)
            except ValueError:
                raise WorkspaceEscapeError(f"Path '{path}' resolves to '{target}', outside workspace '{self.root}'")
        elif self.mode == "shared":
            # Allow access to parent shared dir
            parent_root = self.root.parent
            try:
                target.relative_to(parent_root)
            except ValueError:
                raise WorkspaceEscapeError(
                    f"Path '{path}' resolves to '{target}', outside shared workspace '{parent_root}'"
                )
        elif self.mode == "home":
            # Allow access to anywhere inside AGENT_HOME
            jail = self.home if self.home is not None else self.root
            try:
                target.relative_to(jail)
            except ValueError:
                raise WorkspaceEscapeError(f"Path '{path}' resolves to '{target}', outside agent home '{jail}'")
        return target

    def check_write(self) -> None:
        """Raise WorkspaceWriteError if in readonly mode."""
        if self.mode == "readonly":
            raise WorkspaceWriteError(f"Workspace '{self.root}' is readonly; write operations are not allowed.")

    def child(self, agent_id: str) -> "Workspace":
        """Create an isolated subdirectory workspace for a sub-agent."""
        child_root = self.root / "agents" / agent_id
        child_root.mkdir(parents=True, exist_ok=True)
        return Workspace(
            agent_id=agent_id,
            root=child_root,
            home=self.home,
            parent_id=self.agent_id,
            mode=self.mode,
            extra_mounts=list(self.extra_mounts),
        )

    def shared_dir(self) -> Path:
        """Parent-child shared directory for passing results between agents."""
        shared = self.root / "shared"
        shared.mkdir(exist_ok=True)
        return shared
