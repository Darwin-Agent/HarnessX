# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import abc
import base64
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..workspace.workspace import Workspace


# ---------------------------------------------------------------------------
# Mount
# ---------------------------------------------------------------------------


@dataclass
class Mount:
    """Volume mount: host_path is mounted into the sandbox at container_path."""

    host_path: Path
    container_path: str
    read_only: bool = False

    def __post_init__(self) -> None:
        self.host_path = Path(self.host_path).expanduser().resolve()


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class Sandbox(abc.ABC):
    """Abstract execution environment.

    All filesystem tool operations route through a Sandbox instance so the
    same tool code works locally, inside a Docker container, or on a remote
    sandbox API — no tool rewriting required.
    """

    @property
    @abc.abstractmethod
    def workspace_path(self) -> str:
        """Absolute path to the workspace *inside* this sandbox.

        For LocalSandbox this is the host-side workspace root.
        For container-based sandboxes this is the container-side mount point
        (e.g. ``/workspace`` or ``/app``).
        """

    @abc.abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> str:
        """Run a shell command.  Returns combined stdout + stderr string."""

    async def read_file(self, path: str) -> str:
        """Return file contents.  *path* must be an absolute path."""
        return await self.exec(f"cat -- {path!r}", cwd="/")

    async def write_file(self, path: str, content: str) -> None:
        """Write *content* to *path*.  Creates parent directories as needed.
        *path* must be an absolute path."""
        parent = str(Path(path).parent)
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        await self.exec(f"mkdir -p {parent!r} && echo {encoded!r} | base64 -d > {path!r}", cwd="/")

    async def list_dir(self, path: str) -> list[str]:
        """Return directory entries as strings (dirs suffixed with ``/``).
        *path* must be an absolute path."""
        result = await self.exec(f"ls -p -- {path!r}", cwd="/")
        return [line.strip() for line in result.splitlines() if line.strip()]

    async def kill_running(self) -> None:
        """Best-effort: kill all processes currently running inside this sandbox.

        Called by the Bash tool after a command timeout to prevent orphaned
        processes (e.g. a training loop) from consuming resources and eventually
        crashing the container via OOM.

        Default implementation is a no-op.  Container-backed sandboxes should
        override this to send SIGTERM/SIGKILL to all container processes.
        """

    def resolve(self, path: str) -> str:
        """Resolve *path* relative to :attr:`workspace_path`.

        If *path* is already absolute it is returned as-is.
        Subclasses may add jail-checking here (see ``LocalSandbox``).
        """
        if Path(path).is_absolute():
            return path
        return str(Path(self.workspace_path) / path)

    # ------------------------------------------------------------------
    # Default exec-based implementations for Glob / Grep.
    # LocalSandbox overrides these with faster Python implementations.
    # ------------------------------------------------------------------

    async def glob_files(self, pattern: str, base: str | None = None) -> list[str]:
        """Return file paths matching *pattern* under *base*."""
        cwd = base or self.workspace_path
        # Use find; handle both directory-prefix patterns and filename patterns
        result = await self.exec(
            f"find . -path './{pattern}' 2>/dev/null | sort",
            cwd=cwd,
        )
        return [line.lstrip("./").strip() for line in result.splitlines() if line.strip()]

    async def grep_files(
        self,
        pattern: str,
        path: str | None = None,
        glob: str = "*",
        output_mode: str = "content",
    ) -> str:
        """Search files under *path* for *pattern*."""
        search_path = path or self.workspace_path
        if output_mode == "files_with_matches":
            flags = "-rl"
        elif output_mode == "count":
            flags = "-rc"
        else:
            flags = "-rn"
        result = await self.exec(
            f"grep {flags} --include='{glob}' -E '{pattern}' . 2>/dev/null | head -250",
            cwd=search_path,
        )
        return result.strip() or "No matches found."


# ---------------------------------------------------------------------------
# SandboxProvider
# ---------------------------------------------------------------------------


class SandboxProvider(abc.ABC):
    """Manages Sandbox lifecycle.

    Implementations: LocalSandboxProvider (default), HarborSandboxProvider,
    AgentInfraSandboxProvider, E2BSandboxProvider, …

    Swap providers in HarnessConfig.sandbox_provider to change where all
    tool execution happens without modifying any tool or processor code.
    """

    @abc.abstractmethod
    async def acquire(
        self,
        hint_id: str | None = None,
        workspace: "Workspace | None" = None,
    ) -> Sandbox:
        """Return a Sandbox for use during a run.

        *hint_id*   — stable identifier enabling warm-pool reuse across runs
                      (e.g. ``agent_id`` for a personal assistant that keeps
                      its container alive between turns).  Pass ``None`` for
                      one-shot isolated runs (batch eval, TB2, …).

        *workspace* — if provided, the provider should mount
                      ``workspace.root → /workspace`` (or equivalent) so that
                      agent work files survive container restarts.
        """

    @abc.abstractmethod
    async def release(self, sandbox: Sandbox) -> None:
        """Return *sandbox* to the pool or mark it idle.  No-op for local."""

    async def shutdown(self) -> None:
        """Destroy all managed sandboxes.  Called at process exit."""


# ---------------------------------------------------------------------------
# ContextVar — runtime injection
# ---------------------------------------------------------------------------

_sandbox_ctx: ContextVar[Sandbox | None] = ContextVar("_sandbox_ctx", default=None)


def get_current_sandbox() -> Sandbox | None:
    """Return the active :class:`Sandbox` for the current async context, or ``None``."""
    return _sandbox_ctx.get()
