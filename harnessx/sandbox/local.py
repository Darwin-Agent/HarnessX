# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Sandbox, SandboxProvider

if TYPE_CHECKING:
    from ..workspace.workspace import Workspace


class LocalSandbox(Sandbox):
    """Executes commands and file I/O directly on the local filesystem."""

    def __init__(
        self,
        root: str | Path,
        mode: str = "isolated",
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._mode = mode  # "isolated" | "shared" | "readonly"

    # ── Sandbox interface ────────────────────────────────────────────────────

    @property
    def workspace_path(self) -> str:
        return str(self._root)

    def resolve(self, path: str) -> str:
        """Resolve *path* relative to workspace root with jail checking."""
        from ..workspace.workspace import WorkspaceEscapeError

        p = Path(path)
        target = p.resolve() if p.is_absolute() else (self._root / path).resolve()

        if self._mode == "isolated":
            try:
                target.relative_to(self._root)
            except ValueError:
                raise WorkspaceEscapeError(f"Path '{path}' resolves to '{target}', outside workspace '{self._root}'")
        elif self._mode == "shared":
            parent_root = self._root.parent
            try:
                target.relative_to(parent_root)
            except ValueError:
                raise WorkspaceEscapeError(
                    f"Path '{path}' resolves to '{target}', outside shared workspace '{parent_root}'"
                )
        return str(target)

    def check_write(self) -> None:
        from ..workspace.workspace import WorkspaceWriteError

        if self._mode == "readonly":
            raise WorkspaceWriteError(
                f"Sandbox rooted at '{self._root}' is readonly; write operations are not allowed."
            )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd or str(self._root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return f"Error: Command timed out after {timeout}s"
            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            if err:
                return f"{out}\nSTDERR: {err}" if out else f"STDERR: {err}"
            return out
        except Exception as e:
            return f"Error: {e}"

    async def read_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    async def write_file(self, path: str, content: str) -> None:
        self.check_write()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)

    async def list_dir(self, path: str) -> list[str]:
        p = Path(path)
        if not p.exists():
            return []
        if p.is_file():
            return [p.name]
        return [
            e.name + ("/" if e.is_dir() else "") for e in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        ]

    # ── Fast Python overrides for Glob / Grep ────────────────────────────────

    async def glob_files(self, pattern: str, base: str | None = None) -> list[str]:
        base_path = Path(base) if base else self._root
        try:
            matches = list(base_path.glob(pattern))
            result = []
            for m in sorted(matches):
                try:
                    result.append(str(m.relative_to(self._root)))
                except ValueError:
                    pass  # Outside workspace root — skip
            return result
        except Exception:
            return []

    async def grep_files(
        self,
        pattern: str,
        path: str | None = None,
        glob: str = "*",
        output_mode: str = "content",
    ) -> str:
        base = Path(path) if path else self._root
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        results: list[str] = []
        try:
            for filepath in sorted(base.rglob(glob)):
                if not filepath.is_file():
                    continue
                try:
                    filepath.relative_to(self._root)
                except ValueError:
                    continue  # Outside workspace — skip
                try:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    for lineno, line in enumerate(lines, 1):
                        if regex.search(line):
                            rel = str(filepath.relative_to(self._root))
                            if output_mode == "files_with_matches":
                                results.append(rel)
                                break
                            elif output_mode == "count":
                                count = sum(1 for ln in lines if regex.search(ln))
                                results.append(f"{rel}: {count}")
                                break
                            else:
                                results.append(f"{rel}:{lineno}:{line.rstrip()}")
                except Exception:
                    pass
        except Exception as e:
            return f"Error: {e}"

        if not results:
            return "No matches found."
        return "\n".join(results[:250])


# ---------------------------------------------------------------------------
# LocalSandboxProvider
# ---------------------------------------------------------------------------


class LocalSandboxProvider(SandboxProvider):
    """Creates LocalSandbox instances.  acquire/release are lightweight no-ops.

    This is the default provider when no sandbox_provider is set in
    HarnessConfig.  Behavior is identical to the old workspace-bound tools.
    """

    async def acquire(
        self,
        hint_id: str | None = None,
        workspace: "Workspace | None" = None,
    ) -> LocalSandbox:
        if workspace is not None:
            return LocalSandbox(root=workspace.root, mode=workspace.mode)
        return LocalSandbox(root=os.getcwd(), mode="shared")

    async def release(self, sandbox: Sandbox) -> None:
        pass  # No-op: nothing to pool or destroy for local execution

    async def shutdown(self) -> None:
        pass  # No-op
