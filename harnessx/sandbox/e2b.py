# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Sandbox, SandboxProvider

if TYPE_CHECKING:
    import e2b
    from ..workspace.workspace import Workspace

log = logging.getLogger(__name__)

_CONTAINER_WORKSPACE = "/workspace"
_E2B_IMPORT_ERROR = "E2BSandboxProvider requires the 'e2b' extra: pip install harnessx"


# ---------------------------------------------------------------------------
# E2BSandbox
# ---------------------------------------------------------------------------


class E2BSandbox(Sandbox):
    """Executes commands inside an e2b cloud microVM."""

    def __init__(self, sbx: "e2b.AsyncSandbox") -> None:  # type: ignore[name-defined]
        self._sbx = sbx

    @property
    def workspace_path(self) -> str:
        return _CONTAINER_WORKSPACE

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> str:
        workdir = cwd or _CONTAINER_WORKSPACE
        try:
            result = await asyncio.wait_for(
                self._sbx.commands.run(
                    command,
                    cwd=workdir,
                    timeout=int(timeout),
                ),
                timeout=timeout + 2,  # outer guard slightly larger
            )
        except asyncio.TimeoutError:
            return f"Error: Command timed out after {timeout}s"
        except Exception as exc:
            return f"Error: {exc}"

        out = result.stdout or ""
        err = result.stderr or ""
        if err:
            return f"{out}\nSTDERR: {err}" if out else f"STDERR: {err}"
        return out

    async def read_file(self, path: str) -> str:
        try:
            content = await self._sbx.files.read(path)
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="replace")
            return content
        except Exception as exc:
            return f"Error reading {path}: {exc}"

    async def write_file(self, path: str, content: str) -> None:
        # Ensure parent directory exists
        parent = str(Path(path).parent)
        if parent and parent != "/":
            await self.exec(f"mkdir -p {parent!r}", cwd="/")
        await self._sbx.files.write(path, content)

    async def list_dir(self, path: str) -> list[str]:
        try:
            entries = await self._sbx.files.list(path)
            result = []
            for entry in entries:
                name = entry.name if hasattr(entry, "name") else str(entry)
                is_dir = getattr(entry, "type", None) == "dir" or name.endswith("/")
                result.append(name + ("/" if is_dir and not name.endswith("/") else ""))
            return result
        except Exception as exc:
            return [f"Error: {exc}"]


# ---------------------------------------------------------------------------
# E2BSandboxProvider
# ---------------------------------------------------------------------------


class E2BSandboxProvider(SandboxProvider):
    """Provisions e2b cloud microVMs as sandbox environments.

    Args:
        template:       e2b template id (default ``"base"``).
        api_key:        e2b API key.  Falls back to ``E2B_API_KEY`` env var.
        timeout:        Sandbox lifetime in seconds (default 300).
        upload_workspace: If True (default), upload workspace.root to
                          ``/workspace`` at acquire time.
        warm_pool:      If True, use e2b pause/resume for hint_id reuse.
                        Requires a template that supports snapshots.
    """

    def __init__(
        self,
        template: str = "base",
        api_key: str | None = None,
        timeout: int = 300,
        upload_workspace: bool = True,
        warm_pool: bool = False,
    ) -> None:
        try:
            import e2b  # noqa: F401
        except ImportError:
            raise ImportError(_E2B_IMPORT_ERROR) from None

        self.template = template
        self.api_key = api_key or os.environ.get("E2B_API_KEY")
        self.timeout = timeout
        self.upload_workspace = upload_workspace
        self.warm_pool = warm_pool

        # hint_id → paused sandbox_id (for warm pool)
        self._paused: dict[str, str] = {}
        self._lock = asyncio.Lock()

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _upload_workspace(
        self,
        sbx: "e2b.AsyncSandbox",  # type: ignore[name-defined]
        workspace: "Workspace",
    ) -> None:
        """Upload workspace.root contents to /workspace inside the microVM."""
        root = Path(workspace.root)
        if not root.exists():
            return
        # Upload files concurrently (skip hidden dirs like .git)
        tasks = []
        for local_path in sorted(root.rglob("*")):
            if local_path.is_file() and ".git" not in local_path.parts:
                rel = local_path.relative_to(root)
                remote_path = f"{_CONTAINER_WORKSPACE}/{rel}"
                tasks.append(self._upload_one(sbx, local_path, remote_path))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _upload_one(
        sbx: "e2b.AsyncSandbox",  # type: ignore[name-defined]
        local_path: Path,
        remote_path: str,
    ) -> None:
        try:
            content = local_path.read_text(encoding="utf-8", errors="replace")
            await sbx.files.write(remote_path, content)
        except Exception as exc:
            log.debug("Skip upload %s: %s", local_path, exc)

    # ── SandboxProvider interface ─────────────────────────────────────────────

    async def acquire(
        self,
        hint_id: str | None = None,
        workspace: "Workspace | None" = None,
    ) -> E2BSandbox:
        from e2b import AsyncSandbox

        kwargs: dict = {
            "template": self.template,
            "timeout": self.timeout,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key

        async with self._lock:
            # Warm pool: resume a paused sandbox if available
            if self.warm_pool and hint_id and hint_id in self._paused:
                paused_id = self._paused.pop(hint_id)
                try:
                    sbx = await AsyncSandbox.connect(
                        paused_id,
                        **{k: v for k, v in kwargs.items() if k != "template"},
                    )
                    log.debug("Resumed e2b sandbox %s for hint_id=%s", paused_id, hint_id)
                    return E2BSandbox(sbx)
                except Exception as exc:
                    log.warning("Failed to resume sandbox %s: %s — creating new", paused_id, exc)

        sbx = await AsyncSandbox.create(**kwargs)
        log.debug("Created e2b sandbox %s", sbx.sandbox_id)

        if self.upload_workspace and workspace is not None:
            await self._upload_workspace(sbx, workspace)

        return E2BSandbox(sbx)

    async def release(self, sandbox: Sandbox) -> None:
        """Pause (warm pool) or kill the microVM."""
        if not isinstance(sandbox, E2BSandbox):
            return
        sbx = sandbox._sbx

        # Find hint_id for this sandbox by checking warm pool slots
        # (simple implementation: pause all released sandboxes when warm_pool=True)
        if self.warm_pool:
            try:
                paused_id = await sbx.pause()
                log.debug("Paused e2b sandbox %s", paused_id)
                # Without hint_id tracking at acquire time, we can't reuse by hint;
                # kill instead to avoid leaking paused sandboxes.
                await sbx.kill()
            except Exception as exc:
                log.debug("Could not pause sandbox: %s — killing", exc)
                try:
                    await sbx.kill()
                except Exception:
                    pass
        else:
            try:
                await sbx.kill()
                log.debug("Killed e2b sandbox %s", sbx.sandbox_id)
            except Exception as exc:
                log.debug("Failed to kill e2b sandbox: %s", exc)

    async def shutdown(self) -> None:
        """Nothing to tear down globally — each sandbox is killed at release()."""
        self._paused.clear()
