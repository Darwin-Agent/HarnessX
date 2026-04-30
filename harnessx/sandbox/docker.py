# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Sandbox, SandboxProvider

if TYPE_CHECKING:
    import docker
    from ..workspace.workspace import Workspace

log = logging.getLogger(__name__)

_CONTAINER_WORKSPACE = "/workspace"
_CDP_CONTAINER_PORT = 9222
_CHROMIUM_STARTUP_RETRIES = 20  # × 0.5 s = up to 10 s wait
_DOCKER_IMPORT_ERROR = "DockerSandboxProvider requires the 'docker' extra: pip install harnessx"


# ---------------------------------------------------------------------------
# DockerSandbox
# ---------------------------------------------------------------------------


class DockerSandbox(Sandbox):
    """Executes commands inside a running Docker container.

    Attributes:
        cdp_url: HTTP URL for Chromium CDP remote debugging
                 (e.g. ``http://127.0.0.1:49321``).  ``None`` when
                 ``DockerSandboxProvider(enable_browser=False)`` (the default).
                 ``browser_tool`` reads this attribute to decide whether to
                 connect via CDP or launch a local Playwright instance.
    """

    def __init__(
        self,
        container: "docker.models.containers.Container",  # type: ignore[name-defined]
        cdp_url: str | None = None,
    ) -> None:
        self._container = container
        self.cdp_url = cdp_url

    # ── Sandbox interface ────────────────────────────────────────────────────

    @property
    def workspace_path(self) -> str:
        return _CONTAINER_WORKSPACE

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> str:
        """Run *command* inside the container via docker exec."""
        workdir = cwd or _CONTAINER_WORKSPACE
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._container.exec_run(
                        ["sh", "-c", command],
                        workdir=workdir,
                        demux=True,
                    ),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return f"Error: Command timed out after {timeout}s"
        except Exception as exc:
            return f"Error: {exc}"

        exit_code, (stdout_bytes, stderr_bytes) = result
        out = (stdout_bytes or b"").decode("utf-8", errors="replace")
        err = (stderr_bytes or b"").decode("utf-8", errors="replace")
        if err:
            return f"{out}\nSTDERR: {err}" if out else f"STDERR: {err}"
        return out

    async def kill_running(self) -> None:
        """Send SIGTERM then SIGKILL to all container processes."""
        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._container.exec_run(
                        ["sh", "-c", "kill -15 -1 2>/dev/null; sleep 1; kill -9 -1 2>/dev/null; true"],
                        detach=False,
                    ),
                ),
                timeout=10,
            )
        except Exception:
            pass

    async def read_file(self, path: str) -> str:
        return await self.exec(f"cat -- {path!r}", cwd="/")

    async def write_file(self, path: str, content: str) -> None:
        import base64

        parent = str(Path(path).parent)
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        await self.exec(
            f"mkdir -p {parent!r} && printf '%s' {encoded!r} | base64 -d > {path!r}",
            cwd="/",
        )

    async def list_dir(self, path: str) -> list[str]:
        result = await self.exec(f"ls -p -- {path!r}", cwd="/")
        return [line.strip() for line in result.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# DockerSandboxProvider
# ---------------------------------------------------------------------------


class DockerSandboxProvider(SandboxProvider):
    """Manages Docker containers as isolated sandbox environments.

    Args:
        image:          Docker image to use.  Build from ``container/Dockerfile``
                        or use ``harnessx/agent:latest`` from GHCR.
        network:        Docker network mode.  Default ``"none"`` (fully isolated).
                        Automatically promoted to ``"bridge"`` when
                        ``enable_browser=True`` so the host can reach the CDP port.
        mem_limit:      Container memory limit (e.g. ``"2g"``).
        cpu_count:      Number of CPUs available to the container.
        extra_env:      Extra environment variables injected into every container.
        pull_policy:    ``"never"`` | ``"missing"`` | ``"always"`` (default ``"missing"``).
        docker_url:     Docker daemon socket URL.  ``None`` uses the system default.
        enable_browser: If ``True``, start Chromium inside the container with
                        ``--remote-debugging-port=9222`` and expose it on a
                        randomly-assigned host port.  ``DockerSandbox.cdp_url``
                        will point to that port so ``browser_tool`` can connect
                        via CDP instead of launching a local Playwright instance.
    """

    def __init__(
        self,
        image: str = "harnessx/agent:latest",
        network: str = "none",
        mem_limit: str = "2g",
        cpu_count: int = 2,
        extra_env: dict[str, str] | None = None,
        pull_policy: str = "missing",
        docker_url: str | None = None,
        enable_browser: bool = False,
    ) -> None:
        try:
            import docker as _docker  # noqa: F401
        except ImportError:
            raise ImportError(_DOCKER_IMPORT_ERROR) from None

        self.image = image
        self.mem_limit = mem_limit
        self.cpu_count = cpu_count
        self.extra_env = extra_env or {}
        self.pull_policy = pull_policy
        self._docker_url = docker_url
        self.enable_browser = enable_browser

        # CDP requires the host to reach the container — promote to bridge.
        if enable_browser and network == "none":
            log.info(
                "DockerSandboxProvider: enable_browser=True requires network "
                "access; promoting network mode from 'none' to 'bridge'."
            )
            network = "bridge"
        self.network = network

        self._client: "docker.DockerClient | None" = None  # type: ignore[name-defined]
        # warm pool: hint_id → container
        self._pool: dict[str, "docker.models.containers.Container"] = {}  # type: ignore[name-defined]
        self._lock = asyncio.Lock()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_client(self) -> "docker.DockerClient":  # type: ignore[name-defined]
        if self._client is None:
            import docker

            self._client = docker.DockerClient(base_url=self._docker_url) if self._docker_url else docker.from_env()
        return self._client

    def _pull_if_needed(self) -> None:
        import docker

        client = self._get_client()
        if self.pull_policy == "always":
            client.images.pull(self.image)
        elif self.pull_policy == "missing":
            try:
                client.images.get(self.image)
            except docker.errors.ImageNotFound:
                log.info("Pulling Docker image %s …", self.image)
                client.images.pull(self.image)

    def _build_volumes(self, workspace: "Workspace | None") -> dict:
        """Build the volumes dict for docker-py (host_path → {bind, mode})."""
        vols: dict[str, dict] = {}
        if workspace is not None:
            vols[str(workspace.root)] = {
                "bind": _CONTAINER_WORKSPACE,
                "mode": "rw",
            }
            for mount in getattr(workspace, "extra_mounts", []) or []:
                vols[str(mount.host_path)] = {
                    "bind": mount.container_path,
                    "mode": "ro" if mount.read_only else "rw",
                }
        return vols

    def _start_container(
        self,
        hint_id: str | None,
        workspace: "Workspace | None",
    ) -> "docker.models.containers.Container":  # type: ignore[name-defined]
        self._pull_if_needed()
        client = self._get_client()
        volumes = self._build_volumes(workspace)
        name = f"oh-agent-{hint_id}" if hint_id else None

        # When browser is enabled start chromium alongside the sleep sentinel.
        # --remote-debugging-address=0.0.0.0 allows Docker port mapping to reach
        # it from the host; port 0 on the host lets Docker pick a free port.
        if self.enable_browser:
            command = (
                "sh -c 'chromium --headless --no-sandbox "
                f"--remote-debugging-port={_CDP_CONTAINER_PORT} "
                "--remote-debugging-address=0.0.0.0 "
                "--disable-gpu --disable-dev-shm-usage "
                f"& sleep infinity'"
            )
            ports = {f"{_CDP_CONTAINER_PORT}/tcp": ("127.0.0.1", 0)}
        else:
            command = "sleep infinity"
            ports = {}

        container = client.containers.run(
            self.image,
            command=command,
            name=name,
            detach=True,
            remove=True,
            network_mode=self.network,
            mem_limit=self.mem_limit,
            cpu_count=self.cpu_count,
            volumes=volumes,
            environment=self.extra_env,
            ports=ports,
        )
        log.debug("Started container %s (id=%s)", name or "ephemeral", container.short_id)
        return container

    def _resolve_cdp_url(
        self,
        container: "docker.models.containers.Container",  # type: ignore[name-defined]
    ) -> str | None:
        """Read the host-assigned CDP port from Docker's port bindings."""
        if not self.enable_browser:
            return None
        container.reload()
        bindings = container.ports.get(f"{_CDP_CONTAINER_PORT}/tcp") or []
        if not bindings:
            log.warning("CDP port %s not found in container port bindings", _CDP_CONTAINER_PORT)
            return None
        host_port = bindings[0]["HostPort"]
        return f"http://127.0.0.1:{host_port}"

    # ── SandboxProvider interface ─────────────────────────────────────────────

    async def acquire(
        self,
        hint_id: str | None = None,
        workspace: "Workspace | None" = None,
    ) -> DockerSandbox:
        """Return a running container sandbox.

        If *hint_id* is given and a container with that name already exists and
        is running, it is reused (warm pool).  Otherwise a new container is started.

        .. warning::
            If *workspace* is ``None``, no host path is mounted.  All files
            written inside the container are ephemeral and will be lost when the
            container stops.  Pass a :class:`~harnessx.workspace.workspace.Workspace`
            to persist agent work across turns and container restarts.
        """
        if workspace is None:
            log.warning(
                "DockerSandboxProvider.acquire() called without a workspace — "
                "container filesystem is ephemeral; all agent data will be lost "
                "when the container stops.  Pass workspace=... to mount a host path."
            )
        loop = asyncio.get_event_loop()
        async with self._lock:
            # Check warm pool first
            if hint_id and hint_id in self._pool:
                existing = self._pool[hint_id]
                try:
                    existing.reload()
                    if existing.status == "running":
                        cdp_url = self._resolve_cdp_url(existing)
                        return DockerSandbox(existing, cdp_url=cdp_url)
                except Exception:
                    pass  # container gone — fall through to create new
                del self._pool[hint_id]

            # Start new container in thread (blocking docker-py call)
            container = await loop.run_in_executor(
                None,
                self._start_container,
                hint_id,
                workspace,
            )
            cdp_url = self._resolve_cdp_url(container)
            if cdp_url:
                log.debug("Chromium CDP available at %s", cdp_url)

            if hint_id:
                self._pool[hint_id] = container
            return DockerSandbox(container, cdp_url=cdp_url)

    async def release(self, sandbox: Sandbox) -> None:
        """Leave warm-pool containers running; stop ephemeral ones.

        Also closes any cached CDP connection for this sandbox so
        ``browser_tool`` reconnects fresh on the next container.
        """
        if not isinstance(sandbox, DockerSandbox):
            return

        # Close CDP connection before stopping the container
        if sandbox.cdp_url:
            try:
                from ..tools.builtin.browser import _close_cdp_connection

                await _close_cdp_connection(sandbox.cdp_url)
            except Exception as exc:
                log.debug("Failed to close CDP connection %s: %s", sandbox.cdp_url, exc)

        container = sandbox._container
        # If container is not in any warm-pool slot it's ephemeral — stop it
        if container not in self._pool.values():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: container.stop(timeout=5))

    async def shutdown(self) -> None:
        """Stop and remove all warm-pool containers."""
        loop = asyncio.get_event_loop()
        for hint_id, container in list(self._pool.items()):
            try:
                await loop.run_in_executor(None, lambda c=container: c.stop(timeout=5))
                log.debug("Stopped warm-pool container for hint_id=%s", hint_id)
            except Exception as exc:
                log.warning("Failed to stop container %s: %s", hint_id, exc)
        self._pool.clear()
        if self._client:
            self._client.close()
            self._client = None
