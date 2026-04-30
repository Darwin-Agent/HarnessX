# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import Any, Callable

import httpx

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.paths import EnvironmentPaths

_EXECD_PORT = 44772
_POLL_INTERVAL = 5  # seconds between status polls
_START_TIMEOUT = 600  # seconds to wait for Running state (image pulls can take minutes)
_POLL_READ_TIMEOUT = 10  # per-request read timeout during polling
_DEFAULT_SERVER = os.environ.get("OPENSANDBOX_URL", "http://127.0.0.1:12081")

# Common host paths where `uv` is pre-installed; first match is used for injection.
_UV_CANDIDATE_PATHS = [
    os.environ.get("TB2_UV_BINARY", ""),
    "/root/.local/bin/uv",
    "/usr/local/bin/uv",
    "/opt/tb2-tools/uv",
]

# Retry config for transient sandbox errors (400/500/502/503/504 or network blips)
_RETRYABLE_STATUS = {400, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF = (1.0, 2.0, 4.0)  # seconds before each successive retry
_STATE_CHECK_TIMEOUT = httpx.Timeout(5.0, connect=3.0)


class SandboxExecdDeadError(RuntimeError):
    """Raised when the sandbox's execd process is no longer alive.

    Diagnosed during 502 handling: if the sandbox state is not 'Running'
    when the proxy returns 502, execd has been killed (e.g. container OOM).
    Retrying is pointless — this exception surfaces the root cause immediately.
    """


def _proxy(sid: str, path: str) -> str:
    return f"/v1/sandboxes/{sid}/proxy/{_EXECD_PORT}/{path.lstrip('/')}"


def _parse_exec_stream(text: str) -> tuple[str, str, int]:
    """Parse execd streaming JSON response into (stdout, stderr, return_code)."""
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    return_code = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = event.get("type", "")
        if t == "stdout":
            stdout_parts.append(event.get("text", ""))
        elif t == "stderr":
            stderr_parts.append(event.get("text", ""))
        elif t == "error":
            try:
                return_code = int(event.get("error", {}).get("evalue", "1"))
            except (ValueError, TypeError):
                return_code = 1

    return "\n".join(stdout_parts), "\n".join(stderr_parts), return_code


class OpenSandboxEnvironment(BaseEnvironment):
    """Harbor environment backed by a self-hosted OpenSandbox server.

    Args:
        server_url: OpenSandbox server base URL (default: OPENSANDBOX_URL env var
            or http://127.0.0.1:12081).
        sandbox_timeout_sec: Max sandbox lifetime in seconds. Defaults to None
            (no server-side timeout). Harbor manages lifecycle via explicit DELETE
            in stop(), so a server timeout is not needed in normal operation.
        registry_auth: Optional dict with ``username`` and ``password`` for
            private registry images.
    """

    def __init__(
        self,
        *args,
        server_url: str = _DEFAULT_SERVER,
        sandbox_timeout_sec: int | None = None,
        registry_auth: dict[str, str] | None = None,
        proxy_url: str | None = (os.environ.get("OPENSANDBOX_PROXY_URL") or os.environ.get("OPENSANDBOX_PROXY")),
        no_proxy: str | None = os.environ.get("OPENSANDBOX_NO_PROXY"),
        inject_uv: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._server_url = server_url.rstrip("/")
        self._sandbox_timeout = sandbox_timeout_sec
        self._registry_auth = registry_auth
        self._sandbox_id: str | None = None
        self._client: httpx.AsyncClient | None = None
        # Proxy env vars injected into every exec() call.
        # Only set when OPENSANDBOX_PROXY_URL is explicitly provided;
        # otherwise leave the container environment untouched.
        if proxy_url:
            self._proxy_env: dict[str, str] = {
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
            }
            if no_proxy:
                self._proxy_env["no_proxy"] = no_proxy
                self._proxy_env["NO_PROXY"] = no_proxy
        else:
            self._proxy_env = {}
        # Resolve host-side uv binary for injection (None = skip injection).
        self._uv_host_path: str | None = None
        if inject_uv:
            for candidate in _UV_CANDIDATE_PATHS:
                if candidate and Path(candidate).is_file():
                    self._uv_host_path = candidate
                    break

    # ── Abstract property implementations ────────────────────────────────────

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER  # closest available type; used for display only

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return False

    def _validate_definition(self) -> None:
        if not self.task_env_config.docker_image:
            raise ValueError("OpenSandboxEnvironment requires task_env_config.docker_image to be set.")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, force_build: bool = False) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._server_url,
            timeout=httpx.Timeout(60.0, read=300.0),
        )

        image_spec: dict[str, Any] = {"uri": self.task_env_config.docker_image}
        if self._registry_auth:
            image_spec["auth"] = self._registry_auth

        # Resource limits are intentionally omitted: applying cgroup cpu/memory
        # constraints inside a container fails with cgroupv2 domain-controller
        # errors on some kernels.  The physical host has sufficient headroom.
        sleep_arg = str(self._sandbox_timeout) if self._sandbox_timeout else "infinity"
        payload: dict[str, Any] = {
            "image": image_spec,
            "entrypoint": ["/bin/sleep", sleep_arg],
            "resourceLimits": {},
        }
        if self._sandbox_timeout:
            payload["timeout"] = self._sandbox_timeout

        self.logger.debug(f"OpenSandbox: creating sandbox payload={payload}")
        _max_attempts = 5
        for _attempt in range(1, _max_attempts + 1):
            r = await self._client.post("/v1/sandboxes", json=payload)
            if not r.is_error:
                break
            self.logger.warning(
                f"OpenSandbox: POST /v1/sandboxes {r.status_code} (attempt {_attempt}/{_max_attempts}): {r.text[:300]}"
            )
            if _attempt == _max_attempts:
                r.raise_for_status()
            await asyncio.sleep(10 * _attempt)
        self._sandbox_id = r.json()["id"]
        self.logger.info(f"OpenSandbox: created sandbox {self._sandbox_id}")

        await self._wait_running()
        await self._wait_execd()
        await self._inject_uv()
        await self._ensure_curl()

        # (Re-)create standard Harbor log directories from a clean state.
        # rm -rf first ensures that if this container was reused from a previous
        # trial, any leftover /logs/agent or /logs/verifier content is wiped —
        # preventing cross-trial trace contamination.
        await self.exec(
            f"rm -rf {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir} "
            f"&& mkdir -p {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir} "
            f"&& chmod 777 {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}"
        )

    async def stop(self, delete: bool = True) -> None:
        # Always delete the sandbox container — `delete=False` in Harbor means
        # "keep the Docker image", which doesn't apply to OpenSandbox.
        if self._sandbox_id and self._client:
            try:
                await self._client.delete(f"/v1/sandboxes/{self._sandbox_id}")
                self.logger.info(f"OpenSandbox: deleted sandbox {self._sandbox_id}")
            except Exception as e:
                self.logger.warning(f"OpenSandbox: failed to delete sandbox: {e}")
            self._sandbox_id = None
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _wait_running(self) -> None:
        """Poll until sandbox reaches Running state or timeout expires.

        Uses a short per-request read timeout (_POLL_READ_TIMEOUT) so a slow
        server response during image pulls does not block the loop indefinitely.
        Total wait is bounded by _START_TIMEOUT seconds.
        """
        poll_timeout = httpx.Timeout(_POLL_READ_TIMEOUT, connect=10.0)
        elapsed = 0
        last_state = ""
        while elapsed < _START_TIMEOUT:
            try:
                r = await self._client.get(
                    f"/v1/sandboxes/{self._sandbox_id}",
                    timeout=poll_timeout,
                )
                state = r.json().get("status", {}).get("state", "unknown")
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                state = f"unreachable ({exc.__class__.__name__})"

            if state != last_state:
                self.logger.info(f"OpenSandbox: sandbox {self._sandbox_id} state={state} elapsed={elapsed}s")
                last_state = state

            if state == "Running":
                return
            if state in ("Failed", "Stopped", "Error"):
                raise RuntimeError(f"OpenSandbox: sandbox {self._sandbox_id} entered state {state}")

            await asyncio.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL

        raise TimeoutError(
            f"OpenSandbox: sandbox {self._sandbox_id} did not reach Running "
            f"within {_START_TIMEOUT}s (last state: {last_state})"
        )

    async def _inject_uv(self) -> None:
        """Upload the host-side uv binary into the container so verifiers that
        expect uv/uvx don't need to download it from the internet.

        Installs to /usr/local/bin/uv (always in PATH, including non-interactive
        shells used by verifiers) and creates a uvx symlink there.  Also mirrors
        to /root/.local/bin/ for scripts that hard-code the default uv install
        path, and writes /root/.local/bin/env so ``source $HOME/.local/bin/env``
        succeeds in verifier scripts.  No-ops silently if no host binary was
        found or injection is disabled.
        """
        if not self._uv_host_path:
            self.logger.debug(f"OpenSandbox: _inject_uv skipped — no uv host binary found (sandbox {self._sandbox_id})")
            return
        self.logger.info(f"OpenSandbox: _inject_uv uploading {self._uv_host_path} into sandbox {self._sandbox_id}")
        try:
            await self.upload_file(self._uv_host_path, "/usr/local/bin/uv")
            await self.exec(
                "chmod +x /usr/local/bin/uv"
                " && ln -sf /usr/local/bin/uv /usr/local/bin/uvx"
                " && mkdir -p /root/.local/bin"
                " && ln -sf /usr/local/bin/uv /root/.local/bin/uv"
                " && ln -sf /usr/local/bin/uv /root/.local/bin/uvx"
                # Create the env activation script that the uv installer normally
                # writes; verifiers source it to add ~/.local/bin to PATH.
                " && printf '#!/bin/sh\\nexport PATH=\"/root/.local/bin:/usr/local/bin:$PATH\"\\n'"
                " > /root/.local/bin/env"
                " && chmod +x /root/.local/bin/env"
            )
            self.logger.info(f"OpenSandbox: injected uv from {self._uv_host_path} into sandbox {self._sandbox_id}")
        except Exception as exc:
            self.logger.warning(f"OpenSandbox: uv injection failed (non-fatal): {exc}")

    async def _ensure_curl(self, max_attempts: int = 5) -> None:
        """Ensure curl is installed in the container, retrying if apt is locked.

        Many TB2 verifiers run ``apt-get install -y curl`` themselves; pre-installing
        it here means that step becomes a cheap no-op instead of a full download.
        Retries up to *max_attempts* times with linear backoff to handle the
        common race where the image's init scripts hold the dpkg lock at startup.
        Non-fatal: logs a warning if all attempts fail.
        """
        # Fast path: curl already present (e.g. image already has it)
        check = await self.exec("command -v curl")
        if check.return_code == 0:
            self.logger.debug(f"OpenSandbox: curl already present in sandbox {self._sandbox_id}")
            return

        install_cmd = (
            "DEBIAN_FRONTEND=noninteractive apt-get update -qq"
            " && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl"
            " && rm -rf /var/lib/apt/lists/*"
        )
        # apt-get needs direct internet access; unset any proxy env vars that
        # may have been injected into exec() by default (they break apt's
        # connection to archive.ubuntu.com if the proxy is not reachable from
        # inside the container).
        no_proxy_env = {
            k: "" for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY")
        }
        for attempt in range(1, max_attempts + 1):
            try:
                result = await self.exec(install_cmd, timeout_sec=120, env=no_proxy_env)
                if result.return_code == 0:
                    self.logger.info(f"OpenSandbox: installed curl in sandbox {self._sandbox_id} (attempt {attempt})")
                    return
                # dpkg lock held — wait and retry
                if "Could not get lock" in result.stderr or "Unable to acquire" in result.stderr:
                    self.logger.debug(f"OpenSandbox: dpkg lock busy, retrying curl install ({attempt}/{max_attempts})")
                    await asyncio.sleep(attempt * 3)
                    continue
                # Other apt failure — log and give up
                self.logger.warning(f"OpenSandbox: curl install failed (non-fatal): {result.stderr[:200]}")
                return
            except Exception as exc:
                self.logger.warning(f"OpenSandbox: curl install error on attempt {attempt}: {exc}")
                if attempt < max_attempts:
                    await asyncio.sleep(attempt * 3)
        self.logger.warning(
            f"OpenSandbox: could not install curl after {max_attempts} attempts"
            f" in sandbox {self._sandbox_id} (non-fatal)"
        )

    async def _wait_execd(self, timeout: int = 60) -> None:
        """Poll execd via a simple echo until it responds (or timeout expires).

        The container may report state=Running before execd has finished
        initialising.  This method retries a no-op command every 2 seconds
        until execd accepts the request, so the subsequent rm/mkdir in
        start() never sees a 502.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        interval = 2.0
        while True:
            try:
                r = await self._client.post(
                    _proxy(self._sandbox_id, "/command"),
                    json={"Command": "echo ready"},
                    timeout=httpx.Timeout(5.0, connect=3.0),
                )
                if r.status_code == 200:
                    self.logger.debug(f"OpenSandbox: execd ready on sandbox {self._sandbox_id}")
                    return
            except (httpx.TimeoutException, httpx.RequestError):
                pass
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"OpenSandbox: execd on sandbox {self._sandbox_id} did not become ready within {timeout}s"
                )
            await asyncio.sleep(min(interval, remaining))

    # ── Retry helper ─────────────────────────────────────────────────────────

    async def _get_sandbox_state(self) -> str:
        """Query current sandbox state from the lifecycle API.

        Returns the state string (e.g. 'Running', 'Stopped', 'Error') or
        'unknown' if the request fails — callers must handle 'unknown' as
        indeterminate (i.e. still attempt a retry).
        """
        try:
            r = await self._client.get(
                f"/v1/sandboxes/{self._sandbox_id}",
                timeout=_STATE_CHECK_TIMEOUT,
            )
            return r.json().get("status", {}).get("state", "unknown")
        except Exception:
            return "unknown"

    async def _retry_request(self, request_fn: Callable) -> httpx.Response:
        """Call request_fn() and retry on transient 5xx / network errors.

        On HTTP 502 specifically, the sandbox state is queried before each
        retry.  If the state is not 'Running', execd has been killed (OOM or
        container crash) and a :exc:`SandboxExecdDeadError` is raised
        immediately instead of wasting retry attempts on a dead container.

        request_fn must be a zero-argument callable that returns an awaitable
        httpx.Response (i.e. a lambda over self._client.post/get).
        Up to _MAX_RETRIES additional attempts are made with exponential backoff.
        The final response (or exception) is returned/raised to the caller.
        """
        last_exc: Exception | None = None
        r: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                r = await request_fn()
                if r.status_code not in _RETRYABLE_STATUS:
                    return r
                # 502 from the proxy almost always means execd is dead.
                # Check sandbox state to distinguish OOM/crash from a transient blip.
                if r.status_code == 502:
                    state = await self._get_sandbox_state()
                    if state not in ("Running", "unknown"):
                        raise SandboxExecdDeadError(
                            f"Sandbox {self._sandbox_id} state={state!r} while proxy "
                            f"returned 502 — execd is dead (OOM or container crash)."
                        )
                last_exc = None  # retryable HTTP status — will retry
            except SandboxExecdDeadError:
                raise  # never swallow — surfaces root cause immediately
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
            ) as exc:
                last_exc = exc
                r = None
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                self.logger.warning(
                    f"OpenSandbox: transient error on attempt {attempt + 1} "
                    f"({'HTTP ' + str(r.status_code) if r is not None else str(last_exc)}), "
                    f"retrying in {delay}s"
                )
                await asyncio.sleep(delay)
        if last_exc is not None:
            raise last_exc
        return r  # type: ignore[return-value]  # retryable status exhausted

    # ── Command execution ─────────────────────────────────────────────────────

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        # Proxy vars are defaults; call-level env takes precedence
        effective_env = {**self._proxy_env, **(env or {})} if self._proxy_env else env
        merged_env = self._merge_env(effective_env)

        # Build shell-wrapped command with optional cwd + env
        parts: list[str] = []
        if cwd:
            parts.append(f"cd {shlex.quote(cwd)}")
        if merged_env:
            for k, v in merged_env.items():
                parts.append(f"export {k}={shlex.quote(v)}")
        parts.append(command)
        shell_cmd = " && ".join(parts) if len(parts) > 1 else command

        timeout = httpx.Timeout(timeout_sec or 300.0, connect=10.0)
        r = await self._retry_request(
            lambda: self._client.post(
                _proxy(self._sandbox_id, "/command"),
                json={"Command": shell_cmd},
                timeout=timeout,
            )
        )
        r.raise_for_status()
        stdout, stderr, return_code = _parse_exec_stream(r.text)
        return ExecResult(stdout=stdout, stderr=stderr, return_code=return_code)

    # ── File operations ───────────────────────────────────────────────────────

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source_path = Path(source_path)
        content = source_path.read_bytes()
        metadata = json.dumps({"path": target_path}).encode()

        r = await self._retry_request(
            lambda: self._client.post(
                _proxy(self._sandbox_id, "/files/upload"),
                files={
                    "metadata": ("metadata", metadata, "application/json"),
                    "file": (source_path.name, content, "application/octet-stream"),
                },
            )
        )
        r.raise_for_status()

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source_dir = Path(source_dir)
        tasks = [
            self.upload_file(
                local_file,
                f"{target_dir.rstrip('/')}/{local_file.relative_to(source_dir)}",
            )
            for local_file in sorted(source_dir.rglob("*"))
            if local_file.is_file()
        ]
        if tasks:
            await asyncio.gather(*tasks)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        r = await self._retry_request(
            lambda: self._client.get(
                _proxy(self._sandbox_id, "/files/download"),
                params={"path": source_path},
            )
        )
        r.raise_for_status()
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(r.content)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        result = await self.exec(f"find {shlex.quote(source_dir)} -type f 2>/dev/null")
        target_dir = Path(target_dir)
        tasks = [
            self.download_file(rp, target_dir / os.path.relpath(rp, source_dir))
            for rp in (p.strip() for p in result.stdout.splitlines())
            if rp
        ]
        if tasks:
            await asyncio.gather(*tasks)
