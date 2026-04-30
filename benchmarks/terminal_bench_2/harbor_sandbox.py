# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import shlex
from typing import TYPE_CHECKING, Any

from harnessx.sandbox.base import Sandbox, SandboxProvider

if TYPE_CHECKING:
    from harnessx.workspace.workspace import Workspace

_OUTPUT_LIMIT = 8000

# Temp file inside the container that records the PID of the setsid'd bash
# started by the most recent exec() call.  kill_running() reads this file to
# send SIGTERM/SIGKILL only to that process group, leaving other container
# processes (e.g. background servers started by earlier commands) untouched.
_PGID_FILE = "/tmp/_hx_exec_pgid"


class HarborSandbox(Sandbox):
    """Routes exec/read/write through Harbor's ``environment.exec()``."""

    def __init__(
        self,
        environment: Any,
        workspace_path: str = "/workspace",
        output_limit: int | None = _OUTPUT_LIMIT,
    ) -> None:
        self._env = environment
        self._workspace_path = workspace_path
        self._output_limit = output_limit

    @property
    def workspace_path(self) -> str:
        return self._workspace_path

    def _trim(self, text: str) -> str:
        if self._output_limit and len(text) > self._output_limit:
            removed = len(text) - self._output_limit
            return (
                text[: self._output_limit]
                + f"\n[...output truncated: {removed} chars not shown. Use head/tail/grep to target specific output.]"
            )
        return text

    async def kill_running(self) -> None:
        """Kill only the process group of the most recent exec() call.

        Reads the PGID saved by exec()'s setsid wrapper and sends SIGTERM then
        SIGKILL to that group.  Processes started by *earlier* exec() calls
        (e.g. background servers) are in different process groups and are
        therefore unaffected.
        """
        try:
            await self._env.exec(
                f"_pid=$(cat {_PGID_FILE} 2>/dev/null);"
                f' [ -n "$_pid" ] || exit 0;'
                f' kill -15 -"$_pid" 2>/dev/null;'
                f" sleep 1;"
                f' kill -9 -"$_pid" 2>/dev/null;'
                f" rm -f {_PGID_FILE}",
                timeout_sec=10,
            )
        except Exception:
            pass

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> str:
        # Run the command inside a new session via setsid so it gets its own
        # process group (PGID == PID of the setsid'd process).  The PID is
        # saved to _PGID_FILE so kill_running() can target exactly this group
        # without affecting other container processes.
        #
        # stdout/stderr flow through the background job's inherited file
        # descriptors to docker compose exec's client as normal.  The exit
        # code is forwarded via `wait`.
        wrapped = (
            f"setsid bash -c {shlex.quote(command)} &"
            f" _hx_pid=$!;"
            f" printf '%s' \"$_hx_pid\" > {_PGID_FILE};"
            f" wait $_hx_pid"
        )
        try:
            result = await self._env.exec(
                wrapped,
                cwd=cwd or self._workspace_path,
                timeout_sec=int(timeout),
            )
        except BaseException:
            # Harbor raises RuntimeError on timeout (not asyncio.TimeoutError).
            # CancelledError arrives when an outer asyncio.wait_for fires.
            # Either way the setsid'd process is still running — kill it.
            try:
                await asyncio.shield(self.kill_running())
            except Exception:
                pass
            raise
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = stdout
        if stderr:
            output = f"{output}\nSTDERR: {stderr}" if output else f"STDERR: {stderr}"
        if not output:
            output = f"(exit {result.return_code}, no output captured)"
        elif result.return_code != 0:
            output = f"{output}\n(exit {result.return_code})"
        return self._trim(output)


class HarborSandboxProvider(SandboxProvider):
    """Provider that wraps a Harbor ``BaseEnvironment``.

    Harbor manages the container lifecycle; this provider has no lifecycle of
    its own — acquire/release are no-ops.

    Args:
        environment:    A Harbor ``BaseEnvironment`` instance passed in from
                        ``HarnessXAgent.run()``.
        workspace_path: Container-side path used as the default cwd for all
                        tool calls.  Defaults to ``/workspace``; use ``/app``
                        for Terminal Bench 2.0 tasks.
        output_limit:   Truncate exec output to this many characters.
                        Default 4 000 (matches the old hand-rolled limit).
                        Pass ``None`` to disable truncation.
    """

    def __init__(
        self,
        environment: Any,
        workspace_path: str = "/workspace",
        output_limit: int | None = _OUTPUT_LIMIT,
    ) -> None:
        self._environment = environment
        self._workspace_path = workspace_path
        self._output_limit = output_limit

    async def acquire(
        self,
        hint_id: str | None = None,
        workspace: "Workspace | None" = None,
    ) -> HarborSandbox:
        return HarborSandbox(
            self._environment,
            workspace_path=self._workspace_path,
            output_limit=self._output_limit,
        )

    async def release(self, sandbox: Sandbox) -> None:
        pass  # Harbor owns the container lifecycle

    async def shutdown(self) -> None:
        pass
