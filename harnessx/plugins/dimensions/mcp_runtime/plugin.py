# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import dataclasses
import json
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from ....core.processor import MultiHookProcessor
from ....tools.base import Tool
from ....tools.mcp import MCPClient
from ...base import HarnessPlugin
from .utils import (
    ensure_agent_home_mcp_servers_json,
    normalise_mcp_config,
    resolve_mcp_servers,
)

if TYPE_CHECKING:
    from ....core.events import TaskStartEvent

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds; delay = _BACKOFF_BASE * 2 ** (attempt - 1), capped at 30 s
_CONNECT_TIMEOUT = 30.0  # seconds; per-server initialize handshake budget
_DISCONNECT_TIMEOUT = 10.0  # seconds; per-server graceful close budget


@dataclass
class _ServerState:
    name: str
    spec: dict[str, Any]
    client: MCPClient | None = None
    attempt: int = 0
    give_up: bool = False
    registered_tools: set[str] = field(default_factory=set)


class _LifecycleSupervisor:
    """Single dedicated asyncio task that owns all MCPClient connect/disconnect.

    The MCP SDK + anyio open TaskGroups and CancelScopes inside ``connect()``
    that are bound to the entering task. Exiting them from a different task
    raises ``RuntimeError("different task")`` *before* anyio runs its cleanup,
    leaving an orphan scope whose ``_deliver_cancellation`` re-arms forever via
    ``call_soon`` — a 100% CPU hot loop in the event loop.

    Routing every ``connect()`` / ``disconnect()`` through this single supervisor
    guarantees ``__aenter__`` and ``__aexit__`` happen on the same task, so the
    cross-task path is never hit.

    The supervisor is lazily started on first submit, rebound to whichever event
    loop is running. Callers ``await`` per-request futures shielded against
    cancellation so a cancelled caller does not abort an in-flight cleanup that
    would otherwise leak resources.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue = asyncio.Queue()
        self._task = loop.create_task(self._run(self._queue), name="mcp-lifecycle-supervisor")

    async def submit(
        self,
        label: str,
        coro_factory: Callable[[], Awaitable[Any]],
        timeout: float | None = None,
    ) -> Any:
        loop = asyncio.get_running_loop()
        if self._task is None or self._task.done() or self._loop is not loop:
            self._start(loop)
        assert self._queue is not None
        fut: asyncio.Future = loop.create_future()
        self._queue.put_nowait((label, coro_factory, fut, timeout))
        return await asyncio.shield(fut)

    async def _run(self, queue: asyncio.Queue) -> None:
        while True:
            try:
                item = await queue.get()
            except asyncio.CancelledError:
                self._drain(queue)
                raise
            if item is None:
                break
            label, coro_factory, fut, timeout = item
            try:
                result = await self._run_one(label, coro_factory, timeout)
            except asyncio.CancelledError:
                if not fut.done():
                    fut.cancel()
                raise
            except BaseException as exc:
                if not fut.done():
                    fut.set_exception(exc)
            else:
                if not fut.done():
                    fut.set_result(result)

    @staticmethod
    async def _run_one(
        label: str,
        coro_factory: Callable[[], Awaitable[Any]],
        timeout: float | None,
    ) -> Any:
        """Run ``coro_factory()`` on the supervisor task with an optional budget.

        We deliberately avoid ``asyncio.wait_for`` because it wraps the coroutine
        in a child Task — that would make ``__aenter__`` and ``__aexit__`` of the
        MCP/anyio scopes run on different tasks, recreating the cross-task bug
        the supervisor exists to prevent. Instead we schedule a one-shot timer
        that cancels the supervisor task itself; the cancellation propagates
        into ``coro_factory()`` (same task — anyio scopes unwind cleanly), and
        we then ``uncancel()`` so the supervisor loop survives.
        """
        if timeout is None:
            return await coro_factory()
        loop = asyncio.get_running_loop()
        current = asyncio.current_task()
        timed_out = False

        def _on_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            if current is not None and not current.done():
                current.cancel()

        handle = loop.call_later(timeout, _on_timeout)
        try:
            return await coro_factory()
        except asyncio.CancelledError:
            if timed_out:
                if current is not None and callable(getattr(current, "uncancel", None)):
                    try:
                        while current.cancelling():
                            current.uncancel()
                    except Exception:
                        pass
                raise TimeoutError(f"MCP lifecycle '{label}' exceeded {timeout}s") from None
            raise
        finally:
            handle.cancel()

    @staticmethod
    def _drain(queue: asyncio.Queue) -> None:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item is None:
                continue
            fut = item[2] if len(item) >= 3 else None
            if fut is not None and not fut.done():
                fut.cancel()

    async def stop(self, timeout: float = 10.0) -> None:
        task = self._task
        queue = self._queue
        if task is None or task.done() or queue is None:
            self._task = None
            self._queue = None
            self._loop = None
            return
        try:
            queue.put_nowait(None)
        except Exception:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        finally:
            self._task = None
            self._queue = None
            self._loop = None


class McpRuntimeTaskStartProcessor(MultiHookProcessor):
    """Thin task_start trigger that delegates MCP hot-reload to the plugin."""

    _order = 0
    _singleton_group = "_mcp_runtime"
    __hx_runtime_only__ = True

    def __init__(self, plugin: "McpRuntimePlugin") -> None:
        self._plugin = plugin

    async def on_task_start(self, event: "TaskStartEvent") -> AsyncIterator:
        yield await self._plugin.on_task_start(event)


class McpRuntimePlugin(HarnessPlugin):
    """HarnessX built-in core MCP runtime plugin.

    Reusable by any agent/harness entrypoint that needs MCP runtime management.
    This plugin is built-in and is not discovered from AGENT_HOME/plugins.
    """

    name = "_builtin_mcp_runtime"
    version = "0.1.0"
    description = "Built-in core MCP runtime management"

    def __init__(
        self,
        mcp_config: dict[str, Any] | None = None,
        base_dir: str | Path | None = None,
        ensure_primary: bool = True,
    ) -> None:
        super().__init__()
        self._mcp_config: dict[str, Any] = normalise_mcp_config(mcp_config)
        self._base_dir = base_dir
        self._ensure_primary = bool(ensure_primary)
        if self._mcp_config.get("source") == "agent_home" and self._ensure_primary:
            ensure_agent_home_mcp_servers_json()

        self._servers_sig = ""
        self._servers: "OrderedDict[str, _ServerState]" = OrderedDict()
        self._tool_registry: Any = None
        self._harness_config: Any = None
        self._supervisor = _LifecycleSupervisor()
        self.processors = [McpRuntimeTaskStartProcessor(self)]

    def add_inline_servers(self, servers: list[dict[str, Any]]) -> None:
        """Merge additional inline server specs into this runtime config."""
        if self._mcp_config.get("source") != "inline":
            self._mcp_config = {"source": "inline", "path": None, "servers": []}

        merged = self._mcp_config.get("servers")
        if not isinstance(merged, list):
            merged = []
        self._mcp_config["servers"] = merged

        for spec in servers:
            if isinstance(spec, dict):
                merged.append(dict(spec))

    def setup(self, config) -> None:
        self._harness_config = config
        self._tool_registry = getattr(config, "tool_registry", None)
        # Auto-derive base_dir from workspace when not explicitly set.
        if self._base_dir is None:
            ws = getattr(config, "workspace", None)
            if ws is not None:
                root = getattr(ws, "root", None)
                if root is not None:
                    self._base_dir = str(root)

    async def on_task_start(self, event: "TaskStartEvent") -> "TaskStartEvent":
        tools_changed = False
        servers = self._load_servers()
        sig = self._servers_signature(servers)
        if sig != self._servers_sig:
            tools_changed = await self._replace_servers(servers)
            self._servers_sig = sig

        for state in self._servers.values():
            if state.client is not None or state.give_up:
                continue
            if await self._connect_server(state, event):
                tools_changed = True

        if tools_changed:
            return self._refresh_event_tools(event)
        return event

    async def warmup_summary(self) -> dict[str, int]:
        """Preload runtime connections and return summary counts.

        Returns:
            {"servers": configured_server_count,
             "connected_servers": connected_server_count,
             "tools": registered_tool_count}
        """
        servers = self._load_servers()
        sig = self._servers_signature(servers)
        if sig != self._servers_sig:
            await self._replace_servers(servers)
            self._servers_sig = sig

        for state in self._servers.values():
            if state.client is None and not state.give_up:
                await self._connect_server(state, None)

        return {
            "servers": len(self._servers),
            "connected_servers": sum(1 for s in self._servers.values() if s.client is not None),
            "tools": sum(len(s.registered_tools) for s in self._servers.values()),
        }

    async def stop(self) -> None:
        try:
            await self._disconnect_all(drop_tools=True)
        finally:
            self._servers_sig = ""
            try:
                await self._supervisor.stop()
            except Exception:
                pass

    def _load_servers(self) -> list[dict[str, Any]]:
        loaded = resolve_mcp_servers(
            self._mcp_config,
            base_dir=self._base_dir,
            ensure_primary=self._ensure_primary,
        )
        return self._normalise_servers([s for s in loaded if isinstance(s, dict)])

    @staticmethod
    def _servers_signature(servers: list[dict[str, Any]]) -> str:
        try:
            return json.dumps(servers, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            return ""

    @staticmethod
    def _normalise_servers(raw_servers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        used_names: set[str] = set()
        for i, spec in enumerate(raw_servers):
            normalized = dict(spec)
            base_name = str(normalized.get("name", "")).strip() or f"mcp-{i + 1}"
            name = base_name
            suffix = 2
            while name in used_names:
                name = f"{base_name}-{suffix}"
                suffix += 1
            normalized["name"] = name
            used_names.add(name)
            out.append(normalized)
        return out

    def _resolve_registry(self, event: "TaskStartEvent | None" = None) -> Any | None:
        registry = self._tool_registry
        if registry is None and self._harness_config is not None:
            registry = getattr(self._harness_config, "tool_registry", None)
        if registry is None and event is not None:
            config = getattr(event, "config", None)
            if config is not None:
                registry = getattr(config, "tool_registry", None)
        return registry

    async def _replace_servers(self, servers: list[dict[str, Any]]) -> bool:
        had_tools = any(state.registered_tools for state in self._servers.values())
        await self._disconnect_all(drop_tools=True)
        self._servers.clear()
        for spec in servers:
            name = str(spec.get("name", "")).strip()
            if not name:
                continue
            self._servers[name] = _ServerState(name=name, spec=spec)
        return had_tools

    async def _disconnect_all(self, *, drop_tools: bool) -> None:
        for state in reversed(list(self._servers.values())):
            await self._disconnect_server(state, drop_tools=drop_tools)
        self._servers.clear()

    async def _disconnect_server(self, state: _ServerState, *, drop_tools: bool) -> None:
        client = state.client
        if client is not None:
            try:
                await self._run_disconnect(state.name, client)
            finally:
                state.client = None
        if drop_tools:
            self._drop_tools(state.registered_tools)
            state.registered_tools.clear()

    async def _run_connect(self, name: str, client: MCPClient) -> None:
        await self._supervisor.submit(
            f"connect[{name}]",
            lambda c=client: c.connect(),
            timeout=_CONNECT_TIMEOUT,
        )

    async def _run_disconnect(self, name: str, client: MCPClient) -> None:
        try:
            await self._supervisor.submit(
                f"disconnect[{name}]",
                lambda c=client: c.disconnect(),
                timeout=_DISCONNECT_TIMEOUT,
            )
        except BaseException as exc:
            warnings.warn(
                f"McpRuntimePlugin: disconnect failed for '{name}': {exc!r}",
                stacklevel=2,
            )

    async def _connect_server(self, state: _ServerState, event: "TaskStartEvent | None") -> bool:
        spec = state.spec
        name = state.name
        command = str(spec.get("command", "")).strip()
        args = spec.get("args", [])
        if isinstance(args, list) and args and command:
            command = command + " " + " ".join(str(a) for a in args)

        if state.attempt > 0:
            await asyncio.sleep(min(_BACKOFF_BASE * (2 ** (state.attempt - 1)), 30.0))
        state.attempt += 1

        try:
            client = MCPClient(
                transport=str(spec.get("transport", "stdio") or "stdio"),
                command=command,
                url=str(spec.get("url", "") or ""),
                env=spec.get("env") or None,
            )
            await self._run_connect(name, client)
            state.client = client

            registry = self._resolve_registry(event)
            if registry is None:
                warnings.warn(
                    f"McpRuntimePlugin: connected to '{name}' but no tool_registry available.",
                    stacklevel=2,
                )
                await self._run_disconnect(name, client)
                state.client = None
                return False

            await self._register_tools(state, registry)
            state.attempt = 0
            state.give_up = False
            return bool(state.registered_tools)
        except asyncio.CancelledError as exc:
            # Warmup preloading must never block CLI startup: treat internal
            # cancel-scope cancellation as a transient connect failure.
            if event is None:
                warnings.warn(
                    f"McpRuntimePlugin: warmup connect cancelled for '{name}': {exc}",
                    stacklevel=2,
                )
                if state.client is not None:
                    await self._run_disconnect(name, state.client)
                    state.client = None
                # Python 3.12 structured cancellation: catching CancelledError without
                # re-raising leaves the task's cancellation counter incremented.
                # Call uncancel() so subsequent awaits in the caller are not poisoned.
                _task = asyncio.current_task()
                if _task is not None and callable(getattr(_task, "uncancel", None)):
                    try:
                        while _task.cancelling():
                            _task.uncancel()
                    except Exception:
                        pass
                return False
            raise
        except Exception as exc:
            if _MAX_RETRIES - state.attempt <= 0:
                state.give_up = True
                warnings.warn(
                    f"McpRuntimePlugin: giving up on '{name}' after {_MAX_RETRIES} attempts. Last error: {exc}",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"McpRuntimePlugin: attempt {state.attempt}/{_MAX_RETRIES} failed for '{name}': {exc} "
                    "(will retry on next task_start)",
                    stacklevel=2,
                )
            if state.client is not None:
                try:
                    await self._run_disconnect(name, state.client)
                except Exception:
                    pass
                finally:
                    state.client = None
            return False

    async def _register_tools(self, state: _ServerState, registry: Any) -> None:
        if state.client is None:
            return
        for i, tool_def in enumerate(await state.client.list_tools()):
            tool_name = str(tool_def.get("name", "")).strip() or f"{state.name}_tool_{i}"

            async def _call(
                _client=state.client,
                _name=tool_name,
                **kwargs: "Any",
            ) -> str:
                result, _blocks, _is_error = await _client.call_tool(_name, kwargs)
                return result

            t = Tool(
                name=tool_name,
                description=tool_def.get("description", ""),
                input_schema=tool_def.get("inputSchema", {"type": "object", "properties": {}}),
                fn=_call,
                tags=["mcp", state.name],
            )
            try:
                registry.register(t, replace=True)
            except Exception:
                registry.register(t.__class__(**{**t.__dict__}), replace=True)
            state.registered_tools.add(tool_name)

    def _drop_tools(self, tool_names: set[str]) -> None:
        if not tool_names:
            return
        reg = self._resolve_registry()
        if reg is None:
            return
        tools_dict = getattr(reg, "_tools", None)
        if not isinstance(tools_dict, dict):
            return
        for name in list(tool_names):
            tools_dict.pop(name, None)

    def _refresh_event_tools(self, event: "TaskStartEvent") -> "TaskStartEvent":
        registry = self._resolve_registry(event)
        if registry is None or not hasattr(registry, "get_schemas"):
            return event
        try:
            schemas = tuple(registry.get_schemas())
        except Exception:
            return event
        return dataclasses.replace(event, tools=schemas)
