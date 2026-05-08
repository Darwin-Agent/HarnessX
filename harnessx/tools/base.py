# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from ..core.events import ToolSchema


class ToolConflictError(Exception):
    """Raised when registering a tool whose name is already taken.

    Example::

        registry.register(my_tool)   # first registration — OK
        registry.register(my_tool)   # same name again → ToolConflictError

    Pass ``replace=True`` to :meth:`~harnessx.tools._dict_registry_mixin
    ._DictRegistryMixin.register` to explicitly overwrite an existing tool.
    """


@dataclass
class ToolResult:
    output: str
    error: str | None = None
    content_blocks: list | None = None  # native multimodal blocks (Anthropic content format)


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict  # JSON Schema
    fn: Callable
    tags: list[str] = field(default_factory=list)
    execution_target: str = "local"  # "local" | "cloud"

    def to_schema(self) -> ToolSchema:
        meta: dict = {}
        if self.tags:
            meta["tags"] = list(self.tags)
        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            metadata=meta,
        )


@runtime_checkable
class BaseToolRegistry(Protocol):
    def register(self, tool: Tool) -> None: ...
    def get_schemas(self) -> list[ToolSchema]: ...
    async def execute(self, name: str, input: dict) -> ToolResult: ...
    def list_names(self) -> list[str]: ...


_log = logging.getLogger(__name__)


async def _execute_tool(t: Tool, input: dict) -> ToolResult:
    """Execute a Tool's fn, handling both sync and async callables."""
    try:
        if inspect.iscoroutinefunction(t.fn):
            result = await t.fn(**input)
        else:
            result = await asyncio.to_thread(t.fn, **input)
        if isinstance(result, ToolResult):
            return _truncate_result(result, t.name)
        output = str(result) if result is not None else ""
        return _truncate_result(ToolResult(output=output), t.name)
    except Exception as e:
        _log.debug("Tool %r raised an exception: %s", t.name, e, exc_info=True)
        return ToolResult(output="", error=str(e))


def _truncate_result(result: ToolResult, tool_name: str) -> ToolResult:
    """Truncate oversized tool output and save the full content to disk."""
    from .mcp import _MCP_TEXT_THRESHOLD, _save_text

    output = result.output
    if not isinstance(output, str) or len(output) <= _MCP_TEXT_THRESHOLD:
        return result
    if "[truncated" in output:
        return result
    path = _save_text(output, tool_name=tool_name)
    truncated = (
        output[:_MCP_TEXT_THRESHOLD] + f"\n\n[truncated — complete output ({len(output)} chars) saved to {path}]"
    )
    return ToolResult(output=truncated, error=result.error, content_blocks=result.content_blocks)


def _infer_schema(fn: Callable) -> dict:
    """Infer a basic JSON Schema from function signature.

    Uses typing.get_type_hints() to resolve annotations, which handles
    ``from __future__ import annotations`` (PEP 563 stringified annotations).
    Falls back to inspect.Parameter.empty when hints aren't resolvable.
    """
    import typing

    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    _TYPE_MAP = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
    }

    properties = {}
    required = []
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        annotation = hints.get(param_name, inspect.Parameter.empty)
        prop_type = _TYPE_MAP.get(annotation, "string")
        properties[param_name] = {"type": prop_type}
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool(
    name: str | None = None,
    description: str = "",
    tags: list[str] | None = None,
    input_schema: dict | None = None,
    execution_target: str = "local",
) -> Callable:
    """Decorator to create a Tool from a function."""

    def decorator(fn: Callable) -> Tool:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip()
        schema = input_schema or _infer_schema(fn)
        return Tool(
            name=tool_name,
            description=tool_desc,
            input_schema=schema,
            fn=fn,
            tags=tags or [],
            execution_target=execution_target,
        )

    return decorator
