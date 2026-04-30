"""
Minimal tool framework inlined from HarnessX for self-contained verl_harnessX usage.

Provides: ToolSchema, ToolResult, Tool, @tool decorator, InMemoryToolRegistry.
"""

from __future__ import annotations

import asyncio
import inspect
import typing
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    input_schema: dict
    metadata: dict = field(default_factory=dict, hash=False, compare=False)


@dataclass
class ToolResult:
    output: str
    error: str | None = None


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable
    tags: list[str] = field(default_factory=list)

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


async def _execute_tool(t: Tool, input: dict) -> ToolResult:
    try:
        if inspect.iscoroutinefunction(t.fn):
            result = await t.fn(**input)
        else:
            result = await asyncio.to_thread(t.fn, **input)
        return ToolResult(output=str(result) if result is not None else "")
    except Exception as e:
        return ToolResult(output="", error=str(e))


def _infer_schema(fn: Callable) -> dict:
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    _TYPE_MAP = {int: "integer", float: "number", bool: "boolean", str: "string"}
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
    return {"type": "object", "properties": properties, "required": required}


def tool(
    name: str | None = None,
    description: str = "",
    tags: list[str] | None = None,
    input_schema: dict | None = None,
) -> Callable:
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
        )

    return decorator


class InMemoryToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, replace: bool = False) -> None:
        if not replace and tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[ToolSchema]:
        return [t.to_schema() for t in self._tools.values()]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, input: dict) -> ToolResult:
        t = self._tools.get(name)
        if t is None:
            return ToolResult(output="", error=f"Tool '{name}' not found")
        return await _execute_tool(t, input)
