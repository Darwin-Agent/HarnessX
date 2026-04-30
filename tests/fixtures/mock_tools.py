# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from harnessx.tools import tool
from harnessx.tools.inmemory import InMemoryToolRegistry


@tool(name="add", description="Add two numbers")
def add_tool(a: int, b: int) -> int:
    return a + b


@tool(name="echo", description="Echo back input")
def echo_tool(message: str) -> str:
    return message


@tool(name="fail_tool", description="Always fails")
def fail_tool(message: str) -> str:
    raise RuntimeError("This tool always fails")


def make_registry(*tools_list) -> InMemoryToolRegistry:
    registry = InMemoryToolRegistry()
    for t in tools_list:
        registry.register(t)
    return registry
