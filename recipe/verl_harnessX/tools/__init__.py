from .base import (
    Tool,
    ToolResult as ToolResult,
    ToolSchema as ToolSchema,
    tool as tool,
    InMemoryToolRegistry,
)
from .web_search import web_search_tool
from .web_fetch import web_fetch_tool
from .browser import browser_tool
from .bash import bash_tool
from .code import code_tool
from .read import read_tool

ALL_TOOLS: list[Tool] = [
    web_search_tool,
    web_fetch_tool,
    browser_tool,
    bash_tool,
    code_tool,
    read_tool,
]

TOOL_MAP: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}


def register_tools(
    tool_names: list[str] | None = None,
    registry: InMemoryToolRegistry | None = None,
) -> InMemoryToolRegistry:
    """Register tools by name from config. None or empty = register all.

    Usage in agent_loop:
        registry = register_tools()                          # all tools
        registry = register_tools(["Bash", "Read"])          # only these two
        registry = register_tools(config.tool_names)         # from config

    Available tool names: WebSearch, WebFetch, Browser, Bash, CodeInterpreter, Read
    """
    if registry is None:
        registry = InMemoryToolRegistry()
    if not tool_names:
        tools = ALL_TOOLS
    else:
        tools = []
        for name in tool_names:
            if name not in TOOL_MAP:
                raise ValueError(f"Unknown tool '{name}'. Available: {list(TOOL_MAP.keys())}")
            tools.append(TOOL_MAP[name])
    for t in tools:
        registry.register(t, replace=True)
    return registry
