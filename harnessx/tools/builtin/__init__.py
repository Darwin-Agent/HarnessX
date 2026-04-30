# Claude-native aligned tools — sandbox-aware via get_current_sandbox() ContextVar.
# When Harness.run() is active, all tools route through the configured sandbox
# (path resolution, cwd, read/write).  Without a sandbox they fall back to direct
# os / subprocess calls.  Web tools (web_fetch, browser) added separately.
from .bash import bash_tool
from .read import read_tool
from .write import write_tool
from .edit import edit_tool
from .glob_tool import glob_tool
from .grep_tool import grep_tool
from .web_search import web_search_tool
from .web_fetch import web_fetch_tool
from .browser import browser_tool
from ..spawn_subagent import spawn_subagent_tool

__all__ = [
    "bash_tool",
    "read_tool",
    "write_tool",
    "edit_tool",
    "glob_tool",
    "grep_tool",
    "web_search_tool",
    "web_fetch_tool",
    "browser_tool",
    "spawn_subagent_tool",
    "build_web_tools",
    "build_gaia_tools",
    "build_gaia_tools_qw",
    "build_gaia_tools_full",
    "build_default_tools",
]


_GAIA_ALLOWED_MODULES = {
    "math",
    "statistics",
    "json",
    "re",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "csv",
    "decimal",
    "fractions",
    "operator",
    "string",
    "textwrap",
    "unicodedata",
    "hashlib",
    "base64",
    "struct",
    "calendar",
    "time",
    "random",
    "bisect",
    "heapq",
    "copy",
    "pprint",
    "difflib",
    "html",
    "urllib",
    "io",
    "typing",
    "dataclasses",
    "enum",
    "numbers",
    "cmath",
    "abc",
}


def build_web_tools():
    """
    Registry with web_search + web_fetch + browser tools.
    Use for deep research and web-capable agents.
    """
    from ..inmemory import InMemoryToolRegistry

    registry = InMemoryToolRegistry()
    for t in [web_search_tool, web_fetch_tool, browser_tool]:
        registry.register(t)
    return registry


def build_gaia_tools():
    """
    Registry with web tools + full filesystem tool set.
    Use for GAIA benchmark where tasks may include file attachments
    (DOCX, XLSX, PDF, CSV, PPTX, images, etc.) AND where the agent
    needs to draft intermediate notes or navigate its workspace.
    """
    from ..inmemory import InMemoryToolRegistry

    registry = InMemoryToolRegistry()
    for t in [
        web_search_tool,
        web_fetch_tool,
        browser_tool,
        read_tool,
        write_tool,
        edit_tool,
        glob_tool,
        grep_tool,
        bash_tool,
    ]:
        registry.register(t)

    return registry


def build_gaia_tools_qw():
    """
    Registry for small Qwen models: web tools + full filesystem tool set.

    Mirrors build_gaia_tools() (no code_interpreter — small Qwen models
    reject that tool name). Filesystem tools let the agent open attached
    files (DOCX, XLSX, PDF, CSV, PPTX, images) and keep intermediate notes.
    """
    from ..inmemory import InMemoryToolRegistry

    registry = InMemoryToolRegistry()
    for t in [
        web_search_tool,
        web_fetch_tool,
        browser_tool,
        read_tool,
        write_tool,
        edit_tool,
        glob_tool,
        grep_tool,
        bash_tool,
    ]:
        registry.register(t)

    return registry


def build_gaia_tools_full():
    """
    Extended GAIA tool set: web tools + file tools + code interpreter.
    Use for GAIA benchmark where tasks require computation, data analysis,
    file processing (DOCX, XLSX, PDF, CSV, etc.), and web research.
    """
    from ..inmemory import InMemoryToolRegistry

    registry = InMemoryToolRegistry()
    for t in [web_search_tool, web_fetch_tool, browser_tool, read_tool, bash_tool]:
        registry.register(t)

    return registry


def build_default_tools():
    """
    Full default tool set: filesystem (sandbox-aware) + web tools.
    bash, read, write, edit, glob, grep + web_search, web_fetch, browser.

    All filesystem tools route through the active sandbox (set by Harness.run via
    get_current_sandbox()).  Without a sandbox they fall back to direct os/subprocess
    calls.  Use for CLI agents and HarnessDescriptors that need the full tool set.
    """
    from ..inmemory import InMemoryToolRegistry

    registry = InMemoryToolRegistry()
    for t in [
        bash_tool,
        read_tool,
        write_tool,
        edit_tool,
        glob_tool,
        grep_tool,
        web_search_tool,
        web_fetch_tool,
        browser_tool,
        spawn_subagent_tool,
    ]:
        registry.register(t)
    return registry
