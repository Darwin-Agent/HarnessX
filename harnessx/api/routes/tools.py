# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from fastapi import APIRouter

from harnessx.api.models import ToolInfo

router = APIRouter()

_TOOLS: list[ToolInfo] = [
    # Filesystem (sandbox-aware)
    ToolInfo(name="Bash", group="filesystem", description="Run shell commands via sandbox"),
    ToolInfo(
        name="Read",
        group="filesystem",
        description="Read files (text, PDF, DOCX, XLSX)",
    ),
    ToolInfo(name="Write", group="filesystem", description="Write or overwrite a file"),
    ToolInfo(
        name="Edit",
        group="filesystem",
        description="Exact string replacement in a file",
    ),
    ToolInfo(name="Glob", group="filesystem", description="Find files by glob pattern"),
    ToolInfo(name="Grep", group="filesystem", description="Search file contents with regex"),
    # Web
    ToolInfo(name="WebSearch", group="web", description="Search the web (SerpAPI / Tavily)"),
    ToolInfo(name="WebFetch", group="web", description="Fetch and parse a URL"),
    ToolInfo(name="Browser", group="web", description="Full browser automation (Playwright)"),
]


@router.get("/tools", response_model=list[ToolInfo])
async def get_tools():
    """Return the list of built-in tools with their group and description."""
    return _TOOLS
