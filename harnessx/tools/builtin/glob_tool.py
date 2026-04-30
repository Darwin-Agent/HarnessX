# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from ..base import tool

_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
        "path": {
            "type": "string",
            "description": "The directory to search in (default: current directory)",
        },
    },
    "required": ["pattern"],
}


@tool(
    name="Glob",
    description="Find files matching a glob pattern. Returns matching file paths sorted by modification time.",
    input_schema=_SCHEMA,
)
async def glob_tool(pattern: str, path: str = ".") -> str:
    # Resolve base: relative paths are anchored to sandbox workspace_path when active.
    # File enumeration uses the Python stdlib directly (richer than sandbox.glob_files).
    from ...sandbox.base import get_current_sandbox

    sandbox = get_current_sandbox()
    if not Path(path).is_absolute() and sandbox is not None:
        base = Path(sandbox.workspace_path) / path
    else:
        base = Path(path)
    base = base.resolve()
    try:
        matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return "No files found matching pattern."
        return "\n".join(str(m) for m in matches)
    except Exception as e:
        return f"Error: {e}"
