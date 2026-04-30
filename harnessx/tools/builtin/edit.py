# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from ..base import tool

_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "The path to the file to edit"},
        "old_string": {"type": "string", "description": "The text to replace"},
        "new_string": {"type": "string", "description": "The replacement text"},
        "replace_all": {
            "type": "boolean",
            "description": "Replace all occurrences (default false)",
        },
    },
    "required": ["file_path", "old_string", "new_string"],
}


@tool(
    name="Edit",
    description="Make targeted edits to a file by replacing old_string with new_string.",
    input_schema=_SCHEMA,
)
async def edit_tool(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    from ...sandbox.base import get_current_sandbox

    sandbox = get_current_sandbox()
    resolved = sandbox.resolve(file_path) if sandbox is not None else file_path
    try:
        if sandbox is not None:
            content = await sandbox.read_file(resolved)
        else:
            with open(resolved, "r", encoding="utf-8") as f:
                content = f.read()
        if old_string not in content:
            return f"Error: old_string not found in {file_path}"
        if replace_all:
            new_content = content.replace(old_string, new_string)
            count = content.count(old_string)
        else:
            new_content = content.replace(old_string, new_string, 1)
            count = 1
        if sandbox is not None:
            await sandbox.write_file(resolved, new_content)
        else:
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(new_content)
        return f"Replaced {count} occurrence(s) in {file_path}"
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error: {e}"
