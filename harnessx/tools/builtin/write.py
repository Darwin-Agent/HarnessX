# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os

from ..base import tool

_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "The absolute path to the file to write",
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file",
        },
    },
    "required": ["file_path", "content"],
}


@tool(
    name="Write",
    description="Write a file to the local filesystem. Creates parent directories as needed.",
    input_schema=_SCHEMA,
)
async def write_tool(file_path: str, content: str) -> str:
    from ...sandbox.base import get_current_sandbox

    sandbox = get_current_sandbox()
    resolved = sandbox.resolve(file_path) if sandbox is not None else os.path.abspath(file_path)
    try:
        if sandbox is not None:
            await sandbox.write_file(resolved, content)
        else:
            os.makedirs(os.path.dirname(resolved), exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
        return f"File written successfully to {file_path}"
    except Exception as e:
        return f"Error: {e}"
