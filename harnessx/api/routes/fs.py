# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(tags=["filesystem"])

# File extensions that may be written via the API
WRITABLE_EXTENSIONS = {".md", ".txt", ".yaml", ".yml", ".json", ".toml"}
# Max file size for reading (512 KB)
MAX_READ_BYTES = 512 * 1024


class FsEntry(BaseModel):
    name: str
    type: str  # "file" | "dir"
    size: int
    mtime: str  # ISO-8601


class FsListResponse(BaseModel):
    path: str
    entries: list[FsEntry]


class FsFileResponse(BaseModel):
    path: str
    content: str


class FsWriteRequest(BaseModel):
    path: str
    content: str


def _resolve(raw: str) -> Path:
    """Expand ~ and resolve to absolute path; reject traversal attempts."""
    expanded = os.path.expanduser(raw)
    resolved = Path(expanded).resolve()
    return resolved


def _entry(p: Path) -> FsEntry:
    st = p.stat()
    return FsEntry(
        name=p.name,
        type="dir" if stat.S_ISDIR(st.st_mode) else "file",
        size=st.st_size,
        mtime=datetime.fromtimestamp(st.st_mtime).isoformat(),
    )


@router.get("/fs", response_model=FsListResponse)
async def list_directory(
    path: str = Query(..., description="Absolute or ~-relative path"),
) -> Any:
    """List directory entries at the given path."""
    resolved = _resolve(path)
    if not resolved.exists():
        raise HTTPException(404, f"Path does not exist: {path}")
    if not resolved.is_dir():
        raise HTTPException(400, f"Not a directory: {path}")

    try:
        entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        return FsListResponse(
            path=str(resolved),
            entries=[_entry(e) for e in entries],
        )
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {path}")


@router.get("/fs/file", response_model=FsFileResponse)
async def read_file(
    path: str = Query(..., description="Absolute or ~-relative path"),
) -> Any:
    """Read text file content."""
    resolved = _resolve(path)
    if not resolved.exists():
        raise HTTPException(404, f"File does not exist: {path}")
    if not resolved.is_file():
        raise HTTPException(400, f"Not a file: {path}")

    size = resolved.stat().st_size
    if size > MAX_READ_BYTES:
        raise HTTPException(413, f"File too large ({size} bytes); limit is {MAX_READ_BYTES}")

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
        return FsFileResponse(path=str(resolved), content=content)
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {path}")


@router.put("/fs/file", response_model=FsFileResponse)
async def write_file(req: FsWriteRequest) -> Any:
    """Write text content to a file.  Only certain extensions are allowed."""
    resolved = _resolve(req.path)

    if resolved.suffix.lower() not in WRITABLE_EXTENSIONS:
        raise HTTPException(
            400,
            f"Writing to '{resolved.suffix}' files is not allowed. Allowed: {', '.join(sorted(WRITABLE_EXTENSIONS))}",
        )

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(req.content, encoding="utf-8")
        return FsFileResponse(path=str(resolved), content=req.content)
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {req.path}")
