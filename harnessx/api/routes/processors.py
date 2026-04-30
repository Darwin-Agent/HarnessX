# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from harnessx.api.custom_processors.direct_targets import (
    import_processor_from_content,
    import_processor_from_path,
    list_custom_processors,
    remove_custom_processor,
    scan_path_for_processors,
    scan_text_for_processors,
    test_processor_from_content,
    test_processor_from_path,
)

router = APIRouter(tags=["custom_processors"])


class ScanPathRequest(BaseModel):
    path: str


class ScanFileRequest(BaseModel):
    filename: str
    content: str


class ProcessorCandidate(BaseModel):
    class_name: str
    label: str
    file_path: str
    doc: str = ""


class ScanResponse(BaseModel):
    candidates: list[ProcessorCandidate]


class CustomProcessorInfo(BaseModel):
    id: str
    label: str
    class_name: str
    target: str
    source_path: str
    installed_path: str


class TestImportRequest(BaseModel):
    mode: Literal["path", "file"]
    # path mode
    path: str | None = None
    file_path: str | None = None
    # file mode
    filename: str | None = None
    content: str | None = None
    # shared
    class_name: str


class TestImportResponse(BaseModel):
    ok: bool
    instantiable: bool
    required_args: list[str]
    message: str


class ImportRequest(BaseModel):
    mode: Literal["path", "file"]
    # path mode
    path: str | None = None
    file_path: str | None = None
    # file mode
    filename: str | None = None
    content: str | None = None
    # shared
    class_name: str
    label: str | None = None


def _resolve_scan_file_from_path(path: str, file_path: str | None) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise HTTPException(404, f"Path does not exist: {path}")
    if root.is_file():
        if root.suffix != ".py":
            raise HTTPException(400, "Path must point to a .py file or a directory")
        return root

    # directory mode
    if not file_path:
        raise HTTPException(400, "file_path is required when path is a directory")
    candidate = Path(file_path).expanduser().resolve()
    try:
        candidate.relative_to(root)
    except Exception as exc:
        raise HTTPException(400, "file_path must be inside the provided directory") from exc
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(404, f"file_path does not exist: {file_path}")
    if candidate.suffix != ".py":
        raise HTTPException(400, "file_path must be a .py file")
    return candidate


@router.get("/processors/custom", response_model=list[CustomProcessorInfo])
async def list_processors() -> Any:
    rows = list_custom_processors()
    return [
        CustomProcessorInfo(
            id=str(r.get("id", "")),
            label=str(r.get("label", "")),
            class_name=str(r.get("class_name", "")),
            target=str(r.get("target", "")),
            source_path=str(r.get("source_path", "")),
            installed_path=str(r.get("installed_path", "")),
        )
        for r in rows
    ]


@router.post("/processors/scan-path", response_model=ScanResponse)
async def scan_path(req: ScanPathRequest) -> Any:
    try:
        candidates = scan_path_for_processors(req.path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"Failed to scan path: {exc}")
    return ScanResponse(candidates=[ProcessorCandidate(**c) for c in candidates])


@router.post("/processors/scan-file", response_model=ScanResponse)
async def scan_file(req: ScanFileRequest) -> Any:
    try:
        candidates = scan_text_for_processors(req.filename, req.content)
    except Exception as exc:
        raise HTTPException(400, f"Failed to scan file content: {exc}")
    return ScanResponse(candidates=[ProcessorCandidate(**c) for c in candidates])


@router.post("/processors/test", response_model=TestImportResponse)
async def test_import(req: TestImportRequest) -> Any:
    try:
        if req.mode == "path":
            if not req.path:
                raise HTTPException(400, "path is required for mode=path")
            src = _resolve_scan_file_from_path(req.path, req.file_path)
            res = test_processor_from_path(str(src), req.class_name)
        else:
            if not req.content:
                raise HTTPException(400, "content is required for mode=file")
            res = test_processor_from_content(req.filename or "uploaded_processor.py", req.content, req.class_name)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Import test failed: {type(exc).__name__}: {exc}")
    return TestImportResponse(
        ok=bool(res.get("ok")),
        instantiable=bool(res.get("instantiable")),
        required_args=list(res.get("required_args", [])),
        message=str(res.get("message", "")),
    )


@router.post("/processors/import", response_model=CustomProcessorInfo, status_code=201)
async def import_processor(req: ImportRequest) -> Any:
    try:
        if req.mode == "path":
            if not req.path:
                raise HTTPException(400, "path is required for mode=path")
            src = _resolve_scan_file_from_path(req.path, req.file_path)
            row = import_processor_from_path(str(src), req.class_name, label=req.label)
        else:
            if not req.content:
                raise HTTPException(400, "content is required for mode=file")
            row = import_processor_from_content(
                req.filename or "uploaded_processor.py",
                req.content,
                req.class_name,
                label=req.label,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Import failed: {type(exc).__name__}: {exc}")

    return CustomProcessorInfo(
        id=str(row.get("id", "")),
        label=str(row.get("label", "")),
        class_name=str(row.get("class_name", "")),
        target=str(row.get("target", "")),
        source_path=str(row.get("source_path", "")),
        installed_path=str(row.get("installed_path", "")),
    )


@router.delete("/processors/custom/{processor_id}", status_code=204)
async def delete_processor(processor_id: str) -> None:
    ok = remove_custom_processor(processor_id)
    if not ok:
        raise HTTPException(404, f"Processor not found: {processor_id}")
