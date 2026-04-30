# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["help"])

# Resolve docs root relative to this file: harnessx/api/routes/help.py → docs/
DOCS_ROOT = Path(__file__).parents[3] / "docs"

# Display names and rendering order for top-level sections
_SECTION_LABELS: dict[str, str] = {
    "guide": "Guide",
    "concepts": "Concepts",
    "feats": "Features",
    "recipes": "Recipes",
}
_SECTION_ORDER = ["guide", "concepts", "feats", "recipes"]


# ── Models ────────────────────────────────────────────────────────────────────


class DocEntry(BaseModel):
    path: str  # relative path without .md, e.g. "feats/plugins"
    title: str  # first H1 heading or humanised filename


class DocSection(BaseModel):
    name: str
    items: list[DocEntry]


class DocTree(BaseModel):
    sections: list[DocSection]


class DocContent(BaseModel):
    path: str
    title: str
    content: str  # raw markdown


# ── Helpers ───────────────────────────────────────────────────────────────────


def _first_h1(md_file: Path) -> str:
    """Extract the first # heading from a markdown file."""
    try:
        for line in md_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
    except Exception:
        pass
    return md_file.stem.replace("-", " ").replace("_", " ").title()


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/help", response_model=DocTree)
async def get_doc_tree() -> Any:
    """Return the full documentation tree grouped by section."""
    if not DOCS_ROOT.is_dir():
        return DocTree(sections=[])

    sections_map: dict[str, list[DocEntry]] = {}
    root_items: list[DocEntry] = []

    for md_file in sorted(DOCS_ROOT.rglob("*.md")):
        rel = md_file.relative_to(DOCS_ROOT)
        parts = rel.parts
        path = str(rel.with_suffix(""))  # e.g. "guide/quickstart"
        entry = DocEntry(path=path, title=_first_h1(md_file))

        if len(parts) == 1:
            root_items.append(entry)
        else:
            sections_map.setdefault(parts[0], []).append(entry)

    sections: list[DocSection] = []
    for key in _SECTION_ORDER:
        if key in sections_map:
            sections.append(
                DocSection(
                    name=_SECTION_LABELS.get(key, key.title()),
                    items=sections_map[key],
                )
            )
    for key, items in sections_map.items():
        if key not in _SECTION_ORDER:
            sections.append(DocSection(name=key.title(), items=items))
    if root_items:
        sections.append(DocSection(name="Reference", items=root_items))

    return DocTree(sections=sections)


@router.get("/help/{path:path}", response_model=DocContent)
async def get_doc_content(path: str) -> Any:
    """Return raw markdown content for a doc entry."""
    clean = path.strip("/")
    if ".." in clean:
        raise HTTPException(400, "Invalid path")

    md_file = (DOCS_ROOT / clean).with_suffix(".md")
    if not md_file.is_file():
        raise HTTPException(404, f"Doc not found: {path}")

    try:
        content = md_file.read_text(encoding="utf-8")
    except Exception:
        raise HTTPException(500, "Failed to read doc")

    return DocContent(path=clean, title=_first_h1(md_file), content=content)
