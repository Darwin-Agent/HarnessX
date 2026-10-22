# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

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
_SECTION_LABELS_ZH = {
    "guide": "指南",
    "concepts": "核心概念",
    "feats": "功能",
    "recipes": "配方",
}
_RECIPE_PAGES = {
    "recipes/coevolution": "coevolution",
    "recipes/verl": "verl",
}


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


def _localized_file(md_file: Path, lang: str) -> Path:
    if lang != "zh":
        return md_file
    candidate = md_file.with_name(f"{md_file.stem}.zh-CN.md")
    return candidate if candidate.is_file() else md_file


def _recipe_file(recipe: str, lang: str) -> Path:
    root = DOCS_ROOT.parent / "recipe" / recipe
    return _localized_file(root / "README.md", lang)


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/help", response_model=DocTree)
async def get_doc_tree(lang: Literal["en", "zh"] = "en") -> Any:
    """Return the full documentation tree grouped by section."""
    if not DOCS_ROOT.is_dir():
        return DocTree(sections=[])

    sections_map: dict[str, list[DocEntry]] = {}
    root_items: list[DocEntry] = []

    for md_file in sorted(DOCS_ROOT.rglob("*.md")):
        rel = md_file.relative_to(DOCS_ROOT)
        if md_file.stem.endswith(".zh-CN") or rel.parts[0] == "gateway" or rel.name == "pr18-remediation.md":
            continue
        parts = rel.parts
        path = str(rel.with_suffix(""))  # e.g. "guide/quickstart"
        entry = DocEntry(path=path, title=_first_h1(_localized_file(md_file, lang)))

        if len(parts) == 1:
            root_items.append(entry)
        else:
            sections_map.setdefault(parts[0], []).append(entry)

    for path, recipe in _RECIPE_PAGES.items():
        md_file = _recipe_file(recipe, lang)
        if md_file.is_file():
            sections_map.setdefault("recipes", []).append(DocEntry(path=path, title=_first_h1(md_file)))

    labels = _SECTION_LABELS_ZH if lang == "zh" else _SECTION_LABELS
    sections: list[DocSection] = []
    for key in _SECTION_ORDER:
        if key in sections_map:
            sections.append(
                DocSection(
                    name=labels.get(key, key.title()),
                    items=sections_map[key],
                )
            )
    for key, items in sections_map.items():
        if key not in _SECTION_ORDER:
            sections.append(DocSection(name=key.title(), items=items))
    if root_items:
        sections.append(DocSection(name="参考" if lang == "zh" else "Reference", items=root_items))

    return DocTree(sections=sections)


@router.get("/help/{path:path}", response_model=DocContent)
async def get_doc_content(path: str, lang: Literal["en", "zh"] = "en") -> Any:
    """Return raw markdown content for a doc entry."""
    clean = path.strip("/").removesuffix(".md").removesuffix(".zh-CN")
    if ".." in clean:
        raise HTTPException(400, "Invalid path")

    md_file = (
        _recipe_file(_RECIPE_PAGES[clean], lang)
        if clean in _RECIPE_PAGES
        else _localized_file(DOCS_ROOT / f"{clean}.md", lang)
    )
    if not md_file.is_file():
        raise HTTPException(404, f"Doc not found: {path}")

    try:
        content = md_file.read_text(encoding="utf-8")
    except Exception:
        raise HTTPException(500, "Failed to read doc")

    return DocContent(path=clean, title=_first_h1(md_file), content=content)
