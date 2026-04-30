# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from .backend import truncate_text
from .types import MemoryDocument, PluginConfig

INDEX_FILENAME = "_index.md"
INDEX_ENTRY_LIMIT = 200


def get_index_file_path(cfg: PluginConfig) -> str:
    return os.path.join(cfg.memory_root, INDEX_FILENAME)


def _atomic_write(file_path: str, content: str) -> None:
    tmp = file_path + ".tmp"
    Path(tmp).write_text(content, encoding="utf-8")
    os.replace(tmp, file_path)


def _section_name(relative_path: str) -> str:
    return relative_path.split("/")[0] or "other"


def _build_entry_line(doc: MemoryDocument) -> str:
    fm = doc.frontmatter
    summary = truncate_text(fm.summary, 50)
    updated = fm.updated_at[:10]
    return f"| {doc.relative_path} | {fm.type} | {summary} | {fm.importance} | {updated} |"


def _group_by_section(
    documents: list[MemoryDocument],
) -> list[tuple[str, list[MemoryDocument]]]:
    groups: dict[str, list[MemoryDocument]] = {}
    for doc in documents:
        sec = _section_name(doc.relative_path)
        groups.setdefault(sec, []).append(doc)

    order = ["user", "knowledge", "entity", "sessions", "daily"]
    result: list[tuple[str, list[MemoryDocument]]] = []
    for sec in order:
        if sec in groups:
            result.append((sec, groups.pop(sec)))
    for sec, docs in groups.items():
        result.append((sec, docs))
    return result


def generate_index_file(cfg: PluginConfig, documents: list[MemoryDocument]) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    effective = documents
    if len(documents) > INDEX_ENTRY_LIMIT:
        effective = [d for d in documents if d.frontmatter.status == "active"]

    grouped = _group_by_section(effective)
    lines = [
        "---",
        f"updated: {now}",
        f"total_files: {len(documents)}",
        "---",
        "",
        "# Memory Index",
    ]

    for section, docs in grouped:
        lines.append("")
        lines.append(f"## {section}")
        lines.append("| file | type | summary | importance | updated |")
        lines.append("|------|------|---------|------------|---------|")
        for doc in docs:
            lines.append(_build_entry_line(doc))

    lines.append("")
    content = "\n".join(lines)
    _atomic_write(get_index_file_path(cfg), content)
    return INDEX_FILENAME


# ── Incremental index updates ────────────────────────────────────────────────

_ENTRY_RE = re.compile(r"^\| ([^|]+) \|")


def _read_raw_index(cfg: "PluginConfig") -> list[str]:
    path = get_index_file_path(cfg)
    try:
        return Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def _write_raw_index(cfg: "PluginConfig", lines: list[str]) -> None:
    _atomic_write(get_index_file_path(cfg), "\n".join(lines) + "\n")


def update_index_entry(cfg: "PluginConfig", document: "MemoryDocument") -> None:
    """Add or replace the index row for *document* without a full rebuild.

    Reads the current ``_index.md``, locates the section for the document's
    top-level directory, updates or inserts the row, and writes the file back.
    If the index file does not exist, falls back to a full rebuild.
    """
    from .backend import get_all_memory_documents

    index_path = get_index_file_path(cfg)
    if not os.path.isfile(index_path):
        all_docs = get_all_memory_documents(cfg)
        generate_index_file(cfg, all_docs)
        return

    new_line = _build_entry_line(document)
    rel = document.relative_path
    section = _section_name(rel)

    lines = _read_raw_index(cfg)
    # Find the section header and its table rows
    section_header = f"## {section}"
    in_section = False
    found = False
    insert_pos: int | None = None

    new_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == section_header:
            in_section = True
            new_lines.append(line)
            i += 1
            continue
        if in_section:
            m = _ENTRY_RE.match(line)
            if m:
                existing_path = m.group(1).strip()
                if existing_path == rel:
                    new_lines.append(new_line)
                    found = True
                    i += 1
                    continue
                else:
                    if insert_pos is None:
                        insert_pos = len(new_lines)
                    new_lines.append(line)
                    i += 1
                    continue
            elif line.startswith("## ") or (
                line.strip() == "" and i + 1 < len(lines) and lines[i + 1].startswith("## ")
            ):
                # Leaving section without having found the entry
                if not found and insert_pos is not None:
                    new_lines.insert(insert_pos + 1, new_line)
                    found = True
                in_section = False
                new_lines.append(line)
                i += 1
                continue
            else:
                new_lines.append(line)
                i += 1
                continue
        else:
            new_lines.append(line)
            i += 1

    if not found:
        # Section didn't exist, or we reached EOF still in the section
        if in_section and insert_pos is not None:
            new_lines.insert(insert_pos + 1, new_line)
            found = True
        if not found:
            # Append a new section
            if new_lines and new_lines[-1].strip():
                new_lines.append("")
            new_lines.append(f"## {section}")
            new_lines.append("| file | type | summary | importance | updated |")
            new_lines.append("|------|------|---------|------------|---------|")
            new_lines.append(new_line)

    _write_raw_index(cfg, new_lines)


def remove_index_entry(cfg: "PluginConfig", relative_path: str) -> None:
    """Remove the index row for *relative_path* if present."""
    index_path = get_index_file_path(cfg)
    if not os.path.isfile(index_path):
        return
    lines = _read_raw_index(cfg)
    new_lines = [
        line
        for line in lines
        if not (_ENTRY_RE.match(line) and _ENTRY_RE.match(line).group(1).strip() == relative_path)  # type: ignore[union-attr]
    ]
    if len(new_lines) != len(lines):
        _write_raw_index(cfg, new_lines)
