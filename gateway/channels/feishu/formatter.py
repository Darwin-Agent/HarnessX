from __future__ import annotations

import json
import re
from typing import Any

from .constants import FEISHU_MAX_TEXT_LEN


def text_content(text: str) -> str:
    """Build feishu text message content JSON string."""
    return json.dumps({"text": text[:FEISHU_MAX_TEXT_LEN]}, ensure_ascii=False)


def normalize_feishu_md(text: str) -> str:
    """
    Light markdown normalization for Feishu rendering.
    """
    if not text or not text.strip():
        return text
    # Ensure newline before code fence to avoid parse glitches.
    return re.sub(r"([^\n])(```)", r"\1\n\2", text)


def build_post_md_content(text: str) -> str:
    """
    Build Feishu `post` payload with markdown text.
    """
    body = normalize_feishu_md((text or "")[:FEISHU_MAX_TEXT_LEN])
    content_rows: list[list[dict[str, str]]] = []
    if body:
        content_rows.append([{"tag": "md", "text": body}])
    if not content_rows:
        content_rows = [[{"tag": "md", "text": "[empty]"}]]
    post = {"zh_cn": {"content": content_rows}}
    return json.dumps(post, ensure_ascii=False)


def has_markdown_table(text: str) -> bool:
    """
    Heuristic: detect a markdown table block by leading pipe lines.
    """
    if not text:
        return False
    return bool(re.search(r"^\s*\|", text, flags=re.MULTILINE))


def _parse_md_table(table_lines: list[str]) -> dict[str, Any] | None:
    """
    Parse GFM table lines into a Feishu interactive table element.
    """
    lines = [ln for ln in table_lines if ln.strip()]
    if len(lines) < 2:
        return None
    sep_idx = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*\|[\s\-\:\|]+\|\s*$", ln):
            sep_idx = i
            break
    if sep_idx is None or sep_idx == 0:
        return None

    def _split_row(line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [c.strip() for c in stripped.split("|")]

    headers = _split_row(lines[0])
    if not headers:
        return None

    col_keys = [f"col_{i}" for i in range(len(headers))]
    columns = [
        {
            "name": col_keys[i],
            "display_name": headers[i] if headers[i] else f"Column {i + 1}",
            "data_type": "text",
        }
        for i in range(len(headers))
    ]
    rows: list[dict[str, str]] = []
    for line in lines[sep_idx + 1 :]:
        cells = _split_row(line)
        row: dict[str, str] = {}
        for i, key in enumerate(col_keys):
            cell_text = cells[i] if i < len(cells) else ""
            cell_text = re.sub(r"[*_]{1,2}(.+?)[*_]{1,2}", r"\1", cell_text)
            row[key] = cell_text
        rows.append(row)
    if not rows:
        return None
    return {
        "tag": "table",
        "page_size": min(max(len(rows), 10), 50),
        "columns": columns,
        "rows": rows,
    }


def _convert_md_headings_to_bold(text: str) -> str:
    """
    Convert markdown headings to bold text for Feishu card markdown blocks.
    """
    return re.sub(r"^#{1,6}\s+(.+)$", r"**\1**", text, flags=re.MULTILINE)


def _build_elements(text: str) -> list[dict[str, Any]]:
    lines = text.split("\n")
    elements: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*\|", line):
            table_block: list[str] = []
            while i < len(lines) and re.match(r"^\s*\|", lines[i]):
                table_block.append(lines[i])
                i += 1
            table_elem = _parse_md_table(table_block)
            if table_elem:
                elements.append(table_elem)
            else:
                elements.append(
                    {
                        "tag": "markdown",
                        "content": _convert_md_headings_to_bold("\n".join(table_block)),
                    }
                )
            continue

        text_block: list[str] = []
        while i < len(lines) and not re.match(r"^\s*\|", lines[i]):
            text_block.append(lines[i])
            i += 1
        content = "\n".join(text_block).strip()
        if content:
            elements.append(
                {
                    "tag": "markdown",
                    "content": _convert_md_headings_to_bold(content),
                }
            )
    if not elements:
        elements = [
            {
                "tag": "markdown",
                "content": _convert_md_headings_to_bold(text),
            }
        ]
    return elements


def build_interactive_content(text: str) -> str:
    """
    Build an interactive card body JSON with markdown and native table elements.
    """
    body = (text or "")[:FEISHU_MAX_TEXT_LEN]
    return json.dumps({"elements": _build_elements(body)}, ensure_ascii=False)


def build_text_payload_candidates(
    text: str,
    *,
    prefer_interactive_table: bool = True,
    allow_interactive: bool = True,
) -> list[tuple[str, str]]:
    """
    Build outbound text payload candidates in fallback order.

    Order:
    1) interactive (only when markdown table exists and allowed)
    2) post (md)
    3) text (plain)
    """
    body = text or ""
    candidates: list[tuple[str, str]] = []
    if allow_interactive:
        candidates.append(("interactive", build_interactive_content(body)))
    candidates.append(("post", build_post_md_content(body)))
    candidates.append(("text", text_content(body)))
    return candidates


def card_content(text: str, title: str = "") -> str:
    """Build a minimal interactive card with markdown body."""
    card: dict = {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "body": {"elements": [{"tag": "markdown", "content": text[:FEISHU_MAX_TEXT_LEN]}]},
    }
    if title:
        card["header"] = {
            "template": "blue",
            "title": {"tag": "plain_text", "content": title[:100]},
        }
    return json.dumps(card, ensure_ascii=False)
