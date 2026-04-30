# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
from datetime import datetime, timezone

from .types import MemoryDocument


def _parse_iso(s: str) -> datetime | None:
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _negate_iso(s: str) -> str:
    """Return a string that sorts in the reverse order of ``s``.

    Used as a sort key so that *descending* timestamp order can be expressed
    in a tuple-based ascending sort.
    """
    return "".join(chr(0x10FFFF - ord(c)) for c in s)


def compute_decayed_importance(
    importance: float,
    last_accessed_at: str,
    half_life_days: int,
    now: datetime | None = None,
) -> float:
    if now is None:
        now = datetime.now(timezone.utc)
    last = _parse_iso(last_accessed_at)
    if last is None:
        return importance
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    days = max(0.0, (now - last).total_seconds() / 86400)
    effective = max(1, half_life_days)
    return importance * math.exp(-math.log(2) * days / effective)


def sort_by_decayed_importance(
    documents: list[MemoryDocument],
    half_life_days: int,
    now: datetime | None = None,
) -> list[tuple[MemoryDocument, float]]:
    items = [
        (
            doc,
            compute_decayed_importance(
                doc.frontmatter.importance,
                doc.frontmatter.last_accessed_at,
                half_life_days,
                now,
            ),
        )
        for doc in documents
    ]
    # Sort by decayed importance desc, then updated_at desc (most recent first)
    # Negate the ISO timestamp comparison by reversing the string via a helper
    items.sort(key=lambda x: (-x[1], _negate_iso(x[0].frontmatter.updated_at)))
    return items


def find_decayed_memories(
    documents: list[MemoryDocument],
    half_life_days: int,
    threshold: float = 0.05,
    now: datetime | None = None,
) -> list[MemoryDocument]:
    return [
        doc
        for doc in documents
        if doc.frontmatter.status == "active"
        and compute_decayed_importance(
            doc.frontmatter.importance,
            doc.frontmatter.last_accessed_at,
            half_life_days,
            now,
        )
        < threshold
    ]
