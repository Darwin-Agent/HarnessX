# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .types import PluginConfig


def _format_date(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"


def _format_time(dt: datetime) -> str:
    return f"{_format_date(dt)} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"


def get_daily_file_path(cfg: PluginConfig, date: datetime | None = None) -> str:
    if date is None:
        date = datetime.now(timezone.utc)
    return os.path.join(cfg.memory_root, "daily", f"{_format_date(date)}.md")


def append_to_daily(
    cfg: PluginConfig,
    entry: str,
    date: datetime | None = None,
) -> tuple[str, str]:
    """Append entry to daily log. Returns (relative_path, line_range)."""
    if date is None:
        date = datetime.now(timezone.utc)
    daily_path = get_daily_file_path(cfg, date)
    os.makedirs(os.path.dirname(daily_path), exist_ok=True)

    timestamp = _format_time(date)
    block = f"\n### [{timestamp}]\n\n{entry.strip()}\n"

    p = Path(daily_path)
    if not p.exists():
        header = f"# Daily Observations — {_format_date(date)}\n"
        content = header + block
        p.write_text(content, encoding="utf-8")
        start_line = header.count("\n") + 1
    else:
        existing = p.read_text(encoding="utf-8")
        start_line = existing.count("\n") + 1
        with p.open("a", encoding="utf-8") as f:
            f.write(block)

    end_line = start_line + block.count("\n")
    rel = os.path.relpath(daily_path, cfg.memory_root).replace(os.sep, "/")
    return rel, f"{rel}#L{start_line}-L{end_line}"


def read_recent_daily_entries(
    cfg: PluginConfig,
    days: int = 3,
    now: datetime | None = None,
) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    entries: list[str] = []
    for i in range(days):
        dt = now - timedelta(days=i)
        path = get_daily_file_path(cfg, dt)
        if os.path.isfile(path):
            entries.append(Path(path).read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(entries)
