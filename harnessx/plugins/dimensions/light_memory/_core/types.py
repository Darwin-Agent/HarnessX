# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MemoryType = Literal["style", "profile", "session", "skill", "learning", "entity", "daily"]
MemoryStatus = Literal["active", "deprecated", "archived"]
MemoryConfidence = Literal["high", "medium", "low", "deprecated"]


@dataclass
class PluginConfig:
    memory_root: str
    user_id: str = "user"
    top_k: int = 15
    access_half_life_days: int = 30
    organization_enabled: bool = True
    organization_interval_ms: int = 1_800_000
    organization_timeout_ms: int = 30_000
    decay_enabled: bool = True
    # Git integration
    git_mode: str = "optional"  # "optional" | "required" | "disabled"
    auto_commit: bool = True
    # Recall / capture toggles
    auto_recall: bool = True
    auto_capture: bool = False


@dataclass
class MemoryFrontmatter:
    id: str
    type: MemoryType
    status: MemoryStatus
    confidence: MemoryConfidence
    importance: float
    access_count: int
    skip_count: int
    token_estimate: int
    created_at: str
    updated_at: str
    last_accessed_at: str
    title: str
    summary: str
    keywords: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    source: str = ""
    supersedes: list[str] = field(default_factory=list)
    user_id: str | None = None
    # Optional provenance fields
    project: str | None = None
    source_channel: str | None = None
    source_run_id: str | None = None


@dataclass
class MemoryDocument:
    file_path: str
    relative_path: str
    frontmatter: MemoryFrontmatter
    body: str
    summary_section: str


@dataclass
class GrepHit:
    relative_path: str
    match_count: float
