# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Machine-computed signature for candidate deduplication.

Agents cannot forge this — it is derived purely from file_changes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class FileChange:
    path: str
    diff_sha_after: str


def compute_signature(changes: list[FileChange]) -> str:
    sorted_pairs = sorted((c.path, c.diff_sha_after) for c in changes)
    joined = "\n".join(f"{p}\t{s}" for p, s in sorted_pairs)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
