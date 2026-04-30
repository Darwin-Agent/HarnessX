# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations


def sp_append(current: str, section: str) -> str:
    """Append *section* to *current* only if an identical block is not already present.

    Strips leading/trailing whitespace from *section* before checking membership
    so minor formatting differences do not defeat deduplication.

    Args:
        current: The accumulated system prompt string so far.
        section: The block of text to append.

    Returns:
        ``current + section`` when the stripped section is not already in
        ``current``, otherwise ``current`` unchanged.
    """
    stripped = section.strip()
    if not stripped:
        return current
    if stripped in current:
        return current
    return current + section
