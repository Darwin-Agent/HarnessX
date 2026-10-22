# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Task routing registry for verl backend."""

from __future__ import annotations

from typing import Any

from recipe.verl.spec import VerlConfigSpec
from recipe.verl.research.registry import RESEARCH_CONFIGS

HARNESS_CONFIGS: dict[str, VerlConfigSpec] = {
    **RESEARCH_CONFIGS,
}


def load_harness_config(sample: Any) -> VerlConfigSpec:
    """Route a sample to its VerlConfigSpec based on task_type metadata."""
    metadata = getattr(sample, "metadata", {}) or {}
    task_type = metadata.get("task_type", "research")
    return HARNESS_CONFIGS.get(task_type, HARNESS_CONFIGS["research"])
