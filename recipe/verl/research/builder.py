# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Task builder for GAIA-style research QA."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from harnessx.rl.task import RLTask
from recipe.verl.research.prompt import QUESTION_PROMPT

if TYPE_CHECKING:
    pass


class ResearchTaskBuilder:
    """Builds an RLTask for GAIA-style research QA.

    Handles two sample prompt formats:
        str:        plain question text
        list[dict]: OpenAI-format messages [{"role": "user", ...}]
    """

    def build(self, sample: Any) -> RLTask:
        raw_prompt = sample.prompt
        label: str = getattr(sample, "label", "") or ""
        metadata: dict = {}

        if isinstance(raw_prompt, list):
            description = next(
                (m["content"] for m in raw_prompt if m.get("role") == "user"), ""
            )
            sys_content = next(
                (m["content"] for m in raw_prompt if m.get("role") == "system"), ""
            )
            if sys_content:
                metadata["system_prompt"] = sys_content
        else:
            description = raw_prompt or ""

        return RLTask(
            description=QUESTION_PROMPT + description,
            label=label,
            task_type="research",
            metadata=metadata,
        )
