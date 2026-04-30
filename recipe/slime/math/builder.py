# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from harnessx.rl.task import RLTask

if TYPE_CHECKING:
    from slime.utils.types import Sample


class MathTaskBuilder:
    """Builds an RLTask for math problem solving.

    Handles two sample prompt formats:
        str:       plain problem text (dapo-math-17k)
        list[dict]: OpenAI-format messages (AIME eval)
                    [{"role": "system", ...}, {"role": "user", ...}]

    System prompt source of truth: _MATH_SYSTEM_PROMPT in registry.py.
    AIME samples may override via task.metadata["system_prompt"].
    """

    def build(self, sample: "Sample") -> RLTask:
        raw_prompt: Any = sample.prompt
        label: str = sample.label or ""
        metadata: dict = {}

        if isinstance(raw_prompt, list):
            # AIME format: messages list with optional system + user roles
            description = next((m["content"] for m in raw_prompt if m.get("role") == "user"), "")
            # Preserve sample-level system prompt (overrides spec default)
            sys_content = next((m["content"] for m in raw_prompt if m.get("role") == "system"), "")
            if sys_content:
                metadata["system_prompt"] = sys_content
        else:
            description = raw_prompt or ""

        # max_steps is intentionally absent — generate() overwrites it from
        # spec.max_steps so the spec registry is the single source of truth.
        return RLTask(
            description=description,
            label=label,
            task_type="math",
            metadata=metadata,
        )
