# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Processor that appends extra guidance to the existing system message.

Used for tau2-bench where tau2 injects its own system prompt (full retail
policy + tools). We must NOT replace it with TemplateSystemPromptBuilder.

This processor runs on_task_start: it reads tau2's system message from
state.raw_messages, appends our guidance text to it, and sets
task_system_prompt on the TaskStartEvent so the runloop uses this
combined prompt for all steps (without touching before_model messages).
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from pathlib import Path

from harnessx import MultiHookProcessor


class SystemAppendProcessor(MultiHookProcessor):
    """Append a markdown guidance file to the existing system message.

    Operates on on_task_start so task_system_prompt is set once and frozen,
    matching HarnessX's dual-track invariants.

    Args:
        append_path: Absolute path to the .md file whose text is appended.
    """

    def __init__(self, append_path: str) -> None:
        self._text = Path(append_path).read_text(encoding="utf-8").strip()

    async def on_task_start(self, event):
        # Find tau2's system message from state.raw_messages.
        state = event.state
        existing_system = ""
        if state is not None:
            for msg in getattr(state, "raw_messages", []):
                if getattr(msg, "role", "") == "system":
                    content = getattr(msg, "content", "")
                    existing_system = content if isinstance(content, str) else str(content)
                    break

        combined = existing_system.rstrip() + "\n\n---\n\n" + self._text if existing_system else self._text

        # Set task_system_prompt so the runloop injects it at every step.
        # When task_system_prompt is non-empty, the runloop strips any stale
        # system message from state.messages and re-inserts this one.
        yield _dc_replace(event, system_prompt=combined)
