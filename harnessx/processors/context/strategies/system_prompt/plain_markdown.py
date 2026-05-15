# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Plain-markdown system prompt builder — no templating, no rendering.

Reads the file as-is and returns its content as the system prompt. Use this
when the prompt is pure prose and you want `{{...}}` / `{%...%}` literals
inside the text to be treated as text, not template syntax.

Rationale: `TemplateSystemPromptBuilder` Jinja-renders the template, which
means any `{{foo}}` in the prose crashes the processor at runtime. This
was the R3 incident in aegis_64_v091_r15_v2 where an Evolver-shipped prompt
containing `{{cite tweet}}` (literal Wikipedia template syntax) crashed
SystemPromptProcessor on every task.

When you actually need substitution (e.g. injecting round number into a
meta-agent prompt), use `render_template()` at build time (see
harnessx/aegis/_prompt.py) or `TemplateSystemPromptBuilder` — do NOT use
this builder.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .....workspace.workspace import Workspace


class PlainMarkdownSystemPromptBuilder:
    """Load a prompt file and return its raw content unchanged.

    Args:
        template_path: Path to a .md (or .txt / any text) file. Not rendered.
    """

    def __init__(self, template_path: str):
        if isinstance(template_path, str) and template_path.startswith("file://"):
            template_path = template_path[len("file://") :]
        self.template_path = template_path
        self._cached: str | None = None

    async def build(self, workspace: "Workspace | None" = None) -> str:  # noqa: ARG002
        if self._cached is None:
            with open(self.template_path, "r", encoding="utf-8") as f:
                self._cached = f.read()
        return self._cached
