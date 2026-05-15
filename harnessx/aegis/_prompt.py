# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Static system-prompt builder.

Workaround for a harnessx core serialization bug: ``TemplateSystemPromptBuilder``
stores ``extra_context`` as a dict, and ``_serialize_processor`` drops dict
values that aren't all primitives (lists-of-str fail the check). When the
HarnessConfig is serialized and re-instantiated, ``extra_context`` is lost,
so the template renders with empty placeholders at runtime.

This builder pre-renders the Jinja2 template at build time and stores the
result as a plain string — which serializes cleanly.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..workspace.workspace import Workspace  # noqa: F401


class StaticSystemPromptBuilder:
    """Returns a fixed string as the system prompt. Serializes cleanly."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def build(self, workspace: "object | None" = None) -> str:
        return self.text


def render_template(template_path: str | Path, **context) -> str:
    """Render a Jinja2 template file with the given context."""
    from jinja2 import Template

    src = Path(template_path).read_text(encoding="utf-8")
    return Template(src).render(**context)
