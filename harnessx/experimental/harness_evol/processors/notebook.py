"""Cross-round evolution notebook processor.

EvolutionNotebookProcessor injects pending hypotheses, open todos, and failed
interventions from a shared notebook file into the agent's system prompt on
each task_start. The agent updates the notebook during the run using the Write
tool; WriteScopeGateProcessor ensures only the notebook file is writable.

Notebook file:  {session_dir}/evolution_notebook.md
Shared between DigestAgent and EvolveAgent within one evolution session.

Notebook sections
-----------------
## Pending hypotheses   — unverified suspicions from previous rounds
## Verified findings    — confirmed facts (agent moves items here when proven)
## Failed interventions — things already tried that did not work
## Open todos           — concrete actions to take in upcoming rounds
## Cross-round patterns — trends observed across multiple rounds
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import TYPE_CHECKING

from harnessx.core.processor import MultiHookProcessor
from harnessx.processors._sp_utils import sp_append

if TYPE_CHECKING:
    from harnessx.core.events import TaskStartEvent

_NOTEBOOK_TAG = "evolution_notebook"

# Sections injected into the system prompt (agent must verify / act on these).
_ACTIVE_SECTIONS = frozenset({
    "Pending hypotheses",
    "Failed interventions",
    "Open todos",
    "Cross-round patterns",
})


class EvolutionNotebookProcessor(MultiHookProcessor):
    """Inject cross-round notebook context into the system prompt at task_start.

    Reads evolution_notebook.md, extracts actionable sections (pending
    hypotheses, open todos, failed interventions, cross-round patterns), and
    appends them to the system prompt as a structured <evolution_notebook> block.

    The agent is responsible for updating the notebook via the Write tool
    during the run.  WriteScopeGateProcessor must be configured alongside this
    processor to restrict writes to the notebook path only.

    Order 2 — runs after DigestSystemPromptProcessor (_order=0) and before
    most other processors.
    """

    _singleton_group = "harness_evol.notebook"
    _order = 2

    def __init__(self, notebook_path: Path | str) -> None:
        self._path = Path(notebook_path)

    async def on_task_start(self, event: "TaskStartEvent"):
        section = _build_injection(self._path)
        if section:
            yield dataclasses.replace(
                event, system_prompt=sp_append(event.system_prompt, section)
            )
        else:
            yield event


# ── notebook parsing ──────────────────────────────────────────────────────────

def _parse_sections(text: str) -> dict[str, str]:
    """Split markdown into {heading: body} mapping (## level only)."""
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []

    for line in text.splitlines():
        m = re.match(r"^##\s+(.+)$", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = m.group(1).strip()
            buf = []
        else:
            if current is not None:
                buf.append(line)

    if current is not None:
        sections[current] = "\n".join(buf).strip()

    return sections


def _build_injection(notebook_path: Path) -> str:
    """Read notebook and build the system-prompt injection block.

    Returns an empty string when there is nothing actionable to inject
    (empty sections).  On the very first run (file absent) returns a
    minimal creation-guidance block so the agent knows to initialise it.
    """
    if not notebook_path.exists():
        return (
            f"\n\n<{_NOTEBOOK_TAG} path=\"{notebook_path}\">\n"
            "(no entries yet)\n"
            f"</{_NOTEBOOK_TAG}>\n"
            f"Initialize `{notebook_path}` with the Write tool using these sections:\n"
            "## Pending hypotheses\n"
            "## Verified findings\n"
            "## Failed interventions\n"
            "## Open todos\n"
            "## Cross-round patterns\n"
        )

    text = notebook_path.read_text(encoding="utf-8")
    sections = _parse_sections(text)

    active = {
        k: v for k, v in sections.items()
        if k in _ACTIVE_SECTIONS and v.strip()
    }
    if not active:
        return ""

    lines = [f"<{_NOTEBOOK_TAG} path=\"{notebook_path}\">"]
    for heading, body in active.items():
        lines.append(f"## {heading}")
        lines.append(body)
        lines.append("")
    lines.append(f"</{_NOTEBOOK_TAG}>")
    lines.append(
        f"Update `{notebook_path}` with new findings using the Write tool "
        "(sections: Pending hypotheses, Verified findings, Failed interventions, "
        "Open todos, Cross-round patterns)."
    )

    return "\n\n" + "\n".join(lines) + "\n"
