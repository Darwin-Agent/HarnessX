# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import re

from ...core.events import ToolResultEvent
from ...core.processor import MultiHookProcessor

_ERROR_RE = re.compile(
    r"(?:^|\b)(error|failed|failure|exception|traceback|segmentation fault|"
    r"permission denied|no such file|not found|assertionerror|runtimeerror|"
    r"moduleNotFoundError|importerror)\b",
    re.IGNORECASE,
)

_APT_PROGRESS_RE = re.compile(
    r"^(Get:\d+|Hit:|Ign:|Fetched\s+\d+|Reading package lists|Building dependency tree|"
    r"Reading state information|Selecting previously unselected package|Preparing to unpack|"
    r"Unpacking |Setting up |Processing triggers for |After this operation|Need to get|"
    r"debconf:)",
    re.IGNORECASE,
)
_PIP_PROGRESS_RE = re.compile(
    r"^(Looking in indexes:|Collecting |Downloading |Using cached |Installing collected packages|"
    r"Requirement already satisfied:|Building wheels for collected packages|"
    r"Getting requirements to build wheel|Preparing metadata \(|Successfully installed|"
    r"WARNING: Running pip as the 'root' user|\[[ =#>.-]*\]\s*\d+%|\d+%\s*\|)",
    re.IGNORECASE,
)
_NPM_PROGRESS_RE = re.compile(
    r"^(npm (notice|info|timing)|added \d+ packages?|removed \d+ packages?|changed \d+ packages?|"
    r"audited \d+ packages?|up to date, audited \d+ packages?|"
    r"\d+ packages? are looking for funding|found \d+ vulnerabilities)",
    re.IGNORECASE,
)
_NPM_ERR_RE = re.compile(r"^npm ERR!", re.IGNORECASE)

_SHELL_TOOLS = frozenset({"bash", "terminal", "shell", "exec", "exec_command"})


class ToolResultNoiseFilterProcessor(MultiHookProcessor):
    """Filter package-manager progress noise from shell tool outputs.

    The processor is intentionally conservative:
    - Only applies to shell-like tools (for example ``Bash``).
    - Only activates when enough lines match apt/pip/npm progress patterns.
    - Always preserves error-like lines and a small tail window for context.
    """

    _singleton_group = "tool_result_noise_filter"
    _order = 10

    def __init__(self, *, min_noise_lines: int = 6, tail_lines: int = 6) -> None:
        self.min_noise_lines = max(1, min_noise_lines)
        self.tail_lines = max(1, tail_lines)

    @staticmethod
    def _is_error_line(line: str) -> bool:
        return bool(_ERROR_RE.search(line) or _NPM_ERR_RE.search(line) or line.startswith("E: "))

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        return bool(_APT_PROGRESS_RE.search(line) or _PIP_PROGRESS_RE.search(line) or _NPM_PROGRESS_RE.search(line))

    def _filter_text(self, text: str) -> tuple[str, bool]:
        lines = text.splitlines()
        if not lines:
            return text, False

        noise_idx: set[int] = set()
        kept_lines: list[str] = []
        for idx, line in enumerate(lines):
            is_noise = self._is_noise_line(line)
            is_error = self._is_error_line(line)
            if is_noise and not is_error:
                noise_idx.add(idx)
                continue
            kept_lines.append(line)

        if len(noise_idx) < self.min_noise_lines:
            return text, False

        # Keep short tail context so agents still see final command state.
        tail: list[str] = []
        for line in lines[-self.tail_lines :]:
            if line not in kept_lines and line.strip() and (self._is_error_line(line) or not self._is_noise_line(line)):
                tail.append(line)

        filtered = kept_lines.copy()
        if tail:
            filtered.append("")
            filtered.append(
                f"[noise_filter] dropped {len(noise_idx)} install/download progress line(s); tail kept below:"
            )
            filtered.extend(tail)
        return "\n".join(filtered).strip(), True

    async def on_after_tool(self, event: ToolResultEvent):
        tool = (event.tool_name or "").strip().lower()
        if tool not in _SHELL_TOOLS:
            yield event
            return

        if not event.result or not isinstance(event.result, str):
            yield event
            return

        filtered, changed = self._filter_text(event.result)
        if not changed:
            yield event
            return

        yield dataclasses.replace(event, result=filtered)
