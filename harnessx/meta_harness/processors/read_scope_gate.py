# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Read-scope gate processor for the meta-agent.

Blocks Read / Grep / Glob calls that target restricted root directories,
with explicit per-file exceptions.  Intended to prevent the meta-agent from
spending steps deep-diving into harnessx source code when the SKILL.md files
already provide the necessary API surface.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ...core.events import ToolCallEvent
from ...core.processor import MultiHookProcessor


def _resolve(value: str) -> Path | None:
    raw = (value or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        return None
    try:
        return p.resolve()
    except Exception:
        return None


class ReadScopeGateProcessor(MultiHookProcessor):
    """Block Read / Grep / Glob calls that target restricted root directories.

    Paths listed in ``allowed_files`` are always permitted even if they fall
    under a ``blocked_roots`` entry.  All other paths under ``blocked_roots``
    are rejected with a helpful error message pointing the agent at the
    relevant SKILL.md instead.

    Only ``Read``, ``Grep``, and ``Glob`` are intercepted.  ``Bash`` commands
    that shell-read files are not blocked here (they are typically less
    frequent and harder to parse reliably).
    """

    _singleton_group = "meta_read_scope_gate"
    _order = 4  # before write-scope gate (order 5)

    def __init__(
        self,
        blocked_roots: "tuple[str, ...] | None" = None,
        allowed_files: "tuple[str, ...] | None" = None,
        hint_message: str = "",
    ) -> None:
        self._blocked_roots: tuple[Path, ...] = tuple(Path(x).resolve() for x in (blocked_roots or ()) if x)
        self._allowed_files: tuple[Path, ...] = tuple(Path(x).resolve() for x in (allowed_files or ()) if x)
        self._hint = hint_message

    def _is_blocked(self, path: Path) -> bool:
        resolved = path.resolve()
        for exc in self._allowed_files:
            if resolved == exc:
                return False
        for root in self._blocked_roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _blocked_msg(self, path: str) -> str:
        allowed = ", ".join(str(p) for p in self._allowed_files) or "(none)"
        msg = f"Read-scope gate: access to `{path}` is restricted.\n"
        if self._hint:
            msg += f"{self._hint}\n"
        msg += f"Allowed exceptions: {allowed}"
        return msg

    async def on_before_tool(self, event: ToolCallEvent):
        tool = event.tool_name
        if tool not in {"Read", "Grep", "Glob"}:
            yield event
            return

        tool_input = event.tool_input or {}

        if tool == "Read":
            fp = tool_input.get("file_path", "")
            if isinstance(fp, str):
                p = _resolve(fp)
                if p is not None and self._is_blocked(p):
                    yield dataclasses.replace(
                        event,
                        approved=False,
                        synthetic_result=self._blocked_msg(fp),
                    )
                    return

        elif tool in {"Grep", "Glob"}:
            path_val = tool_input.get("path", "")
            if isinstance(path_val, str) and path_val.strip():
                p = _resolve(path_val)
                if p is not None and self._is_blocked(p):
                    yield dataclasses.replace(
                        event,
                        approved=False,
                        synthetic_result=self._blocked_msg(path_val),
                    )
                    return

        yield event
