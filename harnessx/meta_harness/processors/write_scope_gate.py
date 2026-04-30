# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Write-scope gate processor for TFAR v2 meta-agent.

This gate intercepts direct file-edit tools (`Write`, `Edit`) and blocks writes
outside the allowed roots/files passed by evolve().

It also inspects `Bash` commands for write-like operations (`>`, `>>`, `tee`,
`cp`, `mv`) and blocks destinations outside the same allowed scope.
"""

from __future__ import annotations

import dataclasses
import re
import shlex
from pathlib import Path

from ...core.events import ToolCallEvent
from ...core.processor import MultiHookProcessor


def _norm_path(value: str) -> Path | None:
    raw = (value or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        # Require explicit absolute paths for deterministic policy checks.
        return None
    try:
        return p.resolve()
    except Exception:
        return None


_SHELL_CONTROL = {"&&", "||", ";", "|"}
_REDIR_TOKENS = {">", ">>", "1>", "1>>", "2>", "2>>", "&>", "&>>"}
_SPECIAL_SINKS = {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"}


def _strip_quotes(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return raw
    # Shell fragments may carry unmatched trailing quote after nested parsing
    # (e.g. bash -lc 'echo x >/tmp/a'). Strip boundary quotes defensively.
    return raw.strip("'\"")


def _is_special_sink(path: Path) -> bool:
    s = str(path)
    if s in _SPECIAL_SINKS:
        return True
    return s.startswith("/dev/fd/")


def _extract_bash_write_targets(command: str) -> tuple[list[str], list[str]]:
    """Extract likely Bash write destinations.

    Returns:
    - absolute_targets: destinations starting with `/`
    - non_absolute_targets: relative/ambiguous destinations
    """
    absolute_targets: list[str] = []
    non_absolute_targets: list[str] = []

    def _record_target(raw: str) -> None:
        v = _strip_quotes(raw)
        if not v or v == "-":
            return
        if v.startswith("/"):
            absolute_targets.append(v)
        else:
            non_absolute_targets.append(v)

    try:
        tokens = shlex.split(command, posix=True)
    except Exception:
        tokens = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # Redirection token followed by destination.
        if tok in _REDIR_TOKENS:
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if nxt not in _SHELL_CONTROL:
                    _record_target(nxt)
                i += 1
            i += 1
            continue

        # Attached redirection (e.g. 2>/tmp/e, >>/tmp/o).
        m = re.match(r"^(?:[12]?>>?|&>>?)(.+)$", tok)
        if m:
            _record_target(m.group(1))
            i += 1
            continue

        # tee [opts] <dest...>
        if tok == "tee":
            j = i + 1
            while j < len(tokens):
                t = tokens[j]
                if t in _SHELL_CONTROL:
                    break
                if t.startswith("-"):
                    j += 1
                    continue
                _record_target(t)
                j += 1
            i = j
            continue

        # cp/mv ... <dest>
        if tok in {"cp", "mv"}:
            j = i + 1
            args: list[str] = []
            while j < len(tokens):
                t = tokens[j]
                if t in _SHELL_CONTROL:
                    break
                if not t.startswith("-"):
                    args.append(t)
                j += 1
            if len(args) >= 2:
                _record_target(args[-1])
            i = j
            continue

        i += 1

    # Regex fallback for redirection patterns potentially missed by shlex.
    for m in re.finditer(
        r"(?:^|[\s;&|])(?:[12]?>>?|&>>?)\s*(\"[^\"]+\"|'[^']+'|[^\s;&|]+)",
        command,
    ):
        _record_target(m.group(1))

    def _dedupe(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return _dedupe(absolute_targets), _dedupe(non_absolute_targets)


class WriteScopeGateProcessor(MultiHookProcessor):
    """Block write-like tool calls outside allowed paths."""

    _singleton_group = "tfar_write_scope_gate"
    _order = 5

    def __init__(
        self,
        allowed_roots: tuple[str, ...] = (),
        allowed_files: tuple[str, ...] = (),
    ) -> None:
        self._allowed_roots = tuple(p.resolve() for p in (Path(x) for x in allowed_roots if x))
        self._allowed_files = tuple(p.resolve() for p in (Path(x) for x in allowed_files if x))

    def _is_allowed(self, path: Path) -> bool:
        if not self._allowed_roots and not self._allowed_files:
            return True  # no restrictions configured — pass everything through
        if path in self._allowed_files:
            return True
        for root in self._allowed_roots:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    async def on_before_tool(self, event: ToolCallEvent):
        if event.tool_name not in {"Write", "Edit", "Bash"}:
            yield event
            return

        tool_input = event.tool_input or {}
        candidates: list[Path] = []
        non_abs_targets: list[str] = []

        if event.tool_name == "Bash":
            cmd = tool_input.get("command")
            if not isinstance(cmd, str) or not cmd.strip():
                yield event
                return
            abs_targets, non_abs_targets = _extract_bash_write_targets(cmd)
            for raw in abs_targets:
                p = _norm_path(raw)
                if p is None:
                    continue
                if _is_special_sink(p):
                    continue
                candidates.append(p)
        else:
            for key in ("file_path", "path", "new_file_path", "target_path"):
                v = tool_input.get(key)
                if isinstance(v, str):
                    p = _norm_path(v)
                    if p is not None:
                        candidates.append(p)
                    elif v.strip():
                        msg = f"Write-scope gate: file path must be absolute. Rejected `{v}`."
                        yield dataclasses.replace(
                            event,
                            approved=False,
                            synthetic_result=msg,
                        )
                        return

        if non_abs_targets:
            msg = (
                "Write-scope gate: Bash write destinations must be absolute paths. "
                f"Rejected target(s): {', '.join(non_abs_targets)}"
            )
            yield dataclasses.replace(
                event,
                approved=False,
                synthetic_result=msg,
            )
            return

        if not candidates:
            # Unknown shape: allow and let other guards/prompt rules handle it.
            yield event
            return

        blocked = [p for p in candidates if not self._is_allowed(p)]
        if not blocked:
            yield event
            return

        allowed_roots = ", ".join(str(p) for p in self._allowed_roots) or "(none)"
        allowed_files = ", ".join(str(p) for p in self._allowed_files) or "(none)"
        source = "Bash command write target" if event.tool_name == "Bash" else "write"
        msg = (
            f"Write-scope gate: blocked {source} outside allowed paths.\n"
            f"blocked: {', '.join(str(p) for p in blocked)}\n"
            f"allowed_roots: {allowed_roots}\n"
            f"allowed_files: {allowed_files}"
        )
        yield dataclasses.replace(
            event,
            approved=False,
            synthetic_result=msg,
        )
