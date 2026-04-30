# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from ..base import tool

_GREP_TIMEOUT_S = 300  # 5-minute hard ceiling

# File-type → glob extensions (subset of common ripgrep types)
_TYPE_GLOBS: dict[str, list[str]] = {
    "py": ["*.py"],
    "js": ["*.js", "*.mjs", "*.cjs"],
    "ts": ["*.ts", "*.tsx", "*.mts"],
    "jsx": ["*.jsx"],
    "go": ["*.go"],
    "rust": ["*.rs"],
    "java": ["*.java"],
    "c": ["*.c", "*.h"],
    "cpp": ["*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hxx"],
    "sh": ["*.sh", "*.bash"],
    "yaml": ["*.yaml", "*.yml"],
    "json": ["*.json"],
    "toml": ["*.toml"],
    "md": ["*.md", "*.markdown"],
    "html": ["*.html", "*.htm"],
    "css": ["*.css", "*.scss", "*.sass"],
    "sql": ["*.sql"],
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "The regular expression pattern to search for in file contents",
        },
        "path": {
            "type": "string",
            "description": "File or directory to search in. Defaults to current working directory.",
        },
        "glob": {
            "type": "string",
            "description": "Glob pattern to filter files (e.g. '*.js', '**/*.tsx')",
        },
        "type": {
            "type": "string",
            "description": "File type shorthand (e.g. 'py', 'js', 'ts', 'go', 'rust'). Overrides glob.",
        },
        "output_mode": {
            "type": "string",
            "description": (
                "Output mode: 'content' shows matching lines, "
                "'files_with_matches' shows file paths (default), "
                "'count' shows match counts per file."
            ),
        },
        "-i": {"type": "boolean", "description": "Case insensitive search"},
        "-n": {
            "type": "boolean",
            "description": "Show line numbers in output (default true for content mode)",
        },
        "-A": {
            "type": "number",
            "description": "Number of lines to show after each match",
        },
        "-B": {
            "type": "number",
            "description": "Number of lines to show before each match",
        },
        "-C": {
            "type": "number",
            "description": "Number of lines to show before and after each match (alias: context)",
        },
        "context": {"type": "number", "description": "Alias for -C"},
        "head_limit": {
            "type": "number",
            "description": "Limit output to first N lines/entries (default 250). Pass 0 for unlimited.",
        },
        "offset": {
            "type": "number",
            "description": "Skip first N lines/entries before applying head_limit (default 0)",
        },
        "multiline": {
            "type": "boolean",
            "description": (
                "Enable multiline mode where . matches newlines and patterns can span lines. Default: false."
            ),
        },
    },
    "required": ["pattern"],
}


@tool(
    name="Grep",
    description=(
        "Search for a regex pattern in file contents. "
        "Supports case-insensitive search (-i), context lines (-A/-B/-C), "
        "file-type filtering (type), multiline matching, and paginated output (head_limit/offset)."
    ),
    input_schema=_SCHEMA,
)
async def grep_tool(
    pattern: str,
    path: str = ".",
    glob: str = "**/*",
    type: str | None = None,
    output_mode: str = "files_with_matches",
    head_limit: int = 250,
    offset: int = 0,
    multiline: bool = False,
    **kwargs,
) -> str:
    # Hyphen-named params passed via **kwargs (Python can't name params with hyphens)
    case_insensitive: bool = bool(kwargs.get("-i", False))
    show_line_numbers: bool = bool(kwargs.get("-n", True))
    lines_after: int = int(kwargs.get("-A", 0))
    lines_before: int = int(kwargs.get("-B", 0))
    context_lines: int = int(kwargs.get("-C", kwargs.get("context", 0)))
    if context_lines:
        lines_before = lines_before or context_lines
        lines_after = lines_after or context_lines

    flags = re.MULTILINE
    if case_insensitive:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.DOTALL

    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    from ...sandbox.base import get_current_sandbox

    sandbox = get_current_sandbox()
    if not Path(path).is_absolute() and sandbox is not None:
        base = (Path(sandbox.workspace_path) / path).resolve()
    else:
        base = Path(path).resolve()

    if type and type in _TYPE_GLOBS:
        file_globs = _TYPE_GLOBS[type]
    else:
        file_globs = [glob]

    def _run_sync() -> str:
        all_files: list[Path] = []
        if base.is_file():
            all_files = [base]
        else:
            for fg in file_globs:
                all_files.extend(sorted(base.rglob(fg)))
            seen: set[Path] = set()
            deduped: list[Path] = []
            for f in all_files:
                if f not in seen:
                    seen.add(f)
                    deduped.append(f)
            all_files = deduped

        entries: list[str] = []
        for filepath in all_files:
            if not filepath.is_file():
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if multiline:
                entries.extend(_search_multiline(regex, text, filepath, output_mode, show_line_numbers))
            else:
                entries.extend(
                    _search_lines(
                        regex,
                        text,
                        filepath,
                        output_mode,
                        show_line_numbers,
                        lines_before,
                        lines_after,
                    )
                )

        if not entries:
            return "No matches found."

        if offset:
            entries = entries[offset:]
        if head_limit and head_limit > 0:
            entries = entries[:head_limit]

        return "\n".join(entries)

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _run_sync),
            timeout=_GREP_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return (
            f"Grep timed out after {_GREP_TIMEOUT_S}s. "
            "Narrow the search path, add a file-type filter, or simplify the pattern."
        )


def _search_lines(
    regex: re.Pattern,
    text: str,
    filepath: Path,
    output_mode: str,
    show_line_numbers: bool,
    lines_before: int,
    lines_after: int,
) -> list[str]:
    lines = text.splitlines()
    matched_indices = [i for i, line in enumerate(lines) if regex.search(line)]
    if not matched_indices:
        return []

    if output_mode == "files_with_matches":
        return [str(filepath)]
    if output_mode == "count":
        return [f"{filepath}: {len(matched_indices)}"]

    # content — gather context windows, deduplicated
    included: set[int] = set()
    for idx in matched_indices:
        for i in range(max(0, idx - lines_before), min(len(lines), idx + lines_after + 1)):
            included.add(i)

    result: list[str] = []
    prev: int | None = None
    for i in sorted(included):
        if prev is not None and i > prev + 1:
            result.append("--")
        if show_line_numbers:
            result.append(f"{filepath}:{i + 1}:{lines[i]}")
        else:
            result.append(f"{filepath}:{lines[i]}")
        prev = i
    return result


def _search_multiline(
    regex: re.Pattern,
    text: str,
    filepath: Path,
    output_mode: str,
    show_line_numbers: bool,
) -> list[str]:
    matches = list(regex.finditer(text))
    if not matches:
        return []

    if output_mode == "files_with_matches":
        return [str(filepath)]
    if output_mode == "count":
        return [f"{filepath}: {len(matches)}"]

    # Precompute newline positions for line-number lookup
    newline_positions = [i for i, c in enumerate(text) if c == "\n"]

    def line_num(pos: int) -> int:
        lo, hi = 0, len(newline_positions)
        while lo < hi:
            mid = (lo + hi) // 2
            if newline_positions[mid] < pos:
                lo = mid + 1
            else:
                hi = mid
        return lo + 1  # 1-indexed

    result: list[str] = []
    for m in matches:
        snippet = m.group(0).splitlines()[0][:200]
        if show_line_numbers:
            result.append(f"{filepath}:{line_num(m.start())}:{snippet}")
        else:
            result.append(f"{filepath}:{snippet}")
    return result
