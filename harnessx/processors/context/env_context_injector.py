# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path

from ...core.events import BeforeModelEvent, Message, TaskEndEvent, TaskStartEvent
from ...core.processor import MultiHookProcessor
from .._sp_utils import sp_append

_NON_INTERACTIVE = (
    "\n\n## Operating Mode\n"
    "You are running in **non-interactive mode**. There is no human available to "
    "answer questions or provide clarification. You MUST:\n"
    "- Never ask for confirmation, clarification, or additional input.\n"
    "- Make reasonable assumptions and proceed autonomously.\n"
    "- If multiple valid approaches exist, choose the most likely one and continue."
)

_INTEGRITY_RULES = """

---

## INTEGRITY RULES (VIOLATIONS = AUTOMATIC FAILURE)

1. **NO DATA FABRICATION**: NEVER invent, guess, or fabricate data. If you cannot recover/extract actual data, report the failure. Do not make up values.

2. **NO TEST OVERFITTING**: Your solution must work for ANY valid input, not just test examples. Do NOT hardcode test values as answers. You SHOULD read test files to understand the expected interface (field names, CLI flags, file paths, output formats) — that is not overfitting.

3. **NO BENCHMARK GAMING**: Do NOT modify timing functions, measurement code, or evaluation logic. Optimize the actual algorithm, not the measurement.

4. **NO SERVICE KILLING**: When you start a server/service for a task, do NOT terminate it. Verification needs it running. Use background execution (`&`, `nohup`, daemon mode).

---

## SANDBOX REMINDERS

- **BACKGROUND PROCESSES DON'T PERSIST**: Verification runs in a SEPARATE process. Background servers won't be running unless you start them with `&` or `nohup`, or create a startup script.
- **PREFER STDLIB AND CLI TOOLS**: The sandbox may lack third-party packages. Prefer standard library and CLI tools already present."""

_TIME_WARNING_75 = (
    "\n\n[TimeBudget] ⚠️  You have used ~75% of the allotted time. "
    "Wrap up any remaining work, run verification, and prepare to submit your answer."
)

_TIME_WARNING_90 = (
    "\n\n[TimeBudget] 🚨  You have used ~90% of the allotted time. "
    "Stop new exploration immediately. Focus only on verifying and submitting "
    "what you have — a partial solution is better than nothing."
)


class EnvironmentContextInjector(MultiHookProcessor):
    """Inject a deterministic environment context block once per task.

    The block is appended to the system prompt and contains:

    - Agent workspace path (prefers ``TaskStartEvent.workspace.root``)
    - Project path (the launcher process CWD)
    - Optional task timeout
    - Arbitrary key→value constraint pairs
    - Optional shallow workspace tree (depth 1, up to *max_tree_lines* entries)

    Args:
        working_dir:       Override the working directory shown to the model.
        timeout_seconds:   Optional task-level timeout to disclose and track.
        constraints:       Arbitrary key→value facts (e.g. ``{"language": "Python"}``).
        header:            Section header label (default ``"Environment"``).
        inject_workspace_tree: Whether to include a depth-1 directory listing
                           (default ``True``).
        max_tree_lines:    Maximum number of lines in the tree listing (default 20).
        non_interactive:   Inject a non-interactive mode instruction prohibiting
                           clarification requests (default ``True``).
    """

    _singleton_group = "context.env"
    _order = 5  # early — other processors may build on top of it

    def __init__(
        self,
        working_dir: str | None = None,
        timeout_seconds: int | None = None,
        constraints: dict[str, str] | None = None,
        header: str = "Environment",
        inject_workspace_tree: bool = True,
        max_tree_lines: int = 20,
        non_interactive: bool = True,
        inject_integrity_rules: bool = True,
        show_project_dir: bool = True,
    ) -> None:
        self._working_dir = working_dir or os.getcwd()
        self._project_dir = os.getcwd() if show_project_dir else None
        self._timeout = timeout_seconds
        self._constraints = dict(constraints or {})
        self._header = header
        self._inject_tree = inject_workspace_tree
        self._max_tree_lines = max_tree_lines
        self._non_interactive = non_interactive
        self._inject_integrity_rules = inject_integrity_rules
        self._start_time: float | None = None
        self._warned_75 = False
        self._warned_90 = False
        self._tree_cache: str | None = None  # fetched once in on_task_start

    async def _fetch_tree(self) -> str | None:
        """Return a depth-1 directory listing of the working directory, or None on failure."""
        from ...sandbox.base import get_current_sandbox

        sandbox = get_current_sandbox()

        if sandbox is not None:
            cmd = f"find {self._working_dir} -maxdepth 1 ! -path '*/.*' | sort | head -{self._max_tree_lines}"
            try:
                raw = await sandbox.exec(cmd, timeout=10)
            except Exception:
                return None
        else:
            root = Path(self._working_dir)
            if not root.exists():
                return None
            entries: list[str] = []
            try:
                for p in sorted(root.iterdir()):
                    if p.name.startswith("."):
                        continue
                    entries.append(p.name)
                    if len(entries) >= self._max_tree_lines:
                        break
            except Exception:
                return None
            raw = "\n".join(entries)

        if not raw or not raw.strip():
            return None

        # Strip the working_dir prefix so paths are relative to it.
        prefix = self._working_dir.rstrip("/") + "/"
        lines = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or line == self._working_dir.rstrip("/"):
                continue
            lines.append(line[len(prefix) :] if line.startswith(prefix) else line)

        return "\n".join(lines) if lines else None

    def _time_warning(self) -> str | None:
        """Return a time budget warning string if a threshold has been crossed, else None."""
        if not self._timeout or self._start_time is None:
            return None
        elapsed = time.monotonic() - self._start_time
        pct = elapsed / self._timeout
        if pct >= 0.90 and not self._warned_90:
            self._warned_90 = True
            return _TIME_WARNING_90
        if pct >= 0.75 and not self._warned_75:
            self._warned_75 = True
            return _TIME_WARNING_75
        return None

    def _build_block(self, tree: str | None) -> str:
        lines = [f"\n\n## {self._header}"]
        lines.append(f"- Agent workspace path: `{self._working_dir}`")
        if self._project_dir is not None:
            lines.append(f"- Project path: `{self._project_dir}`")
        if self._timeout:
            lines.append(f"- Task timeout: {self._timeout}s")
        for key, val in self._constraints.items():
            lines.append(f"- {key}: {val}")
        if tree:
            lines.append(f"\n### Workspace\n```\n{tree}\n```")
        if self._inject_integrity_rules:
            lines.append(_INTEGRITY_RULES)
        return "\n".join(lines)

    async def on_task_start(self, event: TaskStartEvent):
        self._start_time = time.monotonic()
        if event.workspace is not None:
            try:
                self._working_dir = str(Path(event.workspace.root).expanduser().resolve())
            except Exception:
                self._working_dir = str(event.workspace.root)
        if self._inject_tree:
            self._tree_cache = await self._fetch_tree()

        tree = self._tree_cache if self._inject_tree else None
        prompt = event.system_prompt
        if self._non_interactive:
            prompt = sp_append(prompt, _NON_INTERACTIVE)
        prompt = sp_append(prompt, self._build_block(tree))
        yield dataclasses.replace(event, system_prompt=prompt)

    async def on_before_model(self, event: BeforeModelEvent):
        # Inject time-budget warnings as a user message right before the model call.
        time_warn = self._time_warning()
        if time_warn:
            msg = Message(role="user", content=time_warn.strip())
            yield dataclasses.replace(event, messages=event.messages + (msg,))
        else:
            yield event

    async def on_task_end(self, event: TaskEndEvent):
        self._start_time = None
        self._warned_75 = False
        self._warned_90 = False
        self._tree_cache = None
        yield event
