# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Auto-validate custom processors and tools on write.

When the meta-agent writes or edits a ``processors/*.py`` or
``tools/*.py`` file, this processor automatically runs the appropriate
checker and appends findings to the tool result. The meta-agent sees
the feedback immediately and can fix the issue in the same session.

- ``processors/*.py`` → contract check (hook-mutation rules)
- ``tools/*.py`` → import/syntax check (dry-fire style)
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from ...core.events import ToolCallEvent, ToolResultEvent
from ...core.processor import MultiHookProcessor

logger = logging.getLogger(__name__)


def _classify_file(path: str) -> str | None:
    """Return 'processor', 'tool', or None."""
    p = Path(path)
    if p.suffix != ".py":
        return None
    if "processors" in p.parts:
        return "processor"
    if "tools" in p.parts:
        return "tool"
    return None


class ContractAutoCheckProcessor(MultiHookProcessor):
    """Run validation when meta-agent writes a processor or tool file."""

    _singleton_group = "contract_autocheck"
    _order = 90

    def __init__(self) -> None:
        self._pending: dict[str, tuple[str, str]] = {}  # call_id -> (kind, path)

    async def on_before_tool(self, event: ToolCallEvent):
        if event.tool_name in ("Write", "Edit") and event.approved:
            path = event.tool_input.get("file_path", "")
            if path:
                kind = _classify_file(path)
                if kind:
                    self._pending[event.tool_call_id] = (kind, path)
        yield event

    async def on_after_tool(self, event: ToolResultEvent):
        entry = self._pending.pop(event.tool_call_id, None)
        if entry is None or event.error:
            yield event
            return

        kind, path = entry
        if kind == "processor":
            feedback = await self._check_processor(path)
        else:
            feedback = self._check_tool(path)

        if feedback:
            new_result = event.result + "\n\n" + feedback
            yield dataclasses.replace(event, result=new_result)
        else:
            yield event

    async def _check_processor(self, path: str) -> str:
        """Contract-check a processor file."""
        import importlib.util

        from ...core.contract_check import check_processor_contract

        try:
            spec = importlib.util.spec_from_file_location("_autocheck", path)
            if spec is None or spec.loader is None:
                return ""
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:
            return (
                f"\n[AUTO-CHECK] Failed to import {Path(path).name}: "
                f"{type(exc).__name__}: {exc}\n"
                "Fix the import error before proceeding."
            )

        from ...core.processor import MultiHookProcessor as _MHP

        processors = []
        for name in dir(mod):
            obj = getattr(mod, name, None)
            if isinstance(obj, type) and issubclass(obj, _MHP) and obj is not _MHP and obj.__module__ == mod.__name__:
                try:
                    processors.append(obj())
                except Exception:
                    pass

        if not processors:
            return ""

        all_violations = []
        for proc in processors:
            try:
                violations = await check_processor_contract(proc)
                all_violations.extend(violations)
            except Exception as exc:
                logger.debug("[contract_autocheck] check failed for %s: %s", type(proc).__name__, exc)

        if not all_violations:
            return "\n[AUTO-CHECK] Processor passes hook-mutation contract. No violations."

        lines = [
            "",
            "[AUTO-CHECK] Hook-mutation contract VIOLATED:",
            "",
        ]
        for v in all_violations:
            lines.append(f"  - {v.hook} [{v.violation_type}] fixture={v.fixture} step={v.step_id}: {v.message}")
        lines.append("")
        lines.append(
            "Fix these violations before end_turn. The post-flight gate will "
            "reject this processor otherwise. See the `reference` skill's "
            "Messages-mutation contract section for the rules."
        )
        return "\n".join(lines)

    def _check_tool(self, path: str) -> str:
        """Import-check a tool file (catches SyntaxError, ImportError, NameError)."""
        import importlib.util

        fname = Path(path).name
        try:
            spec = importlib.util.spec_from_file_location("_tool_check", path)
            if spec is None or spec.loader is None:
                return ""
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except SyntaxError as exc:
            return (
                f"\n[AUTO-CHECK] SyntaxError in {fname} "
                f"(line {exc.lineno}): {exc.msg}\n"
                "Fix the syntax error before proceeding."
            )
        except ImportError as exc:
            return (
                f"\n[AUTO-CHECK] ImportError in {fname}: {exc}\n"
                "The import does not exist. Use stdlib or harnessx builtins only."
            )
        except NameError as exc:
            return f"\n[AUTO-CHECK] NameError in {fname}: {exc}\nUndefined symbol — check for typos or missing imports."
        except Exception as exc:
            return f"\n[AUTO-CHECK] Error loading {fname}: {type(exc).__name__}: {exc}"

        # Check that at least one @tool-decorated function exists
        from ...tools.base import Tool

        tools_found = [name for name in dir(mod) if isinstance(getattr(mod, name, None), Tool)]
        if not tools_found:
            return (
                f"\n[AUTO-CHECK] Warning: {fname} loaded OK but no @tool-decorated "
                "functions found. Did you forget the @tool decorator?"
            )

        return f"\n[AUTO-CHECK] Tool {fname} loads OK. Found: {', '.join(tools_found)}."
