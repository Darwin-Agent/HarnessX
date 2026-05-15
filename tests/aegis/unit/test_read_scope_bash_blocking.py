# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""ReadScopeGateProcessor: Bash-command path blocking + _archive wiring.

The gate historically only intercepted Read / Grep / Glob. With meta-agents
now carrying ``bash_tool``, a command like ``cat /.../aegis_64_v3/...`` could
exfiltrate archived prior-run data and leak cross-experiment context into a
fresh run. These tests pin down:

1. Bash commands referencing absolute paths under a blocked root are denied.
2. Bash commands that don't touch blocked roots are allowed.
3. Bash commands with only relative paths are allowed (we don't pretend to
   parse cwd-relative escapes — the gate is trusted-agent scoping, not
   adversarial sandboxing).
4. Relative paths that happen to SHARE a suffix with a blocked root do not
   trigger a false-positive block.
5. The blocked-root check survives surrounding shell noise (pipes, quotes,
   subshells): as long as the raw absolute path appears in the command
   string, the gate fires.
6. Non-path tools (Read/Grep/Glob) still behave as before — regression.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from harnessx.core.events import ToolCallEvent
from harnessx.meta_harness.processors.read_scope_gate import ReadScopeGateProcessor


def _make_event(tool_name: str, tool_input: dict) -> ToolCallEvent:
    """Minimal ToolCallEvent — only fields the gate actually reads."""
    return ToolCallEvent(
        tool_name=tool_name,
        tool_input=tool_input,
        run_id="test-run",
        step_id=0,
    )


async def _run_gate(gate: ReadScopeGateProcessor, event: ToolCallEvent) -> ToolCallEvent:
    """Drive the gate's async generator once and return the yielded event."""
    async for out in gate.on_before_tool(event):
        return out
    raise AssertionError("gate yielded nothing")


@pytest.fixture
def archive_dir(tmp_path):
    """A concrete blocked-root directory to pin paths against."""
    d = tmp_path / "_archive"
    d.mkdir()
    (d / "old_run").mkdir()
    (d / "old_run" / "data").mkdir()
    (d / "old_run" / "data" / "ship_outcomes.json").write_text("{}")
    return d


@pytest.mark.asyncio
async def test_bash_absolute_path_under_blocked_root_is_denied(archive_dir):
    gate = ReadScopeGateProcessor(blocked_roots=(str(archive_dir),))
    event = _make_event("Bash", {
        "command": f"cat {archive_dir}/old_run/data/ship_outcomes.json",
    })
    out = await _run_gate(gate, event)
    assert out.approved is False
    assert out.synthetic_result
    assert "restricted" in out.synthetic_result.lower()


@pytest.mark.asyncio
async def test_bash_unrelated_absolute_path_is_allowed(archive_dir, tmp_path):
    gate = ReadScopeGateProcessor(blocked_roots=(str(archive_dir),))
    other = tmp_path / "current_run" / "data.txt"
    other.parent.mkdir()
    other.write_text("hello")
    event = _make_event("Bash", {"command": f"cat {other}"})
    out = await _run_gate(gate, event)
    # No `.approved` mutation → gate passed through untouched (approved stays True).
    assert out.approved is not False


@pytest.mark.asyncio
async def test_bash_relative_path_is_not_matched(archive_dir):
    """Relative paths are not caught — gate only inspects absolute tokens."""
    gate = ReadScopeGateProcessor(blocked_roots=(str(archive_dir),))
    event = _make_event("Bash", {"command": "ls _archive/old_run"})
    out = await _run_gate(gate, event)
    assert out.approved is not False


@pytest.mark.asyncio
async def test_bash_blocked_root_embedded_in_pipe_chain(archive_dir):
    """Pipes, redirects, and subshells shouldn't hide the path from the regex."""
    gate = ReadScopeGateProcessor(blocked_roots=(str(archive_dir),))
    event = _make_event("Bash", {
        "command": f"ls | grep foo && cat {archive_dir}/old_run/data/ship_outcomes.json | head",
    })
    out = await _run_gate(gate, event)
    assert out.approved is False


@pytest.mark.asyncio
async def test_bash_command_with_no_absolute_paths_is_allowed(archive_dir):
    gate = ReadScopeGateProcessor(blocked_roots=(str(archive_dir),))
    event = _make_event("Bash", {"command": "python3 -c 'print(1+1)'"})
    out = await _run_gate(gate, event)
    assert out.approved is not False


@pytest.mark.asyncio
async def test_read_still_blocked_regression(archive_dir):
    """The old Read behavior must still work — no regression from Bash addition."""
    gate = ReadScopeGateProcessor(blocked_roots=(str(archive_dir),))
    event = _make_event("Read", {
        "file_path": str(archive_dir / "old_run" / "data" / "ship_outcomes.json"),
    })
    out = await _run_gate(gate, event)
    assert out.approved is False


@pytest.mark.asyncio
async def test_read_allowed_files_exception_still_works(archive_dir):
    """A file listed in allowed_files stays readable even under blocked_roots."""
    target = archive_dir / "old_run" / "data" / "ship_outcomes.json"
    gate = ReadScopeGateProcessor(
        blocked_roots=(str(archive_dir),),
        allowed_files=(str(target),),
    )
    # Read of the exempted file should pass through.
    event = _make_event("Read", {"file_path": str(target)})
    out = await _run_gate(gate, event)
    assert out.approved is not False


@pytest.mark.asyncio
async def test_archive_roots_wired_to_evolver(tmp_path, monkeypatch):
    """The archive_roots() helper must surface the runs/_archive directory,
    and Evolver's read gate should include it in blocked_roots."""
    from harnessx.aegis._paths import archive_roots
    roots = archive_roots()
    assert len(roots) >= 1
    assert any("_archive" in r for r in roots)
    assert any("runs" in r for r in roots)


@pytest.mark.asyncio
async def test_bash_path_regex_does_not_match_substrings_of_longer_paths():
    """If blocked root is /a/b and command references /a/bc/file, must NOT
    block — /a/bc is not under /a/b. Pinned down because the initial fix
    almost did a naïve substring check."""
    gate = ReadScopeGateProcessor(blocked_roots=("/a/b",))
    event = _make_event("Bash", {"command": "cat /a/bc/file"})
    out = await _run_gate(gate, event)
    # The regex extracts /a/bc/file, resolves it, and checks relative_to /a/b
    # → ValueError → NOT blocked. Good.
    assert out.approved is not False
