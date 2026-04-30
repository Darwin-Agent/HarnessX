# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Plumbing tests for strict-mode validators.

Validators live under ``harnessx.meta_harness.validators`` — this test
suite checks the same strict-mode behaviour as before but imports the
new module paths. ``StrictValidationError`` is still re-exported from
``harnessx.meta_harness.evolve`` for backward compatibility.

These tests assert:
- strict=False keeps the legacy "write artifact, return" behaviour
- strict=True raises StrictValidationError for blocking findings, and still
  writes the artifact file so the follow-up prompt can point at it
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harnessx.core.builder import HarnessBuilder
from harnessx.core.processor import MultiHookProcessor
from harnessx.meta_harness.validate_workflow import (
    StrictValidationError,
    run_literals_check,
    run_processor_dry_fire,
    run_tool_dry_fire,
)
from harnessx.tools.base import tool
from harnessx.tools.inmemory import InMemoryToolRegistry


class _BrokenProcessor(MultiHookProcessor):
    """Reads a field that does not exist on BeforeModelEvent.

    run_processor_dry_fire feeds a minimal dummy event and watches for
    AttributeError "object has no attribute X" — the exact shape that
    marks a field-name typo as a likely bug.
    """

    async def on_before_model(self, event):
        _ = event.this_field_does_not_exist  # noqa: F841
        yield event


def _cfg_with_broken_processor():
    cfg = HarnessBuilder().slot(tool_registry=InMemoryToolRegistry()).build().canonicalize()
    # Inject the broken processor directly into _rt_procs (the runtime
    # processor list) since HarnessBuilder serializes instances to _target_
    # dicts that can't be resolved for test-local classes.
    # Override __module__ so dry_fire doesn't skip it (it skips "harnessx.*"
    # and "__main__" as known-safe).
    _BrokenProcessor.__module__ = "authored.custom_processor"
    cfg._rt_procs = [_BrokenProcessor()]
    return cfg


def test_processor_dryfire_default_writes_artifact_and_returns(tmp_path: Path) -> None:
    cfg = _cfg_with_broken_processor()
    asyncio.run(run_processor_dry_fire(cfg, tmp_path))
    assert (tmp_path / "DRY_FIRE_WARNINGS.md").is_file()


def test_processor_dryfire_strict_raises_on_likely_bug(tmp_path: Path) -> None:
    cfg = _cfg_with_broken_processor()
    with pytest.raises(StrictValidationError) as exc_info:
        asyncio.run(run_processor_dry_fire(cfg, tmp_path, strict=True))
    err = exc_info.value
    assert err.kind == "processor_dryfire"
    assert err.findings_path == tmp_path / "DRY_FIRE_WARNINGS.md"
    # Artifact is written BEFORE the raise so the follow-up can Read it.
    assert err.findings_path.is_file()


# ---- custom tool dry-fire -------------------------------------------------


@tool(description="tool with hallucinated import")
async def _broken_tool(query: str) -> str:
    import this_module_does_not_exist_xyz  # noqa: F401

    return query


def _cfg_with_broken_tool():
    reg = InMemoryToolRegistry()
    reg.register(_broken_tool)
    return HarnessBuilder().slot(tool_registry=reg).build().canonicalize()


def test_tool_dryfire_default_writes_artifact_and_returns(tmp_path: Path) -> None:
    cfg = _cfg_with_broken_tool()
    asyncio.run(run_tool_dry_fire(cfg, tmp_path))
    assert (tmp_path / "DRY_FIRE_TOOL_WARNINGS.md").is_file()


def test_tool_dryfire_strict_raises_on_likely_bug(tmp_path: Path) -> None:
    cfg = _cfg_with_broken_tool()
    with pytest.raises(StrictValidationError) as exc_info:
        asyncio.run(run_tool_dry_fire(cfg, tmp_path, strict=True))
    err = exc_info.value
    assert err.kind == "tool_dryfire"
    assert err.findings_path == tmp_path / "DRY_FIRE_TOOL_WARNINGS.md"
    assert err.findings_path.is_file()


def test_strict_is_noop_when_no_findings(tmp_path: Path) -> None:
    """Clean configs do not raise under strict=True — the gate only fires
    on blocking findings, not on absence of custom code."""
    cfg = HarnessBuilder().slot(tool_registry=InMemoryToolRegistry()).build().canonicalize()
    asyncio.run(run_processor_dry_fire(cfg, tmp_path, strict=True))
    asyncio.run(run_tool_dry_fire(cfg, tmp_path, strict=True))


# ---- task-specific literal scan -----------------------------------------


def test_task_specific_literals_clean(tmp_path: Path) -> None:
    """No authored files → validator runs cleanly and writes an empty
    findings artifact (so the follow-up can always cite a file)."""
    out_dir = tmp_path / "out"
    scratch = tmp_path / "scratch"
    out_dir.mkdir()
    scratch.mkdir()
    asyncio.run(run_literals_check(out_dir, scratch, strict=True))
    assert (scratch / "TASK_SPECIFIC_LITERALS.md").is_file()


def test_task_specific_literals_warn_under_threshold(tmp_path: Path) -> None:
    """1-2 matches → warn only: artifact written, strict does not raise."""
    out_dir = tmp_path / "out"
    scratch = tmp_path / "scratch"
    tools = out_dir / "tools"
    scratch.mkdir()
    tools.mkdir(parents=True)
    (tools / "probe.py").write_text(
        '# Hardcoded task reference\nKNOWN = "12345678-abcd-ef12-3456-7890abcdef12"  # task UUID\n',
        encoding="utf-8",
    )
    asyncio.run(run_literals_check(out_dir, scratch, strict=True))
    findings = (scratch / "TASK_SPECIFIC_LITERALS.md").read_text()
    assert "12345678-abcd-ef12-3456-7890abcdef12" in findings


def test_task_specific_literals_strict_fails_at_threshold(tmp_path: Path) -> None:
    """≥ 3 matches across authored files → StrictValidationError."""
    out_dir = tmp_path / "out"
    scratch = tmp_path / "scratch"
    tools = out_dir / "tools"
    processors = out_dir / "processors"
    scratch.mkdir()
    tools.mkdir(parents=True)
    processors.mkdir(parents=True)
    (tools / "lookup.py").write_text(
        "TABLE = {\n"
        '    "12345678-abcd-ef12-3456-7890abcdef12": "A",\n'
        '    "abcdef12-3456-7890-abcd-ef1234567890": "B",\n'
        "}\n",
        encoding="utf-8",
    )
    (processors / "route.py").write_text(
        'if task.id == "11111111-2222-3333-4444-555555555555":\n    pass\n',
        encoding="utf-8",
    )
    with pytest.raises(StrictValidationError) as exc_info:
        asyncio.run(run_literals_check(out_dir, scratch, strict=True))
    err = exc_info.value
    assert err.kind == "task_specific_literals"
    assert err.findings_path == scratch / "TASK_SPECIFIC_LITERALS.md"
    assert err.findings_path.is_file()


def test_task_specific_literals_accepts_extra_patterns(tmp_path: Path) -> None:
    """Benchmark-specific id shapes can be injected via extra_patterns.

    Replaces the old test that hardcoded a ``gaia-<id>`` pattern inside
    meta_harness — that pattern was a benchmark leak. Benchmarks now
    pass their own patterns through the ``extra_patterns`` arg; here we
    simulate that with a fake ``bench-`` prefix.
    """
    import re as _re

    out_dir = tmp_path / "out"
    scratch = tmp_path / "scratch"
    tools = out_dir / "tools"
    scratch.mkdir()
    tools.mkdir(parents=True)
    (tools / "bench_hack.py").write_text(
        'BAD = [\n    "bench-validation-123abc",\n    "bench-validation-456def",\n    "bench-validation-789ghi",\n]\n',
        encoding="utf-8",
    )
    extra = [("bench-id", _re.compile(r"\bbench-[a-zA-Z0-9_-]{8,}\b"))]
    with pytest.raises(StrictValidationError) as exc_info:
        asyncio.run(run_literals_check(out_dir, scratch, strict=True, extra_patterns=extra))
    assert exc_info.value.kind == "task_specific_literals"


def test_task_specific_literals_non_strict_never_raises(tmp_path: Path) -> None:
    """Default strict=False writes the artifact but never raises."""
    out_dir = tmp_path / "out"
    scratch = tmp_path / "scratch"
    tools = out_dir / "tools"
    scratch.mkdir()
    tools.mkdir(parents=True)
    (tools / "lookup.py").write_text(
        'ID = "12345678-abcd-ef12-3456-7890abcdef12"\n'
        'ID2 = "abcdef12-3456-7890-abcd-ef1234567890"\n'
        'ID3 = "11111111-2222-3333-4444-555555555555"\n',
        encoding="utf-8",
    )
    # Must not raise even though >=3 matches — default mode is advisory.
    asyncio.run(run_literals_check(out_dir, scratch))
    assert (scratch / "TASK_SPECIFIC_LITERALS.md").is_file()
