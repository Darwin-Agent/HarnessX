# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Paths shared by AEGIS agents' read gates.

Two halves:

1. ``HARNESSX_SRC_ROOT`` — the harnessx/ package root. Blocked from most
   agent reads (prevents prompt injection and source-code exfiltration
   through the meta-agent's tools).

2. ``api_reference_files()`` — curated allowlist of "living documentation"
   inside harnessx/. Evolver, Planner, and Critic are permitted to Read
   these so they can learn the real API from real code (processor base
   class, built-in processors / tools, event types, loader logic) rather
   than hallucinating from prose-only prompt hints.

Intentionally EXCLUDED from the allowlist:
- ``harnessx/aegis/`` — the meta-agent's own source (would leak prompts)
- ``harnessx/meta_harness/`` — legacy, not the API we want agents to model
- ``harnessx/api/``, ``harnessx/lab/`` — internal services
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# harnessx/ package root (the Python module), NOT the repo root —
# recipe/ and runs/ live under the repo so must stay readable.
HARNESSX_SRC_ROOT: Path = Path(__file__).parent.parent.resolve()


def archive_roots() -> tuple[str, ...]:
    """Resolve the set of runs-archive roots that meta-agents must NOT read.

    Keeps cross-experiment leakage out of fresh runs: when a prior experiment's
    outputs (old runs, contaminated runs, analysis dumps) are moved under
    ``recipe/gaia_evolver/runs/_archive/`` they become invisible to the meta-
    agent. The meta-agent can still read its OWN run's data (runs/<tag>/)
    because that's not under _archive/.

    Returned as a list to allow future expansion (e.g. adding per-benchmark
    archive directories) without changing call sites.
    """
    # Walk up from harnessx/ to find the repo root, then resolve the archive
    # dir relative to it. Computed at call time (not import time) so moving
    # the repo or running from a different checkout doesn't break.
    repo_root = HARNESSX_SRC_ROOT.parent
    archive = repo_root / "recipe" / "gaia_evolver" / "runs" / "_archive"
    return (str(archive.resolve()),)


@lru_cache(maxsize=1)
def api_reference_files() -> tuple[str, ...]:
    """Resolve the curated set of harnessx/ files agents may Read.

    Cached because the set is computed by filesystem scan; the test suite
    instantiates many harnesses per run and the scan is a few hundred
    stat()s that we don't need to repeat.
    """
    paths: set[Path] = set()

    # Core base classes + event types — needed to subclass MultiHookProcessor
    # and understand event field shapes.
    for rel in ("core/processor.py", "core/events.py", "core/builder.py",
                "core/harness.py", "core/state.py", "core/runloop.py"):
        p = HARNESSX_SRC_ROOT / rel
        if p.exists():
            paths.add(p.resolve())

    # All public processors — the best place to copy style from.
    proc_root = HARNESSX_SRC_ROOT / "processors"
    if proc_root.exists():
        for p in proc_root.rglob("*.py"):
            paths.add(p.resolve())

    # All built-in tools — reference implementations for the @tool decorator.
    tool_root = HARNESSX_SRC_ROOT / "tools" / "builtin"
    if tool_root.exists():
        for p in tool_root.rglob("*.py"):
            paths.add(p.resolve())
    # Plus tool base + registry modules.
    for rel in ("tools/base.py", "tools/inmemory.py"):
        p = HARNESSX_SRC_ROOT / rel
        if p.exists():
            paths.add(p.resolve())

    return tuple(str(p) for p in sorted(paths))
