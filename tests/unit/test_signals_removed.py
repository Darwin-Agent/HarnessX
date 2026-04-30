# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Guard: recipe/gaia_evolver/signals.py is gone and unreferenced."""

from __future__ import annotations

import pathlib
import re


REPO = pathlib.Path(__file__).resolve().parents[2]


def test_signals_file_deleted():
    path = REPO / "recipe" / "gaia_evolver" / "signals.py"
    assert not path.exists(), "recipe/gaia_evolver/signals.py must be deleted"


def test_no_imports_of_signals_remain():
    # Grep scan: any `import signals` or `from .signals` / `from recipe.gaia_evolver.signals`
    # outside the tests/ directory is a regression.
    pattern = re.compile(r"\b(from\s+(?:\.|recipe\.gaia_evolver\.)?signals|import\s+signals)\b")
    hits: list[str] = []
    for py in (REPO / "recipe" / "gaia_evolver").rglob("*.py"):
        if py.name == "signals.py":
            continue
        for i, line in enumerate(py.read_text().splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{py}:{i}: {line.strip()}")
    assert not hits, "stale signals imports:\n" + "\n".join(hits)
