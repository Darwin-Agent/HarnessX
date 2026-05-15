# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""CLI smoke tests for recipe/gaia_evolver/run_meta_aegis.py.

These tests do NOT call any LLM provider. They verify:
- ``--help`` exits 0 and advertises the expected flags
- ``--dry-run --smoke`` exits 0 without hitting any provider

If the default GAIA data path is missing, the dry-run test falls back to a
tiny mock JSON in ``tmp_path``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_PATH = _REPO_ROOT / "recipe" / "gaia_evolver" / "data" / "webthinker_gaia_dev_classified.json"


def _make_mock_tasks_json(tmp_path: Path) -> Path:
    """Write a tiny classified-GAIA-shaped JSON with 2 tasks."""
    blob = {
        "questions": [
            {
                "task_id": "mock-t1",
                "Question": "What is 2+2?",
                "Level": 1,
                "answer": "4",
                "category": "Multi-hop (Multi-hop)",
                "Annotator_Metadata": {},
            },
            {
                "task_id": "mock-t2",
                "Question": "Capital of France?",
                "Level": 1,
                "answer": "Paris",
                "category": "Factoid (Factoid)",
                "Annotator_Metadata": {},
            },
        ]
    }
    p = tmp_path / "mock_tasks.json"
    p.write_text(json.dumps(blob), encoding="utf-8")
    return p


def test_pilot_cli_help():
    r = subprocess.run(
        [sys.executable, "-m", "recipe.gaia_evolver.run_meta_aegis", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(_REPO_ROOT),
    )
    assert r.returncode == 0, f"stderr={r.stderr}"
    assert "--num-rounds" in r.stdout
    assert "--smoke" in r.stdout
    assert "--dry-run" in r.stdout
    assert "--num-evolvers" in r.stdout


def test_pilot_dry_run(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "recipe.gaia_evolver.run_meta_aegis",
        "--dry-run",
        "--smoke",
        "--run-tag",
        "test_dry",
    ]
    if not _DEFAULT_DATA_PATH.exists():
        cmd.extend(["--tasks", str(_make_mock_tasks_json(tmp_path))])
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
    )
    assert r.returncode == 0, f"stderr={r.stderr}"
    assert "DRY RUN" in r.stdout
    assert "AegisAgent" in r.stdout
