# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""TB2 ↔ AEGIS adapter.

Bridges the TB2 eval output layout (nested trial/session directories) to the
flat JSONL layout expected by AEGIS Stage P.

TB2 trials layout (under trials_dir):
  {task_id}__{trial_id}/
    agent/
      oh_runs/
        {session_id}/
          {run_id}.jsonl          ← event log (fed to AEGIS)
          {run_id}_trace.jsonl    ← trace (ignored by Stage P)
    result.json                   ← verifier reward
    verifier/
      ctrf.json                   ← per-test CTRF output

AEGIS Stage P expects (under raw_dir):
  {task_id}_r0.jsonl
  {task_id}_r1.jsonl
  ...
  (one file per trial, named {task_id}_r{N}.jsonl)
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from recipe.tb2_hx_evolver.score import tb2_score_by_session_dir

logger = logging.getLogger(__name__)


def _find_event_jsonl(session_dir: Path) -> Path | None:
    """Return the primary event JSONL in a session directory (not _trace)."""
    candidates = sorted(
        p for p in session_dir.glob("*.jsonl") if not p.stem.endswith("_trace")
    )
    return candidates[-1] if candidates else None


def flatten_trials_to_raw(
    trials_dir: Path,
    raw_dir: Path,
) -> dict[str, list[bool]]:
    """Copy TB2 trial JSONL files to raw_dir and return pass flags.

    For each task_id found under trials_dir, collects all session directories
    (one per trial), copies the event JSONL to
    ``raw_dir/{task_id}_r{i}.jsonl``, and scores each trial via
    :func:`tb2_score_by_session_dir`.

    Returns:
        pass_flags_by_task: dict[str, list[bool]]
            One bool per trial, in trial order.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Collect session dirs grouped by task_id.
    # Dir names: {task_id}__{trial_id}/agent/oh_runs/{session_id}/
    task_sessions: dict[str, list[Path]] = {}
    for trial_top in sorted(trials_dir.iterdir()):
        if not trial_top.is_dir() or "__" not in trial_top.name:
            continue
        task_id = trial_top.name.split("__")[0]
        oh_runs = trial_top / "agent" / "oh_runs"
        if not oh_runs.is_dir():
            continue
        for session_dir in sorted(oh_runs.iterdir()):
            if session_dir.is_dir():
                task_sessions.setdefault(task_id, []).append(session_dir)

    pass_flags_by_task: dict[str, list[bool]] = {}
    for task_id, session_dirs in sorted(task_sessions.items()):
        flags: list[bool] = []
        for i, session_dir in enumerate(session_dirs):
            # Copy event JSONL → raw_dir/{task_id}_r{i}.jsonl
            src = _find_event_jsonl(session_dir)
            if src is None:
                logger.warning("No event JSONL in %s — skipping", session_dir)
                flags.append(False)
                continue
            dst = raw_dir / f"{task_id}_r{i}.jsonl"
            shutil.copy2(src, dst)
            # Score
            try:
                passed, _score, _fb = tb2_score_by_session_dir(session_dir)
            except Exception as exc:
                logger.warning("Score failed for %s: %s", session_dir, exc)
                passed = False
            flags.append(passed)
        if flags:
            pass_flags_by_task[task_id] = flags
        else:
            logger.warning("No trials for task %s under %s", task_id, trials_dir)

    return pass_flags_by_task


def discover_task_ids(trials_dir: Path) -> list[str]:
    """Return unique task IDs from {task_id}__{trial_id} subdirs."""
    task_ids: set[str] = set()
    for entry in trials_dir.iterdir():
        if entry.is_dir() and "__" in entry.name:
            task_ids.add(entry.name.split("__")[0])
    return sorted(task_ids)
