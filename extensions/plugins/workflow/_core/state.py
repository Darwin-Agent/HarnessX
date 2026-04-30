# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path


class WorkflowPluginState:
    """Persists internalization bookkeeping to disk.

    Fields
    ------
    task_start_idx : int
        Index of the *current* task's first user message in state.messages.
        Updated at the beginning of every on_task_start.
    internalized_idxs : set[int]
        task_start_idx values that have already been internalized into a
        workflow YAML.  Prevents re-processing the same session segment.

    Persistence
    -----------
    State is written to ``<workflow_dir>/.state.json`` as a plain JSON file.
    Reads are lazy (on first access); writes happen immediately after mutation.
    """

    _FILENAME = ".state.json"

    def __init__(self, workflow_dir: str) -> None:
        self._path = Path(workflow_dir) / self._FILENAME
        self._loaded = False
        self._task_start_idx: int = 0
        self._internalized_idxs: set[int] = set()

    # ── Load / Save ────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._task_start_idx = int(data.get("task_start_idx", 0))
            self._internalized_idxs = set(int(x) for x in data.get("internalized_idxs", []))
        except Exception:
            pass  # corrupt file — start fresh

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(
                    {
                        "task_start_idx": self._task_start_idx,
                        "internalized_idxs": sorted(self._internalized_idxs),
                    },
                    indent=2,
                )
            )
        except Exception:
            pass  # non-fatal — state loss only means potential duplicate processing

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def task_start_idx(self) -> int:
        self._ensure_loaded()
        return self._task_start_idx

    @task_start_idx.setter
    def task_start_idx(self, value: int) -> None:
        self._ensure_loaded()
        self._task_start_idx = value
        self._save()

    @property
    def internalized_idxs(self) -> set[int]:
        self._ensure_loaded()
        return self._internalized_idxs

    def mark_internalized(self, idx: int) -> None:
        self._ensure_loaded()
        self._internalized_idxs.add(idx)
        self._save()

    def is_internalized(self, idx: int) -> bool:
        self._ensure_loaded()
        return idx in self._internalized_idxs
