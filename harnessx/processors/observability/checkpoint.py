# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sqlite3

from ...core.events import StepEndEvent
from ...core.processor import MultiHookProcessor
from ...logging import logger


class CheckpointProcessor(MultiHookProcessor):
    """
    Hooks: step_end
    Every ``every_n`` steps, serializes the state snapshot to SQLite.

    Reads the snapshot from ``StepEndEvent.state_snapshot`` — no external
    state injection needed. If ``state_snapshot`` is None, the step is skipped.

    Use ``CheckpointProcessor.load_checkpoint(run_id)`` to retrieve the latest
    snapshot for crash recovery.
    """

    _singleton_group = "checkpoint"
    _order = 10

    def __init__(self, every_n: int = 5, db_path: str = "checkpoints.db"):
        self.every_n = every_n
        self.db_path = db_path
        self._initialized = False

    def _ensure_db(self) -> None:
        if self._initialized:
            return
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id TEXT NOT NULL,
                step   INTEGER NOT NULL,
                snapshot TEXT NOT NULL,
                ts     REAL NOT NULL,
                PRIMARY KEY (run_id, step)
            )
        """)
        conn.commit()
        conn.close()
        self._initialized = True

    async def on_step_end(self, event: StepEndEvent):
        if event.state_snapshot is not None and event.step_id % self.every_n == 0:
            try:
                self._ensure_db()
                conn = sqlite3.connect(self.db_path)
                conn.execute(
                    "INSERT OR REPLACE INTO checkpoints (run_id, step, snapshot, ts) VALUES (?, ?, ?, ?)",
                    (
                        event.run_id,
                        event.step_id,
                        json.dumps(event.state_snapshot),
                        event.ts,
                    ),
                )
                conn.commit()
                conn.close()
                logger.debug(f"Checkpoint saved: run_id={event.run_id}, step={event.step_id}")
            except Exception as exc:
                logger.warning(f"Checkpoint failed: {exc}")

        yield event

    @classmethod
    def load_checkpoint(cls, run_id: str, db_path: str = "checkpoints.db") -> "dict | None":
        """Load the latest checkpoint snapshot for a given run_id."""
        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT snapshot FROM checkpoints WHERE run_id=? ORDER BY step DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            conn.close()
            if row:
                return json.loads(row[0])
        except Exception:
            pass
        return None
