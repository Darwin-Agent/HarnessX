# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from harnessx import HarnessConfig, BaseTask, ModelConfig
from harnessx.processors.control.compaction import CompactionProcessor
from harnessx.tracing.journal import HarnessJournal

try:
    from ._utils import load_provider, make_test_workspace
except ImportError:
    from _utils import load_provider, make_test_workspace


_SESSION_ID = "e2e-compaction-100tok"
_WS_NAME = "compaction_e2e"


def _build_config(provider):
    """HarnessConfig with token_threshold=100 — fires after ~1 model response."""
    ws = make_test_workspace(_WS_NAME)
    compaction = CompactionProcessor(
        token_threshold=10,  # fires almost immediately (any non-trivial response exceeds this)
        message_threshold=999_999,  # disable message trigger
        retention_window=1,
    )
    model_config = ModelConfig(main=provider)
    config = HarnessConfig(
        tracer=HarnessJournal(silent=True, session_id=_SESSION_ID),
        workspace=ws,
        processors=[compaction],
        init_workspace=False,
    )
    return model_config, config


async def run_compaction_e2e(provider) -> dict:
    """Run a short multi-step task and verify segment rotation artifacts."""
    ws = make_test_workspace(_WS_NAME)
    ws_root = ws.root
    model_config, config = _build_config(provider)
    harness = model_config.agentic(config)

    task = BaseTask(
        description=("Count from 1 to 3, one number per message. Say '1', then '2', then '3', then say 'Done'."),
        max_steps=6,
    )

    checks: list[tuple[str, bool, str]] = []

    def check(name, cond, detail=""):
        checks.append((name, cond, detail))
        status = "✓" if cond else "✗"
        print(f"  {status} {name}: {detail}")

    try:
        result = await harness.run(task)

        check(
            "task_completes",
            result.exit_reason in ("done", "budget_exceeded"),
            f"exit_reason={result.exit_reason}",
        )

        # ── File system checks ────────────────────────────────────────────────
        session_dir = ws_root / "sessions" / _SESSION_ID
        check(
            "session_dir_exists",
            session_dir.is_dir(),
            str(session_dir),
        )

        if session_dir.is_dir():
            segment_files = sorted(session_dir.glob("*.jsonl"))
            check(
                "multiple_segments_created",
                len(segment_files) >= 2,
                f"found {len(segment_files)} .jsonl files: {[f.name for f in segment_files]}",
            )

            state_files = sorted(session_dir.glob("*_state.json"))
            check(
                "checkpoint_files_exist",
                len(state_files) >= 1,
                f"found {len(state_files)} _state.json files",
            )

            compaction_checkpoints = []
            for sf in state_files:
                with open(sf) as f:
                    data = json.load(f)
                if data.get("segment_end_reason") == "compaction":
                    compaction_checkpoints.append(sf.name)
            check(
                "compaction_checkpoint_written",
                len(compaction_checkpoints) >= 1,
                f"checkpoints with reason=compaction: {compaction_checkpoints}",
            )

        # ── Session index ─────────────────────────────────────────────────────
        index_path = ws_root / "sessions" / f"{_SESSION_ID}.json"
        check("session_index_exists", index_path.exists(), str(index_path))

        if index_path.exists():
            with open(index_path) as f:
                idx = json.load(f)
            check(
                "index_has_multiple_run_ids",
                len(idx.get("run_ids", [])) >= 2,
                f"run_ids={idx.get('run_ids', [])}",
            )

        # ── wake() recovery ───────────────────────────────────────────────────
        try:
            state = HarnessJournal.wake(_SESSION_ID, str(ws_root))
            check(
                "wake_restores_state",
                state is not None and state.step >= 1,
                f"step={state.step if state else 'N/A'}",
            )
        except Exception as e:
            check("wake_restores_state", False, f"{type(e).__name__}: {e}")

    except Exception as e:
        check("no_exception", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    passed = all(c[1] for c in checks)
    return {"passed": passed, "checks": checks}


async def main():
    print("E2E Compaction Test — token_threshold=10 (extreme, forces immediate compaction)")
    print("=" * 55)

    provider = load_provider()
    print(f"Provider: {getattr(provider, 'model', provider)}")
    print(f"Workspace: {make_test_workspace(_WS_NAME).root}\n")

    result = await run_compaction_e2e(provider)

    print("\n" + ("PASS" if result["passed"] else "FAIL"))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))


# ─── Pytest entry point ────────────────────────────────────────────────────────

import pytest


@pytest.mark.asyncio
async def test_compaction_e2e():
    provider = load_provider()
    result = await run_compaction_e2e(provider)
    failed = [f"  ✗ {name}: {detail}" for name, ok, detail in result["checks"] if not ok]
    assert result["passed"], "Compaction e2e checks failed:\n" + "\n".join(failed)
