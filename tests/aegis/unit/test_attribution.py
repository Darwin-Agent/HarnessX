"""Per-ship per-task attribution: distinguishes which candidate actually
moved a task in a multi-ship round.

Bucket → default mechanical signature:
  tools     → tool_call match
  processor → processor_invocation match
  prompt    → none (joint by default)
  config    → none (joint by default)

Manifest may declare an explicit ``attribution_signature`` to override
the default (e.g. tool name in PascalCase that differs from the .py stem).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from harnessx.aegis.data.attribution import (
    compute_evidence,
    summarize_evidence,
)


def _write_traj(path: Path, tool_call_counts: dict[str, int],
                body_extra: str = "") -> None:
    """Write a minimal trajectory .md whose frontmatter has the supplied
    tool_call_counts dict — that's what attribution reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    counts_json = _json.dumps(tool_call_counts)
    fm = (
        "---\n"
        "task_id: t\n"
        f"tool_call_counts: {counts_json}\n"
        "---\n\n"
        f"{body_extra}\n"
    )
    path.write_text(fm, encoding="utf-8")


def test_tools_bucket_direct_when_tool_called(tmp_path: Path) -> None:
    run = tmp_path
    _write_traj(run / "R1/trajectories/t1_r0.md", {"SmartFetch": 3})
    _write_traj(run / "R1/trajectories/t1_r1.md", {"SmartFetch": 0})
    ev = compute_evidence(
        run, round_n=1, bucket="tools",
        predicted_tasks=["t1"],
        attribution_signature={"type": "tool_call", "tool_name": "SmartFetch"},
    )
    assert ev["t1"] == "direct"


def test_tools_bucket_orphan_when_tool_not_called(tmp_path: Path) -> None:
    run = tmp_path
    _write_traj(run / "R1/trajectories/t1_r0.md", {"WebFetch": 5, "Bash": 2})
    _write_traj(run / "R1/trajectories/t1_r1.md", {"WebFetch": 1})
    ev = compute_evidence(
        run, round_n=1, bucket="tools",
        predicted_tasks=["t1"],
        attribution_signature={"type": "tool_call", "tool_name": "SmartFetch"},
    )
    assert ev["t1"] == "orphan"


def test_prompt_bucket_always_joint(tmp_path: Path) -> None:
    run = tmp_path
    _write_traj(run / "R1/trajectories/t1_r0.md", {"WebFetch": 5})
    ev = compute_evidence(
        run, round_n=1, bucket="prompt", predicted_tasks=["t1"],
    )
    assert ev["t1"] == "joint"


def test_default_signature_inferred_from_file_changes(tmp_path: Path) -> None:
    """tools-bucket candidate without explicit signature: infer from
    file_changes path stem (``smart_fetch.py`` → ``SmartFetch``)."""
    run = tmp_path
    _write_traj(run / "R1/trajectories/t1_r0.md", {"SmartFetch": 2})
    manifest = {
        "bucket": "tools",
        "file_changes": [
            {"path": "/abs/applied/C-X/smart_fetch.py", "action": "create"},
        ],
    }
    ev = compute_evidence(
        run, round_n=1, bucket="tools",
        predicted_tasks=["t1"], manifest=manifest,
    )
    assert ev["t1"] == "direct"


def test_no_trajectory_files_yields_orphan(tmp_path: Path) -> None:
    """Predicted task with no trajectory written (e.g., not part of this
    round's task set) ⇒ orphan, not direct."""
    run = tmp_path
    (run / "R1/trajectories").mkdir(parents=True)
    ev = compute_evidence(
        run, round_n=1, bucket="tools",
        predicted_tasks=["missing_task"],
        attribution_signature={"type": "tool_call", "tool_name": "X"},
    )
    assert ev["missing_task"] == "orphan"


def test_summarize_counts_three_buckets(tmp_path: Path) -> None:
    s = summarize_evidence({"a": "direct", "b": "direct", "c": "joint", "d": "orphan"})
    assert s == {"direct": 2, "joint": 1, "orphan": 1}


def test_unknown_signature_type_falls_back_to_joint(tmp_path: Path) -> None:
    run = tmp_path
    _write_traj(run / "R1/trajectories/t1_r0.md", {"X": 5})
    ev = compute_evidence(
        run, round_n=1, bucket="tools", predicted_tasks=["t1"],
        attribution_signature={"type": "totally_made_up", "tool_name": "X"},
    )
    assert ev["t1"] == "joint"


# ─── Integration with backfill ───────────────────────────────────────────


def test_backfill_writes_evidence_per_task(tmp_path: Path) -> None:
    """End-to-end: record_ship_outcome → backfill_ship_outcomes populates
    evidence_per_task and evidence_summary on the entry."""
    import json
    from harnessx.aegis.data import ledger
    run = tmp_path

    # Seed task_history (R0 PARTIAL, R1 stabilized)
    hist = run / "data" / "task_history.jsonl"
    hist.parent.mkdir(parents=True)
    hist.write_text(
        json.dumps({"round": 0, "task_id": "t1", "passed": True,
                    "passed_flags": [True, False]}) + "\n" +
        json.dumps({"round": 1, "task_id": "t1", "passed": True,
                    "passed_flags": [True, True]}) + "\n",
        encoding="utf-8",
    )

    # Record ship with attribution_signature
    _write_traj(run / "R1/trajectories/t1_r0.md", {"SmartFetch": 4})
    _write_traj(run / "R1/trajectories/t1_r1.md", {"SmartFetch": 2})
    ledger.record_ship_outcome(
        run, round_n=1, shipped_cid="C-R1-T", bucket="tools",
        predicted_tasks=["t1"], rejected_sibling_cids=[],
        signature="sig",
        attribution_signature={
            "type": "tool_call",
            "tool_name": "SmartFetch",
            "expected_min_calls": 1,
        },
    )
    ledger.backfill_ship_outcomes(run)
    [entry] = json.loads((run / "data" / "ship_outcomes.json").read_text())
    assert entry["evidence_per_task"]["t1"] == "direct"
    assert entry["evidence_summary"] == {"direct": 1, "joint": 0, "orphan": 0}


def test_backfill_evidence_for_prompt_bucket_is_all_joint(tmp_path: Path) -> None:
    import json
    from harnessx.aegis.data import ledger
    run = tmp_path

    hist = run / "data" / "task_history.jsonl"
    hist.parent.mkdir(parents=True)
    hist.write_text(
        json.dumps({"round": 0, "task_id": "t1", "passed_flags": [False, False]}) + "\n" +
        json.dumps({"round": 1, "task_id": "t1", "passed_flags": [True, True]}) + "\n",
        encoding="utf-8",
    )
    _write_traj(run / "R1/trajectories/t1_r0.md", {"WebFetch": 3})
    ledger.record_ship_outcome(
        run, round_n=1, shipped_cid="C-R1-P", bucket="prompt",
        predicted_tasks=["t1"], rejected_sibling_cids=[], signature="s",
    )
    ledger.backfill_ship_outcomes(run)
    [entry] = json.loads((run / "data" / "ship_outcomes.json").read_text())
    assert entry["evidence_per_task"]["t1"] == "joint"
    assert entry["evidence_summary"] == {"direct": 0, "joint": 1, "orphan": 0}
