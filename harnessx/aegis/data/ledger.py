# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Cross-round persistent ledgers — the MAS's long-term memory.

Each helper writes to a file under ``<run_root>/data/``. Agents (Planner,
Evolver, Critic) Read these through their existing Read tool; they choose
what to consult. The orchestrator is responsible for keeping them current;
it does not decide what agents should read.

Files written:
- ``data/task_history.jsonl`` — one line per (round, task): pass bit + exit
  + step count. Agents use this to spot always-fail, always-pass, bouncer,
  recently-flipped, etc. Recipe-writer (not orchestrator) appends because
  per-task records live in the recipe layer.
- ``data/ship_outcomes.json`` — LIST (rewritten each round), one object
  per historical ship. Each object is re-filled with the latest
  ``predicted_tasks_status_latest`` so agents can see "this task was
  predicted to pass in R1 but is still failing in R6".
- ``data/rejected_candidates.jsonl`` — one line per rejected candidate
  (appended in commit stage). Backfill step re-reads and adds a
  ``revived_as`` field when a later brief's lead_pointer points to it.

Also writes ``<run_root>/INDEX.md`` — the agent's entry point. Rewritten
at the start of each round so it always reflects current state.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def data_dir(run_root: Path) -> Path:
    d = Path(run_root) / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False))
        fp.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            _log.warning("skipping malformed line in %s: %s", path, exc)
    return out


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# task_history.jsonl
# ---------------------------------------------------------------------------

_TASK_HISTORY = "task_history.jsonl"


@dataclass
class TaskRecord:
    round: int
    task_id: str
    level: str | None = None
    passed: bool = False  # any-k pass (optimistic; matches k=1 legacy)
    passed_flags: list[bool] = field(default_factory=list)  # full bit list
    k: int = 1  # number of rollouts aggregated in this row
    exit: str = ""
    steps: int = 0
    cost_usd: float = 0.0
    final_output_len: int = 0
    tools_used: list[str] = field(default_factory=list)


def append_task_history(run_root: Path, records: list[TaskRecord | dict]) -> Path:
    """Append a batch of per-task outcomes for one round.

    Accepts either TaskRecord instances or raw dicts with the same keys.
    Missing keys fall back to defaults. When ``passed_flags`` is absent
    but ``passed`` is supplied, we synthesize ``passed_flags=[passed]``
    and ``k=1`` for schema consistency.
    """
    path = data_dir(run_root) / _TASK_HISTORY
    for r in records:
        if isinstance(r, TaskRecord):
            row = asdict(r)
        else:
            passed = bool(r.get("passed", False))
            flags = r.get("passed_flags")
            if flags is None:
                flags = [passed]
            else:
                flags = [bool(x) for x in flags]
            row = {
                "round": int(r.get("round", -1)),
                "task_id": str(r.get("task_id", "")),
                "level": r.get("level"),
                "passed": passed,
                "passed_flags": flags,
                "k": int(r.get("k", len(flags)) or len(flags) or 1),
                "exit": str(r.get("exit", r.get("exit_reason", ""))),
                "steps": int(r.get("steps", 0) or 0),
                "cost_usd": float(r.get("cost_usd", 0.0) or 0.0),
                "final_output_len": int(r.get("final_output_len", len(str(r.get("final_output", "") or ""))) or 0),
                "tools_used": list(r.get("tools_used", []) or []),
            }
        if not row["task_id"]:
            continue
        _append_jsonl(path, row)
    return path


def read_task_history(run_root: Path) -> list[dict]:
    return _read_jsonl(data_dir(run_root) / _TASK_HISTORY)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ship_outcomes.json  (list, rewritten)
# ---------------------------------------------------------------------------

_SHIP_OUTCOMES = "ship_outcomes.json"


def record_ship_outcome(
    run_root: Path,
    *,
    round_n: int,
    shipped_cid: str,
    bucket: str,
    predicted_tasks: list[str],
    rejected_sibling_cids: list[str],
    signature: str | None = None,
    attribution_signature: dict | None = None,
    candidate_manifest: dict | None = None,
) -> Path:
    """Append a skeleton ship_outcome row on commit. Backfill later.

    ``attribution_signature`` (optional) is the manifest's declared
    mechanical fingerprint used by ``backfill_ship_outcomes`` to label
    each predicted task as direct/joint/orphan. ``candidate_manifest``
    is stored as a fallback so the backfill can infer a default
    signature when the Evolver did not declare one explicitly.
    """
    path = data_dir(run_root) / _SHIP_OUTCOMES
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []
    # Persist a *minimal* slice of the manifest — file_changes is what
    # the default-signature inferrer needs, and the bucket field is
    # already on the outcome. We deliberately do NOT inline the whole
    # manifest body (it can be tens of KB).
    manifest_slice = None
    if isinstance(candidate_manifest, dict):
        manifest_slice = {
            "bucket": candidate_manifest.get("bucket"),
            "file_changes": candidate_manifest.get("file_changes") or [],
        }
    entry = {
        "ship_id": shipped_cid,
        "round": int(round_n),
        "bucket": bucket,
        "signature": signature or "",
        "predicted_tasks": list(predicted_tasks),
        "rejected_sibling_cids": list(rejected_sibling_cids),
        "flipped_to_pass_in_ship_round": None,  # filled by backfill
        "hit_rate": None,
        "predicted_tasks_status_latest": {},  # filled by backfill
        "evidence_per_task": {},  # filled by backfill (attribution)
        "evidence_summary": {},  # {direct, joint, orphan}
        "attribution_signature": attribution_signature or None,
        "candidate_manifest_slice": manifest_slice,
        "superseded_by": None,  # v0.9.5: set when a later iterate replaces this ship
    }
    # Replace any previous entry with same ship_id (defensive; shouldn't happen
    # since each ship_id is per-round, but avoids duplicate rows if called twice)
    existing = [e for e in existing if e.get("ship_id") != shipped_cid]
    existing.append(entry)
    existing.sort(key=lambda e: int(e.get("round", 0)))
    _write_json_atomic(path, existing)
    return path


def _classify_state(flags: list[bool] | None) -> str | None:
    """Map a rollout pass-bit list to a 3-way state, k-agnostic.

    Returns ``ALL_PASS`` / ``ALL_FAIL`` / ``PARTIAL`` (or None if no rollouts
    recorded). For k=1, only the two extreme states are reachable (PARTIAL
    requires k>=2 with at least one differing bit).
    """
    if not flags:
        return None
    if all(flags):
        return "ALL_PASS"
    if not any(flags):
        return "ALL_FAIL"
    return "PARTIAL"


def _pass_rate(flags: list[bool] | None) -> float | None:
    if not flags:
        return None
    return sum(flags) / len(flags)


# Transition grades. The ship's hit/miss status is the union of "improving"
# grades (full_unlock, partial_unlock, stabilized, improved). Regression
# grades flag risk so the Critic can downweight a brittle ship.
_IMPROVED_GRADES = {"full_unlock", "partial_unlock", "stabilized", "improved"}
_REGRESSED_GRADES = {"regressed_hard", "regressed_soft", "regressed_partial"}


def _grade_transition(prev_flags: list[bool] | None, ship_flags: list[bool] | None) -> str:
    """Classify (prev_state, ship_state) into one of:

    Improving:
      - full_unlock     ALL_FAIL → ALL_PASS
      - partial_unlock  ALL_FAIL → PARTIAL
      - stabilized      PARTIAL  → ALL_PASS  (the canonical k>=2 evolve win)
      - improved        PARTIAL  → higher PARTIAL (k>=3 only)

    Regressing:
      - regressed_hard    ALL_PASS → ALL_FAIL
      - regressed_soft    ALL_PASS → PARTIAL
      - regressed_partial PARTIAL  → lower PARTIAL or ALL_FAIL

    Neutral:
      - noop_pass    ALL_PASS → ALL_PASS
      - noop_fail    ALL_FAIL → ALL_FAIL
      - noop_partial PARTIAL  → same-rate PARTIAL
      - unknown      pre or ship state missing
    """
    p = _pass_rate(prev_flags)
    s = _pass_rate(ship_flags)
    if p is None or s is None:
        return "unknown"
    if p == 0.0 and s == 1.0:
        return "full_unlock"
    if p == 0.0 and s > 0.0:
        return "partial_unlock"
    if 0.0 < p < 1.0 and s == 1.0:
        return "stabilized"
    if 0.0 < p < 1.0 and 0.0 < s < 1.0 and s > p:
        return "improved"
    if p == 1.0 and s == 0.0:
        return "regressed_hard"
    if p == 1.0 and 0.0 < s < 1.0:
        return "regressed_soft"
    if 0.0 < p < 1.0 and s < p:
        return "regressed_partial"
    if p == 0.0 and s == 0.0:
        return "noop_fail"
    if p == 1.0 and s == 1.0:
        return "noop_pass"
    return "noop_partial"


def backfill_ship_outcomes(run_root: Path) -> Path:
    """Re-compute hit metrics + status from task_history.

    Called at the END of each round's Stage P (after rollouts completed +
    task history is appended). Reads ``passed_flags`` (k-agnostic) so a
    PARTIAL_PASS → ALL_PASS stabilization counts as a hit — the previous
    implementation read ``passed = any(flags)`` and only credited
    ALL_FAIL → any-pass transitions, which made k>=2 stability gains
    invisible (every PARTIAL→ALL_PASS scored 0). For each prior ship,
    fills:

      Headline (back-compat fields, redefined slightly so they describe
      the same intent — what the ship achieved — but on the right metric):
        - hit_rate: ``X/Y`` where X = improving transitions on predicted
        - flipped_to_pass_in_ship_round: tasks that left ALL_FAIL
          (full_unlock OR partial_unlock; legacy semantic preserved)

      New (k-aware breakdown):
        - hit_rate_strict: full_unlock-only count (hardest progress class)
        - flipped_by_category: dict[grade -> list[task_id]] for every
          predicted task, including regression and noop classes so the
          Critic can see costs alongside gains.
        - predicted_tasks_status_latest: enriched with PARTIAL.
    """
    path = data_dir(run_root) / _SHIP_OUTCOMES
    if not path.exists():
        return path
    try:
        outcomes = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return path
    if not isinstance(outcomes, list):
        return path

    # Per-task (round -> list[bool]) from task_history. Prefer passed_flags
    # (full per-rollout bits) and fall back to passed (legacy single bool)
    # so old runs without passed_flags still backfill correctly.
    history = read_task_history(run_root)
    per_task: dict[str, dict[int, list[bool]]] = {}
    for row in history:
        tid = row.get("task_id")
        if not tid:
            continue
        flags = row.get("passed_flags")
        if not isinstance(flags, list) or not flags:
            flags = [bool(row.get("passed", False))]
        per_task.setdefault(tid, {})[int(row.get("round", 0))] = [bool(b) for b in flags]

    max_round = max((r for bits in per_task.values() for r in bits), default=-1)

    for entry in outcomes:
        ship_round = int(entry.get("round", 0))
        preds = entry.get("predicted_tasks", []) or []
        status_latest: dict[str, str] = {}
        by_category: dict[str, list[str]] = {}
        flipped_legacy: list[str] = []  # ALL_FAIL → any-pass
        full_unlock: list[str] = []  # ALL_FAIL → ALL_PASS only
        improved: list[str] = []  # any improving grade
        for tid in preds:
            bits = per_task.get(tid, {})
            prev_flags = bits.get(ship_round - 1)
            ship_flags = bits.get(ship_round)
            latest_round = max(bits) if bits else None
            latest_flags = bits.get(latest_round) if latest_round is not None else None

            latest_state = _classify_state(latest_flags)
            if latest_state == "ALL_PASS":
                status_latest[tid] = "passing"
            elif latest_state == "ALL_FAIL":
                status_latest[tid] = "still_failing"
            elif latest_state == "PARTIAL":
                status_latest[tid] = "partial"
            else:
                status_latest[tid] = "unknown"

            grade = _grade_transition(prev_flags, ship_flags)
            by_category.setdefault(grade, []).append(tid)
            if grade in _IMPROVED_GRADES:
                improved.append(tid)
            if grade == "full_unlock":
                full_unlock.append(tid)
            if grade in ("full_unlock", "partial_unlock"):
                flipped_legacy.append(tid)

        n = len(preds)
        entry["flipped_to_pass_in_ship_round"] = flipped_legacy
        entry["flipped_by_category"] = by_category
        entry["predicted_tasks_status_latest"] = status_latest
        entry["hit_rate"] = f"{len(improved)}/{n}" if n else "0/0"
        entry["hit_rate_strict"] = f"{len(full_unlock)}/{n}" if n else "0/0"
        entry["latest_round_in_history"] = max_round

        # Attribution: per-task direct/joint/orphan label, decoupled from
        # the hit-rate metric so the Critic can ask "did this candidate
        # mechanically fire on the task it claimed credit for?" without
        # double-counting same-round bucket-disjoint ships.
        from .attribution import compute_evidence, summarize_evidence

        evidence = compute_evidence(
            run_root,
            round_n=ship_round,
            bucket=entry.get("bucket"),
            predicted_tasks=preds,
            manifest=entry.get("candidate_manifest_slice"),
            attribution_signature=entry.get("attribution_signature"),
        )
        entry["evidence_per_task"] = evidence
        entry["evidence_summary"] = summarize_evidence(evidence)

    _write_json_atomic(path, outcomes)
    return path


def read_ship_outcomes(run_root: Path) -> list[dict]:
    path = data_dir(run_root) / _SHIP_OUTCOMES
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def mark_ship_superseded(run_root: Path, target_ship_id: str, by_cid: str) -> Path:
    """Mark an earlier ship as superseded by ``by_cid`` (v0.9.5 iterates_from).

    No-op if target is not in the ledger or is already superseded.
    Returns the ledger path regardless so callers can log it.
    """
    path = data_dir(run_root) / _SHIP_OUTCOMES
    if not path.exists():
        return path
    try:
        outcomes = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return path
    if not isinstance(outcomes, list):
        return path
    changed = False
    for entry in outcomes:
        if entry.get("ship_id") != target_ship_id:
            continue
        if entry.get("superseded_by"):
            # Already superseded — honor first claim, don't overwrite.
            break
        entry["superseded_by"] = by_cid
        changed = True
        break
    if changed:
        _write_json_atomic(path, outcomes)
    return path


def ship_ledger_for_gate(run_root: Path) -> dict:
    """Compact snapshot of ship_outcomes keyed by ship_id.

    IV-12 gate consumes this. Only the fields the gate cares about are
    kept — no full per-task status, no signatures. Cheap to build per
    round (called once, passed through run_stage_4).
    """
    out: dict[str, dict] = {}
    for entry in read_ship_outcomes(run_root):
        sid = entry.get("ship_id")
        if not sid:
            continue
        out[str(sid)] = {
            "round": entry.get("round"),
            "bucket": entry.get("bucket"),
            "hit_rate": entry.get("hit_rate"),
            "superseded_by": entry.get("superseded_by"),
        }
    return out


# ---------------------------------------------------------------------------
# rejected_candidates.jsonl
# ---------------------------------------------------------------------------

_REJECTED = "rejected_candidates.jsonl"
_LEAD_ARCHIVE_RE = re.compile(r"^\s*lead_pointer\s*:\s*archive:(\S+)", re.MULTILINE)


def append_rejected_candidates(
    run_root: Path,
    round_n: int,
    entries: list[dict],
) -> Path:
    """Append one line per rejected candidate.

    Each entry dict should have at least:
      candidate_id, bucket, predicted_tasks, rejection_text_excerpt
    Optional:
      signature
    """
    path = data_dir(run_root) / _REJECTED
    for e in entries:
        cid = e.get("candidate_id")
        if not cid:
            continue
        row = {
            "round": int(round_n),
            "candidate_id": cid,
            "bucket": e.get("bucket", ""),
            "predicted_tasks": list(e.get("predicted_tasks", []) or []),
            "rejection_text_excerpt": str(e.get("rejection_text_excerpt", ""))[:400],
            "signature": e.get("signature", ""),
            "revived_as": [],  # filled by backfill
        }
        _append_jsonl(path, row)
    return path


def backfill_rejected_revivals(run_root: Path, all_briefs_dirs: list[Path]) -> Path:
    """For each rejected candidate, note future briefs whose lead_pointer is archive:<cid>.

    Called once at end of each round after current round's briefs are written.
    Only rewrites if changes happen (idempotent).
    """
    path = data_dir(run_root) / _REJECTED
    rows = _read_jsonl(path)
    if not rows:
        return path

    # Build map: archived_cid -> [(round, brief_id)]
    revivals: dict[str, list[dict]] = {}
    for bdir in all_briefs_dirs:
        if not bdir.exists():
            continue
        for p in bdir.glob("B-R*.md"):
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                continue
            m = _LEAD_ARCHIVE_RE.search(text)
            if not m:
                continue
            archived_cid = m.group(1)
            # Derive round from directory path: .../R<n>/briefs/B-R<n>-NN.md
            r_match = re.search(r"/R(\d+)/briefs/", str(p))
            if not r_match:
                continue
            revivals.setdefault(archived_cid, []).append(
                {
                    "round": int(r_match.group(1)),
                    "brief_id": p.stem,
                }
            )

    changed = False
    for row in rows:
        cid = row.get("candidate_id")
        if cid in revivals:
            new_val = revivals[cid]
            if row.get("revived_as") != new_val:
                row["revived_as"] = new_val
                changed = True

    if changed:
        # Rewrite atomically: tmp + os.replace so a crash mid-write can't
        # leave a truncated jsonl with missing rejections.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    return path


def read_rejected_candidates(run_root: Path) -> list[dict]:
    return _read_jsonl(data_dir(run_root) / _REJECTED)


# ---------------------------------------------------------------------------
# INDEX.md
# ---------------------------------------------------------------------------

_INDEX = "INDEX.md"


def refresh_index_md(run_root: Path, current_round: int) -> Path:
    """Rewrite INDEX.md at run root.

    This is the agent's entry point. Agents read this first to learn what
    files exist, then decide what to Read based on their own judgment. No
    instruction to read specific files — just a catalog.
    """
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / _INDEX

    # Enumerate round dirs that exist
    existing_rounds = sorted(int(p.name[1:]) for p in run_root.glob("R[0-9]*") if p.is_dir() and p.name[1:].isdigit())

    data_exists = {
        "task_history.jsonl": (data_dir(run_root) / _TASK_HISTORY).exists(),
        "ship_outcomes.json": (data_dir(run_root) / _SHIP_OUTCOMES).exists(),
        "rejected_candidates.jsonl": (data_dir(run_root) / _REJECTED).exists(),
    }

    lines: list[str] = []
    lines.append(f"# Run state index — currently planning R{current_round}")
    lines.append("")
    lines.append(
        "You have Read / Glob / Grep / LS access to everything listed below. "
        "No required reading list — use your judgment about what supports "
        "the decision you need to make. The orchestrator does not decide "
        "what you read."
    )
    lines.append("")
    lines.append("## Per-round artifacts")
    if existing_rounds:
        lines.append(f"Rounds present on disk: {', '.join('R' + str(r) for r in existing_rounds)}")
        lines.append("")
        lines.append("For each round `R{n}`:")
        lines.append("- `R{n}/summary.md` — per-round overview + actionability + C2 follow-up (if any)")
        lines.append("- `R{n}/digests/*.md` — per-task failure analysis (ALL_FAIL / ALL_PASS / PARTIAL_PASS)")
        lines.append(
            "- `R{n}/regressions.md` — tasks whose pass-state worsened vs R{n-1}, "
            "with joint-suspect ships from the prior round. **Required reading** "
            "for Planner / Evolver / Critic; ship_outcomes.json's hit_rate does "
            "not penalise regressions, so this file is the only place collateral "
            "damage is surfaced."
        )
        lines.append("- `R{n}/landscape.md` — Planner's cross-trace synthesis")
        lines.append("- `R{n}/candidates/C-R{n}-NN.md` — Evolver candidate manifests (K candidates, variable)")
        lines.append("- `R{n}/applied/C-R{n}-NN/` — per-candidate scratch dirs (applied config + asset files)")
        lines.append("- `R{n}/decision.md` — Critic's ship / no_op decision")
        lines.append("- `R{n}/verdicts/V-C-R{n}-NN.md` — Critic per-candidate verdicts")
    else:
        lines.append("(no rounds present yet — this is round 0)")
    lines.append("")
    lines.append("## Cross-round ledgers (`data/`)")
    lines.append("")
    lines.append(
        f"- `data/task_history.jsonl` {'(present)' if data_exists['task_history.jsonl'] else '(not yet populated)'}\n"
        "  One line per (round, task). Fields: round, task_id, level, passed, "
        "exit, steps, cost_usd, final_output_len, tools_used.\n"
        "  Use to find: always-failing / always-passing / bouncer tasks, "
        "tasks that flipped recently, per-level pass trends, exit-reason "
        "distribution over time."
    )
    lines.append("")
    lines.append(
        f"- `data/ship_outcomes.json` {'(present)' if data_exists['ship_outcomes.json'] else '(not yet populated)'}\n"
        "  List of every historical ship, retrospectively filled. Each "
        "entry: ship_id, round, bucket, predicted_tasks, "
        "flipped_to_pass_in_ship_round, hit_rate, "
        "predicted_tasks_status_latest.\n"
        "  Use to find: predicted-hit rate per bucket, which ships' "
        "predictions missed, whether a past prediction still holds."
    )
    lines.append("")
    lines.append(
        f"- `data/rejected_candidates.jsonl` {'(present)' if data_exists['rejected_candidates.jsonl'] else '(not yet populated)'}\n"
        "  One line per rejected candidate. Fields: round, candidate_id, "
        "bucket, predicted_tasks, rejection_text_excerpt, signature, "
        "revived_as.\n"
        "  Use to find: ideas proposed repeatedly and rejected, rejection "
        "patterns per bucket (e.g. tools always rejected on uptake), "
        "archive candidates that were later revived."
    )
    lines.append("")
    lines.append("## Snapshots at run root")
    lines.append(
        "- `journal.md` — first-person prose memo per round (short)\n"
        "- `reputation.json` — per-bucket proposal/ship win-rate window\n"
        "- `curves.json` — per-round pass-rate trajectory (recipe-written)\n"
        "- `audit.jsonl` — structured event log (one line per stage / "
        "decision / gate / commit)"
    )
    lines.append("")
    lines.append("## Your constraints")
    lines.append(
        "- You have Read / Glob / Grep / LS — no Write outside your own output path.\n"
        "- Harness source is gated but living docs (base classes, built-in "
        "processors, built-in tools) are readable for verification.\n"
        "- No file under `data/` is required reading. But if you are making a "
        "decision that plausibly benefits from cross-round evidence, consult them."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def read_index(run_root: Path) -> str:
    path = Path(run_root) / _INDEX
    return path.read_text(encoding="utf-8") if path.exists() else ""
