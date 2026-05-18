# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Per-ship per-task attribution evidence.

When two or more candidates ship in the same round, the scoreboard's
predicted-task hit rate cannot tell which candidate actually drove a
PARTIAL→ALL_PASS or ALL_FAIL→PASS transition: the predictions overlap
or the same task moves regardless of which mutation caused it.

This module assigns one of three labels to each (ship, predicted_task)
pair based on mechanical evidence in the round's trajectories:

    direct  — the ship's mechanical signature fired on the task. For a
              tools/processor candidate this means "the new tool/processor
              was actually invoked"; the candidate is a credible cause.
    orphan  — the ship has a mechanical signature but it did NOT fire on
              this task. Whatever moved this task was someone else's
              work. Surfaces miscredits in multi-ship rounds.
    joint   — the ship's bucket has no mechanical signature (prompt /
              config kwarg). Without an ablation we cannot disentangle
              cause; credit is shared with any concurrent ships.

Bucket → default mechanical signature:
    tools     → tool_call match against the candidate's new tool name
    processor → processor_invocation hooks emitted by the new class
    prompt    → none (joint by default)
    config    → none (joint by default)

The Evolver may override the default by declaring an explicit
``attribution_signature`` field in the candidate manifest, e.g.

    attribution_signature:
      type: tool_call
      tool_name: SmartFetch
      expected_min_calls: 1

Pure-prompt rounds therefore degrade gracefully (everything is joint —
same as today, but explicitly so), and any round with at least one
mechanical-signature ship yields clean attribution for that ship.
"""

from __future__ import annotations

import re
from pathlib import Path

# Tools/processor signatures with this many invocations or more on a
# task → "direct" evidence. Below the floor, a single accidental call
# does not credit the ship.
_DEFAULT_MIN_CALLS = 1


_TOOL_COUNT_RE = re.compile(r'"([^"]+)"\s*:\s*(\d+)')


def _parse_tool_call_counts(md_text: str) -> dict[str, int]:
    """Read the ``tool_call_counts`` JSON-ish dict from a trajectory's
    YAML frontmatter. Robust to either ``{}`` or single-line dicts.
    """
    m = re.search(r"^tool_call_counts:\s*(\{[^}]*\})", md_text, re.M)
    if not m:
        return {}
    body = m.group(1)
    out: dict[str, int] = {}
    for tm in _TOOL_COUNT_RE.finditer(body):
        try:
            out[tm.group(1)] = int(tm.group(2))
        except ValueError:
            continue
    return out


def _infer_default_signature(
    bucket: str | None,
    manifest: dict | None,
) -> dict | None:
    """Best-effort signature inference when the manifest does not declare
    one explicitly.

    The Evolver should declare ``attribution_signature`` for tools and
    processor candidates; this fallback exists so pre-existing manifests
    keep producing useful attribution.
    """
    if bucket == "tools":
        # Strongest signal we can guess: the candidate's file_changes
        # mention a tool registration. We look for a ``::ToolName`` token
        # in any "create"-action filename or in the manifest body. If
        # nothing matches, fall back to "no signature" (joint).
        if not isinstance(manifest, dict):
            return None
        for fc in manifest.get("file_changes") or []:
            if not isinstance(fc, dict):
                continue
            path = str(fc.get("path") or "")
            if path.endswith(".py"):
                # Names of the form smart_fetch.py → SmartFetch is a
                # plausible registry name (PascalCase of stem).
                stem = Path(path).stem
                if stem and "_" in stem:
                    pascal = "".join(p.title() for p in stem.split("_"))
                    return {
                        "type": "tool_call",
                        "tool_name": pascal,
                        "expected_min_calls": _DEFAULT_MIN_CALLS,
                    }
        return None
    if bucket == "processor":
        # Same idea — pull a class name from file_changes if any.
        if not isinstance(manifest, dict):
            return None
        for fc in manifest.get("file_changes") or []:
            if not isinstance(fc, dict):
                continue
            path = str(fc.get("path") or "")
            if path.endswith(".py"):
                stem = Path(path).stem
                if stem and "_" in stem:
                    pascal = "".join(p.title() for p in stem.split("_")) + "Processor"
                    return {
                        "type": "processor_invocation",
                        "class_name": pascal,
                    }
        return None
    return None


def _check_signature(
    sig: dict,
    run_root: Path,
    round_n: int,
    task_id: str,
) -> str:
    """Return ``direct`` if the signature fired on this task in any
    rollout, ``orphan`` otherwise. Trajectories named ``<tid>.md`` (k=1)
    or ``<tid>_r<i>.md`` (k>=2) are both checked.
    """
    traj_dir = run_root / f"R{round_n}" / "trajectories"
    if not traj_dir.exists():
        return "orphan"

    candidates = list(traj_dir.glob(f"{task_id}.md")) + list(traj_dir.glob(f"{task_id}_r*.md"))
    if not candidates:
        return "orphan"

    sig_type = str(sig.get("type") or "")

    if sig_type == "tool_call":
        target = str(sig.get("tool_name") or "")
        floor = int(sig.get("expected_min_calls") or _DEFAULT_MIN_CALLS)
        if not target:
            return "joint"
        for md in candidates:
            counts = _parse_tool_call_counts(md.read_text(encoding="utf-8"))
            if counts.get(target, 0) >= floor:
                return "direct"
        return "orphan"

    if sig_type == "processor_invocation":
        # Best mechanical proxy without a tracer-event index: the trajectory
        # body mentions the processor class name in a tool-result/event
        # block. Approximate but cheap and consistent with "did the new
        # processor surface anywhere observable to the model".
        target = str(sig.get("class_name") or "")
        if not target:
            return "joint"
        for md in candidates:
            if target in md.read_text(encoding="utf-8"):
                return "direct"
        return "orphan"

    # Unknown signature type → cannot verify mechanically.
    return "joint"


def compute_evidence(
    run_root: Path,
    *,
    round_n: int,
    bucket: str | None,
    predicted_tasks: list[str],
    manifest: dict | None = None,
    attribution_signature: dict | None = None,
) -> dict[str, str]:
    """Per-task evidence label (``direct`` / ``orphan`` / ``joint``).

    ``attribution_signature`` overrides everything; otherwise we infer a
    default from the bucket+manifest. Buckets without any mechanical
    fingerprint (prompt, config) return ``joint`` for every predicted
    task.
    """
    sig = attribution_signature or _infer_default_signature(bucket, manifest)
    if sig is None:
        return {tid: "joint" for tid in predicted_tasks}
    return {tid: _check_signature(sig, run_root, round_n, tid) for tid in predicted_tasks}


def summarize_evidence(evidence: dict[str, str]) -> dict[str, int]:
    """``{"direct": N, "joint": M, "orphan": K}`` — handy for headlines."""
    out = {"direct": 0, "joint": 0, "orphan": 0}
    for v in evidence.values():
        if v in out:
            out[v] += 1
    return out
