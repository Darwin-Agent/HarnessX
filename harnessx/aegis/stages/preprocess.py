# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Stage P — Preprocess.

P.1 Cleaner (deterministic): dedup repeated tool outputs + externalize large
    content blocks (>2KB) to media/.
P.3 Aggregate (deterministic): cluster digests + emit summary (added in T12).

P.2 Digester (LLM per-task) lives in agents/digester.py (T13).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from ..agents.digester import build_digester_harness, DigesterInputs
from .trace_facts import extract_trace_facts


def _hash_tool_output(event: dict) -> str:
    tool = event.get("tool", "")
    out = event.get("output", "")
    return hashlib.sha1(f"{tool}::{out}".encode("utf-8")).hexdigest()


def clean_trajectory(
    raw_path: Path,
    out_path: Path,
    *,
    externalize_threshold: int = 2048,
    media_dir: Path | None = None,
) -> None:
    seen_tool_hashes: set[str] = set()
    out_events: list[dict] = []

    for line in raw_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)

        if ev.get("type") == "tool_result":
            h = _hash_tool_output(ev)
            if h in seen_tool_hashes:
                continue
            seen_tool_hashes.add(h)

        content = ev.get("content") or ev.get("output")
        if media_dir is not None and content and len(str(content)) > externalize_threshold:
            media_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha1(str(content).encode("utf-8")).hexdigest()[:16]
            media_path = media_dir / f"{digest}.txt"
            media_path.write_text(str(content), encoding="utf-8")
            ev.pop("content", None)
            ev.pop("output", None)
            ev["content_ref"] = str(media_path.relative_to(media_dir.parent))

        out_events.append(ev)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for e in out_events:
            fp.write(json.dumps(e, ensure_ascii=False) + "\n")


def classify_pattern(pass_flags: list[bool]) -> str:
    """Classify outcome pattern from a list of pass/fail flags.

    Returns:
        "PASS" or "FAIL" for single rollout.
        "ALL_PASS", "ALL_FAIL", or "PARTIAL_PASS" for multiple rollouts.
        "UNKNOWN" if empty.
    """
    if not pass_flags:
        return "UNKNOWN"
    if len(pass_flags) == 1:
        return "PASS" if pass_flags[0] else "FAIL"
    if all(pass_flags):
        return "ALL_PASS"
    if not any(pass_flags):
        return "ALL_FAIL"
    return "PARTIAL_PASS"


_FAILURE_MODE_RE = re.compile(r"failure_mode:\s*(\S+)")
_PATTERN_RE = re.compile(r"pattern:\s*(\S+)")


# C4: require the Latent Fragility section to have at least one real bullet
# — not just a header (`## Latent Fragility` followed by "None" or nothing).
# A bare header was causing every ALL_PASS digest to pass the 0.3 actionability
# gate, triggering unnecessary evolve rounds after ceiling was reached.
_FRAGILITY_RE = re.compile(
    r"##\s*Latent Fragility\s*\n\s*-\s+\S",
    re.IGNORECASE,
)


def _compute_actionability(
    pattern_counts: dict[str, int],
    cluster_list: list[dict],
    digest_texts: list[str],
) -> tuple[float, str]:
    """Compute an actionability score in [0, 1] and a human-readable reason.

    Scoring:
      1.0 — any ALL_FAIL or FAIL cluster (strongest signal)
      0.8 — PARTIAL_PASS present (high-signal divergence)
      0.3 — ALL_PASS only but digests surface latent fragilities (worth refining)
      0.0 — ALL_PASS only with no fragility material (true no-op)
    """
    fail = pattern_counts.get("ALL_FAIL", 0) + pattern_counts.get("FAIL", 0)
    partial = pattern_counts.get("PARTIAL_PASS", 0)
    all_pass = pattern_counts.get("ALL_PASS", 0) + pattern_counts.get("PASS", 0)

    if fail > 0:
        return 1.0, f"ALL_FAIL/FAIL clusters present ({fail} digests)"
    if partial > 0:
        return 0.8, f"PARTIAL_PASS digests present ({partial})"
    if all_pass > 0:
        fragile = sum(1 for t in digest_texts if _FRAGILITY_RE.search(t))
        if fragile > 0:
            return 0.3, f"ALL_PASS only but {fragile} digests report latent fragility"
        return 0.0, f"ALL_PASS only ({all_pass}), no fragility surfaced — no actionable signal"
    return 0.0, "no digests"


def aggregate_digests(
    *, digests_dir: Path, summary_path: Path, cluster_path: Path = None,
) -> dict:
    """Aggregate per-task digests into an overview + actionability score.

    No forced clustering. Each digest's self-reported ``failure_mode`` is a
    free-text tag; the overview simply lists `(task_id, pattern, failure_mode)`
    so the downstream Planner can synthesize cross-trace themes itself with
    access to full digest bodies.

    ``cluster_path`` is a deprecated argument kept for back-compat; when
    provided we no longer write anything to it. Nothing downstream reads
    that file in the new design.

    Returns:
        {"actionability": float [0,1], "actionability_reason": str}
    """
    pattern_counts: dict[str, int] = defaultdict(int)
    digest_texts: list[str] = []
    digest_entries: list[dict] = []

    for md in sorted(digests_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        task_id = md.stem
        digest_texts.append(text)

        pat_m = _PATTERN_RE.search(text)
        fm_m = _FAILURE_MODE_RE.search(text)
        pattern = pat_m.group(1) if pat_m else ""
        failure_mode = fm_m.group(1) if fm_m else ""
        if pattern:
            pattern_counts[pattern] += 1
        digest_entries.append({
            "task_id": task_id,
            "pattern": pattern,
            "failure_mode": failure_mode,
        })

    score, reason = _compute_actionability(
        dict(pattern_counts), [], digest_texts,
    )

    # overview.md body: flat digest list. No grouping — Planner synthesizes.
    lines = ["# Round overview", ""]
    lines.append(f"Actionability: {score:.2f} — {reason}")
    lines.append("")
    lines.append("## Pattern distribution (count of digests per pattern)")
    for p, n in sorted(pattern_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {p}: {n}")
    lines.append("")
    lines.append(
        "## Per-task digest index "
        "(each `failure_mode` is the Digester's self-assigned free-text tag)"
    )
    for e in digest_entries:
        tag = e["failure_mode"] or "(no tag)"
        lines.append(f"- `{e['task_id']}` — pattern={e['pattern']} tag=`{tag}`")
    lines.append("")
    lines.append(
        "Full per-task analysis with evidence anchors lives under "
        "`digests/<task_id>.md`. The Planner reads those directly and "
        "decides which tags actually group together semantically — no "
        "structural clustering is imposed here."
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    return {"actionability": score, "actionability_reason": reason}


async def _run_digester(inputs: DigesterInputs, harness) -> None:
    """Run a single digester invocation. Production path uses a real Harness;
    tests monkey-patch this helper to bypass real LLM calls.
    """
    if harness is None:
        return
    from harnessx import BaseTask
    task = BaseTask(
        description=(
            f"Digest task {inputs.task_id} (pattern={inputs.pattern}). "
            f"Read the trajectories listed in the system prompt, then call "
            f"write_tool ONCE with the required digest format. You are not "
            f"done until you have called write_tool — final_output alone "
            f"will be discarded."
        ),
        max_steps=200,
        max_cost_usd=100.0,
    )
    await harness.run(task)


async def run_stage_p(
    *,
    raw_dir: Path,
    trajectories_dir: Path,
    digests_dir: Path,
    summary_path: Path,
    pass_flags_by_task: dict[str, list[bool]],
    cluster_path: Path = None,  # deprecated; no longer written
    harness_factory=None,
    concurrency: int = 4,
) -> dict:
    """Stage P end-to-end: clean raw sessions, dispatch digesters per task, aggregate.

    harness_factory: Callable[[DigesterInputs], Harness] | None
        Production caller supplies a factory that returns a runnable Harness for
        a given digester input (usually `ModelConfig.agentic(build_digester_harness(inputs))`).
        Tests pass None and monkey-patch _run_digester directly.
    """
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    digests_dir.mkdir(parents=True, exist_ok=True)

    # Accept both .jsonl.raw (legacy) and .jsonl (GAIA recipe). If both
    # exist for the same stem prefer the .jsonl.raw to preserve existing
    # behaviour.
    raw_files = sorted(raw_dir.glob("*.jsonl.raw")) + sorted(raw_dir.glob("*.jsonl"))
    seen: set[str] = set()
    raw_files_filtered: list[Path] = []
    for p in raw_files:
        stem = p.name.replace(".jsonl.raw", "").replace(".jsonl", "")
        if stem in seen:
            continue
        seen.add(stem)
        raw_files_filtered.append(p)

    task_to_clean_paths: dict[str, list[Path]] = {}
    for raw in raw_files_filtered:
        stem = raw.name.replace(".jsonl.raw", "").replace(".jsonl", "")
        task_id = stem.rsplit("_r", 1)[0] if "_r" in stem else stem
        out = trajectories_dir / f"{stem}.jsonl"
        clean_trajectory(raw, out, media_dir=trajectories_dir.parent / "media")
        task_to_clean_paths.setdefault(task_id, []).append(out)

    sem = asyncio.Semaphore(concurrency)

    async def _one(task_id: str, traj_paths: list[Path]) -> None:
        async with sem:
            pattern = classify_pattern(pass_flags_by_task.get(task_id, []))
            # Layer A — mechanical extraction (no LLM). Facts are passed to
            # the Digester so it reasons from grounded evidence, then also
            # prepended to the final digest so Planner/Evolver/Critic see
            # them without Digester re-writing (and possibly distorting) them.
            facts = extract_trace_facts(task_id, traj_paths)
            facts_md = facts.to_markdown()
            inputs = DigesterInputs(
                task_id=task_id,
                pattern=pattern,
                trajectory_paths=traj_paths,
                digest_out_path=digests_dir / f"{task_id}.md",
                trace_facts_md=facts_md,
            )
            harness = harness_factory(inputs) if harness_factory else None
            await _run_digester(inputs, harness)
            digest_path = digests_dir / f"{task_id}.md"
            if digest_path.exists():
                existing = digest_path.read_text(encoding="utf-8")
                if "## Trace Facts (Layer A" not in existing:
                    digest_path.write_text(
                        facts_md + "\n\n" + existing, encoding="utf-8"
                    )

    await asyncio.gather(*[
        _one(tid, paths) for tid, paths in task_to_clean_paths.items()
    ])

    aggregate_result = aggregate_digests(
        digests_dir=digests_dir,
        summary_path=summary_path,
    ) or {}

    # IV-1: verify each digest's anchors point at plausibly real trajectories.
    # A digest with broken anchors can't be trusted as evidence for Planner /
    # Evolver / Critic — so flag it in the summary (don't drop silently).
    import logging
    from ..gates.structure import validate_digest_anchors
    _log = logging.getLogger("aegis.stage_p")
    # digest anchors are of the form [trajectories/<file>#<locator>] and
    # validator resolves them as `digest_root / prefix / relpath`, so pass
    # the round dir (parent of digests_dir) as digest_root.
    digest_root = digests_dir.parent
    degraded: list[tuple[str, str]] = []
    for md in sorted(digests_dir.glob("*.md")):
        result = validate_digest_anchors(md.read_text(encoding="utf-8"), digest_root)
        if not result.ok:
            degraded.append((md.name, result.reason))
            _log.warning("Digest %s has broken anchors: %s", md.name, result.reason)
    if degraded:
        with summary_path.open("a", encoding="utf-8") as fh:
            fh.write("\n\n## Degraded digests (anchor validation failed)\n")
            for name, reason in degraded:
                fh.write(f"- **{name}**: {reason}\n")

    return {
        "task_count": len(task_to_clean_paths),
        "digests_dir": str(digests_dir),
        "summary_path": str(summary_path),
        "actionability": aggregate_result.get("actionability", 0.0),
        "actionability_reason": aggregate_result.get("actionability_reason", ""),
        "degraded_digest_count": len(degraded),
    }
