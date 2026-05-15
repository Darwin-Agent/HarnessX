# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Structure gate — enforces evidence-driven invariants IV-1 ~ IV-6."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class GateResult:
    ok: bool
    reason: str = ""


# Accept both bracket-wrapped and backtick-wrapped anchor forms, since LLMs
# naturally produce either depending on prompt phrasing. Three observed
# real-world formats — all equally valid as evidence citations:
#
#   1. `trajectories/foo.jsonl#step_5`         (current round, inline locator)
#   2. `R6/trajectories/foo.jsonl#step_5`      (round-prefixed; C-R6-01 in
#                                                aegis_64_v4 was killed for this)
#   3. `digests/foo.md`                         (whole-digest citation; locator
#                                                optional for digests per schema)
#
# The previous regex rejected #2 (killed v3 R5 + v4 R6 candidates on valid
# one-line-fix config ships) and rejected #3 (over-strict even though the
# docstring-promised contract allowed digests to omit #).
_ANCHOR_RE = re.compile(
    r"(?:\[|`)"                                   # opening bracket OR backtick
    r"(?:R\d+/)?"                                 # optional "R<N>/" round prefix
    r"(sessions|trajectories|digests)/"
    r"([^\]`#]+)"                                 # path (no closers, no '#')
    r"(?:#([^\]`]+))?"                            # OPTIONAL "#<locator>"
    r"(?:\]|`)"                                   # closing bracket OR backtick
)

# Anchor format used in the YAML evidence_anchors list on Critic verdicts.
# Matches: "trajectories/abc.jsonl#step_4" or "digests/xyz.md".
# For trajectories/sessions the "#<locator>" suffix is MANDATORY (you must
# point at a specific step/message). For digests the "#<anchor>" suffix is
# OPTIONAL (a whole-digest citation is meaningful on its own). Split into
# two patterns so the docstring-promised contract is actually enforced.
_VERDICT_ANCHOR_RE_TRAJ = re.compile(
    r"^(sessions|trajectories)/[^\s#]+#[^\s#]+$"
)
_VERDICT_ANCHOR_RE_DIGEST = re.compile(
    r"^digests/[^\s#]+(?:#[^\s#]+)?$"
)


def _verdict_anchor_matches(anchor: str) -> bool:
    """Return True iff ``anchor`` satisfies the verdict-anchor contract.

    The prefix determines which pattern applies:
    - ``trajectories/`` or ``sessions/`` → must include ``#<locator>``
    - ``digests/`` → ``#<anchor>`` optional
    """
    a = anchor.strip()
    if a.startswith("digests/"):
        return bool(_VERDICT_ANCHOR_RE_DIGEST.match(a))
    if a.startswith(("trajectories/", "sessions/")):
        return bool(_VERDICT_ANCHOR_RE_TRAJ.match(a))
    return False

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse_anchors(text: str) -> list[tuple[str, str, str]]:
    return [(m.group(1), m.group(2), m.group(3)) for m in _ANCHOR_RE.finditer(text)]


def validate_digest_anchors(digest_md: str, digest_root: Path) -> GateResult:
    anchors = _parse_anchors(digest_md)
    if not anchors:
        return GateResult(ok=False, reason="digest contains zero citation anchors (IV-1)")
    # E2: memoise per-target line counts. A digest often cites the same
    # trajectory file multiple times (e.g. steps 3, 7, 12 of the same run)
    # and the previous implementation re-opened and full-scanned the file
    # per anchor.
    line_count_cache: dict[Path, int] = {}
    for prefix, relpath, locator in anchors:
        target = digest_root / prefix / relpath
        if not target.exists():
            return GateResult(ok=False, reason=f"anchor target missing: {target}")
        # locator may be ``None`` when the anchor is a whole-digest citation
        # like ``digests/foo.md`` (valid per the verdict-anchor contract:
        # digests may omit #<locator>). Skip the line-range check in that
        # case — file-exists is the full validation we can do.
        if locator is None:
            continue
        if locator.startswith("msg_") or locator.startswith("step_"):
            try:
                line_no = int(locator.split("_")[1])
            except (ValueError, IndexError):
                continue
            if target not in line_count_cache:
                try:
                    with target.open(encoding="utf-8") as fh:
                        line_count_cache[target] = sum(1 for _ in fh)
                except Exception:
                    # Sentinel: treat as unreadable — keep failing the
                    # anchor rather than raise.
                    line_count_cache[target] = -1
            total_lines = line_count_cache[target]
            if total_lines == -1:
                return GateResult(ok=False, reason=f"anchor target unreadable: {target}")
            if line_no >= total_lines:
                return GateResult(ok=False, reason=f"anchor line out of range in {target}")
    return GateResult(ok=True)


_ALLOWED_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "config":    (".yaml", ".yml"),
    "prompt":    (".md", ".yaml", ".yml"),
    "processor": (".py", ".yaml", ".yml"),
    "tools":     (".py", ".yaml", ".yml"),
}


def _normalize_bucket(bucket) -> list[str]:
    """Normalize manifest bucket value to a list of strings.

    Accepts ``str`` (legacy) or ``list[str]`` (v0.9.3 cross-bucket bundle).
    Empty / falsy input returns []. Invalid types return [] as well; the
    main validator will raise a clearer error upstream.
    """
    if bucket is None:
        return []
    if isinstance(bucket, str):
        return [bucket] if bucket else []
    if isinstance(bucket, (list, tuple)):
        return [str(b) for b in bucket if b]
    return []


def _check_bucket_file_consistency(bucket, file_changes: list) -> str | None:
    """v0.9.2 IV-9 helper: ensure file_changes match the declared bucket.

    Rules:
      - bucket=config:    only .yaml (config.yaml) allowed
      - bucket=prompt:    .md + .yaml allowed
      - bucket=processor: .py + .yaml allowed
      - bucket=tools:     .py + .yaml allowed

    v0.9.3: bucket may be a list (cross-bucket bundle). Allowed-extension
    set is the UNION of per-bucket allowlists — e.g. bucket=[prompt,
    processor] permits .md + .py + .yaml. This makes the legitimate
    "one candidate touches multiple facets" case first-class instead of
    forcing Evolver to split or smuggle via sibling scratch directories.

    Returns an error message if inconsistent, None otherwise. Malformed
    entries (no 'path' key) are skipped — structure validation catches them
    on the main path.
    """
    buckets = _normalize_bucket(bucket)
    if not buckets or not file_changes:
        return None
    # Union of allowed extensions across declared buckets.
    ok_ext_set: set[str] = set()
    unknown_bucket = False
    for b in buckets:
        exts = _ALLOWED_EXTENSIONS.get(b)
        if exts is None:
            unknown_bucket = True
            continue
        ok_ext_set.update(exts)
    if unknown_bucket and not ok_ext_set:
        return None  # all unknown — leave to future extension
    violations: list[str] = []
    for fc in file_changes:
        if not isinstance(fc, dict):
            continue
        path = str(fc.get("path", ""))
        if not path:
            continue
        if path.endswith(("__pycache__",)):
            continue
        lower = path.lower()
        if not any(lower.endswith(ext) for ext in ok_ext_set):
            violations.append(f"{path}")
    if violations:
        return (
            f"bucket={buckets!r} file_changes include paths outside allowed "
            f"extensions {sorted(ok_ext_set)}: {violations[:3]}"
        )
    return None


def _check_exploration_response(
    manifest: dict,
    body_md: str,
    strategy_concern_flagged: set[str] | None,
) -> str | None:
    """v0.9.3 IV-11: if prior Critic's strategy_concern flagged a bucket,
    the candidate must either (a) include that bucket in its `bucket`
    field, or (b) provide a ``## Why flagged direction is infeasible``
    body section with concrete evidence.

    ``strategy_concern_flagged``: set of bucket names the previous round's
    Critic flagged as "untouched + has evidence". None / empty → no-op.

    This turns a previously-soft Critic-advisory into a structural gate.
    "Prompt iteration is easier" is NOT acceptable justification; Evolver
    must use bash/web_search/web_fetch to prove the direction is genuinely
    unreachable.
    """
    if not strategy_concern_flagged:
        return None
    candidate_buckets = set(_normalize_bucket(manifest.get("bucket")))
    if candidate_buckets & strategy_concern_flagged:
        return None  # candidate targets at least one flagged bucket
    # Otherwise require a well-formed infeasibility section.
    marker = "## Why flagged direction is infeasible"
    if marker not in body_md:
        flagged_list = sorted(strategy_concern_flagged)
        return (
            f"strategy_concern flagged bucket(s) {flagged_list} not "
            f"targeted by this candidate (bucket={sorted(candidate_buckets) or '?'}) "
            f"and no '{marker}' body section present (IV-11)"
        )
    # Non-trivial content after the marker (≥1 line of non-whitespace prose).
    idx = body_md.find(marker)
    tail = body_md[idx + len(marker):]
    # Strip leading blank lines, then require at least 50 chars of content
    # before the next `## ` header (or EOF).
    next_hdr = tail.find("\n## ")
    section_body = tail[:next_hdr] if next_hdr != -1 else tail
    if len(section_body.strip()) < 50:
        return (
            f"'{marker}' section too short — must contain concrete "
            f"evidence (bash/web_search/web_fetch output) showing the "
            f"direction is genuinely unreachable (IV-11)"
        )
    return None


def _check_iterates_from(
    manifest: dict,
    body_md: str,
    prior_ships: dict | None,
    current_round: int | None,
) -> str | None:
    """IV-12: if manifest declares ``iterates_from: <ship_id>``, validate:

    - target exists in ``prior_ships`` ledger (caller-supplied snapshot of
      ``ship_outcomes.json``)
    - target is from a round strictly before ``current_round``
    - target has not already been superseded by a prior iterate
    - manifest body cites the target (by id) or its ship_outcomes data —
      an iterate without evidence linkage is indistinguishable from a
      rename; Critic relies on this citation to audit intent

    No-op when ``prior_ships`` is None (back-compat with call sites that
    don't have ledger context — tests, legacy runs).
    """
    target = manifest.get("iterates_from")
    if not target:
        return None
    if not isinstance(target, str) or not target.strip():
        return "iterates_from must be a non-empty string (IV-12)"
    if prior_ships is None:
        return None
    info = prior_ships.get(target)
    if info is None:
        return f"iterates_from target {target!r} not found in ship ledger (IV-12)"
    tgt_round = info.get("round")
    if current_round is not None and tgt_round is not None:
        if int(tgt_round) >= int(current_round):
            return (
                f"iterates_from target {target!r} is from round {tgt_round}, "
                f"must be < current round {current_round} (IV-12)"
            )
    superseder = info.get("superseded_by")
    if superseder:
        return (
            f"iterates_from target {target!r} already superseded by "
            f"{superseder!r} (IV-12)"
        )
    body_lower = body_md.lower()
    if (target.lower() not in body_lower
            and "hit_rate" not in body_lower
            and "ship_outcomes" not in body_lower):
        return (
            f"iterates_from={target!r} requires the manifest body to cite the "
            f"target ship id or its ship_outcomes evidence (IV-12)"
        )
    return None


def validate_candidate_manifest(
    manifest: dict, body_md: str, *, slot_type: str = "regular",
    strategy_concern_flagged: set[str] | None = None,
    prior_ships: dict | None = None,
    current_round: int | None = None,
) -> GateResult:
    required = ("candidate_id", "bucket", "file_changes", "predicted_impact")
    for k in required:
        if k not in manifest:
            return GateResult(ok=False, reason=f"manifest missing key: {k} (IV-3)")
    if not manifest["file_changes"]:
        return GateResult(ok=False, reason="file_changes empty (IV-3)")

    if slot_type == "explorer":
        return GateResult(ok=True)

    # v0.9.1 schema: capability_evidence required on 'regular' slot.
    # 'legacy' slot relaxes it (for reading pre-v0.9 run artefacts).
    # locus was dropped in v0.9.1 — interaction-check moved to Critic.
    if slot_type != "legacy":
        if "capability_evidence" not in manifest:
            return GateResult(
                ok=False,
                reason="manifest missing key: capability_evidence (v0.9 IV-8)",
            )
        caps = manifest.get("capability_evidence")
        if not isinstance(caps, list):
            return GateResult(
                ok=False,
                reason="manifest.capability_evidence must be a list (IV-8)",
            )
        for i, entry in enumerate(caps):
            if not isinstance(entry, dict):
                return GateResult(
                    ok=False,
                    reason=(
                        f"manifest.capability_evidence[{i}] must be a mapping "
                        "(IV-8)"
                    ),
                )
            for key in ("type", "claim", "evidence"):
                val = entry.get(key)
                if not isinstance(val, str) or not val.strip():
                    return GateResult(
                        ok=False,
                        reason=(
                            f"manifest.capability_evidence[{i}] missing "
                            f"non-empty '{key}' (IV-8)"
                        ),
                    )

        # v0.9.2 IV-9: bucket vs file_changes consistency.
        # Prevents cross-bucket pollution such as a config-bucket candidate
        # silently shipping a new prompt template (observed in
        # aegis_64_v091_r15_v2 R4 where a broken Jinja-in-j2 crashed
        # SystemPromptProcessor on every task in the following rollout).
        bucket = manifest.get("bucket", "")
        fcs = manifest.get("file_changes") or []
        bad = _check_bucket_file_consistency(bucket, fcs)
        if bad:
            return GateResult(ok=False, reason=f"{bad} (IV-9 bucket-file mismatch)")

        # v0.9.2 dropped IV-10 (Jinja syntax check). Agent-facing prompts
        # are now plain markdown via PlainMarkdownSystemPromptBuilder — no
        # runtime Jinja rendering — so `{{...}}` and `{%...%}` in shipped
        # prompts are literal prose and cannot crash at task start.
        # Meta-agent templates (harnessx/aegis/templates/*.md) still use
        # render_template() which IS Jinja-rendered, but those files are
        # internal source tree and not part of candidate file_changes.

        # v0.9.3 IV-11: exploration enforcement.
        # No-op if caller doesn't pass strategy_concern_flagged (back-compat
        # with tests and with runs whose Planner/Critic templates haven't
        # been upgraded yet). Activated once Critic writes structured
        # strategy_concern_flagged_buckets in decision.md frontmatter and
        # orchestrator forwards it into Stage 4.
        iv11 = _check_exploration_response(
            manifest, body_md, strategy_concern_flagged,
        )
        if iv11:
            return GateResult(ok=False, reason=iv11)

        # v0.9.5 IV-12: iterates_from validation (no-op if field absent or
        # caller didn't pass prior_ships context).
        iv12 = _check_iterates_from(
            manifest, body_md, prior_ships, current_round,
        )
        if iv12:
            return GateResult(ok=False, reason=iv12)

    if "## Failure Evidence" not in body_md:
        return GateResult(ok=False, reason="missing Failure Evidence section (IV-3)")
    anchors = _parse_anchors(body_md)
    if not anchors:
        return GateResult(
            ok=False,
            reason="candidate manifest body has zero evidence anchors (IV-3)",
        )
    return GateResult(ok=True)


def validate_critic_verdict(verdict_md: str) -> GateResult:
    """Validate a Critic verdict file.

    Real Critic verdicts (per ``critic.j2``) emit evidence anchors as a
    YAML list in the frontmatter:

        ---
        candidate_id: C-R1-01
        verdict: accept
        evidence_anchors:
          - trajectories/abc.jsonl#step_4
          - digests/xyz.md
        confidence: 0.7
        ---

    The verdict passes if ``evidence_anchors`` is a non-empty list whose
    entries look like ``(sessions|trajectories|digests)/<path>`` with an
    optional ``#<locator>`` suffix (mandatory for trajectories/sessions,
    optional for digests).
    """
    m = _FRONTMATTER_RE.match(verdict_md.strip() + "\n")
    if not m:
        return GateResult(
            ok=False,
            reason="critic verdict missing YAML frontmatter (IV-4)",
        )
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError as exc:
        return GateResult(
            ok=False,
            reason=f"critic verdict frontmatter YAML invalid: {exc} (IV-4)",
        )
    if not isinstance(fm, dict):
        return GateResult(
            ok=False,
            reason="critic verdict frontmatter must be a mapping (IV-4)",
        )
    anchors = fm.get("evidence_anchors")
    if not isinstance(anchors, list) or not anchors:
        return GateResult(
            ok=False,
            reason="critic verdict lacks evidence_anchors list (IV-4)",
        )
    for a in anchors:
        if not isinstance(a, str) or not _verdict_anchor_matches(a):
            return GateResult(
                ok=False,
                reason=f"critic verdict anchor malformed: {a!r} (IV-4)",
            )
    return GateResult(ok=True)


def validate_decision_chain(
    decision: dict, candidate_manifests: dict[str, tuple[dict, str]],
) -> GateResult:
    if decision.get("decision_type") == "no_op":
        return GateResult(ok=True)
    for item in decision.get("ship_ranking", []):
        cid = item["candidate_id"]
        if cid not in candidate_manifests:
            return GateResult(
                ok=False,
                reason=f"decision cites unknown candidate {cid} (IV-6)",
            )
    return GateResult(ok=True)
