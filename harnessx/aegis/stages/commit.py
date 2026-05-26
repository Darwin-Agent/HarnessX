# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Stage 4 — try-until-pass. Run gates against each candidate in ship_ranking
order. First candidate passing ALL gates → ship. All fail → no-op (but archive
every candidate with failure context)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from harnessx.aegis.data.signatures import compute_signature, FileChange
from harnessx.aegis.gates.structure import (
    validate_candidate_manifest, validate_decision_chain,
)
from harnessx.aegis.gates.novelty import check_novelty
from harnessx.aegis.gates.canonicalize import check_canonicalize
from harnessx.aegis.gates.replay import check_replay_smoke
from harnessx.aegis.gates.counterfactual import check_counterfactual_replay
from harnessx.aegis.agents.evolver import parse_candidate_manifest


# Module-level context for the counterfactual gate. Orchestrator populates
# BEFORE calling run_stage_4; dict keys: passing_task_ids (list[str]),
# trajectories_dir (Path), k_samples (int). None → gate short-circuits ok.
_counterfactual_context: dict | None = None


def set_counterfactual_context(ctx: dict | None) -> None:
    global _counterfactual_context
    _counterfactual_context = ctx


@dataclass
class GateVerdict:
    ok: bool
    reason: str


def _compute_candidate_signature(manifest: dict) -> str:
    """Compute the FileChange-based signature for a candidate manifest.

    Exposed so the orchestrator can populate ``refuted_signatures`` when
    every candidate fails gate-check (Important Fix 14). Kept in sync with
    the signature computed inside :func:`_run_all_gates`.
    """
    import hashlib
    changes = []
    for fc in manifest.get("file_changes", []):
        path = fc.get("path", "")
        sha = fc.get("diff_sha_after")
        if not sha:
            material = f"{path}|{fc.get('action', '')}|{fc.get('diff_summary', '')}"
            sha = hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]
        changes.append(FileChange(path=path, diff_sha_after=sha))
    return compute_signature(changes)


async def _run_all_gates(
    cid: str,
    manifest_path: Path,
    candidate_config_path: Path,
    refuted_signatures: set[str],
    *,
    replay_model=None,
    strategy_concern_flagged: set[str] | None = None,
    prior_ships: dict | None = None,
    current_round: int | None = None,
) -> dict[str, GateVerdict]:
    md = manifest_path.read_text(encoding="utf-8")
    manifest, body = parse_candidate_manifest(md)
    slot_type = manifest.get("slot_type", "regular")

    sr = validate_candidate_manifest(
        manifest, body, slot_type=slot_type,
        strategy_concern_flagged=strategy_concern_flagged,
        prior_ships=prior_ships,
        current_round=current_round,
    )
    structure = GateVerdict(sr.ok, sr.reason)

    sig = _compute_candidate_signature(manifest)
    nr = check_novelty(sig, refuted_signatures=refuted_signatures)
    novelty = GateVerdict(nr.ok, nr.reason)

    # E6: short-circuit. If either of the cheap gates (structure, novelty)
    # already failed, don't spend LLM budget running canonicalize + replay
    # on a candidate that can't ship. Record skipped gates with an explicit
    # "skipped" reason so audit.jsonl still shows all four entries.
    _SKIP = "skipped: earlier gate failed"
    if not structure.ok or not novelty.ok:
        return {
            "structure": structure, "novelty": novelty,
            "canonicalize": GateVerdict(False, _SKIP),
            "replay": GateVerdict(False, _SKIP),
        }

    # E1: load + canonicalize the applied YAML ONCE up front and reuse the
    # resulting cfg across the canonicalize and replay gates. Previously
    # both gates re-parsed the same YAML independently (3 parses counting
    # apply.validate_applied_config).
    from harnessx.core.harness import HarnessConfig
    cfg_loaded = None
    canon_reason = ""
    try:
        cfg_loaded = HarnessConfig.from_yaml_file(str(candidate_config_path))
        cfg_loaded.canonicalize()
    except Exception as exc:
        canon_reason = f"canonicalize failed: {exc!r}"

    if cfg_loaded is None:
        canonical = GateVerdict(False, canon_reason)
        counterfactual = GateVerdict(False, _SKIP)
        replay = GateVerdict(False, _SKIP)
    else:
        cr = check_canonicalize(cfg_loaded)
        canonical = GateVerdict(cr.ok, cr.reason)
        if not canonical.ok:
            counterfactual = GateVerdict(False, _SKIP)
            replay = GateVerdict(False, _SKIP)
        else:
            # v0.9 counterfactual gate — replay new config's processor
            # chain against sampled prior-passing trajectories.
            ctx = _counterfactual_context
            if ctx is None:
                counterfactual = GateVerdict(
                    True, "skipped: no counterfactual context wired",
                )
            else:
                try:
                    cand_yaml_text = Path(candidate_config_path).read_text(
                        encoding="utf-8",
                    )
                except OSError as exc:
                    cand_yaml_text = None
                    counterfactual = GateVerdict(
                        False, f"counterfactual: cfg read failed: {exc!r}",
                    )
                if cand_yaml_text is not None:
                    cf = await check_counterfactual_replay(
                        new_config_yaml_text=cand_yaml_text,
                        passing_task_ids=list(ctx.get("passing_task_ids") or []),
                        trajectories_dir=ctx.get("trajectories_dir"),
                        k_samples=int(ctx.get("k_samples", 3)),
                    )
                    counterfactual = GateVerdict(cf.ok, cf.reason)

            if not counterfactual.ok:
                replay = GateVerdict(False, _SKIP)
            else:
                rr = await check_replay_smoke(cfg_loaded, model_config=replay_model)
                replay = GateVerdict(rr.ok, rr.reason)

    return {
        "structure": structure,
        "novelty": novelty,
        "canonicalize": canonical,
        "counterfactual": counterfactual,
        "replay": replay,
    }


async def run_stage_4(
    *,
    round_n: int,
    decision: dict,
    candidates_info: dict[str, tuple[Path, Path]],
    refuted_signatures: set[str],
    commit_fn: Callable | None,
    archive_fn: Callable[[str, dict], None],
    replay_model=None,
    strategy_concern_flagged: set[str] | None = None,
    prior_ships: dict | None = None,
) -> dict:
    if decision.get("decision_type") == "no_op":
        for cid in candidates_info:
            archive_fn(cid, {"reason": "no_op"})
        return {
            "shipped_cid": None, "gate_results": {},
            "reason": "critic no-op", "candidate_signatures": {},
        }

    # IV-6: every ship-ranked candidate_id MUST exist in candidates_info
    # (i.e. the Critic cited a real, manifested candidate). If broken,
    # no-op the round instead of silently skipping to the next candidate.
    candidate_manifests: dict[str, tuple[dict, str]] = {}
    for cid, (manifest_path, _config_path) in candidates_info.items():
        try:
            md = manifest_path.read_text(encoding="utf-8")
            fm, body = parse_candidate_manifest(md)
            candidate_manifests[cid] = (fm, body)
        except (OSError, ValueError):
            # Keep cid in the map (as best-effort structural presence) but
            # with empty manifest/body; structure gate will catch it later.
            candidate_manifests[cid] = ({}, "")
    chain = validate_decision_chain(decision, candidate_manifests)
    # Compute per-candidate signatures up front so the orchestrator can
    # populate refuted_signatures on gate-failure paths (Important Fix 14).
    candidate_signatures: dict[str, str] = {}
    for cid, (fm, _body) in candidate_manifests.items():
        try:
            candidate_signatures[cid] = _compute_candidate_signature(fm)
        except Exception:
            # Best-effort: malformed manifest → no signature recorded.
            pass
    if not chain.ok:
        for cid in candidates_info:
            archive_fn(cid, {"reason": f"decision_chain_broken:{chain.reason}"})
        return {
            "shipped_cid": None,
            "gate_results": {},
            "reason": f"decision_chain_broken:{chain.reason}",
            "candidate_signatures": candidate_signatures,
        }

    # Multi-ship loop: iterate decision's ship_ranking in order.
    # v0.9.3: bucket-disjoint claim REMOVED. Evolver can legitimately
    # ship a cross-bucket bundle (e.g. prompt + processor together) or
    # two candidates that touch the same bucket but different fields.
    # Conflict detection is now purely compose-layer (same `_target_`
    # processor with different kwargs = last-writer-wins, which Critic
    # should have caught via its interaction-check requirement).
    # Semantic interaction checking (did this candidate's mutation
    # surface overlap an existing processor?) lives in the Critic's
    # verdict, not in the gate — v0.9.1 moved it there after the v0.9.0
    # locus gate proved too rigid.
    gate_results_all: dict[str, dict] = {}
    shipped_cids: list[str] = []
    shipped_by_bucket: dict[str, str] = {}
    skip_reasons: dict[str, str] = {}
    for item in decision.get("ship_ranking", []):
        cid = item["candidate_id"]
        if cid not in candidates_info:
            continue
        manifest_fm, _body = candidate_manifests.get(cid, ({}, ""))
        bucket_val = (manifest_fm or {}).get("bucket")
        # bucket may now be a str OR list[str]. Keep the first for the
        # legacy ``shipped_by_bucket`` observability dict.
        if isinstance(bucket_val, list):
            bucket = str(bucket_val[0]) if bucket_val else ""
        else:
            bucket = str(bucket_val or "")
        manifest_path, config_path = candidates_info[cid]

        results = await _run_all_gates(
            cid, manifest_path, config_path, refuted_signatures,
            replay_model=replay_model,
            strategy_concern_flagged=strategy_concern_flagged,
            prior_ships=prior_ships,
            current_round=round_n,
        )
        gate_results_all[cid] = results

        if all(v.ok for v in results.values()):
            if commit_fn is not None:
                commit_fn(cid, config_path)
            shipped_cids.append(cid)
            if bucket:
                shipped_by_bucket[bucket] = cid
            continue

        archive_fn(
            cid,
            {k: {"ok": v.ok, "reason": v.reason} for k, v in results.items()},
        )

    # Archive anything that wasn't gate-tested AND wasn't already archived
    # (bucket-claim skips above did their own archive).
    for other_cid in candidates_info:
        if other_cid in shipped_cids:
            continue
        if other_cid in gate_results_all:
            continue
        if other_cid in skip_reasons:
            continue
        archive_fn(other_cid, {"reason": "not_selected"})

    if shipped_cids:
        # v0.9.5: surface iterates_from pairs so orchestrator can mark
        # superseded_by on the ship ledger.
        supersedes: list[dict] = []
        for cid in shipped_cids:
            fm, _body = candidate_manifests.get(cid, ({}, ""))
            target = (fm or {}).get("iterates_from")
            if target:
                supersedes.append({"new": cid, "target": str(target)})
        return {
            "shipped_cid": shipped_cids[0],  # back-compat: first shipped
            "shipped_cids": shipped_cids,
            "shipped_by_bucket": shipped_by_bucket,
            "gate_results": gate_results_all,
            "reason": None,
            "candidate_signatures": candidate_signatures,
            "supersedes": supersedes,
        }

    return {
        "shipped_cid": None,
        "shipped_cids": [],
        "shipped_by_bucket": {},
        "gate_results": gate_results_all,
        "reason": "all_candidates_failed_gate",
        "candidate_signatures": candidate_signatures,
    }
