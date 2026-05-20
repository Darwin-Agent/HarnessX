# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""AEGIS Orchestrator — Python driver for the 6-stage loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .data.audit import AuditLog, AuditEvent
from .data.journal import Journal, RoundEntry
from .data.archive import Archive
from .data.reputation import Reputation
from .data import ledger
from .stages.preprocess import run_stage_p
from .stages.plan import run_stage_1
from .stages.propose import run_stage_2
from .stages.judge import run_stage_3, make_evolver_runner
from .stages.commit import run_stage_4
# NOTE: ``adjudicate_previous_round`` is intentionally NOT imported here.
# Stage 5 runs ACROSS rounds in the pilot driver, not inside ``run_round``.
# Callers wiring the pilot loop should import it directly:
#   from harnessx.aegis.stages.adjudicate import adjudicate_previous_round


class ShipNotLandedError(RuntimeError):
    """Raised when commit shipped candidates but compose produced a merged
    config byte-equivalent to the round's base — i.e. the ship had no effect.

    This is a hard error rather than a warning because a silent no-op gets
    mistaken for evolution variance: the next round runs the previous
    round's config under a fresh hash and any pass-rate delta looks like
    candidate impact when in fact the candidate never landed.
    """


def _strip_volatile_keys(cfg: dict) -> dict:
    """Drop fields that legitimately differ across rounds (paths, ids).

    Used by the ship-landed assertion to compare *semantic* config content
    only — base_dir, session_id, etc. are expected to bump every round.
    """
    import copy as _copy

    out = _copy.deepcopy(cfg)
    tracer = out.get("tracer")
    if isinstance(tracer, dict):
        tracer.pop("base_dir", None)
        tracer.pop("session_id", None)
    return out


def _extract_predicted_tasks(fm: dict | None) -> list[str]:
    """Union of every ``predicted_impact.tasks_will_*`` field a manifest may
    declare, deduped while preserving first-seen order.

    Accepts both the legacy ``tasks_will_pass`` field and the granular
    ``tasks_will_unlock`` / ``tasks_will_stabilize`` fields introduced when
    the scoreboard became k-aware. The legacy field is the union of the
    two granular ones; reading all three and deduping produces the right
    set whether the Evolver populated the new fields, the old one, or
    both.
    """
    pi = (fm or {}).get("predicted_impact")
    if not isinstance(pi, dict):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for key in ("tasks_will_unlock", "tasks_will_stabilize", "tasks_will_pass"):
        vals = pi.get(key) or []
        if not isinstance(vals, list):
            continue
        for x in vals:
            s = str(x)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _assert_merged_differs_from_base(
    *,
    base_path: Path,
    merged_path: Path,
    shipped_cids: list[str],
) -> None:
    """Guarantee a non-empty ship actually mutated the merged config.

    Compares semantic content (volatile keys stripped). If equal, a bucket
    applier silently no-op'd and the round would run the parent's config
    while pretending to validate a new candidate.
    """
    import yaml as _yaml

    base = _yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    merged = _yaml.safe_load(merged_path.read_text(encoding="utf-8")) or {}
    if _strip_volatile_keys(base) == _strip_volatile_keys(merged):
        raise ShipNotLandedError(
            f"compose produced merged.yaml semantically equal to base for "
            f"shipped_cids={shipped_cids}. Most likely cause: a candidate's "
            f"`bucket` field is a list but was stringified before reaching "
            f"compose, so the bucket applier dispatch missed. Check "
            f"orchestrator's shipped_entries construction. "
            f"base={base_path} merged={merged_path}"
        )


@dataclass
class AegisOrchestrator:
    run_dir: Path
    num_evolvers: int  # kept for backward-compat recipe signatures; no longer dispatches
    model_config: object
    max_ask_more: int = 2
    max_concurrency: int = 4
    budget_per_round_usd: float = 20.0
    auto_revert_enabled: bool = True
    # Benchmark model config used by the Stage 4 replay gate. This is the
    # model the Harness runs tasks against — NOT the meta-model used by
    # Planner/Evolver/Critic (``model_config`` above). Recipe callers should
    # set this explicitly. If None, the replay gate skips.
    replay_model: object | None = None
    # Minimum Stage P actionability score (0.0-1.0) required to proceed to
    # Stage 1. Below this, the round early-exits with a clear journal entry.
    # 0.0 means ALL_PASS + no fragility; 0.3 means ALL_PASS but fragilities;
    # 0.8 is PARTIAL_PASS; 1.0 is ALL_FAIL. Default 0.3 treats pure all-pass
    # runs as no-op without burning Planner/Evolver/Critic meta budget.
    min_actionability: float = 0.3
    # TODO(stage5): ``adjudicate_previous_round`` is not called from ``run_round``.
    # Stage 5 runs ACROSS rounds in the pilot driver (not inside this class).
    # When the pilot loop is implemented, it should call
    #   from .stages.adjudicate import adjudicate_previous_round
    #   result = adjudicate_previous_round(
    #       ..., auto_revert_enabled=orch.auto_revert_enabled, ...
    #   )
    # after each round completes.

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(self.run_dir / "audit.jsonl")
        self.journal = Journal(self.run_dir / "journal.md")
        self.archive = Archive(self.run_dir / "archive")
        self.reputation = Reputation(window=5)
        # AegisAgent.evolve constructs a fresh AegisOrchestrator each round, so
        # in-memory Reputation would reset every round. Load from disk if a
        # prior round persisted state; fall back to empty on corruption.
        rep_path = self.run_dir / "reputation.json"
        if rep_path.exists():
            import json as _json

            try:
                self.reputation = Reputation.from_dict(
                    _json.loads(rep_path.read_text(encoding="utf-8")),
                )
            except (_json.JSONDecodeError, KeyError, TypeError) as exc:
                import logging

                logging.getLogger("aegis.orchestrator").warning("reputation.json corrupt (%s) — starting fresh", exc)
                self.reputation = Reputation(window=5)
        # Scoreboard — v0.9 structured per-bucket / per-locus state.
        from .data.scoreboard import Scoreboard

        self.scoreboard = Scoreboard.load(self.run_dir / "scoreboard.json")
        # Fail loudly if no meta-agent model is wired. A None model_config
        # would silently short-circuit Digester/Planner/Evolver/Critic and
        # produce a spurious "0 digests → early_exit" with no hint that the
        # real cause is a missing model. (Tests that want a mock should pass
        # ``MagicMock()`` — which is not None.)
        if self.model_config is None:
            raise ValueError(
                "AegisOrchestrator.model_config must be set — the meta-agent "
                "model that runs Digester/Planner/Evolver/Critic. None is not "
                "a valid production configuration. (If you need a test-only "
                "mock, patch individual stage functions.)"
            )

    def _persist_state(self) -> None:
        import json as _json
        import os as _os

        rep_path = self.run_dir / "reputation.json"
        tmp_path = rep_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            _json.dumps(self.reputation.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # os.replace is atomic on POSIX; on Windows it replaces the target.
        # Either way, a partial write can't leave reputation.json half-rewritten
        # if the process is killed mid-_persist_state.
        _os.replace(tmp_path, rep_path)
        # Scoreboard persisted alongside reputation; Scoreboard.save does its
        # own atomic write via tmp + os.replace.
        self.scoreboard.save(self.run_dir / "scoreboard.json")

    async def run_round(
        self,
        *,
        round_n: int,
        raw_sessions_dir: Path,
        pass_flags_by_task: dict,
        current_config_path: Path,
    ) -> dict:
        round_dir = self.run_dir / f"R{round_n}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Refresh INDEX.md so agents see an up-to-date catalog of what's
        # been accumulated. Agents decide what to consult; we decide what
        # to expose.
        ledger.refresh_index_md(self.run_dir, current_round=round_n)

        # Backfill ship_outcomes with this round's pass-flag state BEFORE
        # Stage P: agents reading during Stage 1/2/3 see the latest
        # retrospective view of "did past ships' predictions hold up".
        # (task_history for the current round is appended by the recipe
        # after rollouts complete; the orchestrator's round_n here is the
        # round whose rollouts ALREADY ran.)
        try:
            ledger.backfill_ship_outcomes(self.run_dir)
        except Exception as exc:
            import logging

            logging.getLogger("aegis.orchestrator").warning("ship_outcomes backfill failed (non-fatal): %s", exc)

        # Compute regression watchlist BEFORE Stage P digester runs so the
        # round_dir/regressions.md exists by the time Planner/Evolver/Critic
        # are dispatched. Mechanical, k-aware, joint-attribution; surfaces
        # AP→AF / AP→PP / PARTIAL→worse transitions that ship_outcomes'
        # hit_rate metric does not penalise (it counts predicted-task
        # improvements only).
        #
        # Off-by-one note: at the start of round N evolve, task_history
        # holds rounds 0..N-1 (round N batch has not run yet). The most
        # recent observable comparison is therefore R{N-2}→R{N-1}, with
        # the suspect ships being those tagged round=N-1. The orchestrator
        # writes the result into R{N}/regressions.md (next to evolve
        # materials) but the analysis target is round_n - 1.
        if round_n >= 1:
            try:
                from .data.regressions import write_regressions_md

                write_regressions_md(
                    self.run_dir,
                    round_n - 1,
                    for_evolve_round_n=round_n,
                )
            except Exception as exc:
                import logging

                logging.getLogger("aegis.orchestrator").warning(
                    "regressions watchlist write failed (non-fatal): %s", exc
                )

        # v0.9: backfill scoreboard's per-ship flipped_in_ship_round from the
        # same ship_outcomes ledger we just refreshed. Atomic per-record
        # rebuild (ShipRecord is frozen).
        try:
            import json as _json

            _outcomes_path = self.run_dir / "data" / "ship_outcomes.json"
            if _outcomes_path.exists():
                _outcomes = _json.loads(_outcomes_path.read_text(encoding="utf-8"))
                _by_cid = {o.get("ship_id"): o for o in _outcomes if isinstance(o, dict)}
                from .data.scoreboard import ShipRecord

                updated: list[ShipRecord] = []
                for rec in self.scoreboard.ships:
                    o = _by_cid.get(rec.cid)
                    if not o:
                        updated.append(rec)
                        continue
                    flipped = o.get("flipped_to_pass_in_ship_round") or []
                    if not isinstance(flipped, list):
                        updated.append(rec)
                        continue
                    updated.append(
                        ShipRecord(
                            cid=rec.cid,
                            round=rec.round,
                            bucket=rec.bucket,
                            predicted_tasks=rec.predicted_tasks,
                            flipped_in_ship_round=tuple(str(t) for t in flipped),
                        )
                    )
                self.scoreboard.ships = updated
        except Exception as exc:
            import logging

            logging.getLogger("aegis.orchestrator").warning(
                "scoreboard backfill failed (non-fatal): %s",
                exc,
            )

        # Stage P
        def digester_factory(inputs):
            from .agents.digester import build_digester_harness

            cfg = build_digester_harness(inputs)
            return self.model_config.agentic(cfg)

        stage_p = await run_stage_p(
            raw_dir=raw_sessions_dir,
            trajectories_dir=round_dir / "trajectories",
            digests_dir=round_dir / "digests",
            summary_path=round_dir / "summary.md",
            pass_flags_by_task=pass_flags_by_task,
            harness_factory=digester_factory,
            concurrency=self.max_concurrency,
        )
        self.audit.append(
            AuditEvent(
                round=round_n,
                stage="P",
                kind="preprocess",
                payload=stage_p,
                evidence_refs=[],
            )
        )

        # C2: Ship-follow-up check. If the immediately previous round shipped
        # a candidate with predicted_tasks_pass, compare those predictions
        # against *this* round's actual pass set and surface any misses in
        # summary.md so the Planner sees "ship didn't take" evidence —
        # preventing it from blindly reinforcing a no-op change.
        prev_entries = self.journal.recent(1)
        if prev_entries and prev_entries[-1].action == "ship" and prev_entries[-1].predicted_tasks_pass:
            prev = prev_entries[-1]
            actual_pass = {
                tid for tid, flags in (pass_flags_by_task or {}).items() if flags and all(bool(f) for f in flags)
            }
            still_failing = [t for t in prev.predicted_tasks_pass if t not in actual_pass]
            summary_path = round_dir / "summary.md"
            if summary_path.exists() and still_failing:
                lines = ["", "## Previous-round ship follow-up", ""]
                lines.append(
                    f"Round {prev.round} shipped `{prev.shipped_cid}` predicting "
                    f"{len(prev.predicted_tasks_pass)} tasks would pass, "
                    f"but **{len(still_failing)} are still failing this round**:"
                )
                lines.extend(f"- {t}" for t in still_failing[:10])
                if len(still_failing) > 10:
                    lines.append(f"- ... ({len(still_failing) - 10} more)")
                lines.append("")
                lines.append(
                    "Planner: treat this as evidence the R%d ship's "
                    "assumed root cause was wrong. Do NOT simply reinforce "
                    "the same bucket — either propose a rollback, or target "
                    "the actual failure mode with a DIFFERENT bucket." % prev.round
                )
                with summary_path.open("a", encoding="utf-8") as fp:
                    fp.write("\n".join(lines) + "\n")
                # Record in audit for observability
                self.audit.append(
                    AuditEvent(
                        round=round_n,
                        stage="P",
                        kind="ship_followup",
                        payload={
                            "prev_round": prev.round,
                            "prev_shipped_cid": prev.shipped_cid,
                            "predicted_count": len(prev.predicted_tasks_pass),
                            "still_failing_count": len(still_failing),
                            "actually_passed_count": len(prev.predicted_tasks_pass) - len(still_failing),
                        },
                        evidence_refs=[],
                    )
                )

        # Imp 4: Early-exit when Stage P yields insufficient signal.
        actionability = float(stage_p.get("actionability", 0.0))
        if actionability < self.min_actionability:
            reason = stage_p.get("actionability_reason", "low actionability")
            narrative = f"early_exit: {reason} (score={actionability:.2f} < {self.min_actionability:.2f})"
            self._finalize_round(
                round_n,
                shipped_cids=[],
                hit_rate=None,
                narrative=narrative,
                refuted_signatures=[],
            )
            return {"shipped_cid": None, "shipped_cids": [], "reason": "early_exit_no_signal"}

        # Stage 1 — Planner writes landscape.md
        landscape_path = round_dir / "landscape.md"
        stage_1 = await run_stage_1(
            round_n=round_n,
            overview_path=round_dir / "summary.md",
            journal_path=self.run_dir / "journal.md",
            archive_dir=self.run_dir / "archive",
            current_config_path=current_config_path,
            landscape_path=landscape_path,
            digests_dir=round_dir / "digests",
            reputation_summary=self.reputation.to_dict(),
            model_config=self.model_config,
            max_cost_usd=100.0,
            actionability_score=actionability,
            run_root=self.run_dir,
            sessions_dir=round_dir / "meta_sessions" / "planner",
        )

        self.audit.append(
            AuditEvent(
                round=round_n,
                stage="1",
                kind="plan",
                payload={
                    "landscape_written": stage_1.get("landscape_written", False),
                    "frontmatter": stage_1.get("frontmatter", {}),
                },
                evidence_refs=[str(landscape_path)] if stage_1.get("landscape_written") else [],
            )
        )

        if not stage_1.get("landscape_written"):
            self._finalize_round(
                round_n,
                shipped_cids=[],
                hit_rate=None,
                narrative="Planner produced no landscape.md — aborting round",
                refuted_signatures=[],
            )
            return {"shipped_cid": None, "shipped_cids": [], "reason": "no_landscape"}

        # Stage 2 — single Evolver produces K candidates
        stage_2 = await run_stage_2(
            round_n=round_n,
            landscape_path=landscape_path,
            current_config_path=current_config_path,
            candidates_dir=round_dir / "candidates",
            trajectories_dir=round_dir / "trajectories",
            digests_dir=round_dir / "digests",
            model_config=self.model_config,
            max_cost_usd=100.0,
            sessions_dir=round_dir / "meta_sessions" / "evolver",
        )
        for cid, ok, reason in stage_2["results"]:
            self.audit.append(
                AuditEvent(
                    round=round_n,
                    stage="2",
                    kind="propose" if ok else "propose_fail",
                    payload={"brief_id": cid, "reason": reason},
                    evidence_refs=[],
                )
            )

        # Stage 3 mini-Evolver factory: Critic calls this when it wants to
        # ask the Evolver a clarifying question about a specific candidate.
        # We run in ask-more mode (no scratch, single-file write at the
        # candidate path so the Q/A gets appended by Critic).
        def evolver_harness_factory(cid: str, brief_path: Path):
            from .agents.evolver import build_evolver_harness, EvolverInputs
            import uuid

            askmore_dir = round_dir / "askmore_scratch"
            askmore_dir.mkdir(parents=True, exist_ok=True)
            scratch = askmore_dir / f"{cid}_{uuid.uuid4().hex[:8]}.md"
            return build_evolver_harness(
                EvolverInputs(
                    round=round_n,
                    current_config_path=current_config_path,
                    landscape_path=landscape_path,
                    digests_dir=round_dir / "digests",
                    trajectories_dir=round_dir / "trajectories",
                    candidates_dir=round_dir / "candidates",
                    applied_root=round_dir / "applied",
                    ask_more_brief_path=brief_path,
                    ask_more_candidate_id=cid,
                    ask_more_candidate_path=scratch,
                )
            )

        # No brief-to-candidate alignment any more — the Critic calls
        # ask_evolver(cid, question) and we pass the candidate's own
        # manifest path as the "brief" (there are no separate briefs).
        candidate_paths_by_cid = {p.stem: p for p in stage_2["candidate_paths"]}
        evolver_runner = make_evolver_runner(
            candidate_paths_by_cid,
            evolver_harness_factory,
            self.model_config,
        )
        stage_3 = await run_stage_3(
            round_n=round_n,
            candidates_dir=round_dir / "candidates",
            verdicts_dir=round_dir / "verdicts",
            decision_path=round_dir / "decision.md",
            digests_dir=round_dir / "digests",
            trajectories_dir=round_dir / "trajectories",
            sessions_dir=raw_sessions_dir,
            journal_path=self.run_dir / "journal.md",
            current_config_path=current_config_path,
            evolver_runner=evolver_runner,
            model_config=self.model_config,
            max_ask_more=self.max_ask_more,
            max_cost_usd=100.0,
            meta_sessions_dir=round_dir / "meta_sessions" / "critic",
        )
        self.audit.append(
            AuditEvent(
                round=round_n,
                stage="3",
                kind="decision",
                payload={
                    "decision_type": (stage_3["decision"] or {}).get("decision_type"),
                    "critic_failed": stage_3["critic_failed"],
                },
                evidence_refs=[str(round_dir / "decision.md")],
            )
        )

        if stage_3["critic_failed"] or not stage_3["decision"]:
            self._finalize_round(
                round_n, shipped_cids=[], hit_rate=None, narrative="Critic failed", refuted_signatures=[]
            )
            return {"shipped_cid": None, "shipped_cids": [], "reason": "critic_failed"}

        # Stage 4
        # Stage 2 materialises an applied HarnessConfig YAML per
        # candidate under <round_dir>/applied/{cid}.yaml. The gates
        # (canonicalize, replay) need the applied YAML — not the parent
        # config — otherwise they silently validate the pre-mutation
        # state and every "ship" is a no-op in disguise.
        applied_dir = round_dir / "applied"
        # New layout: applied/<cid>/config.yaml (per-candidate scratch dir
        # where the Evolver wrote any modified template files alongside).
        candidates_info = {cp.stem: (cp, applied_dir / cp.stem / "config.yaml") for cp in stage_2["candidate_paths"]}
        refuted_sigs = self.journal.all_refuted_signatures()
        # v0.9: wire the counterfactual gate's context — prior-round
        # passing tasks + this round's recorded trajectories. Gate skips
        # silently if context is None / lists empty.
        from .stages.commit import set_counterfactual_context

        _passing_tids = sorted(
            tid for tid, flags in (pass_flags_by_task or {}).items() if flags and all(bool(f) for f in flags)
        )
        set_counterfactual_context(
            {
                "passing_task_ids": _passing_tids,
                "trajectories_dir": round_dir / "trajectories",
                "k_samples": 3,
            }
        )
        # v0.9.3 IV-11: extract strategy_concern_flagged_buckets from the
        # current round's landscape.md frontmatter (Planner relays it from
        # the previous round's Critic decision). Gate is a no-op if the
        # field is absent — only activates once Critic/Planner templates
        # write the structured field. Malformed landscape → silent None.
        strategy_concern_flagged: set[str] | None = None
        try:
            if landscape_path.exists():
                import yaml as _yaml
                import re as _re

                _text = landscape_path.read_text(encoding="utf-8")
                _m = _re.match(r"^---\n(.*?)\n---", _text, _re.DOTALL)
                if _m:
                    _fm = _yaml.safe_load(_m.group(1)) or {}
                    _flag = _fm.get("strategy_concern_flagged_buckets")
                    if isinstance(_flag, list) and _flag:
                        strategy_concern_flagged = {str(b) for b in _flag if b}
        except Exception as _exc:
            import logging

            logging.getLogger("aegis.orchestrator").warning(
                "landscape strategy_concern_flagged parse failed (non-fatal): %s",
                _exc,
            )
        # v0.9.5: ledger snapshot so IV-12 (iterates_from) can validate.
        try:
            prior_ships_snapshot = ledger.ship_ledger_for_gate(self.run_dir)
        except Exception:
            prior_ships_snapshot = None
        stage_4 = await run_stage_4(
            round_n=round_n,
            decision=stage_3["decision"],
            candidates_info=candidates_info,
            refuted_signatures=refuted_sigs,
            commit_fn=None,
            archive_fn=lambda cid, ctx: self.archive.store(
                round_n=round_n,
                cid=cid,
                manifest_md=(round_dir / "candidates" / f"{cid}.md").read_text()
                if (round_dir / "candidates" / f"{cid}.md").exists()
                else "",
                failure_context=ctx if isinstance(ctx, dict) else {"reason": str(ctx)},
            ),
            replay_model=self.replay_model,
            strategy_concern_flagged=strategy_concern_flagged,
            prior_ships=prior_ships_snapshot,
        )
        for cid, gr in stage_4["gate_results"].items():
            self.audit.append(
                AuditEvent(
                    round=round_n,
                    stage="4",
                    kind="gate",
                    payload={
                        "cid": cid,
                        "results": {k: v.ok for k, v in gr.items()},
                        # Include per-gate reason for any FAILED gate so a rejected
                        # candidate's cause is persisted in audit.jsonl without
                        # requiring access to the in-memory stage output.
                        "reasons": {k: v.reason for k, v in gr.items() if not v.ok},
                    },
                    evidence_refs=[],
                )
            )
        shipped_cids = stage_4.get("shipped_cids") or ([stage_4["shipped_cid"]] if stage_4.get("shipped_cid") else [])
        self.audit.append(
            AuditEvent(
                round=round_n,
                stage="4",
                kind="commit" if shipped_cids else "revert",
                payload={
                    "shipped_cids": shipped_cids,
                    "shipped_by_bucket": stage_4.get("shipped_by_bucket", {}),
                    "reason": stage_4["reason"],
                },
                evidence_refs=[],
            )
        )

        # Record bucket reputation. Prior semantics appended
        # ``False`` for any non-shipped candidate — which conflated
        # three very different outcomes into one negative signal:
        #   * Direction-wrong (replay gate failed on actual behavior)
        #   * Implementation bug (structure regex mismatch, canonicalize
        #     error on YAML shape)
        #   * Bucket-disjoint skip (another candidate claimed the same
        #     bucket first in ship_ranking)
        # Only the first is a genuine "bucket failed" signal. Observed
        # in aegis_64_v4 R6: a config-bucket candidate (one-line
        # ``tracer.base_dir`` rebase, direction approved by Critic)
        # failed the structure gate on an anchor-format technicality,
        # and the resulting ``config: [False]`` discouraged all later
        # rounds from touching config — the exact lever Planner and
        # Critic had been begging the Evolver to try.
        # New semantics: only record on actual ship. Non-shipped
        # candidates (gate-fail OR bucket-claim-skip) leave reputation
        # untouched; the Critic's decision body is the right channel
        # to surface implementation issues to the next round.
        from .agents.evolver import parse_candidate_manifest

        shipped_set = set(shipped_cids)
        manifests_by_cid: dict[str, dict] = {}
        for cid, (manifest_path, _cfg_path) in candidates_info.items():
            if not manifest_path.exists():
                continue
            try:
                fm, _body = parse_candidate_manifest(
                    manifest_path.read_text(encoding="utf-8"),
                )
                manifests_by_cid[cid] = fm if isinstance(fm, dict) else {}
                bucket = fm.get("bucket")
                if bucket and cid in shipped_set:
                    self.reputation.record(bucket, hit=True)
            except Exception:
                pass

        # v0.9: scoreboard update — one ShipRecord per shipped cid.
        # flipped_in_ship_round filled on the NEXT round's backfill
        # (we don't know flips yet this round).
        from .data.scoreboard import ShipRecord

        for scid in shipped_cids:
            fm = manifests_by_cid.get(scid) or {}
            predicted = _extract_predicted_tasks(fm)
            self.scoreboard.add_ship(
                ShipRecord(
                    cid=scid,
                    round=round_n,
                    bucket=str(fm.get("bucket", "")),
                    predicted_tasks=tuple(predicted),
                    flipped_in_ship_round=(),
                )
            )
        self.scoreboard.last_updated_round = round_n

        # Ledger: one ship_outcomes entry per shipped cid.
        try:
            for scid in shipped_cids:
                if scid not in manifests_by_cid:
                    continue
                shipped_fm = manifests_by_cid[scid]
                predicted = _extract_predicted_tasks(shipped_fm)
                rejected_siblings = [c for c in candidates_info if c not in shipped_set]
                attr_sig = shipped_fm.get("attribution_signature")
                if not isinstance(attr_sig, dict):
                    attr_sig = None
                ledger.record_ship_outcome(
                    self.run_dir,
                    round_n=round_n,
                    shipped_cid=scid,
                    bucket=str(shipped_fm.get("bucket", "")),
                    predicted_tasks=predicted,
                    rejected_sibling_cids=rejected_siblings,
                    signature=stage_4.get("candidate_signatures", {}).get(scid, ""),
                    attribution_signature=attr_sig,
                    candidate_manifest=shipped_fm,
                )

            # v0.9.5: mark any iterates_from targets as superseded by the new ship.
            for pair in stage_4.get("supersedes") or []:
                target = pair.get("target")
                by_cid = pair.get("new")
                if target and by_cid:
                    try:
                        ledger.mark_ship_superseded(self.run_dir, target, by_cid)
                    except Exception as exc:
                        import logging

                        logging.getLogger("aegis.orchestrator").warning(
                            "mark_ship_superseded failed for %s→%s: %s",
                            target,
                            by_cid,
                            exc,
                        )

            rejected_rows = []
            decision_body = ""
            try:
                decision_path = round_dir / "decision.md"
                if decision_path.exists():
                    decision_body = decision_path.read_text(encoding="utf-8")
            except OSError:
                pass
            for cid, fm in manifests_by_cid.items():
                if cid in shipped_set:
                    continue
                # Extract a ~200-char snippet near the cid mention in decision.md
                excerpt = ""
                if decision_body and cid in decision_body:
                    idx = decision_body.find(cid)
                    start = max(0, idx - 40)
                    excerpt = decision_body[start : start + 400]
                predicted = _extract_predicted_tasks(fm)
                rejected_rows.append(
                    {
                        "candidate_id": cid,
                        "bucket": str(fm.get("bucket", "")),
                        "predicted_tasks": predicted,
                        "rejection_text_excerpt": excerpt,
                        # confidence / novelty_dimension were dropped from the
                        # Evolver manifest schema — agents decide without
                        # self-rated scalars, so ledger no longer records them.
                        "signature": stage_4.get("candidate_signatures", {}).get(cid, ""),
                    }
                )
            if rejected_rows:
                ledger.append_rejected_candidates(
                    self.run_dir,
                    round_n,
                    rejected_rows,
                )
        except Exception as exc:
            import logging

            logging.getLogger("aegis.orchestrator").warning(
                "ledger write (ship_outcomes / rejected_candidates) failed (non-fatal): %s",
                exc,
            )

        # Fix 14: when Stage 4 returns shipped_cid=None because every
        # candidate failed its gates, record the candidates' signatures in
        # refuted_signatures so future novelty gates block duplicates.
        # Skip this for critic no-op (legitimate null decision) — only
        # populate on actual gate rejection.
        refuted: list[str] = []
        if not shipped_cids:
            reason = stage_4.get("reason") or ""
            if reason.startswith("all_candidates") or reason.startswith("decision_chain_broken"):
                refuted = list(stage_4.get("candidate_signatures", {}).values())

        # C2: aggregate predicted_tasks_pass across every shipped candidate,
        # plus per-cid map so the follow-up check can say which specific
        # ship's predictions missed.
        predicted_by_cid: dict[str, list[str]] = {}
        for scid in shipped_cids:
            fm = manifests_by_cid.get(scid, {})
            preds = _extract_predicted_tasks(fm)
            if preds:
                predicted_by_cid[scid] = preds
        flat_predicted = [t for cid_preds in predicted_by_cid.values() for t in cid_preds]

        # Compose the merged next-round base config from all shipped
        # candidates. When only one ship, the result equals its applied
        # YAML; when multiple, each candidate's bucket-specific change is
        # overlaid onto the round's base.
        merged_applied_path: Path | None = None
        if shipped_cids:
            from .compose import compose_shipped_configs

            shipped_entries: list[tuple[str, str | list[str], Path]] = []
            for scid in shipped_cids:
                fm = manifests_by_cid.get(scid, {})
                raw_bucket = (fm or {}).get("bucket", "")
                # Preserve list[str] for cross-bucket bundles (e.g.
                # bucket: [prompt, processor]). Earlier code coerced via
                # str(), which turned a list into "['prompt', 'processor']"
                # — a non-existent bucket name that compose silently
                # skipped, leaving merged.yaml ≡ base.
                if isinstance(raw_bucket, list):
                    bucket: str | list[str] = [str(b) for b in raw_bucket if b]
                else:
                    bucket = str(raw_bucket or "").strip()
                applied_path = round_dir / "applied" / scid / "config.yaml"
                if applied_path.exists() and bucket:
                    shipped_entries.append((scid, bucket, applied_path))
            if shipped_entries:
                merged_applied_path = round_dir / "applied" / "merged.yaml"
                compose_shipped_configs(
                    base_config_path=current_config_path,
                    shipped=shipped_entries,
                    output_path=merged_applied_path,
                )
                # Hard safety: a successful ship MUST mutate the config.
                # If merged ≡ base after compose, an applier silently
                # no-op'd (e.g. unknown bucket name). Refuse to proceed
                # so the round runs on a config that actually reflects
                # the ship, instead of repeating the parent and being
                # mistaken for variance.
                _assert_merged_differs_from_base(
                    base_path=current_config_path,
                    merged_path=merged_applied_path,
                    shipped_cids=shipped_cids,
                )

        narrative = stage_4["reason"] or (f"shipped {', '.join(shipped_cids)}" if shipped_cids else "no_op")
        self._finalize_round(
            round_n,
            shipped_cids=shipped_cids,
            hit_rate=None,
            narrative=narrative,
            refuted_signatures=refuted,
            predicted_tasks_pass=flat_predicted,
            predicted_tasks_by_cid=predicted_by_cid,
        )
        return {
            "shipped_cid": shipped_cids[0] if shipped_cids else None,
            "shipped_cids": shipped_cids,
            "merged_applied_path": str(merged_applied_path) if merged_applied_path else None,
            "reason": stage_4["reason"],
        }

    def _finalize_round(
        self,
        round_n: int,
        *,
        shipped_cids: list[str],
        hit_rate: float | None,
        narrative: str,
        refuted_signatures: list[str],
        predicted_tasks_pass: list[str] | None = None,
        predicted_tasks_by_cid: dict[str, list[str]] | None = None,
    ) -> None:
        shipped_cid = shipped_cids[0] if shipped_cids else None
        entry = RoundEntry(
            round=round_n,
            action="ship" if shipped_cids else "no_op",
            shipped_cid=shipped_cid,
            shipped_cids=list(shipped_cids),
            hypothesis_signatures=[],
            refuted_signatures=refuted_signatures,
            hit_rate=hit_rate,
            narrative=narrative,
            predicted_tasks_pass=predicted_tasks_pass or [],
            predicted_tasks_by_cid=predicted_tasks_by_cid or {},
        )
        self.journal.append(entry)
        self.audit.append(
            AuditEvent(
                round=round_n,
                stage="journal",
                kind="journal",
                payload={"action": entry.action, "shipped_cids": shipped_cids},
                evidence_refs=[str(self.journal.path)],
            )
        )
        self._persist_state()
