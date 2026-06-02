#!/usr/bin/env python3
"""tau2 AEGIS Pilot: harness evolution using AegisAgent.

Runs the 6-stage AEGIS loop (Preprocess → Plan → Propose → Judge → Commit → Adjudicate)
on tau2-bench domains. Each round evaluates the current harness config via tau2 simulations,
then feeds trajectories to AegisAgent.evolve() to produce an improved config.

Usage::

    # Smoke test (2 rounds, 5 tasks, 2 evolvers)
    python -m recipe.tau2_evolver.run_meta_aegis --smoke \\
        --domain airline --base-config recipe/tau2_evolver/configs/airline_base \\
        --output-dir recipe/tau2_evolver/runs/aegis_smoke

    # Full pilot
    python -m recipe.tau2_evolver.run_meta_aegis \\
        --domain airline --base-config recipe/tau2_evolver/configs/airline_base \\
        --output-dir recipe/tau2_evolver/runs/aegis_airline \\
        --num-rounds 5 --num-trials 4 --max-concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_env_path = Path(_PROJECT_ROOT) / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import litellm as _litellm

_litellm.suppress_debug_info = True

from harnessx.aegis import AegisAgent
from harnessx.core.harness import HarnessConfig
from harnessx.core.model_config import ModelConfig

from .defaults import (
    DEFAULT_AGENT_API_BASE,
    DEFAULT_AGENT_EXTENDED_THINKING,
    DEFAULT_AGENT_MODEL,
    DEFAULT_AGENT_THINKING_BUDGET,
    DEFAULT_META_API_BASE,
    DEFAULT_META_MODEL,
    DEFAULT_USER_API_BASE,
    DEFAULT_USER_MODEL,
    EVOLVE_COST_CAP_USD,
    MAX_CONCURRENCY,
    MAX_SIM_STEPS,
    NUM_ROUNDS,
    NUM_TRIALS,
    REGRESSION_TOLERANCE,
)
from .run import _make_provider, _register_tau2_agents, _run_tau2_round, _write_task_trajectory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
logger = logging.getLogger("tau2_aegis")


# ---------------------------------------------------------------------------
# Session flattening: tau2 UUID-based sessions → flat raw/ dir for Aegis Stage P
# ---------------------------------------------------------------------------


def _flatten_sessions_to_raw(
    sessions_dir: Path,
    raw_dir: Path,
    records: list[dict],
) -> None:
    """Flatten tau2 session JSONL files into raw/{task_id}_r{trial_idx}.jsonl.

    tau2 sessions are stored as sessions/{session_id}/{run_id}.jsonl where
    session_id is a UUID. Since tau2 does not populate provider_session_id,
    we match sessions to tasks by correlating the first user message content
    between the session JSONL and the simulation's messages.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not sessions_dir.exists():
        return

    # Build index: first_user_msg_prefix → session JSONL path
    session_index: dict[str, Path] = {}
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        jsonl_files = sorted(f for f in session_dir.glob("*.jsonl") if not f.name.endswith("_trace.jsonl"))
        if not jsonl_files:
            continue
        jsonl_path = jsonl_files[0]
        try:
            with open(jsonl_path) as fh:
                for line in fh:
                    evt = json.loads(line)
                    if evt.get("type") == "raw_user":
                        msg = evt.get("message", {})
                        content = msg.get("content", "") if isinstance(msg, dict) else ""
                        key = content[:200]
                        session_index[key] = jsonl_path
                        break
        except (json.JSONDecodeError, OSError):
            continue

    # Match records to sessions
    for record in records:
        task_id = record.get("task_id")
        sims = record.get("_sims", [])
        if not task_id or not sims:
            continue

        for trial_idx, sim in enumerate(sims):
            messages = getattr(sim, "messages", None) or []
            first_user = ""
            for m in messages:
                role = m.role if hasattr(m, "role") else m.get("role", "")
                if role == "user":
                    content = m.content if hasattr(m, "content") else m.get("content", "")
                    first_user = (content or "")[:200]
                    break
            if not first_user:
                continue
            jsonl_path = session_index.get(first_user)
            if jsonl_path:
                dst = raw_dir / f"{task_id}_r{trial_idx}.jsonl"
                if not dst.exists():
                    shutil.copy2(jsonl_path, dst)


# ---------------------------------------------------------------------------
# Gating helpers
# ---------------------------------------------------------------------------


def _compute_pass_flags(records: list[dict]) -> dict[str, list[bool]]:
    """Build pass_flags_by_task from tau2 records (reward > 0 → True per trial)."""
    flags: dict[str, list[bool]] = {}
    for record in records:
        task_id = str(record["task_id"])
        sims = record.get("_sims", [])
        if sims:
            flags[task_id] = [(getattr(s.reward_info, "reward", 0.0) or 0.0) > 0 for s in sims]
        else:
            flags[task_id] = [record.get("reward", 0.0) > 0]
    return flags


def _save_curves(run_dir: Path, curves: list[dict]) -> None:
    (run_dir / "curves.json").write_text(json.dumps(curves, indent=2, ensure_ascii=False))


def _append_rollback_reputation(run_dir: Path, buckets: list[str]) -> None:
    rep_path = run_dir / "reputation.json"
    rep: dict = {}
    if rep_path.exists():
        try:
            rep = json.loads(rep_path.read_text())
        except json.JSONDecodeError:
            rep = {}
    for b in buckets:
        history = rep.setdefault(b, [])
        history.append(False)
        if len(history) > 5:
            rep[b] = history[-5:]
    rep_path.write_text(json.dumps(rep, indent=2))


def _read_latest_commit_shipments(run_dir: Path, round_n: int) -> list[tuple[str, str]]:
    """Read (cid, bucket) pairs from the last commit stage in audit.jsonl."""
    audit_path = run_dir / "audit.jsonl"
    if not audit_path.exists():
        return []
    shipped = []
    for line in reversed(audit_path.read_text().splitlines()):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("round") != round_n:
            continue
        if entry.get("stage") == "4" and entry.get("kind") == "commit":
            payload = entry.get("payload", {})
            by_bucket = payload.get("shipped_by_bucket", {})
            for bucket, cid in by_bucket.items():
                if cid:
                    shipped.append((cid, bucket))
    return shipped


# ---------------------------------------------------------------------------
# Main pilot loop
# ---------------------------------------------------------------------------


async def run_pilot(args: argparse.Namespace) -> None:
    """Run the tau2 AEGIS evolution pilot."""
    unset_vars = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]
    for v in unset_vars:
        os.environ.pop(v, None)

    os.environ.setdefault("TAU2_DATA_DIR", str(Path.home() / "tau2-bench" / "data"))

    _register_tau2_agents()

    from tau2.runner.helpers import get_tasks

    all_tasks = get_tasks(args.domain, task_split_name=args.task_split)
    if args.max_tasks and args.max_tasks > 0:
        all_tasks = all_tasks[: args.max_tasks]
    tasks = all_tasks

    logger.info(
        "tau2 AEGIS pilot: domain=%s split=%s tasks=%d trials=%d rounds=%d",
        args.domain,
        args.task_split,
        len(tasks),
        args.num_trials,
        args.num_rounds,
    )

    # Build meta model provider (for Aegis agents)
    meta_provider = _make_provider(
        model=args.meta_model,
        api_base=args.meta_api_base,
        extended_thinking=args.meta_extended_thinking,
        thinking_budget_tokens=args.meta_thinking_budget,
        max_tokens=args.meta_max_tokens,
    )
    meta_model = ModelConfig(main=meta_provider)

    # Build task model provider (for replay gate — small max_tokens to avoid
    # Anthropic SDK "streaming required" error on large max_tokens values)
    task_provider = _make_provider(
        model=args.agent_model,
        api_base=args.agent_api_base,
        extended_thinking=False,
        thinking_budget_tokens=10_000,
        max_tokens=4096,
    )
    task_model = ModelConfig(main=task_provider)

    meta_agent = AegisAgent(
        num_evolvers=args.num_evolvers,
        budget_per_round_usd=args.evolve_cost,
        max_ask_more=2,
        max_concurrency=args.concurrency,
        model_config=meta_model,
        replay_model=task_model,
        auto_revert_enabled=True,
        benchmark_context="tau2",
    )

    # Output directory
    RUN_DIR = Path(args.output_dir).resolve()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Load base config
    base_config_path = Path(args.base_config).resolve() / "config.yaml"
    original_base = HarnessConfig.from_yaml_file(str(base_config_path))
    current_config = original_base

    best_so_far: tuple[float, int, HarnessConfig, int] | None = None
    curves: list[dict] = []
    noop_streak = 0
    next_evolve_status = "baseline"
    last_ship_info: dict | None = None
    last_validated_reward: float = 0.0
    last_validated_pass_count: int | None = None

    # Resume support
    if args.start_round > 0:
        curves_path = RUN_DIR / "curves.json"
        if curves_path.exists():
            try:
                curves = json.loads(curves_path.read_text())
            except json.JSONDecodeError:
                curves = []
        prev = args.start_round - 1
        merged_path = RUN_DIR / f"R{prev}" / "applied" / "merged.yaml"
        baseline_path = RUN_DIR / f"R{prev}" / "config.yaml"
        seed_path = merged_path if merged_path.exists() else baseline_path
        if not seed_path.exists():
            raise FileNotFoundError(f"--start-round={args.start_round} requires R{prev} state")
        logger.info("resume: loading config from %s", seed_path)
        current_config = HarnessConfig.from_yaml_file(str(seed_path))
        next_evolve_status = "ok"
        if curves:
            last_validated_reward = curves[-1].get("avg_reward", 0.0)
            last_validated_pass_count = int(curves[-1].get("passed", 0))

    # ── Round loop ────────────────────────────────────────────────────────────
    for round_idx in range(args.num_rounds):
        if round_idx < args.start_round:
            continue
        is_last = round_idx == args.num_rounds - 1

        round_dir = RUN_DIR / f"R{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        traj_dir = round_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir = round_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Write current config
        round_config_path = round_dir / "config.yaml"
        current_config.to_yaml_file(round_config_path)
        config_hash = hashlib.sha256(round_config_path.read_bytes()).hexdigest()[:16]

        logger.info("\n" + "=" * 70)
        logger.info(
            "ROUND %d/%d  domain=%s  config=%s  status=%s",
            round_idx,
            args.num_rounds - 1,
            args.domain,
            config_hash,
            next_evolve_status,
        )
        logger.info("=" * 70)

        # ── Eval: run tau2 simulations ────────────────────────────────────────
        report_path = round_dir / "report.json"
        records = await asyncio.to_thread(
            _run_tau2_round,
            domain=args.domain,
            task_split=args.task_split,
            tasks=tasks,
            round_config_path=round_config_path,
            sessions_dir=sessions_dir,
            agent_model=args.agent_model,
            agent_api_base=args.agent_api_base or None,
            agent_extended_thinking=args.agent_extended_thinking,
            agent_thinking_budget=args.agent_thinking_budget,
            user_model=args.user_model,
            user_api_base=args.user_api_base or None,
            user_temperature=args.user_temperature,
            agent_temperature=args.agent_temperature,
            judge_model=args.judge_model or args.user_model,
            judge_api_base=args.judge_api_base,
            num_trials=args.num_trials,
            max_steps=args.max_sim_steps,
            max_concurrency=args.max_concurrency,
            report_path=report_path,
        )

        # ── Write trajectory .md files ────────────────────────────────────────
        harness_cfg = current_config
        for rec in records:
            try:
                sims = rec.get("_sims", [])
                if sims:
                    _write_task_trajectory(
                        traj_dir,
                        str(rec["task_id"]),
                        rec,
                        sims,
                        harness_config=harness_cfg,
                    )
            except Exception:
                logger.warning("Failed to write trajectory for task %s", rec.get("task_id"), exc_info=True)

        # ── Compute metrics ───────────────────────────────────────────────────
        rewards = [r.get("reward", 0.0) for r in records]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        passed = sum(1 for r in rewards if r > 0)
        n_tasks = len(records)
        pass_rate = passed / n_tasks if n_tasks else 0.0
        round_cost = sum(r.get("cost_usd", 0.0) for r in records)

        pass_flags_by_task = _compute_pass_flags(records)

        curve_point = {
            "round": round_idx,
            "config_hash": config_hash,
            "evolve_status": next_evolve_status,
            "total_tasks": n_tasks,
            "passed": passed,
            "pass_rate": pass_rate,
            "avg_reward": round(avg_reward, 4),
            "cost_usd": round(round_cost, 4),
        }
        curves.append(curve_point)
        _save_curves(RUN_DIR, curves)

        # Append task history for Aegis ledger
        try:
            from harnessx.aegis.data import ledger as _ledger

            task_rows = []
            for rec in records:
                tid = str(rec["task_id"])
                sims = rec.get("_sims", [])
                flags = (
                    [(getattr(s.reward_info, "reward", 0) or 0) > 0 for s in sims]
                    if sims
                    else [rec.get("reward", 0) > 0]
                )
                task_rows.append(
                    {
                        "round": round_idx,
                        "task_id": tid,
                        "level": None,
                        "passed": any(flags),
                        "passed_flags": flags,
                        "k": len(flags),
                        "exit": str(rec.get("termination_reason", "")),
                        "steps": int(rec.get("steps") or 0),
                        "cost_usd": float(rec.get("cost_usd") or 0.0),
                        "final_output_len": int(rec.get("final_output_length") or 0),
                        "tools_used": list(rec.get("tool_call_counts", {}).keys()),
                    }
                )
            _ledger.append_task_history(RUN_DIR, task_rows)
        except Exception as _exc:
            logger.warning("task_history append failed (non-fatal): %s", _exc)

        logger.info(
            "[R%d] avg_reward=%.4f  pass=%d/%d (%.1f%%)  cost=$%.2f",
            round_idx,
            avg_reward,
            passed,
            n_tasks,
            pass_rate * 100,
            round_cost,
        )

        # ── Gating: track best ────────────────────────────────────────────────
        if best_so_far is None or avg_reward > best_so_far[0]:
            best_so_far = (avg_reward, passed, current_config, round_idx)

        # ── Ship-aware rollback ───────────────────────────────────────────────
        rollback_fired = False
        if last_ship_info is not None and last_validated_pass_count is not None:
            delta_reward = avg_reward - last_validated_reward
            delta_count = passed - last_validated_pass_count
            tol = args.regression_tolerance
            if delta_reward <= -tol and delta_count <= -3:
                shipped = last_ship_info["shipped"]
                shipped_cids = [cid for cid, _ in shipped]
                shipped_buckets = [bucket for _, bucket in shipped]
                logger.warning(
                    "[R%d] ROLLBACK — post-ship regression Δreward=%+.4f Δcount=%+d "
                    "vs last_validated=%.4f; reverting: %s",
                    round_idx,
                    delta_reward,
                    delta_count,
                    last_validated_reward,
                    shipped_cids,
                )
                current_config = last_ship_info["pre_ship_config"]
                _append_rollback_reputation(RUN_DIR, shipped_buckets)
                rollback_fired = True
            last_ship_info = None

        if not rollback_fired:
            last_validated_reward = avg_reward
            last_validated_pass_count = passed

        if is_last:
            continue

        # ── Evolve via AegisAgent ─────────────────────────────────────────────
        next_round_dir = RUN_DIR / f"R{round_idx + 1}"
        next_round_dir.mkdir(parents=True, exist_ok=True)

        # Flatten sessions → raw/ for Stage P
        raw_dir = round_dir / "raw"
        _flatten_sessions_to_raw(sessions_dir, raw_dir, records)

        logger.info("[R%d] evolve (planning R%d)", round_idx, round_idx + 1)
        try:
            new_yaml = await meta_agent.evolve(
                current_config=round_config_path,
                trajectories_dir=raw_dir,
                output_dir=next_round_dir,
                pass_flags_by_task=pass_flags_by_task,
                round_n=round_idx + 1,
                raw_sessions_dir=raw_dir,
            )

            new_yaml_path = Path(new_yaml)
            if round_config_path.read_bytes() == new_yaml_path.read_bytes():
                next_evolve_status = "noop"
                noop_streak += 1
            else:
                next_evolve_status = "ok"
                noop_streak = 0
                pre_ship_config = current_config
                shipped_this_round = _read_latest_commit_shipments(RUN_DIR, round_n=round_idx + 1)
                current_config = HarnessConfig.from_yaml_file(new_yaml_path).canonicalize()
                if shipped_this_round:
                    last_ship_info = {
                        "pre_ship_config": pre_ship_config,
                        "pre_ship_reward": avg_reward,
                        "pre_ship_passed": passed,
                        "shipped": shipped_this_round,
                    }

            logger.info(
                "[R%d] evolved → %s  status=%s  noop_streak=%d",
                round_idx,
                new_yaml,
                next_evolve_status,
                noop_streak,
            )

            if noop_streak >= 2:
                logger.info("EARLY STOP: %d consecutive unchanged configs", noop_streak)
                break

        except Exception as exc:
            next_evolve_status = "crashed"
            noop_streak = 0
            logger.exception("[R%d] evolve crashed: %s", round_idx, exc)

    # ── Final summary ─────────────────────────────────────────────────────────
    _save_curves(RUN_DIR, curves)
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY — %s/%s (%d rounds)", args.domain, args.task_split, len(curves))
    logger.info("=" * 70)
    for c in curves:
        logger.info(
            "  R%d: avg_reward=%.4f  pass=%d/%d  cost=$%.2f  status=%s",
            c["round"],
            c["avg_reward"],
            c["passed"],
            c["total_tasks"],
            c["cost_usd"],
            c["evolve_status"],
        )
    if best_so_far:
        logger.info("  Best: R%d avg_reward=%.4f", best_so_far[3], best_so_far[0])
    logger.info("Results → %s", RUN_DIR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="tau2 AEGIS Pilot: harness evolution via AegisAgent.")
    p.add_argument("--domain", default="airline")
    p.add_argument("--task-split", default="base")
    p.add_argument("--base-config", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-tasks", type=int, default=0, help="0 = all tasks")
    p.add_argument("--num-rounds", type=int, default=NUM_ROUNDS)
    p.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    p.add_argument("--max-sim-steps", type=int, default=MAX_SIM_STEPS)
    p.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    p.add_argument("--sim-timeout", type=float, default=3600.0)
    # Agent model
    p.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    p.add_argument("--agent-api-base", default=DEFAULT_AGENT_API_BASE)
    p.add_argument("--agent-temperature", type=float, default=None)
    p.add_argument("--agent-extended-thinking", action="store_true", default=DEFAULT_AGENT_EXTENDED_THINKING)
    p.add_argument("--agent-thinking-budget", type=int, default=DEFAULT_AGENT_THINKING_BUDGET)
    # User model
    p.add_argument("--user-model", default=DEFAULT_USER_MODEL)
    p.add_argument("--user-api-base", default=DEFAULT_USER_API_BASE)
    p.add_argument("--user-temperature", type=float, default=0.0)
    # Judge model
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-api-base", default=None)
    # Meta model (for Aegis agents)
    p.add_argument("--meta-model", default=DEFAULT_META_MODEL)
    p.add_argument("--meta-api-base", default=DEFAULT_META_API_BASE)
    p.add_argument("--meta-extended-thinking", action="store_true", default=False)
    p.add_argument("--meta-thinking-budget", type=int, default=DEFAULT_AGENT_THINKING_BUDGET)
    p.add_argument("--meta-max-tokens", type=int, default=16384)
    # Aegis params
    p.add_argument("--evolve-cost", type=float, default=EVOLVE_COST_CAP_USD)
    p.add_argument("--num-evolvers", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=4, help="Aegis meta-agent concurrency")
    # Gating
    p.add_argument("--regression-tolerance", type=float, default=REGRESSION_TOLERANCE)
    # Resume / control
    p.add_argument("--start-round", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="Quick test: 2 rounds, 5 tasks, 2 evolvers")
    return p


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    if args.smoke:
        args.num_rounds = 2
        args.max_tasks = 5
        args.num_evolvers = 2
        args.num_trials = 1
        args.evolve_cost = 20.0

    asyncio.run(run_pilot(args))


if __name__ == "__main__":
    main()
