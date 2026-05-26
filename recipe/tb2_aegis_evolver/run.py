"""
TB2 multi-round harness evolution using AEGIS (3-role adversarial MAS).

Each round:
  1. Run TB2 eval (via eval_local_docker.sh) → produce rollout trials
  2. Flatten trials to raw JSONL (AEGIS Stage P input format)
  3. AegisAgent.evolve() → maybe produce a new config
  4. If new config returned: adopt it for the next round
  5. If unchanged: keep current config and continue

Run layout:
  {run-dir}/
    state.json                  current_round + current_config
    current_config.yaml         symlink → latest accepted config
    R1/
      trials/                   TB2 eval output ({task}__{trial}/agent/oh_runs/...)
        {job-name}/             job subdirectory written by eval_local_docker.sh
      raw/                      flattened JSONL for AEGIS Stage P
      evolve/                   AegisOrchestrator output (per-round artifacts)
        R1/                     AEGIS internal round dir
    R2/
      ...
    R{N+1}/                     Final validation round (trials only, no evolve)

Configuration via .env file:
  AEGIS_CONFIG=benchmarks/terminal_bench_2/harness_baseline_config.yaml
  AEGIS_RUN_DIR=.benchmarks/aegis-runs/my-run
  AEGIS_R0_TRIALS=.benchmarks/tb2/my-existing-trials
  AEGIS_TASKS=recipe/tb2_hx_evolver/tasks_sample16_seed42_lt15m.json
  AEGIS_NUM_ROUNDS=5
  AEGIS_CONCURRENT=16
  AEGIS_K=3
  AEGIS_META_MODEL=anthropic/claude-opus-4-6
  AEGIS_REPLAY_MODEL=anthropic/claude-sonnet-4-6
  AEGIS_NUM_EVOLVERS=4
  AEGIS_BUDGET_USD=20.0
  AEGIS_MAX_CONCURRENCY=4
  ANTHROPIC_BASE_URL=http://...
  ANTHROPIC_API_KEY=sk-...

Usage:
  # Fresh start
  python -m recipe.tb2_aegis_evolver.run \\
    --config benchmarks/terminal_bench_2/harness_baseline_config.yaml

  # Load from .env
  python -m recipe.tb2_aegis_evolver.run --env evol.env

  # Warm-start: reuse existing trials for round 0 (skip first eval)
  python -m recipe.tb2_aegis_evolver.run \\
    --config benchmarks/terminal_bench_2/harness_baseline_config.yaml \\
    --r0-trials .benchmarks/tb2/my-run

  # Resume an interrupted run
  python -m recipe.tb2_aegis_evolver.run --run-dir .benchmarks/aegis-runs/run-20260519 --resume
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from harnessx.aegis import AegisAgent
from harnessx.core.model_config import ModelConfig

from recipe.tb2_aegis_evolver.tb2_adapter import discover_task_ids, flatten_trials_to_raw

logger = logging.getLogger(__name__)

# ── repo-relative paths ───────────────────────────────────────────────────────

_REPO_ROOT    = Path(__file__).parents[2]
_EVAL_SCRIPT  = _REPO_ROOT / "benchmarks" / "terminal_bench_2" / "scripts" / "eval_local_docker.sh"
_DEFAULT_TASKS = _REPO_ROOT / "recipe" / "tb2_hx_evolver" / "tasks_sample16_seed42_lt15m.json"
_BASELINE_CFG  = _REPO_ROOT / "benchmarks" / "terminal_bench_2" / "harness_baseline_config.yaml"

_DEFAULT_META_MODEL   = "anthropic/claude-opus-4-6"
_DEFAULT_REPLAY_MODEL = "anthropic/claude-sonnet-4-6"
_DEFAULT_NUM_EVOLVERS = 4
_DEFAULT_BUDGET_USD   = 20.0
_DEFAULT_MAX_CONCURRENCY = 4


# ── .env loader ───────────────────────────────────────────────────────────────

def _load_env_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: .env file not found: {path}", file=sys.stderr)
        sys.exit(1)
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


# ── provider factory ──────────────────────────────────────────────────────────

def _make_provider(model: str):
    from harnessx.providers.anthropic_provider import AnthropicProvider
    from harnessx.providers.openai_provider import OpenAIProvider

    base_url    = os.environ.get("AEGIS_BASE_URL")
    api_key     = os.environ.get("AEGIS_API_KEY")
    provider_id = os.environ.get("AEGIS_PROVIDER_ID") or None

    if model.startswith("anthropic/"):
        model_name = model[len("anthropic/"):]
        return AnthropicProvider(
            model=model_name,
            base_url=base_url or os.environ.get("ANTHROPIC_BASE_URL"),
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
    extra_headers = {"X-Model-Provider-Id": provider_id} if provider_id else None
    return OpenAIProvider(
        model,
        base_url=base_url,
        api_key=api_key,
        extra_headers=extra_headers,
    )


# ── TB2 eval runner ───────────────────────────────────────────────────────────

def _run_eval(
    config: Path,
    jobs_dir: Path,
    tasks_json: Path,
    job_name: str,
    concurrent: int,
    k: int,
) -> None:
    """Run TB2 eval via eval_local_docker.sh."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash", str(_EVAL_SCRIPT),
        "--harness-config", str(config),
        "--tasks",          str(tasks_json),
        "--job-name",       job_name,
        "-o",               str(jobs_dir),
        "-n",               str(concurrent),
        "--n-attempts",     str(k),
    ]
    wall_clock_timeout = k * 3600
    logger.info("Running eval (timeout=%ds): %s", wall_clock_timeout, " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT), timeout=wall_clock_timeout)
    except subprocess.TimeoutExpired:
        logger.warning(
            "_run_eval: wall-clock timeout (%ds) exceeded — harbor killed. "
            "Completed tasks are intact; stuck task(s) will score 0.",
            wall_clock_timeout,
        )


# ── state helpers ─────────────────────────────────────────────────────────────

def _load_state(run_dir: Path) -> tuple[int, Path]:
    state = json.loads((run_dir / "state.json").read_text())
    return state["current_round"], Path(state["current_config"])


def _save_state(run_dir: Path, current_round: int, current_config: Path) -> None:
    (run_dir / "state.json").write_text(
        json.dumps(
            {"current_round": current_round, "current_config": str(current_config)},
            indent=2,
        ),
        encoding="utf-8",
    )


def _update_config_link(run_dir: Path, config: Path) -> None:
    link = run_dir / "current_config.yaml"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(config)


def _mean_pass_rate(pass_flags_by_task: dict[str, list[bool]]) -> float | None:
    if not pass_flags_by_task:
        return None
    per_task = [sum(flags) / len(flags) for flags in pass_flags_by_task.values() if flags]
    return sum(per_task) / len(per_task) if per_task else None


# ── main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # ── resolve run directory ─────────────────────────────────────────────────
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (
        _REPO_ROOT / ".benchmarks" / "aegis-runs" / f"aegis-{ts}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── resume or init ────────────────────────────────────────────────────────
    if args.resume:
        if not (run_dir / "state.json").exists():
            print(f"ERROR: state.json not found in {run_dir}; cannot resume", file=sys.stderr)
            sys.exit(1)
        current_round, current_config = _load_state(run_dir)
        logger.info("Resuming from round %d, config: %s", current_round, current_config)
    else:
        if not args.config:
            print("ERROR: --config is required (or use --resume)", file=sys.stderr)
            sys.exit(1)
        current_config = Path(args.config).resolve()
        if not current_config.exists():
            print(f"ERROR: config not found: {current_config}", file=sys.stderr)
            sys.exit(1)
        current_round = 0
        _save_state(run_dir, current_round, current_config)

    tasks_json = Path(args.tasks).resolve() if args.tasks else _DEFAULT_TASKS
    if not tasks_json.exists():
        print(f"ERROR: tasks file not found: {tasks_json}", file=sys.stderr)
        sys.exit(1)

    # ── build AegisAgent ──────────────────────────────────────────────────────
    meta_model   = ModelConfig(main=_make_provider(args.meta_model))
    replay_model = ModelConfig(main=_make_provider(args.replay_model))

    meta_agent = AegisAgent(
        num_evolvers=args.num_evolvers,
        budget_per_round_usd=args.budget_usd,
        max_ask_more=2,
        max_concurrency=args.max_concurrency,
        model_config=meta_model,
        replay_model=replay_model,
    )

    _update_config_link(run_dir, current_config)

    print("═" * 60)
    print("  TB2 AEGIS Harness Evolution")
    print("═" * 60)
    print(f"  run-dir            : {run_dir}")
    print(f"  config             : {current_config}")
    print(f"  r0-trials          : {args.r0_trials or '<fresh eval>'}")
    print(f"  tasks              : {tasks_json}")
    print(f"  num-rounds         : {args.num_rounds}")
    print(f"  concurrent / k     : {args.concurrent} / {args.k}")
    print(f"  meta-model         : {args.meta_model}")
    print(f"  replay-model       : {args.replay_model}")
    print(f"  num-evolvers       : {args.num_evolvers}")
    print(f"  budget-usd/round   : {args.budget_usd}")
    print(f"  max-concurrency    : {args.max_concurrency}")
    print("═" * 60)

    # ── main round loop ───────────────────────────────────────────────────────
    round_summaries: list[dict] = []

    while current_round < args.num_rounds:
        round_label = f"R{current_round + 1}"
        round_dir   = run_dir / round_label
        jobs_dir    = round_dir / "trials"
        job_name    = f"aegis-{round_label}-{run_dir.name}"
        trials_dir  = jobs_dir / job_name
        raw_dir     = round_dir / "raw"
        evolve_dir  = round_dir / "evolve"

        print(f"\n{'═' * 60}")
        print(f"  Round {current_round + 1} / {args.num_rounds}   [{round_label}]   config: {current_config.name}")
        print("═" * 60)

        # ── step 1: acquire trials ────────────────────────────────────────────
        if trials_dir.exists() and any(trials_dir.iterdir()):
            print(f"  [skip eval] trials already exist: {trials_dir}")
        elif current_round == 0 and args.r0_trials:
            r0 = Path(args.r0_trials).resolve()
            if not r0.exists():
                print(f"ERROR: --r0-trials not found: {r0}", file=sys.stderr)
                sys.exit(1)
            jobs_dir.mkdir(parents=True, exist_ok=True)
            trials_dir.symlink_to(r0)
            print(f"  [R1 warm-start] linked existing trials: {r0}")
        else:
            print("  [eval] running TB2 …")
            _run_eval(current_config, jobs_dir, tasks_json, job_name, args.concurrent, args.k)

        # ── step 2: discover task IDs ─────────────────────────────────────────
        task_ids = discover_task_ids(trials_dir)
        if not task_ids:
            print(f"  ERROR: no tasks found under {trials_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"  tasks discovered: {len(task_ids)}")

        # ── step 3: flatten trials → raw JSONL ────────────────────────────────
        if raw_dir.exists() and any(raw_dir.glob("*.jsonl")):
            print(f"  [skip flatten] raw JSONL already exists: {raw_dir}")
            # Rebuild pass_flags from existing trials (need for history logging)
            pass_flags = flatten_trials_to_raw(trials_dir, raw_dir)
        else:
            print("  [flatten] converting trials to raw JSONL for AEGIS …")
            pass_flags = flatten_trials_to_raw(trials_dir, raw_dir)
            print(f"  flattened {sum(len(v) for v in pass_flags.values())} trials "
                  f"across {len(pass_flags)} tasks")

        mean_pr = _mean_pass_rate(pass_flags)
        pr_str  = f"{mean_pr:.1%}" if mean_pr is not None else "n/a"
        print(f"  pass rate (mean_k): {pr_str}")

        # ── step 4: AEGIS evolve ──────────────────────────────────────────────
        aegis_done_marker = evolve_dir / "aegis_done.json"
        if aegis_done_marker.exists():
            print(f"  [skip evolve] aegis_done.json already exists: {evolve_dir}")
            done_data    = json.loads(aegis_done_marker.read_text())
            new_cfg_path = Path(done_data["new_config_path"])
        else:
            print("  [evolve] running AegisAgent …")
            evolve_dir.mkdir(parents=True, exist_ok=True)
            new_cfg_path = await meta_agent.evolve(
                current_config,
                trajectories_dir=raw_dir,
                output_dir=evolve_dir,
                pass_flags_by_task=pass_flags,
                round_n=current_round + 1,
            )
            aegis_done_marker.write_text(
                json.dumps({"new_config_path": str(new_cfg_path)}, indent=2),
                encoding="utf-8",
            )

        # ── step 5: adopt or keep config ──────────────────────────────────────
        prev_config = current_config
        changed = (
            new_cfg_path != current_config
            and new_cfg_path.exists()
            and new_cfg_path.read_text(encoding="utf-8") != current_config.read_text(encoding="utf-8")
        )
        if changed:
            current_config = new_cfg_path
            _update_config_link(run_dir, current_config)
            print(f"  ✓ ACCEPTED — new config: {current_config}")
        else:
            print("  ✗ NOT ACCEPTED — keeping current config")

        round_summaries.append({
            "round":       current_round + 1,
            "label":       round_label,
            "accepted":    changed,
            "pass_rate":   mean_pr,
            "new_config":  str(current_config),
        })

        # ── step 6: persist state ─────────────────────────────────────────────
        current_round += 1
        _save_state(run_dir, current_round, current_config)

    # ── final validation trial ─────────────────────────────────────────────────
    final_label    = f"R{current_round + 1}"
    final_round_dir = run_dir / final_label
    final_job_name  = f"aegis-{final_label}-{run_dir.name}"
    final_jobs_dir  = final_round_dir / "trials"
    final_trials_dir = final_jobs_dir / final_job_name

    print(f"\n{'═' * 60}")
    print(f"  Final validation [{final_label}] — config: {current_config.name}")
    print("═" * 60)
    if final_trials_dir.exists() and any(final_trials_dir.iterdir()):
        print(f"  [skip final-eval] trials already exist: {final_trials_dir}")
    else:
        print("  [final-eval] running TB2 …")
        _run_eval(current_config, final_jobs_dir, tasks_json, final_job_name, args.concurrent, args.k)

    # Compute final pass rate
    final_raw_dir = final_round_dir / "raw"
    final_pass_flags = flatten_trials_to_raw(final_trials_dir, final_raw_dir)
    final_pr = _mean_pass_rate(final_pass_flags)
    final_pr_str = f"{final_pr:.1%}" if final_pr is not None else "n/a"

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("═" * 60)
    print(f"  AEGIS evolution complete: {args.num_rounds} rounds")
    print(f"  Final config : {current_config}")
    print(f"  Run dir      : {run_dir}")
    print("═" * 60)
    print()
    print(f"  {'Round':>6} | {'Acc':^3} | {'mean_k':>8} | new_config")
    print(f"  {'------':>6}-+-{'---':^3}-+-{'--------':>8}-+-" + "-" * 40)
    for s in round_summaries:
        acc = "✓" if s["accepted"] else "✗"
        pr  = f"{s['pass_rate']:.1%}" if s["pass_rate"] is not None else "n/a"
        cfg = Path(s["new_config"]).name
        print(f"  {s['label']:>6} | {acc:^3} | {pr:>8} | {cfg}")
    print(f"  {final_label:>6} | val | {final_pr_str:>8} | (final validation)")
    print()


if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--env", type=str, default=None)
    _pre_args, _ = _pre.parse_known_args()
    if _pre_args.env:
        _load_env_file(_pre_args.env)

    parser = argparse.ArgumentParser(description="TB2 multi-round harness evolution with AEGIS")
    parser.add_argument("--env",             type=str, default=None,
                        help="Path to .env file (AEGIS_* vars). Env overrides default.")
    parser.add_argument("--config",          type=str,
                        default=_env("AEGIS_CONFIG"),
                        help="Initial harness config YAML (required unless --resume). Env: AEGIS_CONFIG")
    parser.add_argument("--run-dir",         type=str,
                        default=_env("AEGIS_RUN_DIR"),
                        help="Base output directory. Env: AEGIS_RUN_DIR")
    parser.add_argument("--r0-trials",       type=str,
                        default=_env("AEGIS_R0_TRIALS"),
                        help="Warm-start: link existing trials dir for round 1. Env: AEGIS_R0_TRIALS")
    parser.add_argument("--tasks",           type=str,
                        default=_env("AEGIS_TASKS"),
                        help=f"Task list JSON. Default: {_DEFAULT_TASKS.name}. Env: AEGIS_TASKS")
    parser.add_argument("--num-rounds",      type=int,
                        default=int(_env("AEGIS_NUM_ROUNDS", "5")),
                        help="Number of evolution rounds. Env: AEGIS_NUM_ROUNDS")
    parser.add_argument("--concurrent",      type=int,
                        default=int(_env("AEGIS_CONCURRENT", "16")),
                        help="TB2 eval concurrency. Env: AEGIS_CONCURRENT")
    parser.add_argument("-k",                type=int,
                        default=int(_env("AEGIS_K", "3")),
                        help="Rollouts per task. Env: AEGIS_K")
    parser.add_argument("--meta-model",      type=str,
                        default=_env("AEGIS_META_MODEL", _DEFAULT_META_MODEL),
                        help="AEGIS meta model (Planner/Critic). Env: AEGIS_META_MODEL")
    parser.add_argument("--replay-model",    type=str,
                        default=_env("AEGIS_REPLAY_MODEL", _DEFAULT_REPLAY_MODEL),
                        help="Stage 4 replay gate model. Env: AEGIS_REPLAY_MODEL")
    parser.add_argument("--num-evolvers",    type=int,
                        default=int(_env("AEGIS_NUM_EVOLVERS", str(_DEFAULT_NUM_EVOLVERS))),
                        help="Number of parallel Evolvers. Env: AEGIS_NUM_EVOLVERS")
    parser.add_argument("--budget-usd",      type=float,
                        default=float(_env("AEGIS_BUDGET_USD", str(_DEFAULT_BUDGET_USD))),
                        help="Budget per round in USD. Env: AEGIS_BUDGET_USD")
    parser.add_argument("--max-concurrency", type=int,
                        default=int(_env("AEGIS_MAX_CONCURRENCY", str(_DEFAULT_MAX_CONCURRENCY))),
                        help="AEGIS internal concurrency. Env: AEGIS_MAX_CONCURRENCY")
    parser.add_argument("--resume",          action="store_true",
                        help="Continue from last completed round in --run-dir")
    args = parser.parse_args()
    asyncio.run(main(args))
