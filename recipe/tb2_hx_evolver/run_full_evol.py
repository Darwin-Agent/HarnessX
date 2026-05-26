"""
TB2 multi-round harness evolution runner.

Each round:
  1. Run TB2 eval (via eval_local_docker.sh) → produce rollout trials
  2. Run EvolOrchestrator (L1 → L2 → L3) → maybe produce a new config
  3. If accepted: adopt the new config for the next round
  4. If rejected: keep the current config and continue

Run layout:
  {run-dir}/
    state.json                  current_round + current_config (updated after every round)
    solvability.json            cross-round SolvabilityJournal
    evolution_notebook.md       DigestAgent / EvolveAgent shared notes (cross-round)
    current_config.yaml         symlink → latest accepted config
    round_000/
      trials/                   TB2 eval output ({task}__{trial}/agent/oh_runs/...)
      evolve/                   EvolOrchestrator output (L1+L2+L3 artifacts)
        evolve_result.json      {accepted, new_config_path, pass_rate, total_tasks, ...}
    round_001/
      ...
    round_{N}/                  Final validation round (trials only, no evolve)
      trials/                   TB2 eval of the last evolved config

Configuration via .env file (loaded with --env; any CLI flag can be set here):
  EVOL_CONFIG=benchmarks/terminal_bench_2/harness_baseline_config.yaml
  EVOL_RUN_DIR=.benchmarks/evolve-runs/my-run
  EVOL_R0_TRIALS=.benchmarks/tb2/my-existing-trials
  EVOL_TASKS=recipe/tb2_hx_evolver/tasks_sample16_seed42_lt15m.json
  EVOL_NUM_ROUNDS=5
  EVOL_CONCURRENT=4
  EVOL_K=3
  EVOL_DIGEST_MODEL=ppio/pa/claude-haiku-4-5-20251001
  EVOL_EVOLVE_MODEL=ppio/pa/claude-sonnet-4-6
  EVOL_DIGEST_MAX_STEPS=200
  EVOL_EVOLVE_MAX_STEPS=300
  ANTHROPIC_BASE_URL=http://...
  ANTHROPIC_API_KEY=sk-...

Usage:
  # Fresh start (flags override .env)
  python -m recipe.tb2_hx_evolver.run_full_evol \\
    --config benchmarks/terminal_bench_2/harness_baseline_config.yaml

  # Load from .env in cwd
  python -m recipe.tb2_hx_evolver.run_full_evol --env .env

  # Warm-start: reuse existing trials for round 0, skip first eval
  python -m recipe.tb2_hx_evolver.run_full_evol \\
    --config benchmarks/terminal_bench_2/harness_baseline_config.yaml \\
    --r0-trials .benchmarks/tb2/my-run

  # Resume an interrupted run
  python -m recipe.tb2_hx_evolver.run_full_evol \\
    --run-dir .benchmarks/evolve-runs/evolve-20260513-120000 \\
    --resume
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from harnessx.core.model_config import ModelConfig
from harnessx.experimental.harness_evol.orchestrator import (
    EvolOrchestrator,
    EvolRoundOutput,
    TaskContext,
)
from harnessx.experimental.harness_evol.signals.parser import parse_session_rollout
from harnessx.experimental.harness_evol.signals.runner import ScoreFn
from harnessx.experimental.harness_evol.signals.schema import RolloutData

from recipe.tb2_hx_evolver.score import tb2_score_by_session_dir

logger = logging.getLogger(__name__)

# ── repo-relative paths ───────────────────────────────────────────────────────

_REPO_ROOT   = Path(__file__).parents[2]
_EVAL_SCRIPT = _REPO_ROOT / "benchmarks" / "terminal_bench_2" / "scripts" / "eval_local_docker.sh"
_DEFAULT_TASKS = _REPO_ROOT / "recipe" / "tb2_hx_evolver" / "tasks_sample16_seed42_lt15m.json"
_BASELINE_CFG  = _REPO_ROOT / "benchmarks" / "terminal_bench_2" / "harness_baseline_config.yaml"

# TB2-specific benchmark knowledge for EvolveAgent (sandbox topology, evolvable config surface).
_TB2_SKILL_DIRS = [
    Path(__file__).parent.parent / "tb2_evolver" / "skills" / "tb2-playbook"
]

_DEFAULT_DIGEST_MODEL = "anthropic/claude-haiku-4-5-20251001"
_DEFAULT_EVOLVE_MODEL = "anthropic/claude-sonnet-4-6"
_DEFAULT_DIGEST_MAX_STEPS = 200
_DEFAULT_EVOLVE_MAX_STEPS = 300


# ── .env loader ───────────────────────────────────────────────────────────────

def _load_env_file(path: str) -> None:
    """Parse a simple KEY=VALUE .env file and set missing env vars."""
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
        # Only set if not already in env (explicit CLI env takes precedence)
        os.environ.setdefault(key, value)


def _env(key: str, default: str | None = None) -> str | None:
    """Read a value from env (populated from .env if --env was passed)."""
    return os.environ.get(key, default)


# ── TB2 adapter ───────────────────────────────────────────────────────────────

class TB2EvolOrchestrator(EvolOrchestrator):
    """
    EvolOrchestrator adapted for the TB2 / OpenHands session directory layout.

    TB2 layout under trials_dir:
      {task_name}__{trial_id}/agent/oh_runs/{run_id}/
    Each session dir may contain multiple *_trace.jsonl segments;
    parse_session_rollout merges them into a single RolloutData.

    Overrides _load_task_results to:
      1. glob session dirs matching {task_id}__* under trajectories_dir
      2. parse each session dir with parse_session_rollout
      3. apply tb2_score_fn to set eval_passed / eval_score / eval_feedback
    """

    def __init__(self, tb2_score_fn: ScoreFn, **kwargs):
        super().__init__(**kwargs)
        self._tb2_score_fn = tb2_score_fn

    def _load_task_results(
        self,
        trajectories_dir: Path,
        task_ids: list[str],
    ) -> dict[str, list[RolloutData]]:
        result: dict[str, list[RolloutData]] = {}
        for task_id in task_ids:
            rollouts: list[RolloutData] = []
            for session_dir in sorted(
                trajectories_dir.glob(f"{task_id}__*/agent/oh_runs/*/")
            ):
                if not session_dir.is_dir():
                    continue
                rollout = parse_session_rollout(session_dir)
                if rollout is None:
                    logger.warning("parse_session_rollout failed: %s", session_dir)
                    continue
                try:
                    passed, score, feedback = self._tb2_score_fn(session_dir)
                    rollout = dataclasses.replace(
                        rollout,
                        eval_passed=passed,
                        eval_score=score,
                        eval_feedback=feedback,
                    )
                except Exception as exc:
                    logger.warning("tb2_score_fn failed for %s: %s", session_dir, exc)
                rollouts.append(rollout)
            if not rollouts:
                logger.warning("No rollouts for task %s under %s", task_id, trajectories_dir)
            result[task_id] = rollouts
        return result


# ── helpers ───────────────────────────────────────────────────────────────────

def discover_task_ids(trials_dir: Path) -> list[str]:
    """Return unique task names from {task_name}__{trial_id} subdirs."""
    task_ids: set[str] = set()
    for entry in trials_dir.iterdir():
        if entry.is_dir() and "__" in entry.name:
            task_ids.add(entry.name.split("__")[0])
    return sorted(task_ids)


def _make_provider(model: str):
    from harnessx.providers.anthropic_provider import AnthropicProvider
    from harnessx.providers.openai_provider import OpenAIProvider

    base_url   = os.environ.get("EVOL_BASE_URL")
    api_key    = os.environ.get("EVOL_API_KEY")
    provider_id = os.environ.get("EVOL_PROVIDER_ID") or None

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


def _run_eval(
    config: Path,
    jobs_dir: Path,
    tasks_json: Path,
    job_name: str,
    concurrent: int,
    k: int,
) -> None:
    """Run TB2 eval via eval_local_docker.sh.

    eval_local_docker.sh passes -o/--jobs-dir and --job-name to tb2_eval.py,
    which writes results to {jobs_dir}/{job_name}/{task}__{trial}/...

    A wall-clock timeout of k * 3600 seconds (1 hour per rollout round) is
    applied to prevent infinite blocking when a verifier hangs.  Tasks that
    completed before the timeout are unaffected; the stuck task gets reward=0.
    """
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
    # 1 hour per rollout round; generous enough for slow tasks but prevents
    # infinite blocks when a verifier (e.g. write-compressor) hangs forever.
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


def _trial_pass_rate(trials_dir: Path) -> float | None:
    """Read mean pass rate from a completed trial's result.json."""
    result_json = trials_dir / "result.json"
    if not result_json.exists():
        return None
    try:
        data = json.loads(result_json.read_text())
        for eval_data in data.get("stats", {}).get("evals", {}).values():
            metrics = eval_data.get("metrics", [])
            if metrics:
                return metrics[0].get("mean")
    except Exception:
        pass
    return None


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


# ── single round ──────────────────────────────────────────────────────────────

async def _run_one_round(
    orchestrator: TB2EvolOrchestrator,
    trials_dir: Path,
    evolve_dir: Path,
    current_config: Path | None,
    round_idx: int,
    task_ids: list[str],
) -> EvolRoundOutput:
    evolve_dir.mkdir(parents=True, exist_ok=True)
    output = await orchestrator.run_single(
        harness_config=current_config,
        trajectories_dir=trials_dir,
        task_context=TaskContext(round_idx=round_idx, task_ids=task_ids),
        output_dir=evolve_dir,
    )
    # Write machine-readable result consumed by resume logic and summary table.
    (evolve_dir / "evolve_result.json").write_text(
        json.dumps(
            {
                "accepted":        output.accepted,
                "new_config_path": str(output.new_config_path) if output.new_config_path else None,
                "pass_rate":       output.digest.pass_rate,
                "total_tasks":     output.digest.total_tasks,
                "needs_revert":    output.digest.needs_revert,
                "has_search":      output.digest.has_search_targets,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


# ── main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # ── resolve run directory ─────────────────────────────────────────────────
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (
        _REPO_ROOT / ".benchmarks" / "evolve-runs" / f"evolve-{ts}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    journal_path  = run_dir / "solvability.json"
    notebook_path = run_dir / "evolution_notebook.md"

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

    # ── orchestrator (shared across rounds; carries SolvabilityJournal state) ─
    orchestrator = TB2EvolOrchestrator(
        tb2_score_fn=tb2_score_by_session_dir,
        evolve_model=ModelConfig(main=_make_provider(args.evolve_model)),
        digest_model=ModelConfig(main=_make_provider(args.digest_model)),
        digest_skill_dirs=_TB2_SKILL_DIRS,
        evolve_skill_dirs=_TB2_SKILL_DIRS,
        journal_path=journal_path,
        notebook_path=notebook_path,
        digest_max_steps=args.digest_max_steps,
        evolve_max_steps=args.evolve_max_steps,
    )

    _update_config_link(run_dir, current_config)

    print("═" * 58)
    print("  TB2 Harness Evolution — full loop")
    print("═" * 58)
    print(f"  run-dir            : {run_dir}")
    print(f"  current-cfg        : {current_config}")
    print(f"  r0-trials          : {args.r0_trials or '<fresh eval>'}")
    print(f"  tasks              : {tasks_json}")
    print(f"  num-rounds         : {args.num_rounds}")
    print(f"  concurrent         : {args.concurrent}  k={args.k}")
    print(f"  digest-model       : {args.digest_model}")
    print(f"  evolve-model       : {args.evolve_model}")
    print(f"  digest-max-steps   : {args.digest_max_steps}")
    print(f"  evolve-max-steps   : {args.evolve_max_steps}")
    print("═" * 58)

    # ── main round loop ───────────────────────────────────────────────────────
    while current_round < args.num_rounds:
        round_dir  = run_dir / f"round_{current_round:03d}"
        evolve_dir = round_dir / "evolve"

        print(f"\n{'═' * 58}")
        print(f"  Round {current_round} / {args.num_rounds - 1}   config: {current_config.name}")
        print("═" * 58)

        # tb2_eval.py writes to {jobs_dir}/{job_name}/{task}__{trial}/...
        # We use round_dir/"trials" as jobs_dir so data lands at trials/{job_name}/
        job_name   = f"evolve-r{current_round}-{run_dir.name}"
        jobs_dir   = round_dir / "trials"
        trials_dir = jobs_dir / job_name   # actual task data lives here

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
            print(f"  [R0 warm-start] linked existing trials: {r0}")
        else:
            print("  [eval] running TB2 …")
            _run_eval(current_config, jobs_dir, tasks_json, job_name, args.concurrent, args.k)

        # ── step 2: discover task IDs from trials ─────────────────────────────
        task_ids = discover_task_ids(trials_dir)
        if not task_ids:
            print(f"  ERROR: no tasks found under {trials_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"  tasks discovered: {len(task_ids)}")

        # ── step 3: evolve ────────────────────────────────────────────────────
        result_file = evolve_dir / "evolve_result.json"
        if result_file.exists():
            print(f"  [skip evolve] evolve_result.json already exists: {evolve_dir}")
            r = json.loads(result_file.read_text())
            accepted    = r.get("accepted", False)
            new_cfg_str = r.get("new_config_path") or ""
        else:
            print("  [evolve] running EvolOrchestrator …")
            output = await _run_one_round(
                orchestrator, trials_dir, evolve_dir,
                current_config, current_round, task_ids,
            )
            accepted    = output.accepted
            new_cfg_str = str(output.new_config_path) if output.new_config_path else ""
            print(
                f"  digest: pass_rate={output.digest.pass_rate:.1%}"
                f"  revert={output.digest.needs_revert}"
                f"  search={output.digest.has_search_targets}"
            )

        # ── step 4: adopt or keep config ──────────────────────────────────────
        prev_config = current_config
        if accepted and new_cfg_str and Path(new_cfg_str).exists():
            new_config = Path(new_cfg_str)
            # Early stop: new config identical to previous (convergence)
            try:
                if new_config.read_text(encoding="utf-8") == prev_config.read_text(encoding="utf-8"):
                    print("  ✗ EARLY STOP — evolved config is identical to current config (converged)")
                    current_round += 1
                    _save_state(run_dir, current_round, current_config)
                    break
            except Exception:
                pass
            current_config = new_config
            _update_config_link(run_dir, current_config)
            print(f"  ✓ ACCEPTED — new config: {current_config}")
        else:
            # Not accepted (validation failed, skipped, or error) — keep current config and continue.
            print("  ✗ NOT ACCEPTED — keeping current config, continuing to next round")

        # ── step 5: persist state and advance round ───────────────────────────
        current_round += 1
        _save_state(run_dir, current_round, current_config)
        print()

    # ── final validation trial ─────────────────────────────────────────────────
    # Run one extra trial with the last evolved config so every config has a
    # measured pass_rate.  No digest/evolve — trials only.
    final_round    = current_round
    final_round_dir = run_dir / f"round_{final_round:03d}"
    final_job_name  = f"evolve-r{final_round}-{run_dir.name}"
    final_jobs_dir  = final_round_dir / "trials"
    final_trials_dir = final_jobs_dir / final_job_name

    print(f"\n{'═' * 58}")
    print(f"  Final validation (round {final_round}) — config: {current_config.name}")
    print("═" * 58)
    if final_trials_dir.exists() and any(final_trials_dir.iterdir()):
        print(f"  [skip final-eval] trials already exist: {final_trials_dir}")
    else:
        print("  [final-eval] running TB2 …")
        _run_eval(current_config, final_jobs_dir, tasks_json, final_job_name, args.concurrent, args.k)

    # ── final summary ─────────────────────────────────────────────────────────
    print()
    print("═" * 58)
    print(f"  Evolution complete: {args.num_rounds} rounds")
    print(f"  Final config: {current_config}")
    print(f"  Run dir: {run_dir}")
    print("═" * 58)
    print()
    print(f"  {'Round':>5} | {'Acc':^3} | {'pass_rate':>9} | new_config")
    print(f"  {'-----':>5}-+-{'---':^3}-+-{'---------':>9}-+-" + "-" * 38)
    for i in range(args.num_rounds):
        rf = run_dir / f"round_{i:03d}" / "evolve" / "evolve_result.json"
        if rf.exists():
            r   = json.loads(rf.read_text())
            acc = "✓" if r.get("accepted") else "✗"
            pr  = f"{r.get('pass_rate', 0):.1%}"
            cfg = Path(r["new_config_path"]).name if r.get("new_config_path") else "-"
            print(f"  {i:>5} | {acc:^3} | {pr:>9} | {cfg}")
        else:
            print(f"  {i:>5} | -   |         - | (no result)")
    # Final validation row (trials only, no evolve result)
    final_pr = _trial_pass_rate(final_trials_dir)
    pr_str = f"{final_pr:.1%}" if final_pr is not None else "  (pending)"
    print(f"  {final_round:>5} | val | {pr_str:>9} | (final validation)")
    print()


if __name__ == "__main__":
    # ── pre-parse --env to load .env before building defaults ─────────────────
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--env", type=str, default=None)
    _pre_args, _ = _pre.parse_known_args()
    if _pre_args.env:
        _load_env_file(_pre_args.env)

    parser = argparse.ArgumentParser(description="TB2 multi-round harness evolution")
    parser.add_argument("--env",           type=str, default=None,
                        help="Path to .env file (EVOL_* vars). Explicit CLI flags override .env.")
    parser.add_argument("--config",        type=str,
                        default=_env("EVOL_CONFIG"),
                        help="Initial harness config YAML (required unless --resume). Env: EVOL_CONFIG")
    parser.add_argument("--run-dir",       type=str,
                        default=_env("EVOL_RUN_DIR"),
                        help="Base output directory. Env: EVOL_RUN_DIR")
    parser.add_argument("--r0-trials",     type=str,
                        default=_env("EVOL_R0_TRIALS"),
                        help="Warm-start: link existing trials dir for round 0. Env: EVOL_R0_TRIALS")
    parser.add_argument("--tasks",         type=str,
                        default=_env("EVOL_TASKS"),
                        help=f"Task list JSON. Default: {_DEFAULT_TASKS.name}. Env: EVOL_TASKS")
    parser.add_argument("--num-rounds",    type=int,
                        default=int(_env("EVOL_NUM_ROUNDS", "5")),
                        help="Number of evolution rounds. Env: EVOL_NUM_ROUNDS")
    parser.add_argument("--concurrent",    type=int,
                        default=int(_env("EVOL_CONCURRENT", "16")),
                        help="TB2 eval concurrency. Env: EVOL_CONCURRENT")
    parser.add_argument("-k",              type=int,
                        default=int(_env("EVOL_K", "3")),
                        help="Rollouts per task. Env: EVOL_K")
    parser.add_argument("--digest-model",  type=str,
                        default=_env("EVOL_DIGEST_MODEL", _DEFAULT_DIGEST_MODEL),
                        help="DigestAgent model. Env: EVOL_DIGEST_MODEL")
    parser.add_argument("--evolve-model",  type=str,
                        default=_env("EVOL_EVOLVE_MODEL", _DEFAULT_EVOLVE_MODEL),
                        help="EvolveAgent model. Env: EVOL_EVOLVE_MODEL")
    parser.add_argument("--digest-max-steps", type=int,
                        default=int(_env("EVOL_DIGEST_MAX_STEPS", str(_DEFAULT_DIGEST_MAX_STEPS))),
                        help="Max steps for DigestAgent. Env: EVOL_DIGEST_MAX_STEPS")
    parser.add_argument("--evolve-max-steps", type=int,
                        default=int(_env("EVOL_EVOLVE_MAX_STEPS", str(_DEFAULT_EVOLVE_MAX_STEPS))),
                        help="Max steps for EvolveAgent. Env: EVOL_EVOLVE_MAX_STEPS")
    parser.add_argument("--resume",        action="store_true",
                        help="Continue from last completed round in --run-dir")
    args = parser.parse_args()
    asyncio.run(main(args))
