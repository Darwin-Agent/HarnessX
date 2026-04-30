# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""TB2 + meta_harness experiment runner.

Workflow:
1) Load task list from a JSON file (list of task name strings).
2) Materialize subset directory as symlinks into r0-dir trial results.
3) Use subset directory directly as trajectory root (no markdown ingestion).
4) Instantiate ``MetaAgent`` once and call ``meta_agent.evolve`` each
   round (recipe-level for-loop).

Example:
python -m recipe.tb2_evolver.run \\
  --r0-dir .benchmarks/tb2/r0-baseline \\
  --tasks recipe/tb2_evolver/tasks.json \\
  --run-tag tb2-evolver-r0-10tasks \\
  --num-rounds 1

# Resume same run-tag and continue one more round with same session_id
python -m recipe.tb2_evolver.run \\
  --run-tag tb2-evolver-r0-10tasks \\
  --resume \\
  --num-rounds 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for _line in path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# Priority: recipe-local .env first, then project-root .env as fallback.
_RECIPE_ENV = Path(__file__).resolve().parent / ".env"
_PROJECT_ENV = Path(_PROJECT_ROOT) / ".env"
_load_env_file(_RECIPE_ENV)
_load_env_file(_PROJECT_ENV)

from harnessx.core.model_config import ModelConfig
from harnessx.meta_harness import MetaAgent
from recipe.tb2_evolver.tb2_trajspec import TB2RoundAdapter


def _save_state(state_path: Path, state: dict) -> None:
    """Atomic-rename write of the evolve state JSON."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, ensure_ascii=False, indent=2)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, state_path)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
logger = logging.getLogger("tb2_metav2")

_RECIPE_DIR = Path(__file__).resolve().parent
RUNS_DIR = _RECIPE_DIR / "runs"


def _env_str(name: str, default: str) -> str:
    value = (os.environ.get(name) or "").strip()
    return value if value else default


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; fallback to default %s", name, raw, default)
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; fallback to default %s", name, raw, default)
        return int(default)


DEFAULT_META_MODEL = _env_str("META_MODEL", "anthropic/YOUR_PROVIDER/claude-sonnet-4-6")
DEFAULT_PROVIDER_ID = _env_str("PROVIDER_ID", "YOUR_PROVIDER_ID")
DEFAULT_EVOLVE_COST_CAP_USD: "float | None" = (
    None
    if (os.environ.get("EVOLVE_COST_CAP_USD") or "").strip().lower() in ("", "none", "null")
    else _env_float("EVOLVE_COST_CAP_USD", 30.0)
)
DEFAULT_EVOLVE_MAX_STEPS = _env_int("EVOLVE_MAX_STEPS", 200)
DEFAULT_EVOLVE_EARLY_REMINDER_STEP = _env_int("EVOLVE_EARLY_REMINDER_STEP", 80)
DEFAULT_EVOLVE_REMINDER_STEP = _env_int("EVOLVE_REMINDER_STEP", 120)
DEFAULT_EVOLVE_WALL_CLOCK_S = _env_int("EVOLVE_WALL_CLOCK_S", 3600)
DEFAULT_TASK_TIMEOUT = _env_int("TB2_TASK_TIMEOUT", 900)


def _make_provider(model: str, provider_id: str = None):
    from harnessx.providers.anthropic_provider import AnthropicProvider
    from harnessx.providers.litellm_provider import LiteLLMProvider

    if model.startswith("anthropic/"):
        model_name = model[len("anthropic/") :]
        return AnthropicProvider(
            model=model_name,
            base_url=os.environ.get("ANTHROPIC_API_BASE"),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
    return LiteLLMProvider(model, extra_headers={"X-Model-Provider-Id": provider_id})


def _env_is_set(name: str) -> bool:
    return bool((os.environ.get(name) or "").strip())


def _auth_preflight(model: str, provider_id: str | None) -> None:
    """Fail fast with actionable guidance when model auth is clearly missing."""
    found: list[str] = []

    def _check_any(vars_: list[str]) -> bool:
        nonlocal found
        matched = [k for k in vars_ if _env_is_set(k)]
        found.extend(matched)
        return bool(matched)

    if model.startswith("anthropic/"):
        required = ["ANTHROPIC_API_KEY"]
        if not _check_any(required):
            raise RuntimeError(
                "Anthropic provider selected but auth is missing.\n"
                f"- model: {model}\n"
                f"- checked env: {', '.join(required)}\n"
                "Set `ANTHROPIC_API_KEY` (gateway keys are fine if your base_url routes through a proxy)."
            )
    else:
        prefix = (model.split("/", 1)[0] if "/" in model else "").lower()
        prefix_candidates: dict[str, list[str]] = {
            "openai": ["OPENAI_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
            "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        }
        provider_vars: list[str] = []
        if provider_id:
            provider_vars.append(f"{provider_id.upper()}_API_KEY")
        generic = [
            "LITELLM_API_KEY",
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "TOGETHER_API_KEY",
            "AZURE_OPENAI_API_KEY",
        ]
        candidates = provider_vars + prefix_candidates.get(prefix, []) + generic
        # Deduplicate while preserving order for readable error text.
        deduped: list[str] = []
        seen: set[str] = set()
        for k in candidates:
            if k and k not in seen:
                seen.add(k)
                deduped.append(k)
        if not _check_any(deduped):
            provider_hint = f"\n- provider-id hint: set `{provider_id.upper()}_API_KEY`" if provider_id else ""
            raise RuntimeError(
                "LiteLLM provider selected but no obvious auth env var is set.\n"
                f"- model: {model}\n"
                f"- checked env: {', '.join(deduped)}"
                f"{provider_hint}\n"
                "Set one credential env var before running."
            )

    if found:
        logger.info(
            "Auth preflight passed for model=%s (detected: %s)",
            model,
            ", ".join(dict.fromkeys(found)),
        )


def _dump_baseline_config(round_dir: Path, timeout_seconds: int = 900) -> Path:
    from benchmarks.terminal_bench_2.harness import make_tb2_harness_config

    cfg = make_tb2_harness_config(timeout_seconds=timeout_seconds)
    config_path = round_dir / "config.yaml"
    cfg.to_yaml_file(config_path)
    logger.info("Dumped baseline config → %s", config_path)
    return config_path


def _reward_of_trial_dir(trial_dir: Path) -> float | None:
    rp = trial_dir / "result.json"
    if not rp.is_file():
        return None
    try:
        result = json.loads(rp.read_text(encoding="utf-8"))
    except Exception:
        return None
    vr = result.get("verifier_result") or {}
    rw = (vr.get("rewards") or {}).get("reward")
    if isinstance(rw, (int, float)):
        return float(rw)
    return None


def _load_trials_from_tasks_json(r0_dir: Path, tasks_json: Path) -> list[Path]:
    """Return sorted trial dirs from r0_dir matching the task names in tasks_json."""
    task_names: list[str] = json.loads(tasks_json.read_text(encoding="utf-8"))
    if not isinstance(task_names, list) or not task_names:
        raise ValueError(f"--tasks file must be a non-empty JSON list: {tasks_json}")

    # Build a map from task_name -> trial dir (prefer result.json task_name field,
    # fall back to directory name prefix before the first '__').
    name_to_dir: dict[str, Path] = {}
    for p in r0_dir.iterdir():
        if not p.is_dir():
            continue
        rp = p / "result.json"
        if rp.is_file():
            try:
                task_name = json.loads(rp.read_text(encoding="utf-8")).get("task_name") or ""
            except Exception:
                task_name = ""
        else:
            task_name = p.name.split("__")[0]
        if task_name:
            name_to_dir[task_name] = p

    picked: list[Path] = []
    missing: list[str] = []
    for name in task_names:
        if name in name_to_dir:
            picked.append(name_to_dir[name])
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Tasks not found in {r0_dir}: {missing}\nAvailable: {sorted(name_to_dir)}")
    return sorted(picked, key=lambda p: p.name)


def _print_eval_summary(eval_records: list[dict], title: str) -> None:
    passed = sum(1 for r in eval_records if r.get("reward") and r["reward"] > 0)
    total = len(eval_records)
    print(f"\n{title}: {passed}/{total} passed ({100 * passed / total:.1f}%)")
    print()
    print(f"  {'task_name':<45} {'reward':>6}  {'elapsed':>8}  {'tokens':>8}")
    print("  " + "-" * 72)
    for r in eval_records:
        status = "PASS" if r.get("reward") and r["reward"] > 0 else "FAIL"
        rew = f"{r.get('reward', '?')}"
        elapsed = f"{r.get('elapsed_s', '?')}s"
        tokens = f"{r.get('total_tokens', '?'):,}" if r.get("total_tokens") else "?"
        print(f"  [{status}] {r['task_name']:<41} {rew:>6}  {elapsed:>8}  {tokens:>8}")


def _parse_iso_utc(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _collect_eval_records_from_trials(trials: list[Path]) -> list[dict]:
    records: list[dict] = []
    for t in trials:
        rp = t / "result.json"
        reward = 0.0
        elapsed_s: float | None = None
        total_tokens: int | None = None
        task_name = t.name
        if rp.is_file():
            try:
                obj = json.loads(rp.read_text(encoding="utf-8"))
                task_name = str(obj.get("task_name") or t.name)
                vr = obj.get("verifier_result") or {}
                reward_raw = (vr.get("rewards") or {}).get("reward")
                if isinstance(reward_raw, (int, float)):
                    reward = float(reward_raw)
                ts0 = _parse_iso_utc(obj.get("started_at"))
                ts1 = _parse_iso_utc(obj.get("finished_at"))
                if ts0 is not None and ts1 is not None and ts1 >= ts0:
                    elapsed_s = round(ts1 - ts0, 1)
                ar = obj.get("agent_result") or {}
                ni = ar.get("n_input_tokens")
                no = ar.get("n_output_tokens")
                if isinstance(ni, int) and isinstance(no, int):
                    total_tokens = ni + no
            except Exception:
                pass
        records.append(
            {
                "task_name": task_name,
                "reward": reward,
                "elapsed_s": elapsed_s,
                "total_tokens": total_tokens,
            }
        )
    return records


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="TB2 subset evolve with meta_harness (v1)",
    )
    parser.add_argument(
        "--r0-dir",
        default=None,
        help=(
            "Source TB2 run directory with prior trajectories. "
            "Required for --trajectory-mode reuse. "
            "Optional for rerun: if provided, warm-starts the R0 manifest with pass/fail info; "
            "if omitted, R0 runs fresh with the baseline config."
        ),
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Output tag under recipe/terminal_bench_2_with_metav2/runs/",
    )
    parser.add_argument(
        "--tasks",
        default=str(_RECIPE_DIR / "tasks.json"),
        help="JSON file containing a list of task name strings (default: recipe dir tasks.json)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_META_MODEL,
        help=f"Meta-agent model (default: {DEFAULT_META_MODEL})",
    )
    parser.add_argument(
        "--provider-id",
        default=DEFAULT_PROVIDER_ID,
        help=f"Provider ID for non-Anthropic models (default: {DEFAULT_PROVIDER_ID})",
    )
    parser.add_argument(
        "--evolve-cost",
        type=lambda x: None if x.lower() in ("none", "null", "") else float(x),
        default=DEFAULT_EVOLVE_COST_CAP_USD,
        help=f"Meta-agent cost cap USD, or 'none' to disable (default: {DEFAULT_EVOLVE_COST_CAP_USD})",
    )
    parser.add_argument(
        "--evolve-steps",
        type=int,
        default=DEFAULT_EVOLVE_MAX_STEPS,
        help=f"Meta-agent step cap (default: {DEFAULT_EVOLVE_MAX_STEPS})",
    )
    parser.add_argument(
        "--evolve-early-reminder",
        type=int,
        default=DEFAULT_EVOLVE_EARLY_REMINDER_STEP,
        help=f"Step at which to inject soft convergence warning (default: {DEFAULT_EVOLVE_EARLY_REMINDER_STEP})",
    )
    parser.add_argument(
        "--evolve-reminder",
        type=int,
        default=DEFAULT_EVOLVE_REMINDER_STEP,
        help=f"Step at which to inject hard deadline reminder (default: {DEFAULT_EVOLVE_REMINDER_STEP})",
    )
    parser.add_argument(
        "--evolve-wall-clock",
        type=int,
        default=DEFAULT_EVOLVE_WALL_CLOCK_S,
        help=f"Meta-agent wall-clock cap in seconds (default: {DEFAULT_EVOLVE_WALL_CLOCK_S})",
    )
    parser.add_argument(
        "--no-require-evidence",
        action="store_true",
        default=False,
        help="Skip the evidence gate (candidates.md not required even when config changes)",
    )
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=DEFAULT_TASK_TIMEOUT,
        help=f"TB2 task timeout used by dumped baseline config (default: {DEFAULT_TASK_TIMEOUT})",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="How many evolve rounds to execute in this invocation (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run-tag evolve state, keeping the same session continuity.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional explicit meta-agent session_id. Default: tb2-metav2:{run_tag}",
    )
    parser.add_argument(
        "--start-round",
        type=int,
        default=None,
        help="Optional start input round index when not resuming (default: 0).",
    )
    parser.add_argument(
        "--allow-round-fallback",
        action="store_true",
        help=(
            "Allow fallback to older trajectories when R{n}/trajectories is missing. "
            "Default is strict (missing round trajectories raises an error)."
        ),
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=("rerun", "reuse"),
        default="rerun",
        help=(
            "How to provide trajectories for R1+ rounds. "
            "`rerun` = run TB2 with each new config to generate fresh trajectories "
            "(true evolution). `reuse` = keep using existing provided trajectories."
        ),
    )
    parser.add_argument(
        "--tb2-eval-script",
        default=str((Path(_PROJECT_ROOT) / "benchmarks/terminal_bench_2/scripts/eval_opensandbox.sh").resolve()),
        help="Path to TB2 eval script used when --trajectory-mode=rerun.",
    )
    parser.add_argument(
        "--tb2-eval-concurrent",
        type=int,
        default=2,
        help="TB2 eval concurrency used when --trajectory-mode=rerun (default: 2).",
    )
    parser.add_argument(
        "--tb2-eval-resume",
        action="store_true",
        help="Pass --resume to TB2 eval script when --trajectory-mode=rerun.",
    )
    args = parser.parse_args()

    if args.resume and not args.run_tag:
        raise ValueError("--resume requires --run-tag so it can locate previous state.")

    r0_dir = Path(args.r0_dir).resolve() if args.r0_dir else None
    tasks_json = Path(args.tasks).resolve()
    if not args.resume and args.trajectory_mode == "reuse":
        if r0_dir is None or not r0_dir.is_dir():
            raise FileNotFoundError(f"--trajectory-mode reuse requires an existing --r0-dir: {r0_dir or '(not set)'}")
    if not args.resume and r0_dir is not None and not r0_dir.is_dir():
        raise FileNotFoundError(f"--r0-dir not found: {r0_dir}")
    if not args.resume and not tasks_json.is_file():
        raise FileNotFoundError(f"--tasks file not found: {tasks_json}")
    try:
        _auth_preflight(args.model, args.provider_id)
    except RuntimeError as exc:
        raise SystemExit(f"\n[auth-preflight] {exc}\n") from exc

    run_tag = args.run_tag or time.strftime("run_%Y%m%d-%H%M%S")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_root = RUNS_DIR / run_tag
    run_root.mkdir(parents=True, exist_ok=True)

    subset_dir = run_root / "subset_r0_failed"  # only materialized for reuse mode
    r0_round_dir = run_root / "R0"
    baseline_config = r0_round_dir / "config.yaml"
    legacy_traj_dir = r0_round_dir / "trajectories"

    if legacy_traj_dir.exists():
        shutil.rmtree(legacy_traj_dir)
        logger.info("Removed legacy ingested markdown directory: %s", legacy_traj_dir)

    task_names: list[str] | None = None
    trajectories_root: Path | None = None  # only needed for reuse mode

    if args.resume:
        if args.trajectory_mode == "reuse":
            if not subset_dir.is_dir():
                raise FileNotFoundError(f"Resume requested but subset trajectory dir not found: {subset_dir}")
            trajectories_root = subset_dir
        if not baseline_config.is_file():
            raise FileNotFoundError(f"Resume requested but R0 baseline config not found: {baseline_config}")
        if tasks_json.is_file():
            task_names = json.loads(tasks_json.read_text(encoding="utf-8"))
        if args.trajectory_mode == "reuse" and (subset_dir / "_manifest.json").is_file():
            manifest = json.loads((subset_dir / "_manifest.json").read_text(encoding="utf-8"))
            trial_count = manifest.get("num_trials")
            n_failed = manifest.get("num_failed")
            n_passed_m = manifest.get("num_passed")
            if isinstance(n_failed, int) and isinstance(n_passed_m, int):
                print(
                    f"\nResuming run `{run_tag}` with subset ({trial_count} trials: "
                    f"{n_failed} failed + {n_passed_m} passed)."
                )
            else:
                print(f"\nResuming run `{run_tag}` with subset ({trial_count} trials).")
        elif task_names:
            print(f"\nResuming run `{run_tag}` with {len(task_names)} tasks ({args.trajectory_mode} mode).")
        else:
            print(f"\nResuming run `{run_tag}`.")
    else:
        r0_round_dir.mkdir(parents=True, exist_ok=True)
        task_names = json.loads(tasks_json.read_text(encoding="utf-8"))

        picked: list[Path] = []
        if r0_dir is not None and r0_dir.is_dir():
            picked = _load_trials_from_tasks_json(r0_dir, tasks_json)

        if args.trajectory_mode == "reuse":
            if not picked:
                raise RuntimeError("--trajectory-mode reuse requires --r0-dir with prior trial directories.")
            if subset_dir.exists():
                shutil.rmtree(subset_dir)
            subset_dir.mkdir(parents=True, exist_ok=True)
            for src in picked:
                (subset_dir / src.name).symlink_to(src.resolve(), target_is_directory=True)
            trajectories_root = subset_dir
            logger.info("Materialized subset dir with %d trial symlinks → %s", len(picked), subset_dir)

        if picked:
            eval_records = _collect_eval_records_from_trials(picked)
            n_passed = sum(1 for r in eval_records if r.get("reward") and r["reward"] > 0)
            print(
                f"\nLoaded {len(picked)} tasks from {tasks_json.name} "
                f"({n_passed} passed, {len(picked) - n_passed} failed) — "
                f"using prior trajectories as R0 evidence:"
            )
            for p in picked:
                rw = _reward_of_trial_dir(p)
                status = "PASSED" if rw and rw > 0 else "FAILED"
                print(f"  - [{status}] {p.name.split('__')[0]}")
            _print_eval_summary(eval_records, "Subset R0 (prior)")
            (r0_round_dir / "eval_summary.json").write_text(
                json.dumps(
                    {
                        "round": 0,
                        "r0_eval_mode": "reuse_existing_results_only"
                        if args.trajectory_mode == "reuse"
                        else "warm_start_provenance",
                        "source_run_dir": str(r0_dir),
                        "tasks_json": str(tasks_json),
                        "records": eval_records,
                        "passed": n_passed,
                        "total": len(eval_records),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (r0_round_dir / "provenance.json").write_text(
                json.dumps(
                    {
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "policy": "warm_start_r0",
                        "detail": "R0 evidence loaded from prior run for provenance display.",
                        "source_run_dir": str(r0_dir),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            print(f"\nFresh start: {len(task_names)} tasks from {tasks_json.name} (R0 will run with baseline config):")
            for name in task_names:
                print(f"  - {name}")
            (r0_round_dir / "eval_summary.json").write_text(
                json.dumps(
                    {
                        "round": 0,
                        "r0_eval_mode": "fresh_rerun",
                        "source_run_dir": None,
                        "tasks_json": str(tasks_json),
                        "records": [],
                        "passed": 0,
                        "total": 0,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (r0_round_dir / "provenance.json").write_text(
                json.dumps(
                    {
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "policy": "fresh_r0_rerun",
                        "detail": "No prior trajectories; R0 will be generated by running TB2 eval with baseline config.",
                        "tasks": task_names,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        # baseline config
        baseline_config = _dump_baseline_config(
            r0_round_dir,
            timeout_seconds=args.task_timeout,
        )

    # 4) run meta_harness evolve loop (recipe-level iteration)
    provider = _make_provider(args.model, args.provider_id)
    meta_model = ModelConfig(main=provider)
    evolve_dir = run_root / "_meta_v2"
    evolve_dir.mkdir(parents=True, exist_ok=True)
    memo_path = run_root / "learnings.md"
    session_id = args.session_id or f"tb2-metav2:{run_tag}"
    state_path = evolve_dir / "_meta_scratch" / "harness_evolve_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    _TB2_SKILLS_DIR = _RECIPE_DIR / "skills"
    meta_agent = MetaAgent(
        inner_model=meta_model,
        memo_path=memo_path,
        extra_skills_dirs=([_TB2_SKILLS_DIR] if _TB2_SKILLS_DIR.is_dir() else None),
        max_cost_usd=args.evolve_cost,
        wall_clock_s=float(args.evolve_wall_clock),
        max_steps=args.evolve_steps,
        step_deadline_early_reminder_step=args.evolve_early_reminder,
        step_deadline_reminder_step=args.evolve_reminder,
        extra_harness_kws={"loop_detection": False},
        require_evidence=not args.no_require_evidence,
    )

    adapter = TB2RoundAdapter(
        baseline_config=baseline_config,
        trajectories_root=trajectories_root,
        task_names=task_names,
        r0_trajectories=r0_dir if (r0_dir is not None and args.trajectory_mode == "rerun") else None,
        run_mode=args.trajectory_mode,
        strict_round_trajectories=not args.allow_round_fallback,
        repo_root=Path(_PROJECT_ROOT),
        eval_script=Path(args.tb2_eval_script),
        eval_concurrent=args.tb2_eval_concurrent,
        eval_resume=args.tb2_eval_resume,
    )

    # Load or initialise evolve state.
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"Resume requested but state not found: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("session_id") != session_id:
            raise ValueError(
                "resume requested but session_id mismatched with previous state: "
                f"state={state.get('session_id')!r}, requested={session_id!r}"
            )
        next_input_round = int(state.get("next_input_round", 0))
        current_config = Path(str(state.get("last_output_config", ""))).resolve()
        if not current_config.is_file():
            raise FileNotFoundError(f"resume state points to missing last_output_config: {current_config}")
    else:
        next_input_round = int(args.start_round) if args.start_round is not None else 0
        current_config = adapter.initial_config()
        now = int(time.time())
        state = {
            "version": 1,
            "session_id": session_id,
            "status": "running",
            "created_at_epoch_s": now,
            "updated_at_epoch_s": now,
            "next_input_round": next_input_round,
            "last_output_round": next_input_round - 1,
            "last_output_config": str(current_config),
            "history": [],
        }
        _save_state(state_path, state)

    # Per-round evolve loop.
    last_rx_output_dir: Path | None = None
    last_promoted: Path = current_config
    for i in range(args.num_rounds):
        input_round = next_input_round + i
        output_round = input_round + 1
        try:
            trajectories_dir = await adapter.resolve_trajectories_for_round(
                run_root=run_root,
                input_round=input_round,
                current_config=current_config,
            )
            if not trajectories_dir.is_dir():
                raise FileNotFoundError(f"resolve_trajectories_for_round returned non-directory: {trajectories_dir}")

            started_at = time.time()
            round_evolve_dir = evolve_dir / f"R{output_round}"
            round_evolve_dir.mkdir(parents=True, exist_ok=True)
            output_config_path = await meta_agent.evolve(
                current_config=current_config,
                trajectories_dir=trajectories_dir,
                output_dir=round_evolve_dir,
            )
            elapsed = time.time() - started_at
            output_config_path = Path(output_config_path).resolve()

            promoted_config = adapter.promote_round_output(
                run_root=run_root,
                output_round=output_round,
                rx_output_dir=round_evolve_dir,
                output_config=output_config_path,
            ).resolve()

            changed = current_config.read_bytes() != output_config_path.read_bytes()
            score = adapter.read_round_score(trajectories_dir)

            rr = {
                "input_round": input_round,
                "output_round": output_round,
                "input_config": str(current_config),
                "trajectories_dir": str(trajectories_dir),
                "rx_output_dir": str(round_evolve_dir),
                "output_config": str(output_config_path),
                "promoted_config": str(promoted_config),
                "changed": bool(changed),
                "elapsed_s": round(float(elapsed), 3),
                "score": score,
            }
            history = state.setdefault("history", [])
            history.append(rr)
            finished_at = int(time.time())
            state["status"] = "running"
            state["next_input_round"] = output_round
            state["last_output_round"] = output_round
            state["last_output_config"] = str(promoted_config)
            state["updated_at_epoch_s"] = finished_at
            state.pop("last_error", None)
            _save_state(state_path, state)

            current_config = promoted_config
            last_rx_output_dir = round_evolve_dir
            last_promoted = promoted_config
        except Exception as exc:
            failed_at = int(time.time())
            state["status"] = "failed"
            state["last_error"] = {
                "input_round": input_round,
                "output_round": output_round,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "failed_at_epoch_s": failed_at,
            }
            state["updated_at_epoch_s"] = failed_at
            _save_state(state_path, state)
            raise

    state["status"] = "completed"
    state["updated_at_epoch_s"] = int(time.time())
    _save_state(state_path, state)

    if last_rx_output_dir is None:
        raise RuntimeError("No evolve rounds executed.")
    new_yaml = last_rx_output_dir / "config.yaml"

    print("\n" + "=" * 72)
    print("meta_harness evolve done")
    print(f"  run_tag:   {run_tag}")
    print(f"  session:   {session_id}")
    print(f"  rounds:    +{args.num_rounds} ({'resume' if args.resume else 'fresh'})")
    print(f"  trajectory_mode: {args.trajectory_mode}")
    print(f"  trajectories_root: {trajectories_root or '(generated per-round)'}")
    print(f"  baseline: {baseline_config}")
    print(f"  evolved_latest: {new_yaml} (bundle: {last_rx_output_dir})")
    print(f"  promoted_latest: {last_promoted}")
    print(f"  state:    {state_path}")
    print(f"  memo:     {memo_path}")
    print(
        "  verify:   "
        f"TB2_HARNESS_CONFIG={last_promoted} "
        "bash benchmarks/terminal_bench_2/scripts/eval_opensandbox.sh "
        "--job-name <verify-job> -n 2 "
        f"--tasks <tasks.json>"
    )
    print("=" * 72 + "\n")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()
