# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# ── project root on sys.path ────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── load .env ────────────────────────────────────────────────────────────────
_env_path = Path(_PROJECT_ROOT) / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from typing import Any

from harnessx.core.harness import HarnessResult
from harnessx.core.model_config import ModelConfig
from harnessx.meta_harness import MetaAgent

from benchmarks.gaia.evaluator import GAIAPipelineEvaluator
from benchmarks.gaia.harness import make_gaia_builder_gpt5
from benchmarks.gaia.task import GAIATask, load_gaia_tasks, load_gaia_tasks_from_json

from .defaults import (
    DEFAULT_CONCURRENCY,
    DEFAULT_META_MODEL,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER_ID,
    EVOLVE_COST_CAP_USD,
    EVOLVE_MAX_STEPS,
    EVOLVE_WALL_CLOCK_S,
    MAX_COST_USD,
    MAX_STEPS,
    MAX_TASKS,
    NUM_ROUNDS,
    PASS_COUNT_NOISE_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
logger = logging.getLogger("gaia_evolver")

# Suppress LiteLLM's own debug/verbose output — it defaults to noisy
# debug logging that drowns out the evolver's progress lines.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
try:
    import litellm as _litellm

    _litellm.suppress_debug_info = True
    _litellm.set_verbose = False
except ImportError:
    pass

# All recipe outputs live under runs/ — one subdir per --run-tag.
# Each run_tag dir is self-contained: configs, trajectories, sessions, evolve
# artifacts all nest under R{N}/ subfolders for easy inspection/cleanup.
_RECIPE_DIR = Path(__file__).resolve().parent
RUNS_DIR = _RECIPE_DIR / "runs"
# Benchmark-specific meta-agent skills. Mounted into the meta-agent's system
# prompt via ``extra_skills_dirs`` so the generic persona (under
# ``harnessx/meta_harness/workspace/``) stays benchmark-agnostic. If this
# directory does not exist the meta-agent simply sees no GAIA playbook,
# which is the desired fallback.
_GAIA_SKILLS_DIR = _RECIPE_DIR / "skills"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(
    model: str,
    provider_id: str,
    *,
    extended_thinking: bool = False,
    thinking_budget_tokens: int = 10_000,
    max_tokens: int = 8192,
    api_base: str | None = None,
    api_key: str | None = None,
):
    """Create a model provider from CLI args.

    ``extended_thinking`` / ``thinking_budget_tokens`` / ``max_tokens`` only
    take effect when the ``AnthropicProvider`` branch is selected — LiteLLM
    routes do not expose Anthropic's thinking API directly. Callers that
    need thinking on a non-Anthropic deployment should instead set the
    equivalent kwarg through the LiteLLM path (no-op here).

    ``api_base`` routes the LiteLLM branch to a custom OpenAI-compatible
    endpoint (local vLLM / SGLang / etc.). The ``X-Model-Provider-Id`` header is dropped in that case — it's
    vendor-specific and will be rejected or ignored by other backends.
    """
    from harnessx.providers.anthropic_provider import AnthropicProvider
    from harnessx.providers.litellm_provider import LiteLLMProvider

    if model.startswith("anthropic/"):
        model_name = model[len("anthropic/") :]
        return AnthropicProvider(
            model=model_name,
            base_url=os.environ.get("ANTHROPIC_API_BASE"),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            extended_thinking=extended_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            max_tokens=max_tokens,
        )
    if api_base:
        return LiteLLMProvider(
            model,
            api_base=api_base,
            api_key=api_key or "EMPTY",
        )
    extra_headers = {"X-Model-Provider-Id": provider_id}
    return LiteLLMProvider(model, extra_headers=extra_headers)


async def _run_task(
    harness: Any,
    task: GAIATask,
    label: str,
    *,
    pipeline_eval: "GAIAPipelineEvaluator",
    harness_config: Any | None = None,
) -> dict:
    """Run a single task + externally evaluate the answer.

    The harness no longer contains an EvaluationProcessor. We call
    ``pipeline_eval.evaluate_answer`` after the run completes. The
    result flows into two places:

    * ``comparison.json`` — the external run-level report (includes
      the dataset's expected answer for offline analysis; not read
      by the meta-agent).
    * The per-task trajectory ``.md`` **frontmatter** — as
      ``eval_passed`` / ``eval_score`` only, so the meta-agent can
      see correctness outcomes (pass/fail) without seeing the
      expected answer text or any reason string that could
      smuggle it.

    The trajectory **body** (conversation transcript) and its
    **frontmatter** are both ground-truth-free: neither the task-agent
    nor the meta-agent sees the expected answer. The meta-agent only
    sees pass/fail signals, so it evolves on correctness, not on a
    known target string.

    ``harness_config`` is the HarnessConfig (typed Any to avoid a core
    import) threaded through so the caller can dump processor / tool
    info into the trajectory markdown and read `tool_registry` for the
    `unused_tools` signal.
    """
    t0 = time.time()
    task_id = task.task_id or "?"
    logger.info("[%s] Running %s (Level %d)...", label, task_id, task.level)

    try:
        result = await harness.run(task, session_id=f"{label}-{task_id}")
        elapsed = time.time() - t0

        # External evaluation — does NOT feed back into trajectory/stats.
        eval_result = await pipeline_eval.evaluate_answer(
            result.final_output or "",
            task.final_answer or "",
        )
        passed = bool(eval_result.passed)
        score = float(eval_result.score)
        reason = (eval_result.reason or "")[:200]

        # Behavioural signals + external eval outcome. Eval fields
        # (passed/score/reason/expected) feed both comparison.json and
        # the trajectory frontmatter's eval_* block; the trajectory
        # body remains ground-truth-free.
        output = result.final_output or ""
        state_snapshot = getattr(getattr(result, "task_end", None), "state_snapshot", None) or {}
        slots = state_snapshot.get("slots") if isinstance(state_snapshot, dict) else {}
        if not isinstance(slots, dict):
            slots = {}
        model_empty_end_turn = bool((slots.get("__model_empty_end_turn_seen") or {}).get("content"))
        empty_end_turn_recovered = bool((slots.get("__empty_end_turn_recovered") or {}).get("content"))
        record = {
            "task_id": task_id,
            "level": task.level,
            "question": task.question[:150],
            "expected": task.final_answer,
            "output": output[:300],
            "final_output": output,
            "passed": passed,
            "score": score,
            "reason": reason,
            "steps": result.total_steps,
            "total_tokens": result.total_tokens,
            "cost_usd": result.total_cost_usd,
            "elapsed_s": round(elapsed, 1),
            "exit_reason": getattr(result, "exit_reason", "?"),
            "model_empty_end_turn": model_empty_end_turn,
            "empty_end_turn_recovered": empty_end_turn_recovered,
        }

        status = "PASS" if passed else "FAIL"
        logger.info(
            "[%s] %s %s — steps=%d cost=$%.3f time=%.1fs",
            label,
            task_id,
            status,
            result.total_steps,
            result.total_cost_usd,
            elapsed,
        )
        if model_empty_end_turn:
            logger.warning(
                "[%s] %s model_empty_end_turn=true recovered=%s",
                label,
                task_id,
                empty_end_turn_recovered,
            )
        return {**record, "_result": result}

    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("[%s] %s ERROR: %s (%.1fs)", label, task_id, exc, elapsed)
        return {
            "task_id": task_id,
            "level": task.level,
            "question": task.question[:150],
            "expected": task.final_answer,
            "passed": False,
            "score": 0.0,
            "reason": f"error: {exc}",
            "elapsed_s": round(elapsed, 1),
            "exit_reason": "error",
            "tool_call_counts": {},
            "tool_error_counts": {},
            "_result": None,
        }


def print_multiround_comparison(rounds: list[list[dict]]) -> None:
    """Print multi-round comparison as two aligned tables + headline.

    Layout:
      * Per-task pass/fail history — PASS/FAIL per round + best-vs-R0 pp delta.
      * Round totals — pass_rate, cost_usd, tokens, steps per round with a
        dedicated Δ column between consecutive rounds.
      * Headline — one-line callout of the total pass_rate swing.
    """
    if not rounds:
        return
    n_rounds = len(rounds)
    task_ids = [r["task_id"] for r in rounds[0]]
    n_tasks = len(task_ids)

    lines: list[str] = []

    r_word = "round" if n_rounds == 1 else "rounds"
    t_word = "task" if n_tasks == 1 else "tasks"
    lines.append(f"GAIA Evolver — Multi-Round Comparison  ({n_rounds} {r_word} × {n_tasks} {t_word})")
    lines.append("")

    # Compute historical-best round index for "vs-best" deltas.
    totals = [
        {
            "passed": sum(1 for r in rd if r.get("passed")),
            "cost": sum(r.get("cost_usd", 0) or 0 for r in rd),
            "tokens": sum(r.get("total_tokens", 0) or 0 for r in rd),
            "steps": sum(r.get("steps", 0) or 0 for r in rd),
        }
        for rd in rounds
    ]
    best_idx = min(
        range(n_rounds),
        key=lambda i: (-totals[i]["passed"], totals[i]["cost"], i),
    )

    # ── Per-task pass/fail history ──────────────────────────────────────
    TID_W, STAT_W, PP_W, DELTA_W = 20, 11, 12, 14
    lines.append("Per-task pass/fail history")
    hdr = f"  {'task_id':<{TID_W}}"
    for i in range(n_rounds):
        hdr += f" | {f'R{i} result':^{STAT_W}}"
    if n_rounds > 1:
        hdr += f" | {f'R{best_idx}-vs-R0 pass':^{PP_W}}"
        hdr += f" | {f'R{best_idx}-vs-R0 tokens':^{DELTA_W}}"
        hdr += f" | {f'R{best_idx}-vs-R0 steps':^{DELTA_W}}"
    lines.append(hdr)
    sep = f"  {'-' * TID_W}"
    for _ in range(n_rounds):
        sep += f"-+-{'-' * STAT_W}"
    if n_rounds > 1:
        sep += f"-+-{'-' * PP_W}"
        sep += f"-+-{'-' * DELTA_W}"
        sep += f"-+-{'-' * DELTA_W}"
    lines.append(sep)

    for tid in task_ids:
        rec0 = next((r for r in rounds[0] if r["task_id"] == tid), {})
        rec_best = next((r for r in rounds[best_idx] if r["task_id"] == tid), {})
        row = f"  {tid[:TID_W]:<{TID_W}}"
        for i in range(n_rounds):
            rec = next((r for r in rounds[i] if r["task_id"] == tid), {})
            status = "PASS" if rec.get("passed") else "FAIL"
            row += f" | {status:^{STAT_W}}"
        if n_rounds > 1:
            pp = (100 if rec_best.get("passed") else 0) - (100 if rec0.get("passed") else 0)
            row += f" | {f'{pp:+d}pp':^{PP_W}}"
            tok0 = int(rec0.get("total_tokens") or 0)
            tok_best = int(rec_best.get("total_tokens") or 0)
            stp0 = int(rec0.get("steps") or 0)
            stp_best = int(rec_best.get("steps") or 0)
            row += f" | {_pct_delta(tok_best, tok0):^{DELTA_W}}"
            row += f" | {_pct_delta(stp_best, stp0):^{DELTA_W}}"
        lines.append(row)
    lines.append("")

    # ── Round totals ────────────────────────────────────────────────────
    LBL_W, VAL_W, D_W = 12, 15, 10
    lines.append("Round totals")
    hdr = f"  {'metric':<{LBL_W}}"
    for i in range(n_rounds):
        hdr += f" | {f'R{i}':^{VAL_W}}"
        if i > 0:
            hdr += f"  {'Δ':^{D_W}}"
    lines.append(hdr)
    sep = f"  {'-' * LBL_W}"
    for i in range(n_rounds):
        sep += f"-+-{'-' * VAL_W}"
        if i > 0:
            sep += f"--{'-' * D_W}"
    lines.append(sep)

    n = n_tasks

    def _row(label: str, v_fn, d_fn) -> str:
        line = f"  {label:<{LBL_W}}"
        for i, t in enumerate(totals):
            line += f" | {v_fn(t):^{VAL_W}}"
            if i > 0:
                line += f"  {d_fn(t, totals[i - 1]):^{D_W}}"
        return line

    lines.append(
        _row(
            "pass_rate",
            lambda t: f"{100 * t['passed'] / n:.1f}% ({t['passed']}/{n})" if n else "-",
            lambda cur, prev: f"{100 * (cur['passed'] - prev['passed']) / n:+.1f}pp" if n else "-",
        )
    )
    lines.append(
        _row(
            "cost_usd",
            lambda t: f"${t['cost']:.2f}",
            lambda cur, prev: _pct_delta(cur["cost"], prev["cost"]),
        )
    )
    lines.append(
        _row(
            "tokens",
            lambda t: f"{t['tokens']:,}",
            lambda cur, prev: _pct_delta(cur["tokens"], prev["tokens"]),
        )
    )
    lines.append(
        _row(
            "steps",
            lambda t: f"{t['steps']}",
            lambda cur, prev: _pct_delta(cur["steps"], prev["steps"]),
        )
    )

    # ── Headline ────────────────────────────────────────────────────────
    if n_rounds > 1 and n_tasks > 0:
        p0 = 100 * totals[0]["passed"] / n_tasks
        p_best = 100 * totals[best_idx]["passed"] / n_tasks
        pp = p_best - p0
        tok0 = totals[0]["tokens"]
        tok_best = totals[best_idx]["tokens"]
        lines.append("")
        lines.append(
            f"  >>> best-vs-R0 pass_rate: R0 {p0:.1f}% -> R{best_idx} {p_best:.1f}% over {n_rounds} rounds  ({pp:+.1f}pp)"
        )
        lines.append(
            f"  >>> best-vs-R0 tokens:    R0 {tok0:,} -> R{best_idx} {tok_best:,} over {n_rounds} rounds  ({_pct_delta(tok_best, tok0)})"
        )

    width = max(80, max(len(line) for line in lines if line))
    print("\n" + "=" * width)
    for line in lines:
        print(line)
    print("=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="GAIA Evolver: multi-round meta-harness")
    parser.add_argument("--max-tasks", type=int, default=MAX_TASKS)
    parser.add_argument("--max-cost", type=float, default=MAX_COST_USD)
    parser.add_argument("--num-rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(f"Model used by the task-doing (inner) agent on each GAIA task. Default: {DEFAULT_MODEL}."),
    )
    parser.add_argument(
        "--meta-model",
        default=DEFAULT_META_MODEL,
        help=(
            "Model used by the meta-agent during evolve/reflect. Kept "
            "separate from --model so the outer loop can run on a stronger "
            f"tier than the inner loop. Default: {DEFAULT_META_MODEL}."
        ),
    )
    parser.add_argument("--provider-id", default=DEFAULT_PROVIDER_ID)
    parser.add_argument(
        "--api-base",
        default=None,
        help=(
            "Optional OpenAI-compatible endpoint for the inner task agent "
            "(e.g. http://host:port/v1). When set, the --model request is "
            "routed to this URL through LiteLLM "
            "header is dropped. Only affects --model; --meta-model and the "
            "judge still use the default provider. Default: unset."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key paired with --api-base. Ignored when --api-base is "
            "unset. Defaults to 'EMPTY' (works for most open local endpoints)."
        ),
    )
    parser.add_argument("--clean", action="store_true", help="Wipe runs/<tag>/ before starting")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLMJudgeProcessor (verdict fields omitted from frontmatter). Default: judge enabled.",
    )
    parser.add_argument("--evolve-cost", type=float, default=EVOLVE_COST_CAP_USD)
    parser.add_argument("--evolve-steps", type=int, default=EVOLVE_MAX_STEPS)
    parser.add_argument("--evolve-wall-clock", type=int, default=EVOLVE_WALL_CLOCK_S)
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=0.03,
        help=(
            "Score drop allowed before reverting to previous config. "
            "0.0 = any regression reverts. 0.05 = up to 5 pp drop tolerated. "
            "Default 0.03 absorbs typical eval stochastic noise (~2-3 pp) so "
            "a genuinely-useful code fix isn't discarded because an unrelated "
            "task flipped. The best_score baseline still only advances on "
            "strict improvements, so tolerance cannot drift it downward. "
            "Score = pass_rate - cost_weight * max(relative_cost_delta, 0)."
        ),
    )
    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.0,
        help=(
            "How strongly per-round cost increases penalize the gating score. "
            "0.0 = pass_rate only (legacy). 0.1 = a 20%% cost increase is "
            "treated as a 2 pp pass_rate regression."
        ),
    )
    parser.add_argument(
        "--pass-count-noise-threshold",
        type=int,
        default=PASS_COUNT_NOISE_THRESHOLD,
        help=(
            "Absolute passed-task count delta below which a pass_rate "
            "regression is treated as noise (no rollback). A rollback fires "
            "only when BOTH the score-based tolerance is exceeded AND the "
            f"absolute passed-count delta meets this threshold. Default "
            f"{PASS_COUNT_NOISE_THRESHOLD} — small flips on small task sets "
            "are usually eval stochasticity, not regression."
        ),
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Label for this run's directory. "
            "Output goes to recipe/gaia_evolver/runs/{run_tag}/ so repeated "
            "runs don't clobber each other. Defaults to 'run_YYYYMMDD-HHMMSS'."
        ),
    )
    _DEFAULT_DATA_PATH = str(Path(__file__).resolve().parent / "data" / "webthinker_gaia_dev.json")
    parser.add_argument(
        "--data-path",
        default=_DEFAULT_DATA_PATH,
        help=(
            "Path to a local GAIA JSON file (webthinker schema). Pass '' to fall back to HuggingFace dataset download."
        ),
    )
    parser.add_argument(
        "--attachments-dir",
        default=None,
        help=(
            "Optional dir containing per-task attachment files named "
            "'<task_id>.<ext>'. Only needed if your JSON references attachments."
        ),
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="GAIA difficulty level to load (1, 2, or 3). 0 = all levels. Default: 0 (all).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Per-task step cap. Default: {MAX_STEPS}.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=(f"Max concurrent trajectories per round. 1 = serial. Default: {DEFAULT_CONCURRENCY}."),
    )
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Scope this run's outputs under runs/{run_tag}/ so repeated runs
    # don't overwrite each other. --clean wipes THIS run's tree (not all runs).
    run_tag = args.run_tag or time.strftime("run_%Y%m%d-%H%M%S")
    RUN_DIR = RUNS_DIR / run_tag
    if args.clean and RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
        logger.info("Cleaned %s", RUN_DIR)
    elif RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        logger.warning(
            "--run-tag %r already exists and is non-empty; new output will be "
            "interleaved with prior run data. Pass --clean to wipe it first.",
            run_tag,
        )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Run outputs → %s", RUN_DIR)

    provider = _make_provider(
        args.model,
        args.provider_id,
        api_base=args.api_base,
        api_key=args.api_key,
    )
    model_config = ModelConfig(main=provider)

    # Judge runs on --meta-model (Opus by default) rather than the inner
    # agent's model. Correctness signal comes from the stronger tier even
    # when the task agent is on Sonnet. Extended thinking is intentionally
    # OFF here — a judge call returns a short verdict, so a 32k thinking
    # budget per eval would dominate the run's cost for no quality gain.
    # Used for both GAIAPipelineEvaluator (end-of-task scoring) and
    # LLMJudgeProcessor (per-step verdict injected into the config).
    judge_provider = _make_provider(args.meta_model, args.provider_id)
    pipeline_eval = GAIAPipelineEvaluator(judge_provider=judge_provider)

    # The meta-agent's workload is architectural reasoning over trajectories
    # — it benefits materially from Anthropic's extended thinking when the
    # underlying model is served through AnthropicProvider. On non-Anthropic
    # deployments the kwargs are no-ops and this falls back to the same
    # LiteLLM provider the target uses. The meta-agent runs on --meta-model
    # (defaults to Opus) while the inner task agent runs on --model (defaults
    # to Sonnet), so the outer loop can reason at a stronger tier than the
    # inner loop without paying Opus rates on every benchmark task.
    meta_provider = _make_provider(
        args.meta_model,
        args.provider_id,
        extended_thinking=True,
        thinking_budget_tokens=32_000,
        max_tokens=40_000,
    )
    meta_model = ModelConfig(main=meta_provider)

    level_filter = None if args.level == 0 else args.level
    max_tasks_filter = None if args.max_tasks <= 0 else args.max_tasks
    logger.info(
        "Loading GAIA tasks (level=%s, max_tasks=%s)...",
        level_filter if level_filter else "all",
        max_tasks_filter if max_tasks_filter else "all",
    )
    if args.data_path:
        tasks = load_gaia_tasks_from_json(
            args.data_path,
            level=level_filter,
            max_tasks=max_tasks_filter,
            attachments_dir=args.attachments_dir,
        )
    else:
        tasks = load_gaia_tasks(level=level_filter, max_tasks=max_tasks_filter)
    if not tasks:
        logger.error("No tasks loaded!")
        return
    # Apply per-task step cap from CLI (overrides the default baked into GAIATask).
    for t in tasks:
        t.max_steps = args.max_steps
    logger.info("Loaded %d tasks (max_steps=%d)", len(tasks), args.max_steps)

    # Original baseline — never mutated. Each round starts from this (or the
    # previously-accepted compiled config if gating reverted).
    # Uses the GAIA-tuned preset (benchmarks/gaia/prompts/gaia_agent.j2 +
    # build_gaia_tools_full with code_interpreter + GPT-5 token/loop/checkpoint
    # budgets). The actual model
    # is still governed by --model/--provider-id via ModelConfig below.
    original_base = make_gaia_builder_gpt5(
        max_cost_usd=args.max_cost,
    ).build()

    # current_config starts as the original baseline; replaced by the compiled
    # config after each reflect+compile cycle.
    current_config = original_base

    # Add LLMJudgeProcessor to the serializable processors list unless --no-judge
    # was passed. Storing it as a _target_ dict (not in _rt_procs) means it
    # survives every YAML round-trip naturally: from_yaml_file() re-instantiates
    # it via _instantiate_proc, and the meta-agent can see/modify it in config.yaml.
    if not args.no_judge:
        import dataclasses as _dcs
        from harnessx.core.harness import _serialize_processor
        from harnessx.processors.evaluation.llm_judge import LLMJudgeProcessor

        _initial_judge = LLMJudgeProcessor(judge_model=args.meta_model)
        _judge_dict = _serialize_processor(_initial_judge)
        if _judge_dict:
            current_config = _dcs.replace(
                current_config,
                processors=[*current_config.processors, _judge_dict],
            )
            original_base = current_config

    # Cross-round memo for meta-agent continuity (also surfaces "Needs From
    # Human" asks). The file accumulates across the whole experiment.
    LEARNINGS_PATH = RUN_DIR / "learnings.md"

    meta_agent = MetaAgent(
        inner_model=meta_model,
        memo_path=LEARNINGS_PATH,
        extra_skills_dirs=([_GAIA_SKILLS_DIR] if _GAIA_SKILLS_DIR.is_dir() else None),
        max_cost_usd=args.evolve_cost,
        wall_clock_s=float(args.evolve_wall_clock),
        max_steps=args.evolve_steps,
    )

    # Track (pass_rate, round_cost, config_object, round_idx, passed_count)
    # of the historical-best round so tolerance cannot let the baseline
    # drift downward over many rounds. Every round is compared against
    # this; only strictly-better rounds displace the holder. The raw
    # passed_count is tracked alongside the rate so the gate can apply
    # the absolute noise-threshold rule (``--pass-count-noise-threshold``).
    best_so_far: tuple[float, float, Any, int, int] | None = None
    # evolve_status for the round about to start. R0's config came from the
    # baseline (no evolve), every later round's came from the try/except
    # block at the end of the previous iteration.
    next_evolve_status: str = "baseline"

    all_rounds: list[list[dict]] = []
    round_summaries: list[dict] = []

    from harnessx.tracing.journal import HarnessJournal as _HJ

    for round_idx in range(args.num_rounds):
        is_last = round_idx == args.num_rounds - 1

        # Round's self-contained tree: config.yaml + trajectories/ + sessions/
        # + (for R1+) evolve/. Runs/{tag}/R{N}/ answers "everything about R{N}".
        round_dir = RUN_DIR / f"R{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        traj_dir = round_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir = round_dir / "sessions"

        # Per-round journal: isolates this round's JSONL from others, and gives
        # the meta-agent a single directory to glob for "R{N}'s run data".
        # Replaces the cross-round shared tracer that lived across rounds.
        round_journal = _HJ(base_dir=str(sessions_dir), export_jsonl=True)
        round_config = current_config.copy(tracer=round_journal)

        # judge_proc is no longer a round-level handle: each per-task harness
        # instantiates its own LLMJudgeProcessor from the YAML dict, with its
        # own _verdict_sink. Verdict retrieval happens inside _run_one after
        # the harness has run, by scanning harness._rt.processors directly.

        # Dump the config actually executed this round — baseline or evolved —
        # for reproducibility and for evolve() to Read on the next iteration.
        round_config_path = round_dir / "config.yaml"
        round_config.to_yaml_file(round_config_path)

        config_label = "baseline" if round_idx == 0 else f"compiled_R{round_idx}"
        logger.info("\n" + "=" * 60)
        logger.info("ROUND %d/%d  [%s]", round_idx, args.num_rounds - 1, config_label)
        logger.info("=" * 60)

        # ── Run all tasks (up to args.concurrency in parallel) ─────────────
        # Trajectories are independent: each gets its own harness instance,
        # own session_id, and writes to its own per-task file. judge_proc is
        # shared but keyed by run_id. The semaphore bounds concurrent
        # harness.run() calls to stay within LLM provider rate limits; cheap
        # post-processing (judge lookup, trajectory write) runs outside it.
        sem = asyncio.Semaphore(max(1, args.concurrency))

        async def _run_one(task: GAIATask) -> dict:
            # dataclass.replace gives us a per-round copy of the task with
            # this round's budget; avoids mutating the original (which is
            # later consumed by the replay gate through _task_index).
            from dataclasses import replace as _dc_replace

            async with sem:
                task = _dc_replace(task, max_cost_usd=args.max_cost)
                harness = model_config.agentic(round_config)
                record = await _run_task(
                    harness,
                    task,
                    f"R{round_idx}",
                    pipeline_eval=pipeline_eval,
                    harness_config=round_config,
                )

            raw = record.get("_result")
            tid = record.get("task_id") or "unknown"
            record["trajectory_file"] = f"R{round_idx}/trajectories/{tid}.md"

            # Collect judge verdict: find the LLMJudgeProcessor that ran
            # inside this harness instance and pull its verdict for this run_id.
            judge_entry: dict = {}
            if not args.no_judge:
                from harnessx.processors.evaluation.llm_judge import (
                    LLMJudgeProcessor as _LJP,
                )

                run_id = getattr(raw, "run_id", "") or "" if raw is not None else ""
                for _proc in harness._rt.processors.get("*", []):
                    if isinstance(_proc, _LJP):
                        judge_entry = _proc.get_verdict(run_id) or {}
                        break

            # Fold into record (v2 frontmatter fields):
            record["extracted_answer"] = judge_entry.get("extracted_answer") or ""
            record["llm_judge_verdict"] = judge_entry.get("verdict") or {}

            if raw is not None:
                # Populate behavioral fields FIRST so frontmatter has them.
                record["pivotal_tool"] = _pick_pivotal_tool(raw)
                call_counts, error_counts = _compute_tool_counts(raw)
                record["tool_call_counts"] = call_counts
                record["tool_error_counts"] = error_counts
                _, err_count = _compute_tool_stats(raw)
                record["error_count"] = err_count
                traj_text = _build_trajectory_text(task, raw, harness_config=round_config)
                _write_task_trajectory(traj_dir, task, traj_text, record=record)
            return record

        records: list[dict] = list(await asyncio.gather(*(_run_one(t) for t in tasks)))
        gc.collect()

        all_rounds.append(records)

        passed = sum(1 for r in records if r.get("passed"))
        round_cost = sum((r.get("cost_usd") or 0) for r in records)
        totals = _compute_round_totals(records)
        round_pass_rate = round(passed / len(records), 3) if records else 0.0
        round_summaries.append(
            {
                "round": round_idx,
                "config": config_label,
                "tasks": len(records),
                "passed": passed,
                "pass_rate": round_pass_rate,
                "total_cost_usd": round(round_cost, 4),
                "total_tokens": totals["total_tokens"],
                "total_steps": totals["total_steps"],
                "evolve_status": next_evolve_status,
            }
        )
        logger.info("[R%d] pass=%d/%d  cost=$%.3f", round_idx, passed, len(records), round_cost)

        # ── Best-so-far gating ────────────────────────────────────────────
        # Compare against the historical-best round (not the last-accepted)
        # so tolerance cannot silently drift the baseline downward over many
        # rounds. Score = pass_rate - cost_weight * max(cost_delta, 0).
        gate_decision, gate_reason, best_so_far, reverted_cfg = _score_and_gate(
            round_pass_rate=round_pass_rate,
            round_cost=round_cost,
            round_idx=round_idx,
            round_config=current_config,
            round_passed=passed,
            best=best_so_far,
            tolerance=args.regression_tolerance,
            cost_weight=args.cost_weight,
            pass_count_noise_threshold=args.pass_count_noise_threshold,
        )
        if reverted_cfg is not None:
            best_round_for_log = best_so_far[3]
            logger.warning(
                "[R%d] REGRESSION — reverting current_config to R%d for next round",
                round_idx,
                best_round_for_log,
            )
            current_config = reverted_cfg

        # Back-fill the journal entry for this round (if the meta-agent
        # wrote one for it). ``gating_outcome`` + per-task
        # ``gating_attribution`` turn the journal into structured
        # evidence the next evolve's CONTEXT.md can aggregate. R0 is
        # baseline — no meta-agent ran before it — so nothing to fill.
        if round_idx >= 1:
            try:
                from harnessx.meta_harness import journal as _journal

                entries = _journal.read_entries(LEARNINGS_PATH)
                entry = next((e for e in entries if e.round == round_idx), None)
                if entry is not None:
                    prev_records = all_rounds[-2] if len(all_rounds) >= 2 else []
                    prev_passed = {r["task_id"] for r in prev_records if r.get("passed")}
                    prev_appeared = {r["task_id"] for r in prev_records if r.get("task_id")}
                    cur_passed = {r["task_id"] for r in records if r.get("passed")}
                    cur_appeared = {r["task_id"] for r in records if r.get("task_id")}
                    outcome = "reverted" if gate_decision == "REVERTED" else "accepted"
                    # Byte-identical config = explicit noop — surface
                    # that separately from a change that survived gating.
                    if next_evolve_status == "noop":
                        outcome = "noop"
                    attribution = _journal.compute_attribution(
                        entry.predicted_affected,
                        passed_now=cur_passed,
                        passed_before=prev_passed,
                        appeared_now=cur_appeared,
                        appeared_before=prev_appeared,
                    )
                    # Side-effect detection: tasks that regressed this
                    # round but were NOT in predicted_affected. These
                    # reduce the agent's lever precision — a hypothesis
                    # that helps its claimed target but breaks something
                    # else shouldn't read as 100% effective.
                    predicted_set = set(entry.predicted_affected)
                    regressed_unpredicted = sorted(
                        (prev_passed & prev_appeared & cur_appeared) - cur_passed - predicted_set
                    )
                    # Load orchestrator-computed changeset from the evolve
                    # step that produced this round's config. The file is
                    # written under R{round_idx}/evolve/_meta_scratch/.
                    changeset_path = RUN_DIR / f"R{round_idx}" / "evolve" / "_meta_scratch" / "changeset.json"
                    changeset: dict = {}
                    if changeset_path.is_file():
                        try:
                            changeset = json.loads(changeset_path.read_text(encoding="utf-8"))
                        except (json.JSONDecodeError, OSError) as cs_exc:
                            logger.warning(
                                "[R%d] changeset.json unreadable: %s",
                                round_idx,
                                cs_exc,
                            )
                    ok = _journal.fill_gating(
                        LEARNINGS_PATH,
                        round_idx,
                        outcome,
                        attribution,
                        extra_frontmatter={
                            "regressed_unpredicted": regressed_unpredicted,
                            "changeset": changeset,
                        },
                    )
                    if ok:
                        logger.info(
                            "[R%d] journal fill_gating outcome=%s attribution=%s regressed_unpredicted=%d",
                            round_idx,
                            outcome,
                            attribution,
                            len(regressed_unpredicted),
                        )
                else:
                    logger.debug(
                        "[R%d] no journal entry found — meta-agent did not "
                        "append one yet; skipping attribution back-fill",
                        round_idx,
                    )
            except Exception as exc:  # noqa: BLE001
                # Never fail the round over memoisation bookkeeping.
                logger.warning(
                    "[R%d] journal fill_gating failed (non-fatal): %s",
                    round_idx,
                    exc,
                )

        if is_last:
            continue

        # ── Evolve: produce next round's config ───────────────────────────
        next_round_dir = RUN_DIR / f"R{round_idx + 1}"
        next_round_dir.mkdir(parents=True, exist_ok=True)
        evolve_dir = next_round_dir / "evolve"
        logger.info(
            "[R%d] evolve → %s (memo=%s)",
            round_idx,
            evolve_dir,
            LEARNINGS_PATH,
        )

        # Replay gate needs to resolve task_id strings (read from trajectory
        # frontmatter) back to the concrete GAIATask objects we loaded at
        # startup. Index them once per round and expose via a tiny async
        # closure; meta_harness stays benchmark-agnostic and the recipe
        # owns the dataset shape.
        _task_index = {t.task_id: t for t in tasks if t.task_id}

        async def _gaia_task_loader(task_id: str):  # noqa: ANN001
            try:
                return _task_index[task_id]
            except KeyError as exc:
                raise KeyError(
                    f"replay_gate requested task_id={task_id!r} but it is "
                    f"not in the current task set ({len(_task_index)} tasks "
                    "loaded). The gate picks task ids from the last round's "
                    "trajectory frontmatter; mismatches usually mean --data-path "
                    "changed between rounds."
                ) from exc

        try:
            new_yaml = await meta_agent.evolve(
                current_config=round_config_path,
                trajectories_dir=traj_dir,
                output_dir=evolve_dir,
                replay_model=model_config,
                replay_max_cost_usd=min(0.5, args.max_cost),
            )
            from harnessx.core.harness import HarnessConfig as _HC

            candidate_cfg = _HC.from_yaml_file(new_yaml).canonicalize()
            # Byte-identical output = the meta-agent's explicit no-op idiom
            # ("`cp current output/config.yaml`"). Surfacing this lets the
            # next round's summary distinguish "agent decided no change" from
            # "agent produced a meaningful change".
            if round_config_path.read_bytes() == Path(new_yaml).read_bytes():
                next_evolve_status = "noop"
            else:
                next_evolve_status = "ok"
            logger.info(
                "[R%d] R%d config → %s (status=%s)",
                round_idx,
                round_idx + 1,
                new_yaml,
                next_evolve_status,
            )
            current_config = candidate_cfg
        except Exception as exc:  # noqa: BLE001
            next_evolve_status = "crashed"
            logger.exception(
                "[R%d] evolve crashed — R%d reuses current config: %s",
                round_idx,
                round_idx + 1,
                exc,
            )

    # ── Final showcase ─────────────────────────────────────────────────────
    print_multiround_comparison(all_rounds)

    results_path = RUN_DIR / "comparison.json"
    results_path.write_text(
        json.dumps(
            {
                "rounds": [[{k: v for k, v in r.items() if not k.startswith("_")} for r in rd] for rd in all_rounds],
                "round_summaries": round_summaries,
                "run_config": {
                    "model": args.model,
                    "meta_model": args.meta_model,
                    "max_tasks": args.max_tasks,
                    "max_cost_usd": args.max_cost,
                    "num_rounds": args.num_rounds,
                    "evolve_steps": args.evolve_steps,
                    "evolve_cost": args.evolve_cost,
                    "evolve_wall_clock": args.evolve_wall_clock,
                    "run_dir": str(RUN_DIR),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Results → %s", results_path)


def _compute_tool_counts(result: Any) -> tuple[dict[str, int], dict[str, int]]:
    """Per-tool call counts and error counts for one HarnessResult.

    Returns (tool_call_counts, tool_error_counts). Both keyed by tool name.
    Inlined from the deleted recipe/gaia_evolver/signals.py to preserve
    measurement data in the v2 frontmatter.
    """
    call_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    traj = getattr(result, "trajectory", None)
    if not traj or not hasattr(traj, "steps"):
        return call_counts, error_counts
    for step in traj.steps:
        for tr in step.observation or []:
            name = getattr(tr, "tool_name", "") or ""
            if not name:
                continue
            call_counts[name] = call_counts.get(name, 0) + 1
            if getattr(tr, "error", None):
                error_counts[name] = error_counts.get(name, 0) + 1
    return call_counts, error_counts


def _render_trajectory_frontmatter(record: dict) -> str:
    """Render per-task YAML frontmatter (v2 schema) for the trajectory .md file.

    Agent-facing contract: ``Read limit=30`` yields the key measurements + judge
    verdict at the top, so the meta-agent can orient before drilling into the body.

    Fields split into three tiers, in render order:

    1. Behaviour signals — always present (``exit_reason``, ``steps``, counts…).
    2. Evaluation signals — always present. ``eval_passed`` / ``eval_score``
       come from the external pipeline evaluator and are authoritative. The
       ground-truth answer text is intentionally withheld from the
       frontmatter, and so is the evaluator's textual reason — only the
       pass/fail outcome and numeric score are exposed so the meta-agent
       evolves on correctness signals, not on a known target string.
       (``extracted_answer``, written by the judge tier, is what the agent
       committed — not the dataset's expected answer.)
    3. Judge signals — optional, present only when ``llm_judge_verdict`` is
       populated (``--no-judge`` omits all six ``judge_*`` / ``extracted_answer``
       fields). These are opinion, not ground truth.
    """
    import json as _json

    def _yaml_scalar(v: Any) -> str:
        if v is None:
            return '""'
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, list):
            return _json.dumps(v, ensure_ascii=False)
        if isinstance(v, dict):
            return _json.dumps(v, ensure_ascii=False)
        s = str(v).replace("\n", " ").strip()
        return _json.dumps(s, ensure_ascii=False)

    tool_call_counts = record.get("tool_call_counts") or {}
    tool_error_counts = record.get("tool_error_counts") or {}
    tools_used = sorted(tool_call_counts.keys())
    # Prefer `output` as the canonical source. Keep `final_output` as legacy fallback.
    final_output = (record.get("output") or record.get("final_output") or "").strip()

    fields: list[tuple[str, Any]] = [
        ("task_id", record.get("task_id") or "unknown"),
        ("exit_reason", record.get("exit_reason") or ""),
        ("steps", int(record.get("steps") or 0)),
        ("cost_usd", float(record.get("cost_usd") or 0.0)),
        ("final_output_length", len(final_output)),
        ("model_empty_end_turn", bool(record.get("model_empty_end_turn") or False)),
        ("empty_end_turn_recovered", bool(record.get("empty_end_turn_recovered") or False)),
        ("tools_used", tools_used),
        ("tool_call_counts", tool_call_counts),
        ("tool_error_counts", tool_error_counts),
    ]

    total_tokens = record.get("total_tokens")
    if total_tokens is not None:
        fields.insert(4, ("total_tokens", int(total_tokens)))

    # External evaluator (authoritative) — always emitted so the meta-agent
    # can see correctness outcomes. Only pass/fail and numeric score are
    # exposed; the expected answer text AND the evaluator's textual reason
    # are both withheld, because a reason string can still smuggle the
    # agent's extracted answer or correctness-adjacent phrasing. The
    # meta-agent evolves on pure pass/fail signals.
    fields.extend(
        [
            ("eval_passed", bool(record.get("passed") or False)),
            ("eval_score", float(record.get("score") or 0.0)),
        ]
    )

    judge_verdict = record.get("llm_judge_verdict") or {}
    if judge_verdict:
        extracted_answer = record.get("extracted_answer") or ""
        # missing_capability is the richer signal the meta-agent reads when
        # deciding whether to reach for the Action lever (authoring a new
        # tool) over another processor tweak. Always emit when the key
        # exists so it shows up in frontmatter `Read limit=30`; the empty
        # payload ({"present":false,"summary":"","evidence_steps":[]})
        # reads fine as a negative signal too.
        mc = judge_verdict.get("missing_capability")
        if not isinstance(mc, dict):
            mc = {"present": False, "summary": "", "evidence_steps": []}
        fields.extend(
            [
                ("extracted_answer", extracted_answer),
                ("judge_verdict", judge_verdict.get("verdict") or ""),
                ("judge_confidence", float(judge_verdict.get("confidence") or 0.0)),
                ("judge_cause", judge_verdict.get("cause") or ""),
                ("judge_missing", judge_verdict.get("missing") or ""),
                ("judge_lesson", judge_verdict.get("lesson") or ""),
                ("judge_missing_capability", mc),
            ]
        )

    lines = ["---"]
    for k, v in fields:
        lines.append(f"{k}: {_yaml_scalar(v)}")
    lines.append("---")
    return "\n".join(lines)


def _write_task_trajectory(
    round_dir: Path,
    task: Any,
    text: str,
    record: dict | None = None,
) -> None:
    """Write a single task's trajectory to ``round_dir/<task_id>.md``.

    When ``record`` is provided, prepends the YAML frontmatter produced by
    :func:`_render_trajectory_frontmatter`. Legacy callers passing only
    ``(round_dir, task, text)`` still work — they just get a body without
    frontmatter.
    """
    round_dir.mkdir(parents=True, exist_ok=True)
    tid = getattr(task, "task_id", None) or "unknown"
    if record is not None:
        fm = _render_trajectory_frontmatter(record)
        text = f"{fm}\n\n{text.lstrip()}"
    (round_dir / f"{tid}.md").write_text(text, encoding="utf-8")


def write_round_trajectories(
    round_dir: Path,
    task_trajectories: list[tuple[Any, str]],
) -> None:
    """Write per-task trajectories under ``round_dir/<task_id>.md``."""
    for task, text in task_trajectories:
        _write_task_trajectory(round_dir, task, text)


def _compute_round_totals(records: list[dict]) -> dict:
    return {
        "total_tokens": sum(int(r.get("total_tokens") or 0) for r in records),
        "total_steps": sum(int(r.get("steps") or 0) for r in records),
    }


def _pct_delta(new: float, old: float) -> str:
    """Render a percentage change as e.g. '+12%' / '-5%'. '?' when old == 0."""
    if not old:
        return "?"
    return f"{100 * (new - old) / old:+.0f}%"


def _score_and_gate(
    *,
    round_pass_rate: float,
    round_cost: float,
    round_idx: int,
    round_config: Any,
    round_passed: int,
    best: tuple[float, float, Any, int, int] | None,
    tolerance: float,
    cost_weight: float,
    pass_count_noise_threshold: int = 3,
) -> tuple[str, str, tuple[float, float, Any, int, int], Any]:
    """Best-so-far gating kernel.

    Compares this round to the historical best (not the last-accepted round)
    so tolerance cannot drift the baseline downward over many rounds.

    Two guards against noise-driven rollback:

    - ``tolerance`` — relative score drop allowed (pass_rate-based).
    - ``pass_count_noise_threshold`` — absolute passed-task count delta
      below which even a score regression is treated as noise. A rollback
      fires only when BOTH checks fail (score below tolerance AND count
      delta >= threshold). Small task sets (e.g. 6 tasks per round) can
      swing 1-2 passes due to eval stochasticity alone; without this
      guard those would trigger spurious rollbacks.

    Returns ``(decision, reason, new_best, reverted_to_config_or_none)``.

    - ``decision`` ∈ {"ACCEPTED", "REVERTED"}.
    - ``reverted_to_config_or_none`` is ``None`` on ACCEPTED and the
      best-round's config on REVERTED (caller should set ``current_config``
      back to it).
    - ``new_best`` equals ``best`` unless this round strictly beats the best
      score; equal-score rounds do not dethrone the earliest holder.
    """
    score = round_pass_rate  # recomputed below when a baseline exists
    if best is None:
        return (
            "ACCEPTED",
            "first round — no prior to compare against",
            (round_pass_rate, round_cost, round_config, round_idx, round_passed),
            None,
        )
    best_rate, best_cost, best_cfg, best_round, best_passed = best
    cost_delta_ratio = (round_cost - best_cost) / max(best_cost, 1e-3) if best_cost else 0.0
    score = round_pass_rate - cost_weight * max(cost_delta_ratio, 0.0)
    best_score = best_rate  # baseline's own cost-delta against itself is 0
    if score < best_score - tolerance:
        # Score says regress — but first check the absolute count delta.
        # If only 1-2 tasks flipped, that's within eval noise on small
        # task sets and shouldn't wipe out the round's other changes.
        count_delta = abs(round_passed - best_passed)
        if count_delta < pass_count_noise_threshold:
            reason = (
                f"noise-level regression: passed {best_passed}→{round_passed} "
                f"(|Δ|={count_delta} < threshold {pass_count_noise_threshold}); "
                f"score {score:.3f} vs R{best_round} {best_score:.3f} kept despite "
                f"tolerance breach"
            )
            # Don't update best — this round underperformed; next round
            # still gets compared against the same historical high.
            return ("ACCEPTED", reason, best, None)
        reason = (
            f"score {score:.3f} < R{best_round} {best_score:.3f} - "
            f"tolerance {tolerance:.3f} "
            f"(pass_rate {best_rate:.3f}→{round_pass_rate:.3f}; "
            f"passed {best_passed}→{round_passed} |Δ|={count_delta}; "
            f"cost ${best_cost:.2f}→${round_cost:.2f}; "
            f"cost_weight={cost_weight:.2f})"
        )
        return ("REVERTED", reason, best, best_cfg)
    if score > best_score:
        new_best = (
            round_pass_rate,
            round_cost,
            round_config,
            round_idx,
            round_passed,
        )
    else:
        new_best = best
    reason = f"score {score:.3f} ≥ R{best_round} {best_score:.3f} - tolerance {tolerance:.3f}"
    return ("ACCEPTED", reason, new_best, None)


def _compute_tool_stats(result) -> tuple[dict, int]:
    """Return ({tool_name: call_count}, total_error_count) from a HarnessResult."""
    traj = getattr(result, "trajectory", None)
    counts: dict[str, int] = {}
    errors = 0
    if traj and hasattr(traj, "steps"):
        for step in traj.steps:
            for tr in step.observation or []:
                name = getattr(tr, "tool_name", "") or ""
                if name:
                    counts[name] = counts.get(name, 0) + 1
                if getattr(tr, "error", ""):
                    errors += 1
    return counts, errors


def _pick_pivotal_tool(result) -> str:
    """Return the most-used tool name (best-effort)."""
    traj = getattr(result, "trajectory", None)
    counts: dict[str, int] = {}
    if traj and hasattr(traj, "steps"):
        for step in traj.steps:
            for tr in step.observation or []:
                name = getattr(tr, "tool_name", "")
                if name:
                    counts[name] = counts.get(name, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda kv: -kv[1])[0][0]


def _build_trajectory_text(
    task: GAIATask,
    result: HarnessResult,
    harness_config: Any | None = None,
) -> str:
    """Build detailed human-readable trajectory from HarnessResult.

    Layout (same as the original MetaHarness._dump_trajectory):
      1. Task description
      2. Result (pass/fail, exit reason, final output)
      3. Harness Config (processors, tools)
      4. Diagnostics (budget usage, error rates, top tools)
      5. Execution Steps (tool calls + results per step)
    """
    import hashlib as _hashlib

    task_id = getattr(task, "task_id", "") or _hashlib.sha256(str(task.description)[:100].encode()).hexdigest()[:12]

    # ── Aggregate tool stats from trajectory ────────────────────────────
    traj = getattr(result, "trajectory", None)
    total_tool_calls = 0
    tool_errors = 0
    tool_call_counts: dict[str, int] = {}

    if traj and hasattr(traj, "steps"):
        for step in traj.steps:
            for tr in step.observation or []:
                total_tool_calls += 1
                tname = getattr(tr, "tool_name", "?")
                tool_call_counts[tname] = tool_call_counts.get(tname, 0) + 1
                if getattr(tr, "error", ""):
                    tool_errors += 1

    lines: list[str] = [f"# Trajectory: {task_id}"]

    # ── Section 1: Task ─────────────────────────────────────────────────
    desc = task.description if isinstance(task.description, str) else str(task.description)
    lines.append(f"\n## Task\n\n{desc}")

    # ── Section 2: Result ───────────────────────────────────────────────
    task_end = getattr(result, "task_end", None)
    if task_end:
        lines.append("\n## Result\n")
        lines.append(f"- exit_reason: {getattr(task_end, 'exit_reason', '?')}")
        lines.append(f"- total_steps: {getattr(task_end, 'total_steps', '?')}")
        final_out = getattr(task_end, "final_output", "") or ""
        lines.append(f"- final_output: {final_out}")

    # ── Section 3: Harness Config ───────────────────────────────────────
    if harness_config is not None:
        lines.append("\n## Harness Config\n")
        # ``HarnessConfig.processors`` is a flat ``list[dict]`` (pre- or
        # post-canonicalize — the processor instances live in
        # ``_rt_procs`` after canonicalize). Walk both to render cleanly
        # in either lifecycle.
        proc_parts: list[str] = []
        for entry in getattr(harness_config, "processors", None) or []:
            if isinstance(entry, dict):
                target = entry.get("_target_", "") or ""
                label = target.rsplit("::", 1)[-1] if "::" in target else target.rsplit(".", 1)[-1]
                if label:
                    proc_parts.append(label)
            else:
                group = getattr(entry, "_singleton_group", "")
                order = getattr(entry, "_order", "?")
                label = group or type(entry).__name__
                proc_parts.append(f"{label}({order})")
        for p in getattr(harness_config, "_rt_procs", None) or []:
            group = getattr(p, "_singleton_group", "")
            order = getattr(p, "_order", "?")
            label = group or type(p).__name__
            tag = f"{label}({order})"
            if tag not in proc_parts:
                proc_parts.append(tag)
        if proc_parts:
            lines.append(f"Processors: {', '.join(proc_parts)}")
        registry = getattr(harness_config, "tool_registry", None)
        if registry and hasattr(registry, "list_names"):
            try:
                tool_names = list(registry.list_names())
                lines.append(f"Tools: [{', '.join(sorted(tool_names))}]")
            except Exception:
                pass

    # ── Section 4: Diagnostics ──────────────────────────────────────────
    lines.append("\n## Diagnostics\n")
    total_steps = getattr(result, "total_steps", 0) or 0
    max_steps = getattr(task, "max_steps", 0) or 20
    total_tokens = getattr(result, "total_tokens", 0) or 0
    total_cost = getattr(result, "total_cost_usd", 0) or 0
    max_cost = getattr(task, "max_cost_usd", 0) or 0
    exit_reason = getattr(result, "exit_reason", "?")

    step_pct = f"{100 * total_steps // max_steps}%" if max_steps else "?"
    lines.append(f"- steps: {total_steps}/{max_steps} ({step_pct} budget)")
    lines.append(f"- tokens: {total_tokens}")
    if max_cost:
        cost_pct = f"{100 * total_cost / max_cost:.0f}%"
        lines.append(f"- cost: ${total_cost:.3f}/${max_cost:.2f} ({cost_pct} budget)")
    else:
        lines.append(f"- cost: ${total_cost:.3f}")
    error_rate = f"{100 * tool_errors / total_tool_calls:.0f}%" if total_tool_calls else "0%"
    lines.append(f"- tool_calls: {total_tool_calls}, errors: {tool_errors} (error_rate={error_rate})")
    if tool_call_counts:
        top_tools = sorted(tool_call_counts.items(), key=lambda x: -x[1])[:5]
        lines.append(f"- top_tools: {', '.join(f'{n}({c})' for n, c in top_tools)}")
    lines.append(f"- exit_reason: {exit_reason}")

    # Per-tool error counts (inlined from deleted signals.py)
    _, tool_err = _compute_tool_counts(result)
    err_parts = [f"{n}({c})" for n, c in (tool_err or {}).items() if c]
    lines.append(f"- tool_error_counts: {', '.join(err_parts) if err_parts else '-'}")

    # ── Section 5: Execution Steps ──────────────────────────────────────
    if traj and hasattr(traj, "steps"):
        lines.append("\n---\n")
        lines.append("## Execution Steps\n")
        for step in traj.steps:
            lines.append(f"\n### Step {step.step_id}")

            action = step.action
            if action:
                thinking = getattr(action, "thinking", "") or ""
                raw = getattr(action, "content", None)
                content = raw if isinstance(raw, str) else (str(raw) if raw else "")

                if thinking:
                    lines.append(f"\n#### Thinking\n\n{thinking}")
                if content:
                    lines.append(f"\n#### Response\n\n{content}")

                tool_calls = getattr(action, "tool_calls", None) or ()
                if tool_calls:
                    lines.append("\n#### Tool Calls\n")
                    for tc in tool_calls:
                        input_str = json.dumps(tc.input, ensure_ascii=False)
                        lines.append(f"- **{tc.name}**(`{input_str}`)")

            for tr in step.observation or []:
                tname = getattr(tr, "tool_name", "?")
                error_str = getattr(tr, "error", "") or ""
                result_str = getattr(tr, "result", "") or ""
                if error_str:
                    lines.append(f"  -> {tname}: ERROR: {error_str}")
                else:
                    lines.append(f"  -> {tname}: {result_str}")

    return "\n".join(lines)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message=".*Event loop is closed.*")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        gc.collect()
        loop.close()
        gc.collect()
