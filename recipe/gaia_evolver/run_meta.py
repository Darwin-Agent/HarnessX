#!/usr/bin/env python3
"""GAIA Meta-Harness: per-domain evolution with configurable agent + meta-agent.

Splits GAIA tasks by domain category into non-overlapping groups and evolves
each domain independently. Supports gating, early stop, and learnings
accumulation across rounds.

Usage:
    python -m recipe.gaia_evolver.run_meta --list-domains
    python -m recipe.gaia_evolver.run_meta --domain Multi-hop --num-rounds 4
    python -m recipe.gaia_evolver.run_meta   # all domains sequentially
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any  # noqa: F401 — used by type comments

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_env_path = Path(_PROJECT_ROOT) / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from harnessx.core.harness import HarnessConfig
from harnessx.core.model_config import ModelConfig
from harnessx.meta_harness import MetaAgent
from harnessx.tracing.journal import HarnessJournal

from benchmarks.gaia.evaluator import GAIAPipelineEvaluator
from benchmarks.gaia.harness import make_gaia_builder_gpt5
from benchmarks.gaia.task import GAIATask

logging.basicConfig(
    level=logging.INFO,
    format="\033[32m%(asctime)s\033[0m \033[1m%(levelname)-5s\033[0m \033[36m%(name)s\033[0m — \033[1m%(message)s\033[0m",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
try:
    import litellm as _litellm

    _litellm.suppress_debug_info = True
    _litellm.set_verbose = False
except ImportError:
    pass

_RECIPE_DIR = Path(__file__).resolve().parent
RUNS_DIR = _RECIPE_DIR / "runs"
_GAIA_SKILLS_DIR = _RECIPE_DIR / "skills"

# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_DATA_PATH = os.environ.get("GAIA_DATA_PATH", str(_RECIPE_DIR / "data" / "webthinker_gaia_dev_classified.json"))
DEFAULT_MODEL = os.environ.get("GAIA_MODEL", "gpt-5")
DEFAULT_META_MODEL = os.environ.get("GAIA_META_MODEL", "anthropic/YOUR_PROVIDER/claude-opus-4-6")
DEFAULT_API_BASE = os.environ.get("OPENAI_API_BASE", "")
DEFAULT_PROVIDER_ID = os.environ.get("GAIA_PROVIDER_ID", "azure_openai")

NUM_ROUNDS = 8
MAX_STEPS = 40
MAX_COST_USD = 15.0
CONCURRENCY = 6
EVOLVE_COST = 50.0
EVOLVE_STEPS = 200
EVOLVE_WALL_CLOCK = 10000


# ─── Provider factory ────────────────────────────────────────────────────────


def _make_provider(
    model,
    provider_id,
    *,
    api_base=None,
    api_key=None,
    extended_thinking=False,
    thinking_budget_tokens=10_000,
    max_tokens=8192,
):
    from harnessx.providers.anthropic_provider import AnthropicProvider
    from harnessx.providers.openai_provider import OpenAIProvider

    if model.startswith("anthropic/"):
        model_name = model[len("anthropic/") :]
        return AnthropicProvider(
            model=model_name,
            base_url=os.environ.get("ANTHROPIC_API_BASE") or os.environ.get("ANTHROPIC_BASE_URL"),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            extended_thinking=extended_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            max_tokens=max_tokens,
        )
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY")
    extra_headers = {"X-Model-Provider-Id": provider_id}
    return OpenAIProvider(
        model=model,
        base_url=api_base or os.environ.get("OPENAI_API_BASE"),
        api_key=resolved_key,
        extra_headers=extra_headers,
    )


# ─── Load classified tasks ──────────────────────────────────────────────────


def _load_classified_tasks(path: str, attachments_dir: str | None = None) -> dict[str, list[GAIATask]]:
    """Load GAIA tasks from the classified JSON and group by domain category.

    Returns {category_short_name: [GAIATask, ...]}.
    """
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    if isinstance(blob, list):
        questions = blob
    elif isinstance(blob, dict):
        questions = blob.get("questions") or blob
        if isinstance(questions, dict):
            questions = list(questions.values())
    else:
        raise ValueError(f"Unexpected GAIA JSON root type: {type(blob).__name__}")

    from benchmarks.gaia.task import GAIATask as _GT

    domains: dict[str, list[GAIATask]] = {}

    for raw in questions:
        cat = raw.get("category", "unknown")
        # Extract English part from bilingual "Chinese (English)" format
        short = cat.split("(")[-1].rstrip(")").strip() if "(" in cat else cat
        short = short.replace(" ", "_").replace("/", "_")

        lvl = int(raw.get("Level", 1))
        tid = raw.get("task_id", "")

        file_name, _file_path = "", ""
        if attachments_dir and tid:
            import os as _os

            try:
                matches = [n for n in _os.listdir(attachments_dir) if n.startswith(tid + ".")]
            except OSError:
                matches = []
            if matches:
                file_name = matches[0]
                _file_path = _os.path.join(attachments_dir, file_name)

        row = {
            "task_id": tid,
            "Question": raw.get("Question", ""),
            "Level": lvl,
            "Final answer": raw.get("answer", ""),
            "file_name": file_name,
            "file_path": file_name,
            "Annotator Metadata": raw.get("Annotator_Metadata", {}),
        }
        task = _GT.from_hf_row(row, data_dir=attachments_dir or "")
        domains.setdefault(short, []).append(task)

    return domains


# ─── Task runner (adapted from run.py) ────────────────────────────────────────


async def _run_task(harness, task, label, *, pipeline_eval, harness_config=None):
    t0 = time.time()
    task_id = task.task_id or "?"
    logger.info("[%s] Running %s (Level %d)...", label, task_id, task.level)
    try:
        result = await harness.run(task, session_id=f"{label}-{task_id}")
        elapsed = time.time() - t0
        # LLM-judge-primary eval: passes the full trajectory (most recent
        # assistant turns) + ground truth to an LLM judge rather than relying
        # on string-matching ``result.final_output``. The legacy string-match
        # path misgrades tasks where the FINAL ANSWER was emitted one turn
        # before the trajectory ended (e.g. after a CommitNudge injection)
        # because ``final_output`` ends up empty.
        eval_result = await pipeline_eval.evaluate_with_trace_judge(
            task_description=task.question or task.description or "",
            ground_truth=task.final_answer or "",
            final_output=result.final_output or "",
            trajectory_messages=getattr(
                getattr(result, "task_end", None), "final_messages", (),
            ),
        )
        passed = bool(eval_result.passed)
        score = float(eval_result.score)
        reason = (eval_result.reason or "")[:200]

        output = result.final_output or ""
        state_snapshot = getattr(getattr(result, "task_end", None), "state_snapshot", None) or {}
        slots = state_snapshot.get("slots") if isinstance(state_snapshot, dict) else {}
        if not isinstance(slots, dict):
            slots = {}

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
            "model_empty_end_turn": False,
            "empty_end_turn_recovered": False,
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


# ─── Trajectory helpers (from run.py) ─────────────────────────────────────────


def _compute_tool_counts(result):
    call_counts, error_counts = {}, {}
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


def _pick_pivotal_tool(result):
    traj = getattr(result, "trajectory", None)
    counts = {}
    if traj and hasattr(traj, "steps"):
        for step in traj.steps:
            for tr in step.observation or []:
                name = getattr(tr, "tool_name", "")
                if name:
                    counts[name] = counts.get(name, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda kv: -kv[1])[0][0]


def _render_trajectory_frontmatter(record):
    import json as _json

    def _yaml_scalar(v):
        if v is None:
            return '""'
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, (list, dict)):
            return _json.dumps(v, ensure_ascii=False)
        s = str(v).replace("\n", " ").strip()
        return _json.dumps(s, ensure_ascii=False)

    tool_call_counts = record.get("tool_call_counts") or {}
    tool_error_counts = record.get("tool_error_counts") or {}
    final_output = (record.get("output") or record.get("final_output") or "").strip()

    fields = [
        ("task_id", record.get("task_id") or "unknown"),
        ("exit_reason", record.get("exit_reason") or ""),
        ("steps", int(record.get("steps") or 0)),
        ("cost_usd", float(record.get("cost_usd") or 0.0)),
        ("final_output_length", len(final_output)),
        ("tools_used", sorted(tool_call_counts.keys())),
        ("tool_call_counts", tool_call_counts),
        ("tool_error_counts", tool_error_counts),
        ("eval_passed", bool(record.get("passed") or False)),
        ("eval_score", float(record.get("score") or 0.0)),
    ]
    total_tokens = record.get("total_tokens")
    if total_tokens is not None:
        fields.insert(4, ("total_tokens", int(total_tokens)))

    lines = ["---"]
    for k, v in fields:
        lines.append(f"{k}: {_yaml_scalar(v)}")
    lines.append("---")
    return "\n".join(lines)


def _write_task_trajectory(traj_dir, task, text, record=None):
    traj_dir.mkdir(parents=True, exist_ok=True)
    tid = getattr(task, "task_id", None) or "unknown"
    if record is not None:
        fm = _render_trajectory_frontmatter(record)
        text = f"{fm}\n\n{text.lstrip()}"
    (traj_dir / f"{tid}.md").write_text(text, encoding="utf-8")


def _build_trajectory_text(task, result, harness_config=None):
    import hashlib as _hashlib

    task_id = getattr(task, "task_id", "") or _hashlib.sha256(str(task.description)[:100].encode()).hexdigest()[:12]
    traj = getattr(result, "trajectory", None)
    total_tool_calls, tool_errors = 0, 0
    tool_call_counts = {}
    if traj and hasattr(traj, "steps"):
        for step in traj.steps:
            for tr in step.observation or []:
                total_tool_calls += 1
                tname = getattr(tr, "tool_name", "?")
                tool_call_counts[tname] = tool_call_counts.get(tname, 0) + 1
                if getattr(tr, "error", ""):
                    tool_errors += 1

    lines = [f"# Trajectory: {task_id}"]
    desc = task.description if isinstance(task.description, str) else str(task.description)
    lines.append(f"\n## Task\n\n{desc}")
    task_end = getattr(result, "task_end", None)
    if task_end:
        lines.append("\n## Result\n")
        lines.append(f"- exit_reason: {getattr(task_end, 'exit_reason', '?')}")
        lines.append(f"- total_steps: {getattr(task_end, 'total_steps', '?')}")
        lines.append(f"- final_output: {getattr(task_end, 'final_output', '') or ''}")

    lines.append("\n## Diagnostics\n")
    total_steps = getattr(result, "total_steps", 0) or 0
    max_steps_val = getattr(task, "max_steps", 0) or 40
    total_tokens = getattr(result, "total_tokens", 0) or 0
    total_cost = getattr(result, "total_cost_usd", 0) or 0
    lines.append(f"- steps: {total_steps}/{max_steps_val}")
    lines.append(f"- tokens: {total_tokens}")
    lines.append(f"- cost: ${total_cost:.3f}")
    lines.append(f"- tool_calls: {total_tool_calls}, errors: {tool_errors}")
    if tool_call_counts:
        top = sorted(tool_call_counts.items(), key=lambda x: -x[1])[:5]
        lines.append(f"- top_tools: {', '.join(f'{n}({c})' for n, c in top)}")

    if traj and hasattr(traj, "steps"):
        lines.append("\n---\n\n## Execution Steps\n")
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


# ─── Gating ──────────────────────────────────────────────────────────────────


def _score_and_gate(
    *,
    round_pass_rate,
    round_cost,
    round_idx,
    round_config,
    round_passed,
    best,
    tolerance,
    cost_weight,
    pass_count_noise_threshold=3,
):
    score = round_pass_rate
    if best is None:
        return ("ACCEPTED", "first round", (round_pass_rate, round_cost, round_config, round_idx, round_passed), None)
    best_rate, best_cost, best_cfg, best_round, best_passed = best
    cost_delta_ratio = (round_cost - best_cost) / max(best_cost, 1e-3) if best_cost else 0.0
    score = round_pass_rate - cost_weight * max(cost_delta_ratio, 0.0)
    best_score = best_rate
    if score < best_score - tolerance:
        count_delta = abs(round_passed - best_passed)
        if count_delta < pass_count_noise_threshold:
            return ("ACCEPTED", f"noise-level regression |Δ|={count_delta}", best, None)
        return ("REVERTED", f"score {score:.3f} < best {best_score:.3f}", best, best_cfg)
    if score > best_score:
        new_best = (round_pass_rate, round_cost, round_config, round_idx, round_passed)
    else:
        new_best = best
    return ("ACCEPTED", f"score {score:.3f} ≥ best", new_best, None)


# ─── Core evolution loop ─────────────────────────────────────────────────────


async def run_evolve_loop(
    *,
    scope_name: str,
    tasks: list[GAIATask],
    model_config: ModelConfig,
    meta_model: ModelConfig,
    judge_provider,
    original_base: HarnessConfig,
    scope_dir: Path,
    num_rounds: int,
    max_cost: float,
    max_steps: int,
    concurrency: int,
    evolve_cost: float,
    evolve_steps: int,
    evolve_wall_clock: float,
    gaia_skills_dir: Path | None,
) -> dict:
    """Run the full R0→R{N} evolution loop for one scope (domain or global).

    Returns summary dict with performance curves, best round info, etc.
    """
    scope_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = scope_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    pipeline_eval = GAIAPipelineEvaluator(judge_provider=judge_provider)
    LEARNINGS_PATH = scope_dir / "learnings.md"

    meta_agent = MetaAgent(
        inner_model=meta_model,
        memo_path=LEARNINGS_PATH,
        extra_skills_dirs=([gaia_skills_dir] if gaia_skills_dir and gaia_skills_dir.is_dir() else None),
        max_cost_usd=evolve_cost,
        wall_clock_s=float(evolve_wall_clock),
        max_steps=evolve_steps,
    )

    current_config = original_base
    best_so_far = None
    next_evolve_status = "baseline"

    all_rounds_records: list[list[dict]] = []
    curves: list[dict] = []
    noop_streak = 0
    early_stopped = False

    for round_idx in range(num_rounds):
        is_last = round_idx == num_rounds - 1

        round_dir = scope_dir / f"R{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        traj_dir = round_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir = round_dir / "sessions"

        round_journal = HarnessJournal(base_dir=str(sessions_dir), export_jsonl=True)
        round_config = current_config.copy(tracer=round_journal)

        round_config_path = round_dir / "config.yaml"
        round_config.to_yaml_file(round_config_path)
        shutil.copy2(round_config_path, configs_dir / f"R{round_idx}_config.yaml")

        config_hash = hashlib.sha256(round_config_path.read_bytes()).hexdigest()[:16]

        logger.info("\n" + "=" * 70)
        logger.info(
            "[%s] ROUND %d/%d  config=%s  evolve_status=%s",
            scope_name,
            round_idx,
            num_rounds - 1,
            config_hash,
            next_evolve_status,
        )
        logger.info("=" * 70)

        # ── Run all tasks ─────────────────────────────────────────────────
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _run_one(task: GAIATask) -> dict:
            from dataclasses import replace as _dc_replace

            async with sem:
                task = _dc_replace(task, max_cost_usd=max_cost, max_steps=max_steps)
                harness = model_config.agentic(round_config)
                record = await _run_task(
                    harness,
                    task,
                    f"{scope_name}/R{round_idx}",
                    pipeline_eval=pipeline_eval,
                    harness_config=round_config,
                )

            raw = record.get("_result")

            if raw is not None:
                record["pivotal_tool"] = _pick_pivotal_tool(raw)
                cc, ec = _compute_tool_counts(raw)
                record["tool_call_counts"] = cc
                record["tool_error_counts"] = ec
                traj_text = _build_trajectory_text(task, raw, harness_config=round_config)
                _write_task_trajectory(traj_dir, task, traj_text, record=record)
            return record

        records = list(await asyncio.gather(*(_run_one(t) for t in tasks)))
        gc.collect()

        all_rounds_records.append(records)

        # ── Stats ─────────────────────────────────────────────────────────
        passed = sum(1 for r in records if r.get("passed"))
        round_cost_usd = sum((r.get("cost_usd") or 0) for r in records)
        round_pass_rate = round(passed / len(records), 4) if records else 0.0
        total_tokens = sum(int(r.get("total_tokens") or 0) for r in records)
        total_steps = sum(int(r.get("steps") or 0) for r in records)

        level_stats = {}
        for lvl in sorted({r.get("level") for r in records if r.get("level")}):
            lrecs = [r for r in records if r.get("level") == lvl]
            lp = sum(1 for r in lrecs if r.get("passed"))
            level_stats[f"L{lvl}"] = {
                "total": len(lrecs),
                "passed": lp,
                "pass_rate": round(lp / len(lrecs), 4) if lrecs else 0.0,
            }

        curve_point = {
            "round": round_idx,
            "config_hash": config_hash,
            "evolve_status": next_evolve_status,
            "total_tasks": len(records),
            "passed": passed,
            "pass_rate": round_pass_rate,
            "pass_pct": f"{round_pass_rate * 100:.1f}%",
            "cost_usd": round(round_cost_usd, 4),
            "total_tokens": total_tokens,
            "total_steps": total_steps,
            "level_stats": level_stats,
        }
        curves.append(curve_point)

        logger.info(
            "[%s/R%d] pass=%d/%d (%.1f%%)  cost=$%.2f  tokens=%d",
            scope_name,
            round_idx,
            passed,
            len(records),
            round_pass_rate * 100,
            round_cost_usd,
            total_tokens,
        )
        for k, v in level_stats.items():
            logger.info("  %s: %d/%d (%.1f%%)", k, v["passed"], v["total"], v["pass_rate"] * 100)

        # Save curves incrementally
        _save_curves(scope_dir, curves)

        # ── Gating ────────────────────────────────────────────────────────
        gate_decision, gate_reason, best_so_far, reverted_cfg = _score_and_gate(
            round_pass_rate=round_pass_rate,
            round_cost=round_cost_usd,
            round_idx=round_idx,
            round_config=current_config,
            round_passed=passed,
            best=best_so_far,
            tolerance=0.03,
            cost_weight=0.0,
            pass_count_noise_threshold=3,
        )
        if reverted_cfg is not None:
            logger.warning("[%s/R%d] REGRESSION — reverting to best config", scope_name, round_idx)
            current_config = reverted_cfg

        # ── Journal attribution ───────────────────────────────────────────
        if round_idx >= 1:
            _backfill_journal(
                LEARNINGS_PATH, round_idx, next_evolve_status, gate_decision, all_rounds_records, scope_dir, scope_name
            )

        if is_last:
            continue

        # ── Evolve ────────────────────────────────────────────────────────
        next_round_dir = scope_dir / f"R{round_idx + 1}"
        next_round_dir.mkdir(parents=True, exist_ok=True)
        evolve_dir = next_round_dir / "evolve"

        logger.info("[%s/R%d] evolve → %s", scope_name, round_idx, evolve_dir)

        try:
            new_yaml = await meta_agent.evolve(
                current_config=round_config_path,
                trajectories_dir=traj_dir,
                output_dir=evolve_dir,
                replay_model=model_config,
                replay_max_cost_usd=min(0.5, max_cost),
            )

            candidate_cfg = HarnessConfig.from_yaml_file(new_yaml).canonicalize()

            if round_config_path.read_bytes() == Path(new_yaml).read_bytes():
                next_evolve_status = "noop"
                noop_streak += 1
            else:
                next_evolve_status = "ok"
                noop_streak = 0

            current_config = candidate_cfg
            logger.info(
                "[%s/R%d] evolved config → %s  status=%s  noop_streak=%d",
                scope_name,
                round_idx,
                new_yaml,
                next_evolve_status,
                noop_streak,
            )

            # Early stop: 2 consecutive noops
            if noop_streak >= 2:
                logger.info("[%s] EARLY STOP: %d consecutive unchanged configs", scope_name, noop_streak)
                early_stopped = True
                break

        except Exception as exc:
            next_evolve_status = "crashed"
            noop_streak = 0
            logger.exception("[%s/R%d] evolve crashed: %s", scope_name, round_idx, exc)

    # ── Final summary ─────────────────────────────────────────────────────
    _save_curves(scope_dir, curves)
    best_idx = max(range(len(curves)), key=lambda i: (curves[i]["passed"], -curves[i]["cost_usd"]))
    _print_scope_summary(scope_name, curves, best_idx, early_stopped)

    return {
        "scope": scope_name,
        "curves": curves,
        "best_round": best_idx,
        "best_pass_rate": curves[best_idx]["pass_rate"],
        "best_passed": curves[best_idx]["passed"],
        "total_tasks": len(tasks),
        "early_stopped": early_stopped,
        "num_rounds_run": len(curves),
    }


def _save_curves(scope_dir: Path, curves: list[dict]):
    (scope_dir / "curves.json").write_text(json.dumps(curves, indent=2, ensure_ascii=False), encoding="utf-8")


def _backfill_journal(learnings_path, round_idx, evolve_status, gate_decision, all_rounds, scope_dir, scope_name):
    try:
        from harnessx.meta_harness import journal as _journal

        entries = _journal.read_entries(learnings_path)
        entry = next((e for e in entries if e.round == round_idx), None)
        if entry is None:
            return
        records = all_rounds[-1]
        prev_records = all_rounds[-2] if len(all_rounds) >= 2 else []
        prev_passed = {r["task_id"] for r in prev_records if r.get("passed")}
        prev_appeared = {r["task_id"] for r in prev_records}
        cur_passed = {r["task_id"] for r in records if r.get("passed")}
        cur_appeared = {r["task_id"] for r in records}
        outcome = "reverted" if gate_decision == "REVERTED" else "accepted"
        if evolve_status == "noop":
            outcome = "noop"
        attribution = _journal.compute_attribution(
            entry.predicted_affected,
            passed_now=cur_passed,
            passed_before=prev_passed,
            appeared_now=cur_appeared,
            appeared_before=prev_appeared,
        )
        predicted_set = set(entry.predicted_affected)
        regressed_unpredicted = sorted((prev_passed & prev_appeared & cur_appeared) - cur_passed - predicted_set)
        cs_path = scope_dir / f"R{round_idx}" / "evolve" / "_meta_scratch" / "changeset.json"
        changeset = {}
        if cs_path.is_file():
            try:
                changeset = json.loads(cs_path.read_text())
            except Exception:
                pass
        _journal.fill_gating(
            learnings_path,
            round_idx,
            outcome,
            attribution,
            extra_frontmatter={"regressed_unpredicted": regressed_unpredicted, "changeset": changeset},
        )
        logger.info("[%s/R%d] journal attribution: outcome=%s", scope_name, round_idx, outcome)
    except Exception as exc:
        logger.warning("[%s/R%d] journal backfill failed: %s", scope_name, round_idx, exc)


def _print_scope_summary(scope_name, curves, best_idx, early_stopped):
    print(f"\n{'=' * 70}")
    print(f"  {scope_name} Evolution Summary  {'(EARLY STOPPED)' if early_stopped else ''}")
    print(f"{'=' * 70}")
    print(f"  {'Round':<8} {'Pass':>6} {'Rate':>8} {'Cost':>10} {'Tokens':>12} {'Status':<10}")
    print(f"  {'-' * 8} {'-' * 6} {'-' * 8} {'-' * 10} {'-' * 12} {'-' * 10}")
    for c in curves:
        marker = " <<<BEST" if c["round"] == best_idx else ""
        print(
            f"  R{c['round']:<7d} {c['passed']:>5d}/{c['total_tasks']}"
            f"  {c['pass_pct']:>7s}"
            f"  ${c['cost_usd']:>8.2f}"
            f"  {c['total_tokens']:>11,d}"
            f"  {c['evolve_status']:<10s}{marker}"
        )
    if len(curves) > 1:
        r0 = curves[0]["pass_rate"]
        rb = curves[best_idx]["pass_rate"]
        print(f"\n  >>> R0={r0 * 100:.1f}% → R{best_idx}={rb * 100:.1f}%  Δ={100 * (rb - r0):+.1f}pp")
    print(f"{'=' * 70}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="GAIA Meta-Harness: per-domain evolution")
    parser.add_argument(
        "--domain", type=str, default=None, help="Run a single domain (short name). None=all domains sequentially."
    )
    parser.add_argument("--num-rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--meta-model", default=DEFAULT_META_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--provider-id", default=DEFAULT_PROVIDER_ID)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-cost", type=float, default=MAX_COST_USD)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--evolve-cost", type=float, default=EVOLVE_COST)
    parser.add_argument("--evolve-steps", type=int, default=EVOLVE_STEPS)
    parser.add_argument("--evolve-wall-clock", type=int, default=EVOLVE_WALL_CLOCK)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit tasks per domain (0=all).")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--list-domains", action="store_true", help="Print available domains and exit.")
    args = parser.parse_args()

    # ── Load tasks by domain category ────────────────────────────────────
    domains = _load_classified_tasks(args.data_path)

    if args.list_domains:
        print("Available domains:")
        for name, tasks in sorted(domains.items(), key=lambda kv: -len(kv[1])):
            levels = {}
            for t in tasks:
                levels[t.level] = levels.get(t.level, 0) + 1
            lvl_str = ", ".join(f"L{k}={v}" for k, v in sorted(levels.items()))
            print(f"  {name:40s}  {len(tasks):3d} tasks  ({lvl_str})")
        return

    for t_list in domains.values():
        for t in t_list:
            t.max_steps = args.max_steps

    total = sum(len(v) for v in domains.values())
    logger.info("Loaded %d tasks across %d domains", total, len(domains))
    for name, tasks in sorted(domains.items(), key=lambda kv: -len(kv[1])):
        logger.info("  %-40s %3d tasks", name, len(tasks))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_tag = args.run_tag or f"meta_gpt5_opus_{time.strftime('%Y%m%d_%H%M%S')}"
    RUN_DIR = RUNS_DIR / run_tag
    if args.clean and RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Run outputs → %s", RUN_DIR)

    # ── Providers ─────────────────────────────────────────────────────────
    provider = _make_provider(args.model, args.provider_id, api_base=args.api_base, api_key=args.api_key)
    model_config = ModelConfig(main=provider)

    judge_provider = _make_provider(args.meta_model, args.provider_id)
    meta_provider = _make_provider(
        args.meta_model, args.provider_id, extended_thinking=True, thinking_budget_tokens=32_000, max_tokens=40_000
    )
    meta_model = ModelConfig(main=meta_provider)

    # ── Baseline config ───────────────────────────────────────────────────
    original_base = make_gaia_builder_gpt5(max_cost_usd=args.max_cost).build()

    import dataclasses as _dcs
    from harnessx.core.harness import _serialize_processor
    from harnessx.processors.evaluation.llm_judge import LLMJudgeProcessor

    _initial_judge = LLMJudgeProcessor(judge_model=args.meta_model)
    _judge_dict = _serialize_processor(_initial_judge)
    if _judge_dict:
        original_base = _dcs.replace(original_base, processors=[*original_base.processors, _judge_dict])

    # ── Determine which domains to run ────────────────────────────────────
    if args.domain:
        matches = [k for k in domains if args.domain.lower() in k.lower()]
        if not matches:
            logger.error("No domain matching '%s'. Use --list-domains.", args.domain)
            return
        domains_to_run = {k: domains[k] for k in matches}
    else:
        domains_to_run = domains

    gaia_skills = _GAIA_SKILLS_DIR if _GAIA_SKILLS_DIR.is_dir() else None

    all_summaries = []

    # Sort: largest domains first (more signal for evolution)
    for domain_name in sorted(domains_to_run, key=lambda k: -len(domains_to_run[k])):
        domain_tasks = domains_to_run[domain_name]
        if args.max_tasks > 0:
            domain_tasks = domain_tasks[: args.max_tasks]
        if not domain_tasks:
            logger.warning("No tasks for domain '%s', skipping", domain_name)
            continue

        scope_dir = RUN_DIR / "domain" / domain_name
        logger.info("\n" + "#" * 70)
        logger.info("# DOMAIN: %s  (%d tasks, %d rounds)", domain_name, len(domain_tasks), args.num_rounds)
        logger.info("#" * 70)

        summary = await run_evolve_loop(
            scope_name=domain_name,
            tasks=domain_tasks,
            model_config=model_config,
            meta_model=meta_model,
            judge_provider=judge_provider,
            original_base=original_base,
            scope_dir=scope_dir,
            num_rounds=args.num_rounds,
            max_cost=args.max_cost,
            max_steps=args.max_steps,
            concurrency=args.concurrency,
            evolve_cost=args.evolve_cost,
            evolve_steps=args.evolve_steps,
            evolve_wall_clock=args.evolve_wall_clock,
            gaia_skills_dir=gaia_skills,
        )
        all_summaries.append(summary)

        (RUN_DIR / "summary.json").write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Final report ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL REPORT — All Domains")
    print("=" * 70)
    total_best_passed = 0
    total_all_tasks = 0
    for s in all_summaries:
        total_best_passed += s["best_passed"]
        total_all_tasks += s["total_tasks"]
        print(
            f"  {s['scope']:40s}  best R{s['best_round']} = {s['best_passed']}/{s['total_tasks']}"
            f" ({s['best_pass_rate'] * 100:.1f}%)"
            f"{'  [EARLY STOP]' if s['early_stopped'] else ''}"
        )
    if total_all_tasks:
        print(
            f"\n  {'AGGREGATE':40s}  {total_best_passed}/{total_all_tasks}"
            f" ({total_best_passed / total_all_tasks * 100:.1f}%)"
        )
    print("=" * 70)

    logger.info("All results → %s", RUN_DIR)


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
