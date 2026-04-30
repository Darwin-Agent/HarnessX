# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""tau2-bench Evolver: multi-round meta-harness optimization for tau2 domains.

Usage::

    # Retail domain, 3 rounds, 4 parallel simulations
    python -m recipe.tau2_evolver.run \\
        --domain retail \\
        --num-rounds 3 \\
        --run-tag my_run

    # Override models
    python -m recipe.tau2_evolver.run \\
        --domain retail \\
        --agent-model anthropic/tongyi/qwen3.5-27b \\
        --agent-api-base https://your-api-base.example.com/anthropic \\
        --user-model azure_openai/gpt-5.2 \\
        --user-api-base https://your-api-base.example.com/v1 \\
        --meta-model azure_openai/gpt-4.1 \\
        --meta-api-base https://your-api-base.example.com/v1 \\
        --num-rounds 3 --max-tasks 20

    # Bootstrap from an existing tau2 JSON report (skips R0 simulation)
    python -m recipe.tau2_evolver.run \\
        --domain retail \\
        --from-report /path/to/retail_0421_qwen3.5-27B.json \\
        --base-config /path/to/harness_config_base.yaml \\
        --num-rounds 2 \\
        --run-tag retail_from_report

How it works
------------
Round 0: run tau2 simulations with baseline harness_config.yaml.
         Or skip with ``--from-report``: load trajectories from an existing
         tau2 JSON report file (``results.json`` / report JSON) instead.
Round N: meta-agent reads trajectories from R(N-1), produces a new config.yaml.
         Simulations run with the evolved config.
         If avg_reward drops more than REGRESSION_TOLERANCE below the historical
         best, the evolved config is rejected and the previous best is reused.

Key differences from gaia_evolver
----------------------------------
- tau2 drives its own simulation loop (HalfDuplexAgent interface).
  HarnessX config is threaded in via ``llm_args_agent["harness_config"]``.
- Score = ``reward_info.reward`` (float 0–1) rather than pass/fail.
- No LLMJudgeProcessor (tau2 evaluates actions against ground truth).
- ``run_tasks()`` is synchronous; called via asyncio.to_thread().
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── load .env ────────────────────────────────────────────────────────────────
_env_path = Path(_PROJECT_ROOT) / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# Silence litellm's cost-tracker "Provider List:" / "Give Feedback" banner that
# prints on every completion when a non-standard subprovider slug (e.g.
# "azure_openai/gpt-5.2") isn't in its registry — harmless but floods stderr
# and masks real Connection errors.
import litellm as _litellm

_litellm.suppress_debug_info = True

from harnessx.meta_harness import MetaAgent
from harnessx.core.harness import HarnessConfig
from harnessx.core.model_config import ModelConfig
from harnessx.providers.anthropic_provider import AnthropicProvider
from harnessx.providers.litellm_provider import LiteLLMProvider

from .defaults import (
    DEFAULT_AGENT_API_BASE,
    DEFAULT_AGENT_EXTENDED_THINKING,
    DEFAULT_AGENT_MODEL,
    DEFAULT_AGENT_THINKING_BUDGET,
    DEFAULT_DOMAIN,
    DEFAULT_META_API_BASE,
    DEFAULT_META_MODEL,
    DEFAULT_TASK_SPLIT,
    DEFAULT_USER_API_BASE,
    DEFAULT_USER_MODEL,
    EVOLVE_COST_CAP_USD,
    EVOLVE_MAX_STEPS,
    EVOLVE_WALL_CLOCK_S,
    MAX_CONCURRENCY,
    MAX_SIM_STEPS,
    MAX_TASKS,
    NUM_ROUNDS,
    NUM_TRIALS,
    REGRESSION_TOLERANCE,
    COST_WEIGHT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
logger = logging.getLogger("tau2_evolver")

_RECIPE_DIR = Path(__file__).resolve().parent
RUNS_DIR = _RECIPE_DIR / "runs"

# Baseline config relative to benchmarks/tau2/ (resolved absolutely below)
_TAU2_BENCHMARK_DIR = Path(_PROJECT_ROOT) / "benchmarks" / "tau2"
_BASELINE_CONFIG = _TAU2_BENCHMARK_DIR / "harness_config.yaml"

# Benchmark-specific meta-agent skills. Mounted into the meta-agent's system
# prompt via ``extra_skills_dirs`` so the generic persona (under
# ``harnessx/meta_harness/workspace/``) stays benchmark-agnostic.
_TAU2_SKILLS_DIR = _RECIPE_DIR / "skills"


# ---------------------------------------------------------------------------
# Infrastructure-error detection
# ---------------------------------------------------------------------------

# Terminations that indicate the *simulation infrastructure* failed, not the
# agent. Lumping them with real agent failures poisons the meta-agent's
# reasoning: "11/11 tasks failed at step 0 with zero messages" looks like an
# agent bug to an LLM, but is really a network / thread-pool outage.
_INFRA_ERROR_TERMINATIONS: frozenset[str] = frozenset(
    {
        "infrastructure_error",
        "too_many_errors",
        "error",
    }
)

# When this fraction of tasks in a round hit infrastructure errors, treat the
# round as poisoned — do not feed it to the meta-agent, do not gate against
# it, and abort the experiment so the user can fix the proxy and rerun.
_POISON_THRESHOLD: float = 0.5


class RoundPoisonedError(RuntimeError):
    """Raised when a tau2 round has too many infrastructure errors to be
    interpretable as a signal about the agent's behaviour.

    ``records`` is attached so the caller can still write trajectory files
    (useful for post-mortem) before aborting the experiment.
    """

    def __init__(self, records: list[dict], infra_count: int, total: int) -> None:
        self.records = records
        self.infra_count = infra_count
        self.total = total
        pct = (100 * infra_count / total) if total else 0
        super().__init__(
            f"Round poisoned: {infra_count}/{total} tasks hit infrastructure "
            f"errors ({pct:.0f}%). Likely proxy / rate-limit / thread-pool "
            "failure. Trajectories preserved for post-mortem; experiment aborted."
        )


def _cleanup_http_state() -> None:
    """Best-effort cleanup of stale httpx clients between rounds.

    A broken proxy in round N can leave ``httpx.Client`` instances in a
    half-closed state and their associated thread pools in "shutdown" mode.
    Round N+1 that tries to reuse a global client inherits the broken
    state and fails with ``cannot schedule new futures after shutdown``.
    This function scans the live object graph and explicitly closes every
    ``httpx.Client`` / ``httpx.AsyncClient`` it finds, then triggers GC.

    All failures are swallowed — this is pure hygiene, never a hard
    dependency.
    """
    import gc as _gc

    try:
        import httpx
    except ImportError:
        return

    closed = 0
    for obj in _gc.get_objects():
        for cls in (
            getattr(httpx, "Client", None),
            getattr(httpx, "AsyncClient", None),
        ):
            if cls is not None and isinstance(obj, cls):
                try:
                    close = getattr(obj, "close", None)
                    if callable(close):
                        close()
                        closed += 1
                except Exception:  # noqa: BLE001
                    pass
    if closed:
        logger.info("[cleanup] closed %d httpx client instances", closed)
    _gc.collect()


# ---------------------------------------------------------------------------
# Provider helper
# ---------------------------------------------------------------------------


def _make_provider(
    model: str,
    api_base: str | None,
    *,
    extended_thinking: bool = False,
    thinking_budget_tokens: int = 10_000,
    max_tokens: int = 8192,
):
    """Create a model provider for the meta-agent.

    Matches ``recipe/gaia_evolver/run.py:_make_provider`` in behaviour:
    Anthropic-routed models get the real ``AnthropicProvider`` (which
    honours ``extended_thinking`` / ``thinking_budget_tokens`` /
    ``max_tokens``); everything else falls back to ``LiteLLMProvider``
    where those kwargs are no-ops.
    """
    if model.startswith("anthropic/"):
        model_name = model[len("anthropic/") :]
        return AnthropicProvider(
            model=model_name,
            base_url=api_base,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            extended_thinking=extended_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            max_tokens=max_tokens,
        )
    return LiteLLMProvider(model, extra_headers={"X-Model-Provider-Id": "YOUR_PROVIDER_ID"})


# ---------------------------------------------------------------------------
# tau2 registration helpers
# ---------------------------------------------------------------------------


def _register_tau2_agents() -> None:
    """Register HarnessXAgent and StopGuardUserSimulator with tau2's registry."""
    from tau2.registry import registry
    from benchmarks.tau2.agent import create_harnessx_agent
    from benchmarks.tau2.stop_guard import StopGuardUserSimulator

    try:
        registry.register_agent_factory(create_harnessx_agent, "harnessx")
    except (ValueError, KeyError):
        pass
    try:
        registry.register_user(StopGuardUserSimulator, "harnessx_stop_guard")
    except (ValueError, KeyError):
        pass


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def _run_tau2_round(
    *,
    domain: str,
    task_split: str,
    tasks: list[Any],
    round_config_path: Path,
    sessions_dir: Path,
    agent_model: str,
    agent_api_base: str | None,
    agent_extended_thinking: bool,
    agent_thinking_budget: int,
    user_model: str,
    user_api_base: str | None,
    user_temperature: float | None,
    agent_temperature: float | None,
    judge_model: str | None,
    judge_api_base: str | None,
    num_trials: int,
    max_steps: int,
    max_concurrency: int,
    report_path: Path,
) -> list[dict]:
    """Run one round of tau2 simulations (synchronous, runs in thread).

    Returns a list of per-task records with keys:
        task_id, reward, termination_reason, num_messages,
        tool_call_counts, tool_error_counts, cost_usd, elapsed_s
    """
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import run_tasks
    import tau2.evaluator.evaluator_nl_assertions as _nl_eval

    # Point NL assertions judge at our endpoint (avoids needing a raw OpenAI key)
    if judge_model:
        _nl_eval.DEFAULT_LLM_NL_ASSERTIONS = judge_model
        _nl_eval.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
            "temperature": 0.0,
            "api_base": judge_api_base or agent_api_base,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }

    agent_llm_args: dict = {
        "api_base": agent_api_base,
        # Absolute path — pathlib's / operator keeps it absolute, so the
        # agent's `Path(__file__).parent / harness_config` resolves correctly.
        "harness_config": str(round_config_path),
        "logs_dir": str(sessions_dir),
        "extended_thinking": agent_extended_thinking,
        "thinking_budget_tokens": agent_thinking_budget,
        "request_timeout": 120.0,
    }
    user_llm_args: dict = {
        "api_base": user_api_base,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if user_temperature is not None:
        user_llm_args["temperature"] = user_temperature
    if agent_temperature is not None:
        agent_llm_args["temperature"] = agent_temperature

    config = TextRunConfig(
        domain=domain,
        agent="harnessx",
        user="harnessx_stop_guard",
        llm_agent=agent_model,
        llm_args_agent=agent_llm_args,
        llm_user=user_model,
        llm_args_user=user_llm_args,
        num_trials=num_trials,
        max_steps=max_steps,
        max_concurrency=max_concurrency,
    )

    # Remove stale report so run_tasks does not prompt "resume?" interactively
    if report_path.exists():
        report_path.unlink()

    # Guard: tau2's runner can raise mid-batch (proxy outage, thread pool
    # shutdown, internal assertion) — we do NOT want that to propagate past
    # this function because the caller has no way to distinguish "tau2
    # crashed" from "agent did badly". Substitute an empty-simulations
    # stand-in so the fallback loop below generates per-task "no simulation
    # run" records, which then trip the poison check and trigger an abort
    # that points at the infrastructure root cause.
    try:
        batch_results = run_tasks(
            config,
            tasks,
            save_path=report_path,
            evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
            console_display=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[round] tau2.run_tasks() raised — treating whole round as infrastructure failure: %s",
            exc,
        )

        class _EmptyBatch:
            simulations: list = []

        batch_results = _EmptyBatch()

    # Aggregate per-task records
    from collections import defaultdict

    sims_by_task: dict[str, list] = defaultdict(list)
    for sim in batch_results.simulations:
        sims_by_task[sim.task_id].append(sim)

    records: list[dict] = []
    for task in tasks:
        sims = sims_by_task.get(task.id, [])
        if not sims:
            records.append(
                {
                    "task_id": task.id,
                    "reward": 0.0,
                    "termination_reason": "error",
                    "eval_reason": "termination=error",
                    "num_messages": 0,
                    "steps": 0,
                    "total_tokens": 0,
                    "final_output_length": 0,
                    "pivotal_tool": "",
                    "error_count": 0,
                    "db_match": None,
                    "failed_actions": [],
                    "expected_actions": [],
                    "failed_nl_assertions": [],
                    "executed_actions": [],
                    "judge_cause": "no simulation run — agent crashed before any step",
                    "tool_call_counts": {},
                    "tool_error_counts": {},
                    "cost_usd": 0.0,
                    "elapsed_s": 0.0,
                    "_sims": [],
                }
            )
            continue

        rewards = [s.reward_info.reward if s.reward_info else 0.0 for s in sims]
        avg_reward = sum(rewards) / len(rewards)
        last_sim = sims[-1]
        term = last_sim.termination_reason.value if last_sim.termination_reason is not None else "unknown"
        num_msgs = sum(len(s.messages) for s in sims if s.messages)
        total_cost = sum((s.agent_cost or 0.0) + (s.user_cost or 0.0) for s in sims)
        total_elapsed = sum(s.duration for s in sims)

        # Tool counts from last simulation's messages
        last_messages = last_sim.messages or []
        tool_call_counts, tool_error_counts = _extract_tool_counts(last_messages)

        # Evaluator signals: decompose reward_info into the four fields the
        # meta-agent reads off frontmatter. We mirror gaia's judge_* shape so
        # the meta-agent's read pattern stays benchmark-agnostic.
        ri = last_sim.reward_info
        ev = _extract_evaluator_signals(ri, is_dict=False)
        judge_cause = _derive_judge_cause(
            reward=avg_reward,
            db_match=ev["db_match"],
            failed_actions=ev["failed_actions"],
            failed_nl_assertions=ev["failed_nl_assertions"],
        )

        # Behaviour fields aligned with gaia_evolver/run.py frontmatter.
        steps = _count_assistant_turns(last_messages)
        total_tokens = _sum_tokens_from_sims(sims)
        final_output_length = _last_assistant_content_length(last_messages)
        pivotal_tool = max(tool_call_counts, key=tool_call_counts.get) if tool_call_counts else ""
        error_count = sum(tool_error_counts.values())
        executed_actions = _executed_actions_from_messages(last_messages)

        records.append(
            {
                "task_id": task.id,
                "reward": round(avg_reward, 4),
                "termination_reason": term,
                "eval_reason": _derive_eval_reason(term, avg_reward, ev["failed_actions"], ev["db_match"]),
                "num_messages": num_msgs,
                "steps": steps,
                "total_tokens": total_tokens,
                "final_output_length": final_output_length,
                "pivotal_tool": pivotal_tool,
                "error_count": error_count,
                # Evaluator decomposition (mirrors gaia judge_* fields).
                "db_match": ev["db_match"],
                "failed_actions": ev["failed_actions"],
                "expected_actions": ev["expected_actions"],
                "failed_nl_assertions": ev["failed_nl_assertions"],
                "executed_actions": executed_actions,
                "judge_cause": judge_cause,
                "tool_call_counts": tool_call_counts,
                "tool_error_counts": tool_error_counts,
                "cost_usd": round(total_cost, 5),
                "elapsed_s": round(total_elapsed, 2),
                "reward_info": ri,  # popped before render
                "_sims": sims,
            }
        )

    # ── Poison check ───────────────────────────────────────────────────────
    # Do NOT silently pass a round dominated by infrastructure errors to the
    # evolve loop. The meta-agent reading "all tasks failed at step 0" will
    # confidently recommend the wrong fix (see real incident 2026-04-24).
    infra_count = sum(1 for r in records if r.get("termination_reason") in _INFRA_ERROR_TERMINATIONS)
    if records and (infra_count / len(records)) >= _POISON_THRESHOLD:
        raise RoundPoisonedError(records=records, infra_count=infra_count, total=len(records))

    # Warn (not raise) when cost metadata is missing — proxy endpoints that
    # don't surface usage/cost info turn cost-weighted gating into a no-op.
    # The meta-agent should still see a hint that the signal is unreliable.
    if records and all((r.get("cost_usd") or 0) == 0 for r in records):
        logger.warning(
            "[round] all %d tasks reported cost_usd=0 — proxy endpoint is "
            "not returning usage/cost metadata; cost-weighted gating is "
            "effectively disabled this round",
            len(records),
        )

    return records


# Tool-name prefixes that identify *read* tools. Mirrors benchmarks/tau2/tool_filter.py
# so `executed_actions` below only contains mutation calls (the ones the
# evaluator's action_checks scores), not information-gathering reads.
_READ_PREFIXES = ("get_", "list_", "search_", "find_", "check_", "look_", "think")


def _is_read_tool(name: str) -> bool:
    return any(name.startswith(p) for p in _READ_PREFIXES)


def _mget(m: Any, *keys: str, default: Any = None) -> Any:
    """Read the first non-None value for any of ``keys`` from ``m`` (object or dict)."""
    for k in keys:
        if isinstance(m, dict):
            if k in m and m[k] is not None:
                return m[k]
        else:
            v = getattr(m, k, None)
            if v is not None:
                return v
    return default


def _sum_tokens_from_sims(sims: list) -> int:
    """Best-effort total-token extraction from tau2 SimulationRun objects.

    tau2 field naming has shifted across versions; probe a few common shapes
    (``usage.total_tokens``, ``usage.input_tokens + output_tokens``,
    per-message ``usage``). Returns 0 when nothing matches rather than raising.
    """
    total = 0
    for s in sims or []:
        u = getattr(s, "usage", None)
        if u:
            tt = getattr(u, "total_tokens", None)
            if tt:
                total += int(tt)
                continue
            it = getattr(u, "input_tokens", 0) or 0
            ot = getattr(u, "output_tokens", 0) or 0
            if it or ot:
                total += int(it) + int(ot)
                continue
        for m in getattr(s, "messages", None) or []:
            mu = getattr(m, "usage", None)
            if not mu:
                continue
            tt = getattr(mu, "total_tokens", None) or (
                (getattr(mu, "input_tokens", 0) or 0) + (getattr(mu, "output_tokens", 0) or 0)
            )
            if tt:
                total += int(tt)
    return total


def _sum_tokens_from_json(sim: dict) -> int:
    """Token extraction from a JSON-report simulation dict. Same fallback ladder."""
    u = sim.get("usage") or {}
    tt = u.get("total_tokens")
    if tt:
        return int(tt)
    it = u.get("input_tokens") or 0
    ot = u.get("output_tokens") or 0
    if it or ot:
        return int(it) + int(ot)
    total = 0
    for m in sim.get("messages") or []:
        mu = (m.get("usage") or {}) if isinstance(m, dict) else {}
        tt = mu.get("total_tokens")
        if tt:
            total += int(tt)
        else:
            total += int(mu.get("input_tokens") or 0) + int(mu.get("output_tokens") or 0)
    return total


def _last_assistant_content_length(messages: list) -> int:
    for m in reversed(messages or []):
        if _mget(m, "role") == "assistant":
            c = _mget(m, "content", default="") or ""
            if isinstance(c, str):
                return len(c)
            return len(str(c))
    return 0


def _count_assistant_turns(messages: list) -> int:
    return sum(1 for m in messages or [] if _mget(m, "role") == "assistant")


def _executed_actions_from_messages(messages: list) -> list[str]:
    """Ordered list of action (non-read) tool calls the agent actually made."""
    out: list[str] = []
    for m in messages or []:
        if _mget(m, "role") != "assistant":
            continue
        for tc in _mget(m, "tool_calls", default=[]) or []:
            name = _mget(tc, "name", default="") or ""
            if name and not _is_read_tool(name):
                out.append(name)
    return out


def _extract_evaluator_signals(ri: Any, is_dict: bool) -> dict:
    """Pull db_match / failed_actions / expected_actions / failed_nl from reward_info.

    Accepts either a live ``RewardInfo`` object (``is_dict=False``) or a dict
    loaded from the JSON report (``is_dict=True``). Returns a dict with the
    four signals; empty / None values are normalised.
    """
    if ri is None:
        return {
            "db_match": None,
            "failed_actions": [],
            "expected_actions": [],
            "failed_nl_assertions": [],
        }

    if is_dict:
        db = (ri.get("db_check") or {}) if isinstance(ri, dict) else {}
        ac = ri.get("action_checks") or []
        nl = ri.get("nl_assertions") or ri.get("communicate_checks") or []
        db_match = db.get("db_match")
        expected = [(a.get("action") or {}).get("name", "?") for a in ac]
        failed_actions = [(a.get("action") or {}).get("name", "?") for a in ac if not a.get("action_match", True)]
        failed_nl = [
            (a.get("nl_assertion") or a.get("assertion") or "")
            for a in nl
            if not a.get("met", a.get("communicate_check_reward", 1.0) >= 1.0)
        ]
    else:
        db = getattr(ri, "db_check", None)
        ac = getattr(ri, "action_checks", None) or []
        nl = getattr(ri, "nl_assertions", None) or getattr(ri, "communicate_checks", None) or []
        db_match = getattr(db, "db_match", None) if db is not None else None
        expected = [_mget(getattr(a, "action", None), "name", default="?") for a in ac]
        failed_actions = [
            _mget(getattr(a, "action", None), "name", default="?") for a in ac if not getattr(a, "action_match", True)
        ]
        failed_nl = [
            _mget(a, "nl_assertion", "assertion", default="")
            for a in nl
            if not getattr(
                a,
                "met",
                getattr(a, "communicate_check_reward", 1.0) >= 1.0,
            )
        ]

    return {
        "db_match": db_match,
        "failed_actions": [x for x in failed_actions if x],
        "expected_actions": [x for x in expected if x],
        "failed_nl_assertions": [str(x).strip() for x in failed_nl if x],
    }


def _derive_judge_cause(
    *,
    reward: float,
    db_match: Any,
    failed_actions: list[str],
    failed_nl_assertions: list[str],
) -> str:
    """One-line hypothesis aligned with gaia's ``judge_cause`` field.

    Picks the first component that failed in the (DB → actions → NL) chain
    so the meta-agent can immediately tell which lever to reach for.
    """
    if reward >= 1.0:
        return ""
    if db_match is False:
        return "db_mismatch — final database state diverges from ground truth"
    if failed_actions:
        shown = ", ".join(failed_actions[:2])
        suffix = f" (+{len(failed_actions) - 2} more)" if len(failed_actions) > 2 else ""
        return f"action_mismatch — expected actions not called / wrong args: {shown}{suffix}"
    if failed_nl_assertions:
        first = failed_nl_assertions[0]
        return f"nl_assertion_failed — {first[:140]}"
    return f"reward={reward:.3f} with no failed component reported"


def _derive_eval_reason(
    termination: str,
    reward: float,
    failed_actions: list[str],
    db_match: Any = None,
) -> str:
    """Build a short ``eval_reason`` string aligned with SOUL.md's schema.

    Combines ``termination_reason`` with the first few failed action names
    and a ``db_mismatch`` flag so the meta-agent can read a single line
    in frontmatter and know roughly why the round dropped points. Keeps
    the richer details (full ``reward_info``) in the trajectory body.
    """
    parts: list[str] = []
    if termination:
        parts.append(f"termination={termination}")
    if reward >= 1.0:
        return "; ".join(parts) if parts else "pass"
    if db_match is False:
        parts.append("db_mismatch")
    if failed_actions:
        shown = ", ".join(failed_actions[:3])
        if len(failed_actions) > 3:
            shown += f", +{len(failed_actions) - 3} more"
        parts.append(f"failed_actions=[{shown}]")
    return "; ".join(parts) if parts else f"reward={reward:.3f}"


def _extract_tool_counts(messages: list) -> tuple[dict, dict]:
    """Count tool calls and errors from a tau2 message list."""
    call_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    for msg in messages:
        role = getattr(msg, "role", None)
        if role == "assistant":
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                name = getattr(tc, "name", "unknown")
                call_counts[name] = call_counts.get(name, 0) + 1
        elif role == "tool":
            # Detect error responses: content contains "Error" or starts with error markers
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str) and (content.startswith("Error") or '"error"' in content.lower()):
                # Associate with the most recent tool name — scan backwards
                for prev in reversed(messages[: messages.index(msg)]):
                    if getattr(prev, "role", None) == "assistant":
                        for tc in getattr(prev, "tool_calls", None) or []:
                            name = getattr(tc, "name", "unknown")
                            error_counts[name] = error_counts.get(name, 0) + 1
                        break
    return call_counts, error_counts


# ---------------------------------------------------------------------------
# Bootstrap from existing JSON report
# ---------------------------------------------------------------------------


def _records_from_json_report(report_path: Path) -> list[dict]:
    """Convert a tau2 JSON report file into the same record format that
    ``_run_tau2_round`` returns.

    Supports the standard tau2 ``Results`` JSON layout::

        { "simulations": [ { "task_id": ..., "messages": [...], ... }, ... ] }

    One record is emitted per unique task_id (last trial wins when num_trials > 1).
    """
    data = json.loads(report_path.read_text(encoding="utf-8"))
    sims_raw: list[dict] = data.get("simulations", [])

    # Last trial per task_id
    by_task: dict[str, dict] = {}
    for sim in sims_raw:
        tid = str(sim.get("task_id", "unknown"))
        by_task[tid] = sim  # later entries overwrite earlier trials

    records: list[dict] = []
    for tid, sim in by_task.items():
        ri = sim.get("reward_info") or {}
        reward = float(ri.get("reward", 0.0))
        messages: list[dict] = sim.get("messages") or []

        # tool counts from message list (dicts, not objects)
        call_counts: dict[str, int] = {}
        error_counts: dict[str, int] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    name = tc.get("name", "unknown")
                    call_counts[name] = call_counts.get(name, 0) + 1
        # tool errors: tool messages whose content looks like an error
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool":
                content = str(msg.get("content") or "")
                if content.startswith("Error") or '"error"' in content.lower():
                    for prev in reversed(messages[:i]):
                        if prev.get("role") == "assistant":
                            for tc in prev.get("tool_calls") or []:
                                name = tc.get("name", "unknown")
                                error_counts[name] = error_counts.get(name, 0) + 1
                            break

        # Evaluator decomposition — dict variant. Same shape as live path so
        # the renderer treats both sources uniformly.
        term = sim.get("termination_reason", "unknown") or "unknown"
        ev = _extract_evaluator_signals(ri, is_dict=True)
        judge_cause = _derive_judge_cause(
            reward=reward,
            db_match=ev["db_match"],
            failed_actions=ev["failed_actions"],
            failed_nl_assertions=ev["failed_nl_assertions"],
        )
        steps = _count_assistant_turns(messages)
        final_output_length = _last_assistant_content_length(messages)
        pivotal_tool = max(call_counts, key=call_counts.get) if call_counts else ""
        error_count = sum(error_counts.values())
        total_tokens = _sum_tokens_from_json(sim)
        executed_actions = _executed_actions_from_messages(messages)

        records.append(
            {
                "task_id": tid,
                "reward": round(reward, 4),
                "termination_reason": term,
                "eval_reason": _derive_eval_reason(term, reward, ev["failed_actions"], ev["db_match"]),
                "num_messages": len(messages),
                "steps": steps,
                "total_tokens": total_tokens,
                "final_output_length": final_output_length,
                "pivotal_tool": pivotal_tool,
                "error_count": error_count,
                "db_match": ev["db_match"],
                "failed_actions": ev["failed_actions"],
                "expected_actions": ev["expected_actions"],
                "failed_nl_assertions": ev["failed_nl_assertions"],
                "executed_actions": executed_actions,
                "judge_cause": judge_cause,
                "tool_call_counts": call_counts,
                "tool_error_counts": error_counts,
                "cost_usd": round((sim.get("agent_cost") or 0.0) + (sim.get("user_cost") or 0.0), 5),
                "elapsed_s": round(sim.get("duration") or 0.0, 2),
                "_reward_info": ri,
                "_messages": messages,
            }
        )

    logger.info(
        "[from-report] loaded %d task records from %s  avg_reward=%.4f",
        len(records),
        report_path.name,
        sum(r["reward"] for r in records) / len(records) if records else 0.0,
    )
    return records


def _write_task_trajectory_from_json(
    traj_dir: Path,
    record: dict,
    *,
    harness_config: Any | None = None,
) -> None:
    """Write trajectory .md for a record built from a JSON report."""
    traj_dir.mkdir(parents=True, exist_ok=True)
    tid = record["task_id"]
    messages = record.pop("_messages", [])
    record.pop("_reward_info", None)  # already decomposed into record fields
    fm = _render_trajectory_frontmatter(record)
    body = _render_trajectory_body_unified(messages, record=record, harness_config=harness_config)
    (traj_dir / f"{tid}.md").write_text(f"{fm}\n\n{body.lstrip()}", encoding="utf-8")


# ---------------------------------------------------------------------------
# Trajectory rendering
# ---------------------------------------------------------------------------


def _render_trajectory_frontmatter(record: dict) -> str:
    """YAML frontmatter for a tau2 task trajectory (meta-agent readable).

    Emits both tau2-native keys (``reward`` / ``termination_reason`` /
    ``num_messages``) and the SOUL.md-canonical ``eval_*`` aliases
    (``eval_passed`` / ``eval_score`` / ``eval_reason``), so the generic
    read pattern documented in the meta-agent's persona works on tau2
    trajectories without any benchmark-specific branching. The playbook
    skill spells out what the tau2-native fields mean; the eval_* aliases
    cover the generic workflow.
    """
    import json as _json

    def _scalar(v: Any) -> str:
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

    tcc = record.get("tool_call_counts") or {}
    tec = record.get("tool_error_counts") or {}
    reward = float(record.get("reward") or 0.0)
    term = record.get("termination_reason") or ""
    # eval_reason may be absent on very old records — fall back to term.
    eval_reason = record.get("eval_reason") or (f"termination={term}" if term else "")

    # Behaviour tier (always present — gaia-parity names).
    fields: list[tuple[str, Any]] = [
        ("task_id", record.get("task_id") or "unknown"),
        ("exit_reason", term),
        ("steps", int(record.get("steps") or 0)),
        ("num_messages", int(record.get("num_messages") or 0)),
        ("total_tokens", int(record.get("total_tokens") or 0)),
        ("cost_usd", float(record.get("cost_usd") or 0.0)),
        ("elapsed_s", float(record.get("elapsed_s") or 0.0)),
        ("final_output_length", int(record.get("final_output_length") or 0)),
        ("pivotal_tool", record.get("pivotal_tool") or ""),
        ("error_count", int(record.get("error_count") or 0)),
        ("tools_used", sorted(tcc.keys())),
        ("tool_call_counts", tcc),
        ("tool_error_counts", tec),
    ]

    # Eval tier — always emitted. Matches SOUL.md's "authoritative
    # ground-truth outcome" contract: the meta-agent should see the truth
    # signal on every task, not only the failing ones.
    #
    # tau2's ground truth is an *action sequence* (tau2 grades by DB state +
    # called-action match), not a single answer string. `expected_actions`
    # is therefore tau2's analogue of gaia's `final_answer` (what should
    # have happened), and `executed_actions` is the analogue of gaia's
    # `extracted_answer` (what the agent actually did). Diffing the two
    # lists is the fastest way for the meta-agent to tell whether the
    # agent missed an action, ran an extra one, or got the order wrong.
    expected_actions = record.get("expected_actions") or []
    executed_actions = record.get("executed_actions") or []
    fields.extend(
        [
            ("expected_actions", expected_actions),  # ← ground truth
            ("executed_actions", executed_actions),  # ← what agent did
            ("reward", reward),
            ("eval_passed", reward >= 1.0),
            ("eval_score", reward),
            ("eval_reason", eval_reason),
        ]
    )

    # Failure-blame tier — only emitted when something failed, so passing
    # tasks keep a slim `Read limit=30` frontmatter. These surface *which
    # component* of tau2's three-part evaluator failed (DB / actions / NL),
    # which maps directly to the intervention lever the meta-agent should
    # reach for.
    if reward < 1.0 or record.get("failed_actions") or record.get("failed_nl_assertions"):
        db_match = record.get("db_match")
        failed_actions = record.get("failed_actions") or []
        failed_nl = record.get("failed_nl_assertions") or []
        judge_cause = record.get("judge_cause") or ""
        fields.extend(
            [
                ("db_match", db_match),
                ("failed_actions", failed_actions),
                ("failed_nl_assertions", failed_nl[:3]),
                ("judge_cause", judge_cause),
            ]
        )

    lines = ["---"]
    for k, v in fields:
        lines.append(f"{k}: {_scalar(v)}")
    lines.append("---")
    return "\n".join(lines)


def _render_trajectory_body_unified(
    messages: list,
    *,
    record: dict,
    harness_config: Any | None = None,
) -> str:
    """Render a gaia-parity 5-section trajectory body.

    Sections (in order):
      1. Task         — first user turn, plain text so the meta-agent reads
                        the problem statement without reconstructing it.
      2. Result       — reward / db_match / failed_actions / eval_reason.
      3. Harness Config — processors executed this round, inline so the
                        meta-agent can orient before deciding what to change.
      4. Diagnostics  — steps / messages / cost / error-rate / top tools.
      5. Execution Steps — numbered ``### Step N`` blocks so the meta-agent
                        can cite specific steps (``R2 step 7``) the way
                        SOUL.md step 3 asks it to.

    ``messages`` may be either a list of tau2 SimulationRun message objects
    (live path) or a list of dicts (``--from-report`` path). Both are handled
    via :func:`_mget` duck-typing so the renderer is source-agnostic.
    """
    lines: list[str] = []

    # ── 1. Task ────────────────────────────────────────────────────────────
    first_user = next(
        (m for m in (messages or []) if _mget(m, "role") == "user"),
        None,
    )
    if first_user is not None:
        task_text = _mget(first_user, "content", default="") or ""
        if not isinstance(task_text, str):
            task_text = str(task_text)
        lines.append("## Task\n")
        lines.append(task_text.strip())
        lines.append("")

    # ── 2. Result ──────────────────────────────────────────────────────────
    reward = float(record.get("reward") or 0.0)
    lines.append(f"## Result  reward={reward:.3f}")
    db_match = record.get("db_match")
    if db_match is not None:
        lines.append(f"db_match={db_match}")
    failed_actions = record.get("failed_actions") or []
    if failed_actions:
        lines.append(f"failed_actions: {', '.join(failed_actions)}")
    failed_nl = record.get("failed_nl_assertions") or []
    if failed_nl:
        shown = "; ".join(a[:120] for a in failed_nl[:3])
        lines.append(f"failed_nl_assertions: {shown}")
    expected = record.get("expected_actions") or []
    executed = record.get("executed_actions") or []
    if expected or executed:
        lines.append(f"expected_actions: {expected}")
        lines.append(f"executed_actions: {executed}")
    judge_cause = record.get("judge_cause") or ""
    if judge_cause:
        lines.append(f"judge_cause: {judge_cause}")
    lines.append(f"eval_reason: {record.get('eval_reason', '')}")
    lines.append("")

    # ── 3. Harness Config ──────────────────────────────────────────────────
    if harness_config is not None:
        lines.append("## Harness Config\n")
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
                label = getattr(entry, "_singleton_group", "") or type(entry).__name__
                order = getattr(entry, "_order", "?")
                proc_parts.append(f"{label}({order})")
        for p in getattr(harness_config, "_rt_procs", None) or []:
            label = getattr(p, "_singleton_group", "") or type(p).__name__
            order = getattr(p, "_order", "?")
            proc_parts.append(f"{label}({order})")
        # dedupe while keeping order
        seen: set[str] = set()
        deduped = [p for p in proc_parts if not (p in seen or seen.add(p))]
        if deduped:
            lines.append(f"Processors: {', '.join(deduped)}")
        lines.append("")

    # ── 4. Diagnostics ─────────────────────────────────────────────────────
    tcc = record.get("tool_call_counts") or {}
    tec = record.get("tool_error_counts") or {}
    total_calls = sum(tcc.values())
    total_errs = sum(tec.values())
    err_rate = f"{100 * total_errs / total_calls:.0f}%" if total_calls else "0%"
    steps = int(record.get("steps") or 0)
    num_msgs = int(record.get("num_messages") or 0)
    total_tokens = int(record.get("total_tokens") or 0)
    cost = float(record.get("cost_usd") or 0.0)
    elapsed = float(record.get("elapsed_s") or 0.0)
    term = record.get("termination_reason") or "?"

    lines.append("## Diagnostics\n")
    lines.append(f"- agent_steps: {steps}  (total messages: {num_msgs})")
    lines.append(f"- tokens: {total_tokens}")
    lines.append(f"- cost: ${cost:.4f}")
    lines.append(f"- elapsed: {elapsed:.1f}s")
    lines.append(f"- tool_calls: {total_calls}, errors: {total_errs} (error_rate={err_rate})")
    if tcc:
        top = sorted(tcc.items(), key=lambda x: -x[1])[:5]
        lines.append(f"- top_tools: {', '.join(f'{n}({c})' for n, c in top)}")
    if tec:
        err_parts = [f"{n}({c})" for n, c in tec.items() if c]
        lines.append(f"- tool_error_counts: {', '.join(err_parts)}")
    lines.append(f"- exit_reason: {term}")
    lines.append("")

    # ── 5. Execution Steps ─────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Execution Steps\n")
    step_id = 0
    pending_user: str | None = None
    for msg in messages or []:
        role = _mget(msg, "role")
        if role == "system":
            continue

        if role == "user":
            # Stash until we render an assistant step — keeps every agent
            # turn paired with the user prompt that triggered it.
            content = _mget(msg, "content", default="") or ""
            pending_user = content if isinstance(content, str) else str(content)
            continue

        if role == "assistant":
            step_id += 1
            lines.append(f"### Step {step_id}\n")
            if pending_user:
                lines.append(f"**User:** {pending_user.strip()}\n")
                pending_user = None
            content = _mget(msg, "content", default="") or ""
            if content:
                text = content if isinstance(content, str) else str(content)
                lines.append(f"#### Response\n\n{text.strip()}\n")
            tool_calls = _mget(msg, "tool_calls", default=[]) or []
            if tool_calls:
                lines.append("#### Tool Calls\n")
                for tc in tool_calls:
                    name = _mget(tc, "name", default="?") or "?"
                    args = _mget(tc, "arguments", default={})
                    try:
                        args_repr = json.dumps(args, ensure_ascii=False)
                    except (TypeError, ValueError):
                        args_repr = str(args)
                    lines.append(f"- **{name}**(`{args_repr}`)")
            continue

        if role == "tool":
            # Inline tool results under the most recent assistant step.
            results = _mget(msg, "results", default=None)
            if results:
                for tr in results:
                    name = _mget(tr, "requestor", "tool_name", "name", default="?") or "?"
                    tr_content = _mget(tr, "content", default="") or ""
                    snippet = str(tr_content).strip().replace("\n", " ")[:400]
                    lines.append(f"  -> **{name}**: {snippet}")
            else:
                content = _mget(msg, "content", default="") or ""
                snippet = str(content).strip().replace("\n", " ")[:400]
                if snippet:
                    lines.append(f"  -> tool: {snippet}")

    # Trailing user-only turn (conversation ended on user feedback with no
    # follow-up agent step) — still visible so the meta-agent sees how the
    # dialogue actually closed.
    if pending_user:
        step_id += 1
        lines.append(f"\n### Step {step_id} — user final turn\n")
        lines.append(pending_user.strip())

    return "\n".join(lines)


def _write_task_trajectory(
    traj_dir: Path,
    task_id: str,
    record: dict,
    sims: list,
    *,
    harness_config: Any | None = None,
) -> None:
    """Write a live-sim trajectory .md file."""
    traj_dir.mkdir(parents=True, exist_ok=True)
    fm = _render_trajectory_frontmatter(record)
    messages = (sims[-1].messages if sims else None) or []
    body = _render_trajectory_body_unified(messages, record=record, harness_config=harness_config)
    (traj_dir / f"{task_id}.md").write_text(f"{fm}\n\n{body.lstrip()}", encoding="utf-8")


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


def _score_and_gate(
    *,
    round_reward: float,
    round_cost: float,
    round_idx: int,
    round_config: Any,
    best: tuple[float, float, Any, int] | None,
    tolerance: float,
    cost_weight: float,
) -> tuple[str, str, tuple, Any | None]:
    """Gate evolved config against historical best.

    Returns (decision, reason, new_best, reverted_config_or_None).
    ``decision`` is "accept" | "reject".
    """
    adjusted = round_reward - cost_weight * max(round_cost - (best[1] if best else 0.0), 0.0)

    if best is None:
        new_best = (round_reward, round_cost, round_config, round_idx)
        return "accept", "first round — establishing baseline", new_best, None

    best_reward, best_cost, best_cfg, best_idx = best
    best_adjusted = best_reward - cost_weight * 0.0

    if adjusted >= best_adjusted - tolerance:
        new_best = (round_reward, round_cost, round_config, round_idx) if round_reward > best_reward else best
        return "accept", f"reward {round_reward:.4f} ≥ best {best_reward:.4f} − tol {tolerance}", new_best, None
    else:
        reason = f"reward {round_reward:.4f} < best {best_reward:.4f} − tol {tolerance} → revert to R{best_idx}"
        return "reject", reason, best, best_cfg


# ---------------------------------------------------------------------------
# Multi-round comparison table
# ---------------------------------------------------------------------------


def _print_comparison(rounds: list[list[dict]]) -> None:
    if not rounds:
        return
    n_rounds = len(rounds)
    task_ids = [r["task_id"] for r in rounds[0]]
    n_tasks = len(task_ids)

    def _pct_delta(cur: float, prev: float) -> str:
        if prev == 0:
            return "n/a"
        d = (cur - prev) / prev * 100
        return f"{d:+.1f}%"

    lines: list[str] = []
    lines.append(f"tau2 Evolver — Multi-Round Comparison  ({n_rounds} rounds × {n_tasks} tasks)")
    lines.append("")

    TID_W, VAL_W, D_W = 20, 12, 10
    hdr = f"  {'task_id':<{TID_W}}"
    for i in range(n_rounds):
        hdr += f" | {f'R{i} reward':^{VAL_W}}"
    if n_rounds > 1:
        hdr += f" | {'vs-R0':^{D_W}}"
    lines.append(hdr)
    sep = f"  {'-' * TID_W}"
    for _ in range(n_rounds):
        sep += f"-+-{'-' * VAL_W}"
    if n_rounds > 1:
        sep += f"-+-{'-' * D_W}"
    lines.append(sep)

    for tid in task_ids:
        rec0 = next((r for r in rounds[0] if r["task_id"] == tid), {})
        row = f"  {tid[:TID_W]:<{TID_W}}"
        for i in range(n_rounds):
            rec = next((r for r in rounds[i] if r["task_id"] == tid), {})
            rw = rec.get("reward", 0.0)
            row += f" | {rw:^{VAL_W}.4f}"
        if n_rounds > 1:
            rec_last = next((r for r in rounds[-1] if r["task_id"] == tid), {})
            delta = rec_last.get("reward", 0.0) - rec0.get("reward", 0.0)
            row += f" | {delta:>+{D_W}.4f}"
        lines.append(row)
    lines.append("")

    # Round totals
    totals = []
    for rd in rounds:
        rewards = [r.get("reward", 0.0) for r in rd]
        totals.append(
            {
                "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "cost": sum(r.get("cost_usd", 0.0) or 0.0 for r in rd),
            }
        )

    LBL_W = 14
    lines.append("Round totals")
    hdr2 = f"  {'metric':<{LBL_W}}"
    for i in range(n_rounds):
        hdr2 += f" | {f'R{i}':^{VAL_W}}"
        if i > 0:
            hdr2 += f"  {'Δ':^{D_W}}"
    lines.append(hdr2)
    sep2 = f"  {'-' * LBL_W}"
    for i in range(n_rounds):
        sep2 += f"-+-{'-' * VAL_W}"
        if i > 0:
            sep2 += f"--{'-' * D_W}"
    lines.append(sep2)

    def _row(label: str, v_fn, d_fn) -> str:
        line = f"  {label:<{LBL_W}}"
        for i, t in enumerate(totals):
            line += f" | {v_fn(t):^{VAL_W}}"
            if i > 0:
                line += f"  {d_fn(t, totals[i - 1]):^{D_W}}"
        return line

    lines.append(
        _row(
            "avg_reward",
            lambda t: f"{t['avg_reward']:.4f}",
            lambda cur, prev: f"{cur['avg_reward'] - prev['avg_reward']:+.4f}",
        )
    )
    lines.append(
        _row(
            "cost_usd",
            lambda t: f"${t['cost']:.3f}",
            lambda cur, prev: _pct_delta(cur["cost"], prev["cost"]),
        )
    )

    if n_rounds > 1 and n_tasks > 0:
        r0 = totals[0]["avg_reward"]
        rN = totals[-1]["avg_reward"]
        lines.append("")
        lines.append(f"  >>> avg_reward: {r0:.4f} → {rN:.4f} over {n_rounds} rounds  ({rN - r0:+.4f})")

    width = max(80, max(len(ln) for ln in lines if ln))
    print("\n" + "=" * width)
    for ln in lines:
        print(ln)
    print("=" * width)


# ---------------------------------------------------------------------------
# Memo helpers
# ---------------------------------------------------------------------------


def _append_gating_note(
    *,
    memo_path: Path,
    round_idx: int,
    decision: str,
    reason: str,
    avg_reward: float,
    round_cost: float,
    config_path: Path,
    baseline_round: int | None,
    baseline_reward: float | None,
) -> None:
    note = (
        f"\n## R{round_idx} gating: {decision.upper()}\n"
        f"- avg_reward: {avg_reward:.4f}  cost: ${round_cost:.3f}\n"
        f"- config: `{config_path}`\n"
        f"- reason: {reason}\n"
    )
    if baseline_round is not None and baseline_reward is not None:
        note += f"- best_so_far: R{baseline_round} avg_reward={baseline_reward:.4f}\n"
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    with memo_path.open("a", encoding="utf-8") as f:
        f.write(note)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="tau2-bench Evolver: multi-round meta-harness optimization")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="tau2 domain")
    parser.add_argument("--task-split", default=DEFAULT_TASK_SPLIT)
    parser.add_argument("--max-tasks", type=int, default=MAX_TASKS)
    parser.add_argument(
        "--task-ids",
        default=None,
        metavar="ID1,ID2,...",
        help="Comma-separated task IDs to run. Overrides --max-tasks when set.",
    )
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--max-sim-steps", type=int, default=MAX_SIM_STEPS)
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--num-rounds", type=int, default=NUM_ROUNDS)

    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--agent-api-base", default=DEFAULT_AGENT_API_BASE)
    parser.add_argument("--extended-thinking", action="store_true", default=DEFAULT_AGENT_EXTENDED_THINKING)
    parser.add_argument("--thinking-budget", type=int, default=DEFAULT_AGENT_THINKING_BUDGET)

    parser.add_argument("--user-model", default=DEFAULT_USER_MODEL)
    parser.add_argument("--user-api-base", default=DEFAULT_USER_API_BASE)
    parser.add_argument(
        "--user-temperature",
        type=float,
        default=None,
        metavar="T",
        help="Sampling temperature for the user simulator (default: model default)",
    )
    parser.add_argument(
        "--agent-temperature",
        type=float,
        default=None,
        metavar="T",
        help="Sampling temperature for the agent (default: model default)",
    )

    parser.add_argument("--meta-model", default=DEFAULT_META_MODEL)
    parser.add_argument("--meta-api-base", default=DEFAULT_META_API_BASE)

    parser.add_argument("--judge-model", default=None, help="NL assertions judge model (defaults to --user-model)")
    parser.add_argument("--judge-api-base", default=None)

    parser.add_argument("--evolve-cost", type=float, default=EVOLVE_COST_CAP_USD)
    parser.add_argument("--evolve-steps", type=int, default=EVOLVE_MAX_STEPS)
    parser.add_argument("--evolve-wall-clock", type=int, default=EVOLVE_WALL_CLOCK_S)
    parser.add_argument("--regression-tolerance", type=float, default=REGRESSION_TOLERANCE)
    parser.add_argument("--cost-weight", type=float, default=COST_WEIGHT)

    parser.add_argument("--run-tag", default=None, help="Output tag. Defaults to domain_YYYYMMDD_HHMMSS.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Wipe runs/{run_tag}/ before starting, so a repeated run with the "
            "same --run-tag starts from a clean slate instead of interleaving "
            "with prior output. Only affects THIS run-tag, never other runs. "
            "Without --clean, re-using an existing run-tag prints a warning "
            "and keeps the old files around."
        ),
    )
    parser.add_argument("--base-config", default=str(_BASELINE_CONFIG), help="Path to baseline harness_config.yaml")
    parser.add_argument(
        "--from-report",
        default=None,
        metavar="REPORT_JSON",
        help=(
            "Path to an existing tau2 JSON report file. "
            "When set, Round 0 loads trajectories from this file instead of "
            "running new simulations. Requires --num-rounds >= 2 to be useful "
            "(R0 = existing data, R1+ = new simulations with evolved config)."
        ),
    )
    args = parser.parse_args()

    # ── Run directory ──────────────────────────────────────────────────────
    if args.run_tag is None:
        import datetime

        args.run_tag = f"{args.domain}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    RUN_DIR = RUNS_DIR / args.run_tag
    if args.clean and RUN_DIR.exists():
        import shutil as _shutil

        _shutil.rmtree(RUN_DIR)
        logger.info("Cleaned %s (--clean)", RUN_DIR)
    elif RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        logger.warning(
            "--run-tag %r already exists and is non-empty; new output will be "
            "interleaved with prior-run data. Pass --clean to wipe it first.",
            args.run_tag,
        )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LEARNINGS_PATH = RUN_DIR / "learnings.md"

    logger.info("Run dir: %s", RUN_DIR)
    logger.info("Domain: %s  split: %s  rounds: %d", args.domain, args.task_split, args.num_rounds)

    # ── tau2 setup ─────────────────────────────────────────────────────────
    _register_tau2_agents()

    from tau2.runner.helpers import get_tasks as _get_tasks

    tasks = _get_tasks(
        args.domain,
        task_split_name=args.task_split,
        num_tasks=args.max_tasks,
    )
    if args.task_ids:
        wanted = {tid.strip() for tid in args.task_ids.split(",") if tid.strip()}
        tasks = [t for t in tasks if str(t.id) in wanted]
        missing = wanted - {str(t.id) for t in tasks}
        if missing:
            logger.warning("task IDs not found in split: %s", ", ".join(sorted(missing)))
    logger.info("Loaded %d tasks from domain=%s split=%s", len(tasks), args.domain, args.task_split)

    # ── Meta-agent model ───────────────────────────────────────────────────
    # Outer loop runs on a stronger tier than the inner task agent. When the
    # meta model routes through Anthropic, we also turn on extended thinking
    # — architectural reasoning over trajectories benefits materially from
    # the thinking budget. On non-Anthropic routes the kwargs are no-ops
    # inside _make_provider. Mirrors recipe/gaia_evolver/run.py:515-522.
    meta_provider = _make_provider(
        args.meta_model,
        args.meta_api_base,
        extended_thinking=True,
        thinking_budget_tokens=32_000,
        max_tokens=40_000,
    )
    meta_model = ModelConfig(main=meta_provider)

    # One MetaAgent per run — persistent config (model, memo, budgets,
    # tau2 playbook skills) lives on the instance; per-round args
    # (current_config, trajectories_dir, output_dir) are passed to
    # .evolve() inside the loop below.
    meta_agent = MetaAgent(
        inner_model=meta_model,
        memo_path=LEARNINGS_PATH,
        extra_skills_dirs=([_TAU2_SKILLS_DIR] if _TAU2_SKILLS_DIR.is_dir() else None),
        max_cost_usd=args.evolve_cost,
        wall_clock_s=float(args.evolve_wall_clock),
        max_steps=args.evolve_steps,
    )

    # ── Baseline config ────────────────────────────────────────────────────
    base_config_path = Path(args.base_config).resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    current_config = HarnessConfig.from_yaml_file(base_config_path)
    current_config.canonicalize()
    logger.info("Baseline config: %s", base_config_path)

    best_so_far: tuple[float, float, Any, int] | None = None
    next_evolve_status: str = "baseline"
    all_rounds: list[list[dict]] = []
    round_summaries: list[dict] = []

    for round_idx in range(args.num_rounds):
        is_last = round_idx == args.num_rounds - 1

        round_dir = RUN_DIR / f"R{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        traj_dir = round_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir = round_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Write the config that will execute this round (for reproducibility
        # and so the meta-agent can Read it on the next iteration).
        round_config_path = round_dir / "config.yaml"
        current_config.to_yaml_file(round_config_path)

        config_label = "baseline" if round_idx == 0 else f"evolved_R{round_idx}"
        logger.info("\n" + "=" * 60)
        logger.info("ROUND %d/%d  [%s]", round_idx, args.num_rounds - 1, config_label)
        logger.info("=" * 60)

        # ── Run tau2 simulations (or load from existing report) ───────────
        report_path = round_dir / "report.json"
        t0 = time.time()

        use_existing = round_idx == 0 and args.from_report is not None
        use_cached = not use_existing and report_path.exists()

        # Flip to True when this round's simulation was poisoned (too many
        # infrastructure errors). We still write the partial trajectories
        # so the user can post-mortem, but then break the loop before
        # calling meta_harness.evolve() — an LLM reading zero-step garbage
        # will hallucinate the wrong fix.
        abort_after_this_round: bool = False

        if use_existing:
            existing_report = Path(args.from_report).resolve()
            logger.info("[R0] loading trajectories from %s (skipping simulation)", existing_report)
            records = _records_from_json_report(existing_report)
            if args.task_ids:
                wanted_ids = {str(t.id) for t in tasks}
                records = [r for r in records if str(r["task_id"]) in wanted_ids]
                logger.info("[from-report] filtered to %d tasks by --task-ids", len(records))
            # Copy the source report into the round dir for reproducibility
            import shutil as _shutil

            _shutil.copy2(existing_report, report_path)
        elif use_cached:
            logger.info("[R%d] report.json already exists — skipping simulation, loading cached results", round_idx)
            records = _records_from_json_report(report_path)
        else:
            # NOTE: we intentionally do NOT call _cleanup_http_state() here.
            # Previous incarnation force-closed every httpx.Client found in the
            # GC graph, which includes litellm/openai SDK's module-level client
            # shared across rounds — closing it made every subsequent request
            # fail with "Connection error." The original motivation (proxy
            # outages leaving half-closed clients) is now prevented at the
            # shell layer by `unset http_proxy ...` in run.sh.
            try:
                records = await asyncio.to_thread(
                    _run_tau2_round,
                    domain=args.domain,
                    task_split=args.task_split,
                    tasks=tasks,
                    round_config_path=round_config_path,
                    sessions_dir=sessions_dir,
                    agent_model=args.agent_model,
                    agent_api_base=args.agent_api_base,
                    agent_extended_thinking=args.extended_thinking,
                    agent_thinking_budget=args.thinking_budget,
                    user_model=args.user_model,
                    user_api_base=args.user_api_base,
                    user_temperature=args.user_temperature,
                    agent_temperature=args.agent_temperature,
                    judge_model=args.judge_model or args.user_model,
                    judge_api_base=args.judge_api_base or args.user_api_base,
                    num_trials=args.num_trials,
                    max_steps=args.max_sim_steps,
                    max_concurrency=args.max_concurrency,
                    report_path=report_path,
                )
            except RoundPoisonedError as exc:
                logger.error(
                    "[R%d] %s — writing partial trajectories for post-mortem, "
                    "then aborting experiment (no evolve will run).",
                    round_idx,
                    exc,
                )
                records = exc.records
                abort_after_this_round = True

        elapsed_round = time.time() - t0
        logger.info(
            "[R%d] %s done in %.1fs",
            round_idx,
            "report loaded" if (use_existing or use_cached) else "simulations",
            elapsed_round,
        )

        # ── Write trajectories ─────────────────────────────────────────────
        # Pass the config that *executed* this round so the Harness Config
        # section of each trajectory body records the processor pipeline
        # the meta-agent is about to reason over. When R0 loads from
        # --from-report, we still pass current_config so the meta-agent
        # sees the same baseline it would have run with.
        #
        # Per-record try/except: a write failure for one task must not
        # drop the other 20 tasks' trajectories silently (as happened in
        # the 2026-04-24 retail_evolve_subset run — 11/21 .md on disk,
        # no traceback logged).
        written = 0
        write_failures: list[tuple[str, str]] = []
        for rec in records:
            tid = rec.get("task_id", "?")
            try:
                if "_messages" in rec:
                    _write_task_trajectory_from_json(traj_dir, rec, harness_config=current_config)
                else:
                    sims = rec.pop("_sims", [])
                    rec.pop("reward_info", None)
                    _write_task_trajectory(traj_dir, tid, rec, sims, harness_config=current_config)
                written += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[R%d] failed to write trajectory for task_id=%s: %s",
                    round_idx,
                    tid,
                    exc,
                )
                write_failures.append((str(tid), f"{type(exc).__name__}: {exc}"))
        if written != len(records):
            logger.warning(
                "[R%d] wrote %d/%d trajectory files; %d failures: %s",
                round_idx,
                written,
                len(records),
                len(write_failures),
                ", ".join(f"{tid}({err[:60]})" for tid, err in write_failures[:5]),
            )
        gc.collect()

        all_rounds.append(records)

        rewards = [r.get("reward", 0.0) for r in records]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        round_cost = sum(r.get("cost_usd", 0.0) or 0.0 for r in records)

        round_summaries.append(
            {
                "round": round_idx,
                "config": config_label,
                "tasks": len(records),
                "avg_reward": round(avg_reward, 4),
                "total_cost_usd": round(round_cost, 4),
                "evolve_status": next_evolve_status,
            }
        )
        logger.info(
            "[R%d] avg_reward=%.4f  cost=$%.3f  tasks=%d",
            round_idx,
            avg_reward,
            round_cost,
            len(records),
        )

        # ── Gating ────────────────────────────────────────────────────────
        gate_decision, gate_reason, best_so_far, reverted_cfg = _score_and_gate(
            round_reward=avg_reward,
            round_cost=round_cost,
            round_idx=round_idx,
            round_config=current_config,
            best=best_so_far,
            tolerance=args.regression_tolerance,
            cost_weight=args.cost_weight,
        )
        if reverted_cfg is not None:
            logger.warning(
                "[R%d] REGRESSION (%s) — reverting to R%d config",
                round_idx,
                gate_reason,
                best_so_far[3],
            )
            current_config = reverted_cfg

        if not is_last:
            _append_gating_note(
                memo_path=LEARNINGS_PATH,
                round_idx=round_idx,
                decision=gate_decision,
                reason=gate_reason,
                avg_reward=avg_reward,
                round_cost=round_cost,
                config_path=round_config_path,
                baseline_round=best_so_far[3] if best_so_far else None,
                baseline_reward=best_so_far[0] if best_so_far else None,
            )

        if is_last:
            continue

        # Abort after a poisoned round: trajectories written, gating note
        # written (so post-mortem has "R{N} gating: ACCEPT/REJECT — …
        # POISONED" in learnings.md), but no evolve. The whole experiment
        # stops here; remaining rounds are skipped.
        if abort_after_this_round:
            logger.error(
                "[R%d] experiment aborted after poisoned round — skipping "
                "evolve and all remaining rounds. Fix the proxy / rate-limit "
                "issue and rerun with a fresh --run-tag.",
                round_idx,
            )
            break

        # ── Evolve: produce next round's config ───────────────────────────
        next_round_dir = RUN_DIR / f"R{round_idx + 1}"
        next_round_dir.mkdir(parents=True, exist_ok=True)
        evolve_dir = next_round_dir / "evolve"
        logger.info("[R%d] evolve → %s", round_idx, evolve_dir)

        # Skip evolve if config.yaml was already produced (e.g. manually fixed
        # after a previous crash or regression).
        _existing_yaml = evolve_dir / "config.yaml"
        if _existing_yaml.is_file():
            logger.info("[R%d] evolve skipped — using existing %s", round_idx, _existing_yaml)
            candidate = HarnessConfig.from_yaml_file(_existing_yaml)
            candidate.canonicalize()
            next_evolve_status = "ok"
            current_config = candidate
        else:
            try:
                new_yaml = await meta_agent.evolve(
                    current_config=round_config_path,
                    trajectories_dir=traj_dir,
                    output_dir=evolve_dir,
                )
                candidate = HarnessConfig.from_yaml_file(new_yaml)
                candidate.canonicalize()

                if round_config_path.read_bytes() == Path(new_yaml).read_bytes():
                    next_evolve_status = "noop"
                else:
                    next_evolve_status = "ok"
                logger.info(
                    "[R%d] → R%d config: %s  (status=%s)",
                    round_idx,
                    round_idx + 1,
                    new_yaml,
                    next_evolve_status,
                )
                current_config = candidate

            except Exception as exc:  # noqa: BLE001
                next_evolve_status = "crashed"
                logger.exception("[R%d] evolve crashed — R%d reuses current config: %s", round_idx, round_idx + 1, exc)

        # Cooldown after evolve: the meta-agent (Claude opus) bursts many
        # requests; starting the next round immediately triggers "Request rate
        # increased too quickly" 400s on the same endpoint. 60s is enough for
        # the rate-limit window to reset.
        _EVOLVE_COOLDOWN_S = 60
        logger.info("[R%d] post-evolve cooldown %ds before next round …", round_idx, _EVOLVE_COOLDOWN_S)
        await asyncio.sleep(_EVOLVE_COOLDOWN_S)

    # ── Final report ───────────────────────────────────────────────────────
    _print_comparison(all_rounds)

    results_path = RUN_DIR / "comparison.json"
    results_path.write_text(
        json.dumps(
            {
                "rounds": [
                    {
                        "summary": round_summaries[i],
                        "records": [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_rounds[i]],
                    }
                    for i in range(len(all_rounds))
                ],
                "meta": {
                    "domain": args.domain,
                    "task_split": args.task_split,
                    "agent_model": args.agent_model,
                    "num_rounds": args.num_rounds,
                    "run_tag": args.run_tag,
                    "evolve_cost": args.evolve_cost,
                    "evolve_steps": args.evolve_steps,
                    "evolve_wall_clock": args.evolve_wall_clock,
                    "regression_tolerance": args.regression_tolerance,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Results written to %s", results_path)


if __name__ == "__main__":
    asyncio.run(main())
