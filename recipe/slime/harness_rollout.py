# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample
from slime.utils.http_utils import post as _slime_post

from harnessx.core.model_config import ModelConfig

from recipe.slime.harness import make_slime_harness
from recipe.slime.registry import load_harness_config
from recipe.slime.formats.slime_format import SlimeRLFormat
from harnessx.providers.sglang import (
    SGLangProvider as SlimeSGLangProvider,
    build_flat_sequence,
    validate_token_annotation_consistency,
)

# Per-sample wall-clock timeout (seconds) for Harness.run().
# Prevents any single sample from blocking the entire rollout batch indefinitely.
# With max_steps=5 and python_timeout=30s, a normal episode finishes well under
# this budget.  Only pathological cases (SGLang queue stall, deadlocked tool)
# would hit this.
_HARNESS_RUN_TIMEOUT_S = float(os.environ.get("HARNESSX_SAMPLE_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# generate — Slime custom generate() interface
# ---------------------------------------------------------------------------


async def generate(args: Any, sample: Sample, sampling_params: dict) -> Sample:
    """
    Multi-turn agent generation using HarnessX Harness.run().

    Implements Slime's custom generate() interface.

    Flow:
        1. Load SlimeConfigSpec for this sample's task_type
        2. Build RLTask (via spec.task_builder.build(sample))
        3. Build SlimeSGLangProvider (captures token IDs + logprobs per turn)
        4. Build HarnessConfig (processors: SystemPrompt, TokenBudget, RLSignal, EpisodeMetrics, Eval)
        5. await ModelConfig(main=provider).agentic(config).run(task)
           - SystemPromptProcessor sets system prompt once (task_start)
           - TokenBudgetProcessor enforces a hard context-window safety guard
           - SlimeSGLangProvider.complete() called each turn → StepCapture saved
           - Tool execution via InMemoryToolRegistry (code_interpreter for math)
           - EvaluationProcessor fires on task_end → eval_result on TaskEndEvent
           - Harness.run() auto-backfills eval_result.reward into all traj steps
           - Harness.run() calls provider.annotate_trajectory(traj) → populates
             TrajectoryStep.token_annotation for every step (via backfill_token_annotations)
        6. traj.to_rl_records(SlimeRLFormat(tokenizer)) → episode dict
           (fallback: build_flat_sequence(provider) if annotations incomplete)
        7. Fill sample.tokens / loss_mask / rollout_log_probs / response

    Training correctness invariants:
        len(loss_mask) == len(rollout_log_probs) == response_length
        step_captures[t].input_ids == apply_chat_template(full_context_at_t)

    Args:
        args:            Slime training args (sglang_router_ip, sglang_router_port,
                         rollout_max_context_len, context_parallel_size,
                         max_tokens_per_gpu, partial_rollout, etc.)
        sample:          Slime Sample (.prompt, .label, .metadata)
        sampling_params: SGLang sampling parameters

    Returns:
        Sample with .tokens, .loss_mask, .rollout_log_probs, .response filled.
    """
    assert not getattr(args, "partial_rollout", False), "Partial rollout is not supported for harness_rollout."

    try:
        return await _generate_inner(args, sample, sampling_params)
    except Exception:
        logger.exception(
            "generate(): unhandled exception for sample %s — marking ABORTED",
            getattr(sample, "id", "?"),
        )
        sample.status = Sample.Status.ABORTED
        return sample


async def _generate_inner(args: Any, sample: Sample, sampling_params: dict) -> Sample:
    """Inner generate logic, separated so the outer generate() can catch all exceptions."""

    # ── 1. Config + Task ─────────────────────────────────────────────────────
    spec = load_harness_config(sample)
    task = spec.task_builder.build(sample)
    task.max_steps = spec.max_steps

    # ── 2. Provider ──────────────────────────────────────────────────────────
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    if args.rollout_max_context_len is not None:
        max_ctx = args.rollout_max_context_len
    else:
        max_ctx = args.context_parallel_size * args.max_tokens_per_gpu

    message_formatter = spec.formatter_factory(state.tokenizer) if spec.formatter_factory is not None else None
    inter_turn_formatter = (
        spec.inter_turn_formatter_factory(state.tokenizer) if spec.inter_turn_formatter_factory is not None else None
    )
    provider = SlimeSGLangProvider(
        url=url,
        tokenizer=state.tokenizer,
        sampling_params=sampling_params,
        rollout_max_context_len=max_ctx,
        message_formatter=message_formatter,
        inter_turn_formatter=inter_turn_formatter,
        post_fn=_slime_post,  # use Slime's process-level client: trust_env=False,
        # Timeout(None), 60-retry — matches sglang_rollout.py behaviour
    )

    # ── 3. HarnessConfig (per-run, evaluator freshly instantiated) ───────────
    config = make_slime_harness(spec, provider, task)

    # ── 4-7. Run Harness + fill sample ───────────────────────────────────────
    # provider holds a persistent httpx.AsyncClient; always close it when done
    # (success, truncation, abort, or unexpected exception).
    try:
        # ── 4. Run Harness ────────────────────────────────────────────────────
        # Note: run_loop catches ALL provider exceptions internally (ContextLengthExceeded,
        # GenerationAborted, etc.) and returns with exit_reason="error".
        # We detect truncation/abort via provider.truncated / provider.aborted flags set
        # BEFORE the exceptions are raised in SlimeSGLangProvider.complete().
        t_harness_start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                ModelConfig(main=provider).agentic(config).run(task),
                timeout=_HARNESS_RUN_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            harness_run_ms = (time.monotonic() - t_harness_start) * 1000
            logger.warning(
                "generate(): Harness.run() timed out after %.0fs for sample %s (had %d captures) — marking TRUNCATED",
                _HARNESS_RUN_TIMEOUT_S,
                getattr(sample, "id", "?"),
                len(provider.step_captures),
            )
            # If we have partial captures, try to salvage the sample
            if provider.step_captures:
                if _fill_sample_from_captures(sample, provider, state.tokenizer, traj=None, result=None):
                    sample.status = Sample.Status.TRUNCATED
                    sample.metadata = dict(sample.metadata or {})
                    sample.metadata["harness_run_ms"] = harness_run_ms
                    sample.metadata["timeout"] = True
                    return sample
            sample.status = Sample.Status.ABORTED
            return sample
        harness_run_ms = (time.monotonic() - t_harness_start) * 1000

        traj = result.trajectory
        # Harness.run() auto-backfills eval_result into all traj.steps AND calls
        # provider.annotate_trajectory(traj) which populates token_annotations.

        # ── 5. Guard: no captures → aborted before first model call ──────────
        if not provider.step_captures:
            sample.status = Sample.Status.ABORTED
            return sample

        # ── 6. Build flat sequence from StepCapture side-channel (primary path)
        t_fill_start = time.monotonic()
        if not _fill_sample_from_captures(sample, provider, state.tokenizer, traj=traj, result=result):
            sample.status = Sample.Status.ABORTED
            return sample
        sample_fill_ms = (time.monotonic() - t_fill_start) * 1000

        # ── 7. Set sample status from Harness exit reason ─────────────────────
        exit_reason = result.task_end.exit_reason
        last_finish = provider.step_captures[-1].finish_reason if provider.step_captures else "stop"
        if provider.aborted or exit_reason == "interrupted":
            sample.status = Sample.Status.ABORTED
        elif provider.truncated or exit_reason in ("budget_exceeded", "loop_detected") or last_finish == "length":
            # loop_detected: RLSignalCollectorProcessor raised LoopDetectedError
            # — episode was forcibly cut short, not a natural completion.
            sample.status = Sample.Status.TRUNCATED
        else:
            sample.status = Sample.Status.COMPLETED

        # ── 8. Timing diagnostics → sample.metadata for wandb ────────────────
        sample.metadata = dict(sample.metadata or {})
        sample.metadata["harness_run_ms"] = harness_run_ms
        sample.metadata["sample_fill_ms"] = sample_fill_ms
        # Cumulative SGLang /generate time across all turns
        sample.metadata["sglang_inference_ms"] = sum(
            getattr(cap, "inference_ms", 0.0) for cap in provider.step_captures
        )

    finally:
        await provider.aclose()

    return sample


def _fill_sample_from_captures(
    sample: Sample,
    provider: SlimeSGLangProvider,
    tokenizer: Any,
    traj: Any,
    result: Any,
) -> bool:
    """Fill sample.tokens / loss_mask / rollout_log_probs from trajectory or captures.

    Returns True on success, False when the sample should be marked ABORTED
    (FlatSequence invariant broken in the fallback path).

    Primary path (trajectory-centric):
        traj.to_rl_records(SlimeRLFormat(tokenizer)) — uses TokenAnnotation per step,
        populated by provider.annotate_trajectory() which Harness.run() calls after
        backfill_rewards().  Produces model-only response text (loss_mask=1 tokens).

    Fallback path (StepCapture side-channel):
        build_flat_sequence(provider) — used when traj is None or annotations
        incomplete (e.g. partial annotation failure).  Warns to logs.

    Validation (HARNESSX_VALIDATE_TOKENS=1, dev/test only):
        Runs build_flat_sequence and validate_token_annotation_consistency to assert
        both paths produce identical token sequences.
    """
    if traj is not None and traj.has_token_annotations():
        episode = traj.to_rl_records(SlimeRLFormat(tokenizer=tokenizer))
        sample.tokens = episode["tokens"]
        sample.loss_mask = episode["loss_mask"]
        sample.rollout_log_probs = episode["rollout_log_probs"]
        sample.response_length = episode["response_length"]
        sample.response = episode["response"]
    else:
        # Fallback: StepCapture path (traj missing or annotation incomplete)
        logger.warning(
            "Token annotations incomplete (traj=%s has_annotations=%s) — falling back to build_flat_sequence",
            traj is not None,
            traj.has_token_annotations() if traj is not None else "n/a",
        )
        flat = build_flat_sequence(provider)
        if len(flat.response_ids) != len(flat.loss_mask) or len(flat.response_ids) != len(flat.rollout_logprobs):
            logger.error(
                "FlatSequence invariant broken in fallback path: resp=%d mask=%d logps=%d — marking sample ABORTED",
                len(flat.response_ids),
                len(flat.loss_mask),
                len(flat.rollout_logprobs),
            )
            return False
        sample.tokens = flat.prompt_ids + flat.response_ids
        sample.loss_mask = flat.loss_mask
        sample.rollout_log_probs = flat.rollout_logprobs
        sample.response_length = len(flat.response_ids)
        sample.response = tokenizer.decode(flat.response_ids, skip_special_tokens=False)

    # Validate both paths agree (dev mode only)
    if os.environ.get("HARNESSX_VALIDATE_TOKENS") and traj is not None:
        flat_ref = build_flat_sequence(provider)
        validate_token_annotation_consistency(traj, flat_ref)

    # ── Free O(n²) token storage — no longer needed after sample.tokens is filled ──
    # step_captures.input_ids grow O(t) per step → total O(n²) across n steps.
    # token_annotation.prompt_ids are the same data duplicated in trajectory form.
    # reward_func() only reads step.reward, step.observation, and token_annotation.token_reward
    # (set later in reward_func itself), so clearing these is safe.
    provider.step_captures.clear()
    if traj is not None:
        for step in traj.steps:
            if step.token_annotation is not None:
                step.token_annotation.prompt_ids = []

    # Save trajectory + eval_result in metadata for reward_func()
    if traj is not None:
        sample.metadata = dict(sample.metadata or {})
        sample.metadata["trajectory"] = traj
        if result is not None:
            sample.metadata["eval_result"] = result.task_end.eval_result
            sample.metadata["exit_reason"] = result.task_end.exit_reason
            sample.metadata["total_steps"] = result.task_end.total_steps
            sample.metadata["total_tokens"] = result.task_end.total_tokens

    return True


# ---------------------------------------------------------------------------
# reward_func — Slime custom reward function
# ---------------------------------------------------------------------------


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> dict:
    """
    Compute reward for a completed rollout.  Task-agnostic entry point.

    Reads:
    - sample.metadata["trajectory"]  → StatefulTrajectory (from generate())
    - sample.metadata["eval_result"] → EvalResult (terminal ±1)
    - sample.metadata["exit_reason"] → str ("done" | "loop_detected" | "budget_exceeded")

    Returns dict with:
    Core (Slime GRPO):
    - "score"              → scalar reward (args.reward_key = "score")

    Reward decomposition (wandb):
    - "terminal"           → raw evaluator score before PRM
    - "prm_adjusted"       → score after PRM shaping, before extra_reward_fn delta

    Episode diagnostics (wandb):
    - "tool_success_count" → total successful tool executions
    - "tool_error_count"   → total failed tool executions
    - "tool_use_rate"      → fraction of steps with ≥1 tool call
    - "total_steps"        → total harness steps
    - "total_tokens"       → total tokens consumed
    - "response_length"    → response token count (model+tool tokens)
    - "exit_reason"        → "done" | "loop_detected" | "budget_exceeded"
    - "trajectory_quality" → "complete" | "truncated" | "aborted"

    Task-specific fields (from spec.extra_reward_fn, merged into return dict):
    - math: "format_score", "pred", "has_boxed_answer"
    - other tasks: whatever extra_reward_fn returns (minus "score_delta")

    Slime extracts args.reward_key = "score" as the scalar for GRPO.
    All other keys are logged to wandb rollout/rewards.
    """
    if not isinstance(sample, Sample):
        raise TypeError(f"Expected Sample, got {type(sample)}")

    spec = load_harness_config(sample)
    metadata = sample.metadata or {}
    traj = metadata.get("trajectory")
    eval_result = metadata.get("eval_result")
    exit_reason = metadata.get("exit_reason", "done")

    # ── Terminal reward ────────────────────────────────────────────────────────
    terminal: float = float(eval_result.reward) if eval_result is not None else -1.0

    # ── PRM scoring ───────────────────────────────────────────────────────────
    step_rewards = await spec.prm.score_steps(traj, exit_reason=exit_reason)

    # is_terminal_only PRMs (RetoolCompatPRM, NullPRM) already return the
    # final adjusted scalar in every step — use step_rewards[-1] directly.
    # prm.aggregate() must NOT be called: it would double-count tool bonus.
    if spec.prm.is_terminal_only:
        prm_adjusted = step_rewards[-1] if step_rewards else terminal
    else:
        prm_adjusted = spec.prm.aggregate(terminal, step_rewards)

    # ── Task-specific extra reward (delegated to spec.extra_reward_fn) ────────
    # extra_reward_fn(sample, eval_result, traj) -> dict:
    #   "score_delta": float  — added to prm_adjusted to form the final score
    #   any other keys        — merged into the return dict for wandb logging
    # Example (math): returns "format_score", "pred", "has_boxed_answer"
    extra: dict = {}
    if spec.extra_reward_fn is not None:
        _extra_result = spec.extra_reward_fn(sample, eval_result, traj)
        if inspect.isawaitable(_extra_result):
            _extra_result = await _extra_result
        extra = dict(_extra_result or {})
    score_delta: float = extra.pop("score_delta", 0.0)
    score: float = prm_adjusted + score_delta

    # ── Episode-level stats ────────────────────────────────────────────────────
    tool_success_count = 0
    tool_error_count = 0
    tool_turn_count = 0
    total_steps = metadata.get("total_steps", 0)
    total_tokens = metadata.get("total_tokens", 0)

    if traj and traj.steps:
        for step in traj.steps:
            if step.observation:
                tool_turn_count += 1
            for obs in step.observation:
                # Two error channels:
                #   obs.error — set when tool registry raises an exception
                #   obs.result — code_interpreter embeds all errors as "Error: ..." strings
                #                (never raises, so obs.error is None for sandbox errors)
                is_error = bool(obs.error) or (isinstance(obs.result, str) and obs.result.startswith("Error:"))
                if is_error:
                    tool_error_count += 1
                else:
                    tool_success_count += 1

    tool_use_rate: float = tool_turn_count / max(total_steps, 1)

    response_length: int = getattr(sample, "response_length", 0) or 0

    # Trajectory quality: complete / truncated / aborted
    if exit_reason in ("interrupted", "error"):
        trajectory_quality = "aborted"
    elif (
        exit_reason in ("budget_exceeded", "loop_detected")
        or getattr(sample, "status", None) == Sample.Status.TRUNCATED
    ):
        trajectory_quality = "truncated"
    else:
        trajectory_quality = "complete"

    # ── Backfill token_reward into TokenAnnotation for offline data export ─────
    # Writes the PRM step score into each step's token_annotation.token_reward
    # so trajectory store exports and offline analysis have per-step reward labels.
    # Must run before the trajectory is popped from metadata below.
    if traj is not None and traj.has_token_annotations() and step_rewards:
        for i, step in enumerate(traj.steps):
            if i < len(step_rewards) and step.token_annotation is not None:
                step.token_annotation.token_reward = step_rewards[i]

    # ── Release heavy objects from metadata to reduce memory pressure ─────
    # trajectory is the heaviest object (full state snapshots per step);
    # lightweight scalars (exit_reason, total_steps, etc.) are kept.
    metadata.pop("trajectory", None)
    metadata.pop("eval_result", None)

    return {
        # ── Core reward (GRPO) ─────────────────────────────────────────────
        "score": score,
        # ── Reward decomposition ───────────────────────────────────────────
        "terminal": terminal,
        "prm_adjusted": prm_adjusted,
        # ── Episode diagnostics ─────────────────────────────────────────────
        "tool_success_count": tool_success_count,
        "tool_error_count": tool_error_count,
        "tool_use_rate": tool_use_rate,
        "total_steps": total_steps,
        "total_tokens": total_tokens,
        "response_length": response_length,
        "exit_reason": exit_reason,
        "trajectory_quality": trajectory_quality,
        # ── Correctness diagnostics (for PG loss debugging) ──────────────────
        # is_correct: 1 if terminal >= 0, 0 otherwise — for pass@k estimation
        # When is_correct is near 0 across all n_samples_per_prompt, the GRPO
        # advantage is ~0 (all scores similar), which explains PG loss ≈ 0.
        "is_correct": int(terminal >= 0),
        # ── Timing diagnostics (ms) ────────────────────────────────────────
        "harness_run_ms": metadata.get("harness_run_ms", 0.0),
        "sample_fill_ms": metadata.get("sample_fill_ms", 0.0),
        "sglang_inference_ms": metadata.get("sglang_inference_ms", 0.0),
        # ── Task-specific fields (from spec.extra_reward_fn) ───────────────
        **extra,
    }
