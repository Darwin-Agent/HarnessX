# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""verl harness rollout — generate() and reward_func().

Same pattern as recipe/slime/harness_rollout.py:
    generate(sample, sampling_params, url, tokenizer) → fills sample with tokens
    reward_func(sample) → computes scalar reward from trajectory
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from typing import Any

from harnessx.core.model_config import ModelConfig
from harnessx.providers.sglang import (
    SGLangProvider as VerlSGLangProvider,
    build_flat_sequence,
)
from recipe.verl.harness import make_verl_harness
from recipe.verl.registry import load_harness_config

logger = logging.getLogger(__name__)

_HARNESS_RUN_TIMEOUT_S = float(os.environ.get("HARNESSX_SAMPLE_TIMEOUT", "600"))


def make_inprocess_post_fn(server_manager):
    """Create a post_fn that calls server_manager.generate() via Ray actor (no HTTP).

    This is the correct path for verl hybrid engine — the SGLang HTTP server
    is not used in training mode. Instead, generate requests go through verl's
    AsyncLLMServerManager → SGLangHttpServer Ray actor → tokenizer_manager.
    """
    from uuid import uuid4

    async def _inprocess_post(url: str, payload: dict) -> dict:
        input_ids = payload["input_ids"]
        sampling_params = payload.get("sampling_params", {})
        return_logprob = payload.get("return_logprob", False)
        image_data = payload.get("image_data", None)

        request_id = uuid4().hex
        sampling_params_copy = {**sampling_params, "logprobs": return_logprob}

        try:
            output = await server_manager.generate(
                request_id=request_id,
                prompt_ids=input_ids,
                sampling_params=sampling_params_copy,
                image_data=image_data,
            )
        except Exception as e:
            logger.error(f"server_manager.generate() failed: {type(e).__name__}: {e}")
            raise

        # Convert TokenOutput to SGLang HTTP-response-compatible dict
        token_ids = list(output.token_ids) if output.token_ids else []
        result = {"output_ids": token_ids}
        meta_info = {}
        if output.stop_reason:
            meta_info["finish_reason"] = {"type": output.stop_reason}
        else:
            meta_info["finish_reason"] = {"type": "stop"}
        if output.log_probs is not None:
            if len(output.log_probs) != len(token_ids):
                raise ValueError("SGLang returned different token and log probability counts")
            meta_info["output_token_logprobs"] = [
                (float(logprob) if logprob is not None else None, int(token_id), None)
                for logprob, token_id in zip(output.log_probs, token_ids, strict=True)
            ]
        result["meta_info"] = meta_info
        return result

    return _inprocess_post


class VerlSample:
    """Minimal sample object compatible with load_harness_config / task_builder."""

    def __init__(self, prompt: Any, label: str = "", metadata: dict | None = None):
        self.prompt = prompt
        self.label = label
        self.metadata = metadata or {}
        self.metadata.setdefault("task_type", "research")


async def generate(
    sample: VerlSample,
    sampling_params: dict,
    url: str,
    tokenizer: Any,
    server_manager: Any = None,
    harness_config: Any = None,
) -> dict | None:
    """Run HarnessX agent loop and return token-level episode data.

    Args:
        sample: VerlSample with .prompt, .label, .metadata
        sampling_params: SGLang sampling parameters
        url: SGLang server URL (configure the host and port for your environment)
        tokenizer: HuggingFace tokenizer instance

    Returns:
        dict with keys:
            "tokens": list[int] — prompt + response token IDs
            "loss_mask": list[int] — 1=model, 0=tool
            "rollout_log_probs": list[float]
            "response_length": int
            "response": str
            "trajectory": StatefulTrajectory (for reward_func)
            "eval_result": EvalResult
            "exit_reason": str
            "status": "completed" | "truncated" | "aborted"
    """
    spec = load_harness_config(sample)
    task = spec.task_builder.build(sample)
    task.max_steps = spec.max_steps

    _max_new_tokens = os.environ.get("HARNESSX_MAX_NEW_TOKENS_PER_TURN")
    if _max_new_tokens:
        sampling_params = dict(sampling_params)
        sampling_params["max_new_tokens"] = min(
            sampling_params.get("max_new_tokens", 8192),
            int(_max_new_tokens),
        )

    post_fn = make_inprocess_post_fn(server_manager) if server_manager else None
    provider = VerlSGLangProvider(
        url=url,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_max_context_len=32768,
        tool_call_format=spec.tool_call_format,
        tool_result_role=spec.tool_result_role,
        post_fn=post_fn,
        strip_think_prefix=True,
        require_logprobs=True,
        stop_strings=["</tool_call>"],
        truncation_strategy="head_tail",
    )

    try:
        from recipe.coevolution.behavior import resolve_behavior

        behavior = resolve_behavior(sample.metadata, explicit=harness_config)
        config = make_verl_harness(spec, provider, task, behavior=behavior)
        t_start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                ModelConfig(main=provider).agentic(config).run(task),
                timeout=_HARNESS_RUN_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning("Harness.run() timed out after %.0fs", _HARNESS_RUN_TIMEOUT_S)
            if provider.step_captures:
                flat = build_flat_sequence(provider)
                return {
                    "tokens": flat.prompt_ids + flat.response_ids,
                    "loss_mask": flat.loss_mask,
                    "rollout_log_probs": flat.rollout_logprobs,
                    "response_length": len(flat.response_ids),
                    "response": tokenizer.decode(flat.response_ids, skip_special_tokens=False),
                    "trajectory": None,
                    "eval_result": None,
                    "exit_reason": "budget_exceeded",
                    "status": "truncated",
                    "harness_run_ms": (time.monotonic() - t_start) * 1000,
                }
            logger.warning("generate: timeout with no step_captures, returning None")
            return None
        harness_run_ms = (time.monotonic() - t_start) * 1000

        traj = result.trajectory

        if provider.aborted:
            return None

        if not provider.step_captures:
            logger.warning(
                f"generate: step_captures empty after Harness.run() "
                f"(exit_reason={result.task_end.exit_reason if result.task_end else 'N/A'})"
            )
            return None

        flat = build_flat_sequence(provider)
        episode = {
            "tokens": flat.prompt_ids + flat.response_ids,
            "loss_mask": flat.loss_mask,
            "rollout_log_probs": flat.rollout_logprobs,
            "response_length": len(flat.response_ids),
            "response": tokenizer.decode(flat.response_ids, skip_special_tokens=False),
        }

        exit_reason = result.task_end.exit_reason if result.task_end else "done"
        if exit_reason in ("budget_exceeded", "loop_detected"):
            status = "truncated"
        elif exit_reason in ("interrupted", "error"):
            status = "aborted"
        else:
            status = "completed"

        episode["trajectory"] = traj
        episode["eval_result"] = result.task_end.eval_result if result.task_end else None
        episode["exit_reason"] = exit_reason
        episode["status"] = status
        episode["harness_run_ms"] = harness_run_ms
        episode["total_steps"] = result.task_end.total_steps if result.task_end else 0
        episode["total_tokens"] = result.task_end.total_tokens if result.task_end else 0

        # MATPO: collect subagent episodes from state
        _state = result.resume_state
        _matpo_slot = "_matpo_sub_episodes"
        episode["sub_episodes"] = getattr(_state, _matpo_slot, []) if _state else []

        return episode

    finally:
        await provider.aclose()


async def reward_func(sample: VerlSample, episode: dict) -> dict:
    """Compute reward from a completed rollout episode.

    Args:
        sample: original VerlSample
        episode: dict returned by generate()

    Returns:
        dict with "score" + decomposition fields for logging.
    """
    spec = load_harness_config(sample)
    traj = episode.get("trajectory")
    eval_result = episode.get("eval_result")
    exit_reason = episode.get("exit_reason", "done")

    terminal: float = float(eval_result.reward) if eval_result is not None else 0.0

    step_rewards = await spec.prm.score_steps(traj, exit_reason=exit_reason)
    if spec.prm.is_terminal_only:
        prm_adjusted = step_rewards[-1] if step_rewards else terminal
    else:
        prm_adjusted = spec.prm.aggregate(terminal, step_rewards)

    extra: dict = {}
    if spec.extra_reward_fn is not None:
        _extra_result = spec.extra_reward_fn(sample, eval_result, traj)
        if inspect.isawaitable(_extra_result):
            _extra_result = await _extra_result
        extra = dict(_extra_result or {})
    score_delta: float = extra.pop("score_delta", 0.0)
    score: float = prm_adjusted + score_delta

    tool_success_count = 0
    tool_error_count = 0
    if traj and traj.steps:
        for step in traj.steps:
            for obs in step.observation:
                is_error = bool(obs.error) or (isinstance(obs.result, str) and obs.result.startswith("Error:"))
                if is_error:
                    tool_error_count += 1
                else:
                    tool_success_count += 1

    return {
        "score": score,
        "terminal": terminal,
        "prm_adjusted": prm_adjusted,
        "tool_success_count": tool_success_count,
        "tool_error_count": tool_error_count,
        "exit_reason": exit_reason,
        "is_correct": int(eval_result is not None and eval_result.passed),
        "harness_run_ms": episode.get("harness_run_ms", 0.0),
        **extra,
    }
