# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""verl AgentLoop — delegates to HarnessX's full RunLoop via harness_rollout.

Registered as "harnessx_full" for verl's agent loop infrastructure.
Internally calls recipe.verl.harness_rollout.generate() which runs
the complete HarnessX Harness.run() with tools, processors, evaluation.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)

from recipe.verl.harness_rollout import VerlSample, generate, reward_func

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _get_sglang_url(server_manager) -> str:
    """Extract SGLang server URL from verl's AsyncLLMServerManager.

    server_manager._server_id_to_handle keys are "{ip}:{port}" strings.
    """
    addresses = list(server_manager._server_id_to_handle.keys())
    if not addresses:
        raise RuntimeError("No SGLang servers available in server_manager")
    return f"http://{addresses[0]}/generate"


@register("harnessx_full")
class HarnessXAgentLoop(AgentLoopBase):
    """verl AgentLoop backed by HarnessX's full RunLoop.

    On each call to run():
    1. Builds a VerlSample from the input messages
    2. Calls harness_rollout.generate() which runs Harness.run() end-to-end
    3. Converts the episode dict to AgentLoopOutput format
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sglang_url: str | None = None
        self.response_length = self.rollout_config.response_length

    def _get_url(self) -> str:
        if self._sglang_url is None:
            self._sglang_url = _get_sglang_url(self.server_manager)
            logger.info("HarnessX agent loop using SGLang at: %s", self._sglang_url)
        return self._sglang_url

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}

        # Build sample — ground_truth from kwargs (verl passes non_tensor_batch fields as kwargs)
        label = ""
        rm = kwargs.get("reward_model") or extra_info.get("reward_model")
        if isinstance(rm, dict):
            label = rm.get("ground_truth", "")
        elif isinstance(rm, str):
            import json as _json
            try:
                label = _json.loads(rm).get("ground_truth", "")
            except (ValueError, AttributeError):
                pass
        if not label:
            label = kwargs.get("ground_truth", "") or extra_info.get("ground_truth", "")
        sample = VerlSample(
            prompt=messages,
            label=label,
            metadata={"task_type": "research", **extra_info},
        )

        # Run HarnessX (in-process mode via server_manager, no HTTP)
        url = self._get_url()
        try:
            episode = await generate(
                sample=sample,
                sampling_params=sampling_params,
                url=url,
                tokenizer=self.tokenizer,
                server_manager=self.server_manager,
            )
        except Exception as e:
            import traceback
            print(f"[HARNESSX_DEBUG] generate() raised {type(e).__name__}: {e}\n{traceback.format_exc()}")
            episode = None

        # Handle failure
        if episode is None:
            prompt_ids = self._encode_fallback(messages)
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=[self.tokenizer.eos_token_id],
                response_mask=[1],
                response_logprobs=[0.0],
                num_turns=1,
                metrics={},
                extra_fields={},
            )

        # Compute format/tool reward from trajectory (structured messages)
        from recipe.verl.research.rewards import research_extra_reward_fn
        from recipe.verl.research.evaluator import compute_accuracy_score
        try:
            traj = episode.get("trajectory")
            eval_result = episode.get("eval_result")
            reward_info = research_extra_reward_fn(sample, eval_result, traj)
        except Exception:
            reward_info = {"format_reward": 0.0, "tool_call_reward": 0.0, "accuracy": 0.0}

        # Compute total reward score directly (bypass verl reward loop)
        accuracy = float(reward_info.get("accuracy", 0.0))
        format_reward = float(reward_info.get("format_reward", 0.0))
        tool_call_reward = float(reward_info.get("tool_call_reward", 0.0))

        # MATPO: blend main and subagent format/tool_call scores
        # Formula: 0.5 * main + 0.5 * avg(sub) for each signal
        sub_episodes = episode.get("sub_episodes", [])
        if sub_episodes:
            n_subs = len(sub_episodes)
            sub_format_avg = sum(ep.get("format_reward", 0.0) for ep in sub_episodes) / n_subs
            sub_tool_call_avg = sum(ep.get("tool_call_reward", 0.0) for ep in sub_episodes) / n_subs
            format_reward = 0.5 * format_reward + 0.5 * sub_format_avg
            tool_call_reward = 0.5 * tool_call_reward + 0.5 * sub_tool_call_avg

        # Weights: 0.7 accuracy + format + tool_call (already weighted 0.1/0.2)
        total_reward = 0.7 * accuracy + format_reward + tool_call_reward

        # Extract token-level data
        try:
            tokens = episode["tokens"]
            resp_len = episode["response_length"]
            if tokens is None or resp_len is None:
                logger.error(f"generate() returned episode with tokens={type(tokens)}, "
                            f"response_length={resp_len}, keys={list(episode.keys())}")
                prompt_ids = self._encode_fallback(messages)
                return AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=[self.tokenizer.eos_token_id],
                    response_mask=[1],
                    response_logprobs=[0.0],
                    num_turns=1,
                    metrics={},
                    extra_fields={},
                )
            prompt_ids = tokens[: len(tokens) - resp_len]
            response_ids = tokens[len(tokens) - resp_len:]
            loss_mask = episode["loss_mask"]
            logprobs = episode["rollout_log_probs"]

            logprobs = logprobs or []

            resp_ids_trunc = response_ids[: self.response_length]
            mask_trunc = loss_mask[: self.response_length]
            logprobs_trunc = [float(lp) if lp is not None else 0.0 for lp in logprobs[: self.response_length]]

            # MATPO: build sub_outputs from subagent episodes
            sub_outputs = None
            sub_episodes = episode.get("sub_episodes", [])
            if sub_episodes:
                sub_outputs = []
                for sub_ep in sub_episodes:
                    sub_tokens = sub_ep.get("prompt_ids", []) + sub_ep.get("response_ids", [])
                    sub_resp_len = sub_ep.get("response_length", len(sub_ep.get("response_ids", [])))
                    sub_prompt = sub_tokens[:len(sub_tokens) - sub_resp_len]
                    sub_resp = sub_tokens[len(sub_tokens) - sub_resp_len:]
                    sub_mask = sub_ep.get("loss_mask", [1] * sub_resp_len)
                    sub_lp = sub_ep.get("rollout_log_probs", [0.0] * sub_resp_len)

                    sub_outputs.append(AgentLoopOutput(
                        prompt_ids=sub_prompt,
                        response_ids=sub_resp[:self.response_length],
                        response_mask=sub_mask[:self.response_length],
                        response_logprobs=[float(lp) if lp is not None else 0.0
                                           for lp in sub_lp[:self.response_length]],
                        num_turns=1,
                        metrics={},
                        extra_fields={
                            "is_subagent": True,
                            "reward_extra_info": {
                                "accuracy": accuracy,
                                "format_reward": sub_ep.get("format_reward", 0.0),
                                "tool_call_reward": sub_ep.get("tool_call_reward", 0.0),
                            },
                        },
                        reward_score=total_reward,
                    ))

            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=resp_ids_trunc,
                response_mask=mask_trunc,
                response_logprobs=logprobs_trunc,
                num_turns=episode.get("total_steps", 1),
                metrics={},
                extra_fields={
                    "reward_extra_info": {
                        "accuracy": accuracy,
                        "format_reward": format_reward,
                        "tool_call_reward": tool_call_reward,
                    },
                },
                reward_score=total_reward,
                sub_outputs=sub_outputs,
            )
        except Exception as e:
            import traceback
            print(f"[HARNESSX_DEBUG] episode postprocess raised {type(e).__name__}: {e}\n"
                        f"episode keys={list(episode.keys()) if episode else None}\n"
                        f"{traceback.format_exc()}")
            prompt_ids = self._encode_fallback(messages)
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=[self.tokenizer.eos_token_id],
                response_mask=[1],
                response_logprobs=[0.0],
                num_turns=1,
                metrics={},
                extra_fields={},
            )

    def _encode_fallback(self, messages: list[dict]) -> list[int]:
        """Encode messages as prompt_ids for error fallback."""
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return [self.tokenizer.bos_token_id or 0]
