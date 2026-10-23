"""Adapt HarnessX rollouts to veRL's token-level agent-loop interface."""

from __future__ import annotations

import json
import logging
import math
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

from recipe.verl.harness_rollout import VerlSample, generate

logger = logging.getLogger(__name__)


def _get_sglang_url(server_manager) -> str:
    addresses = list(server_manager._server_id_to_handle)
    if not addresses:
        raise RuntimeError("No SGLang servers available in server_manager")
    return f"http://{addresses[0]}/generate"


def _validate_tokens(prompt_ids, response_ids, loss_mask, logprobs):
    if not prompt_ids or not response_ids:
        raise ValueError("Rollout must contain both prompt and response tokens")
    if not (len(response_ids) == len(loss_mask) == len(logprobs)):
        raise ValueError("Response tokens, masks, and log probabilities must have equal lengths")
    if any(mask not in (0, 1) for mask in loss_mask):
        raise ValueError("Response masks must be binary")
    if not any(loss_mask):
        raise ValueError("Rollout contains no trainable tokens")
    normalized = []
    for mask, logprob in zip(loss_mask, logprobs, strict=True):
        if mask and (logprob is None or not math.isfinite(float(logprob)) or float(logprob) > 0):
            raise ValueError("Every trainable token requires a finite, non-positive rollout log probability")
        normalized.append(float(logprob) if mask else 0.0)
    return normalized


@register("harnessx_full")
class HarnessXAgentLoop(AgentLoopBase):
    """Run the full HarnessX harness, masking failed samples out of training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sglang_url = None
        self.response_length = self.rollout_config.response_length
        self.prompt_length = getattr(
            self.rollout_config,
            "prompt_length",
            getattr(self.rollout_config, "max_prompt_length", 32768),
        )

    def _get_url(self):
        if self._sglang_url is None:
            self._sglang_url = _get_sglang_url(self.server_manager)
        return self._sglang_url

    def _validate_prompt_length(self, prompt_ids):
        if len(prompt_ids) > self.prompt_length:
            raise ValueError(
                f"Rendered Harness prompt length {len(prompt_ids)} exceeds configured prompt_length {self.prompt_length}"
            )

    def _bounded_response(self, response_ids, response_mask, logprobs):
        if len(response_ids) != len(response_mask) or len(response_ids) != len(logprobs):
            raise ValueError("Response tokens, masks, and log probabilities must have equal lengths")
        limit = min(len(response_ids), self.response_length)
        response_ids = response_ids[:limit]
        response_mask = response_mask[:limit]
        logprobs = logprobs[:limit]
        if not response_ids or not any(response_mask):
            raise ValueError("Rollout response has no trainable tokens within response_length")
        return response_ids, response_mask, logprobs

    def _failed_output(self, messages):
        token_id = self.tokenizer.eos_token_id
        if token_id is None:
            token_id = self.tokenizer.pad_token_id
        if token_id is None:
            raise ValueError("The training tokenizer must define an EOS or padding token")
        return AgentLoopOutput(
            prompt_ids=self._encode_fallback(messages),
            response_ids=[token_id],
            response_mask=[0],
            response_logprobs=[0.0],
            num_turns=0,
            metrics={},
            reward_score=0.0,
            extra_fields={
                "episode_id": uuid4().hex,
                "rollout_failed": True,
                "reward_extra_info": {"accuracy": 0.0, "format_reward": 0.0, "tool_call_reward": 0.0},
            },
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info") or {}
        reward_model = kwargs.get("reward_model") or extra_info.get("reward_model")
        if isinstance(reward_model, str):
            try:
                reward_model = json.loads(reward_model)
            except ValueError:
                reward_model = {}
        label = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        label = label or kwargs.get("ground_truth") or extra_info.get("ground_truth", "")
        metadata = {"task_type": "research", **extra_info}
        metadata["harness_index"] = kwargs.get("harness_index", 0)
        sample = VerlSample(prompt=messages, label=label, metadata=metadata)
        episode_id = uuid4().hex
        url = self._get_url()
        try:
            episode = await generate(
                sample=sample,
                sampling_params=sampling_params,
                url=url,
                tokenizer=self.tokenizer,
                server_manager=self.server_manager,
            )
        except (TypeError, AttributeError, ImportError):
            logger.exception("HarnessX rollout configuration or interface error")
            raise
        except Exception:
            logger.exception("HarnessX rollout failed; masking this sample")
            return self._failed_output(messages)
        if episode is None or episode.get("status") == "aborted":
            return self._failed_output(messages)

        from recipe.verl.research.rewards import research_extra_reward_fn

        try:
            reward_info = research_extra_reward_fn(sample, episode.get("eval_result"), episode.get("trajectory"))
            accuracy = float(reward_info.get("accuracy", 0.0))
            format_reward = float(reward_info.get("format_reward", 0.0))
            tool_reward = float(reward_info.get("tool_call_reward", 0.0))
            sub_episodes = episode.get("sub_episodes", [])
            if sub_episodes:
                format_reward = 0.5 * format_reward + 0.5 * sum(
                    child.get("format_reward", 0.0) for child in sub_episodes
                ) / len(sub_episodes)
                tool_reward = 0.5 * tool_reward + 0.5 * sum(
                    child.get("tool_call_reward", 0.0) for child in sub_episodes
                ) / len(sub_episodes)
            reward = 0.7 * accuracy + format_reward + tool_reward
            if not math.isfinite(reward):
                raise ValueError("Rollout reward must be finite")
            tokens = episode["tokens"]
            response_length = episode["response_length"]
            if not isinstance(response_length, int) or not 0 < response_length < len(tokens):
                raise ValueError("Invalid rollout response length")
            prompt_ids = tokens[:-response_length]
            response_ids = tokens[-response_length:]
            logprobs = _validate_tokens(prompt_ids, response_ids, episode["loss_mask"], episode["rollout_log_probs"])
            self._validate_prompt_length(prompt_ids)
            response_ids, response_mask, logprobs = self._bounded_response(response_ids, episode["loss_mask"], logprobs)
            sub_outputs = []
            for child in sub_episodes:
                child_prompt = child["prompt_ids"]
                child_response = child["response_ids"]
                child_mask = child["loss_mask"]
                child_logprobs = _validate_tokens(child_prompt, child_response, child_mask, child["rollout_log_probs"])
                self._validate_prompt_length(child_prompt)
                child_response, child_mask, child_logprobs = self._bounded_response(
                    child_response, child_mask, child_logprobs
                )
                sub_outputs.append(
                    AgentLoopOutput(
                        prompt_ids=child_prompt,
                        response_ids=child_response,
                        response_mask=child_mask,
                        response_logprobs=child_logprobs,
                        num_turns=child.get("total_steps", 1),
                        metrics={},
                        reward_score=reward,
                        extra_fields={
                            "episode_id": episode_id,
                            "is_subagent": True,
                            "rollout_failed": False,
                            "reward_extra_info": {
                                "accuracy": accuracy,
                                "format_reward": child.get("format_reward", 0.0),
                                "tool_call_reward": child.get("tool_call_reward", 0.0),
                            },
                        },
                    )
                )
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=logprobs,
                num_turns=episode.get("total_steps", 1),
                metrics={},
                reward_score=reward,
                sub_outputs=sub_outputs or None,
                extra_fields={
                    "episode_id": episode_id,
                    "rollout_failed": False,
                    "reward_extra_info": {
                        "accuracy": accuracy,
                        "format_reward": format_reward,
                        "tool_call_reward": tool_reward,
                    },
                },
            )
        except (KeyError, TypeError, ValueError):
            logger.exception("Invalid HarnessX rollout data; masking this sample")
            return self._failed_output(messages)

    def _encode_fallback(self, messages):
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return [self.tokenizer.bos_token_id or 0]
