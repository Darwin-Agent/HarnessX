# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from harnessx.core.trajectory import StatefulTrajectory


class SlimeRLFormat:
    """
    Slime GRPO episode format.

    Implements the RLFormat protocol: to_episode(traj) -> dict.

    Aggregates per-step TokenAnnotations into a single flat episode record
    matching Slime's Sample field layout:

        tokens            = prompt_ids + Σ(response_ids per step)
        loss_mask         = Σ(response_mask per step)
        rollout_log_probs = Σ(response_logprobs per step)
        response_length   = Σ(len(response_ids) per step)
        reward            = traj.steps[-1].reward   (terminal, from backfill_rewards)

    The aggregation order is sequential across steps:
        step 0: [model_tokens_0 | tool_tokens_0]
        step 1: [model_tokens_1 | tool_tokens_1]
        ...
    which matches what build_flat_sequence() produces.

    Args:
        tokenizer: HuggingFace tokenizer for decoding response text.
                   If None, episode["response"] is set to "".
    """

    def __init__(self, tokenizer: Any = None) -> None:
        self._tokenizer = tokenizer

    def to_episode(self, traj: "StatefulTrajectory") -> dict:
        """
        Convert a full trajectory into a single Slime GRPO episode record.

        Args:
            traj: StatefulTrajectory with all steps' token_annotation populated.
                  Call backfill_token_annotations(traj, provider) first.

        Returns:
            dict with keys:
                tokens            : list[int]    prompt + response token IDs
                loss_mask         : list[int]    1=model, 0=tool (response only)
                rollout_log_probs : list[float]  logprobs (response only)
                response_length   : int          total response token count
                reward            : float        terminal episode reward
                response          : str          decoded response text (or "")

        Raises:
            ValueError if any step is missing token_annotation.
        """
        if not traj.steps:
            raise ValueError("SlimeRLFormat.to_episode(): trajectory has no steps")

        prompt_ids: list[int] = traj.steps[0].token_annotation.prompt_ids  # type: ignore[union-attr]
        all_response: list[int] = []
        all_mask: list[int] = []
        all_logprobs: list[float] = []

        for step in traj.steps:
            ta = step.token_annotation
            if ta is None:
                raise ValueError(
                    f"SlimeRLFormat.to_episode(): step {step.step_id} has no "
                    "token_annotation — call backfill_token_annotations() first"
                )
            all_response += ta.response_ids
            all_mask += ta.response_mask
            all_logprobs += ta.response_logprobs

        # Validate flat-sequence invariant
        if not (len(all_response) == len(all_mask) == len(all_logprobs)):
            raise RuntimeError(
                f"SlimeRLFormat invariant broken: "
                f"resp={len(all_response)} mask={len(all_mask)} logps={len(all_logprobs)}"
            )

        # Terminal reward (backfilled by backfill_rewards() before reward_func)
        terminal_reward = traj.steps[-1].reward

        # Decode response text (model tokens only for readability)
        response_text = ""
        if self._tokenizer is not None and all_response:
            try:
                # Decode model tokens only (where loss_mask=1)
                model_ids = [tok for tok, m in zip(all_response, all_mask) if m == 1]
                response_text = self._tokenizer.decode(model_ids, skip_special_tokens=False)
            except Exception as exc:
                logger.warning("SlimeRLFormat: tokenizer.decode failed: %s", exc)

        return {
            "tokens": prompt_ids + all_response,
            "loss_mask": all_mask,
            "rollout_log_probs": all_logprobs,
            "response_length": len(all_response),
            "reward": terminal_reward,
            "response": response_text,
        }
