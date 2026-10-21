# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""VerlRLFormat — incremental token/mask builder.

Instead of post-hoc reconstruction, builds token_ids and attention_mask
segment by segment as each turn completes:
  - assistant turn → append model_ids, mask=[1]*len
  - tool/user turn → append tool_ids, mask=[0]*len

Each segment insertion is logged for verification.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VerlRLFormat:
    """Incremental token/mask builder for verl RL training."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def to_episode(self, traj: Any) -> dict:
        """Build flat episode by iterating trajectory steps incrementally.

        Each step contributes:
          1. model_ids (assistant generation) → mask=1
          2. tool_ids (tool/user injection)   → mask=0

        Logs each segment for debugging.
        """
        if not traj or not traj.steps:
            return {"tokens": [], "loss_mask": [], "rollout_log_probs": [],
                    "response_length": 0, "reward": 0.0, "response": ""}

        prompt_ids: list[int] = []
        response_ids: list[int] = []
        loss_mask: list[int] = []
        logprobs: list[float] = []
        _turn_log: list[dict] = []

        for i, step in enumerate(traj.steps):
            ann = step.token_annotation
            if ann is None:
                continue

            # First step: capture prompt
            if not prompt_ids and ann.prompt_ids:
                prompt_ids = list(ann.prompt_ids)

            if not ann.response_ids:
                continue

            # Split response into model segment and tool segment using response_mask
            if ann.response_mask and len(ann.response_mask) == len(ann.response_ids):
                model_len = sum(1 for m in ann.response_mask if m == 1)
                tool_len = sum(1 for m in ann.response_mask if m == 0)

                model_ids = ann.response_ids[:model_len]
                tool_ids = ann.response_ids[model_len:]
                model_logps = (ann.response_logprobs or [0.0] * len(ann.response_ids))[:model_len]
                tool_logps = [0.0] * tool_len

                response_ids.extend(model_ids)
                loss_mask.extend([1] * model_len)
                logprobs.extend(model_logps)

                _turn_log.append({"step": i, "role": "ASSISTANT", "mask": 1, "mask_len": model_len, "ids_len": len(model_ids)})

                if tool_ids:
                    response_ids.extend(tool_ids)
                    loss_mask.extend([0] * tool_len)
                    logprobs.extend(tool_logps)
                    _turn_log.append({"step": i, "role": "TOOL", "mask": 0, "mask_len": tool_len, "ids_len": len(tool_ids)})
            else:
                response_ids.extend(ann.response_ids)
                loss_mask.extend([1] * len(ann.response_ids))
                lps = ann.response_logprobs or [0.0] * len(ann.response_ids)
                logprobs.extend(lps)
                _turn_log.append({"step": i, "role": "UNKNOWN", "mask": 1, "mask_len": len(ann.response_ids), "ids_len": len(ann.response_ids)})

        # Final reward from last step
        reward = 0.0
        if traj.steps and traj.steps[-1].reward is not None:
            reward = float(traj.steps[-1].reward)

        # Decode only model-generated tokens for response text
        model_token_ids = [tid for tid, m in zip(response_ids, loss_mask) if m == 1]
        response_text = self._tokenizer.decode(model_token_ids, skip_special_tokens=False) if model_token_ids else ""

        # Summary log: print interleaved segments with token_ids and mask values
        total_model = sum(loss_mask)
        total_tool = len(loss_mask) - total_model

        # Build segment report: consecutive runs of same mask value
        segments = []
        if loss_mask:
            seg_start = 0
            seg_val = loss_mask[0]
            for idx in range(1, len(loss_mask)):
                if loss_mask[idx] != seg_val:
                    segments.append((seg_val, seg_start, idx))
                    seg_start = idx
                    seg_val = loss_mask[idx]
            segments.append((seg_val, seg_start, len(loss_mask)))

        seg_report = []
        for mask_val, start, end in segments:
            ids_len = len(response_ids[start:end])
            mask_len = len(loss_mask[start:end])
            lp_len = len(logprobs[start:end])
            role = "MODEL" if mask_val == 1 else "TOOL"
            match = "ok" if ids_len == mask_len == lp_len else "MISMATCH"
            seg_report.append(f"  [{role}] ids={ids_len} mask={mask_len} logp={lp_len} | {match}")

        # Length consistency check
        seg_total_len = sum(end - start for _, start, end in segments)
        len_response = len(response_ids)
        len_mask = len(loss_mask)
        len_logprobs = len(logprobs)
        consistent = (seg_total_len == len_response == len_mask == len_logprobs)

        if not consistent:
            logger.error(
                "[MASK_BUILD] LENGTH MISMATCH! response_ids=%d loss_mask=%d logprobs=%d segments_sum=%d",
                len_response, len_mask, len_logprobs, seg_total_len,
            )

        return {
            "tokens": prompt_ids + response_ids,
            "loss_mask": loss_mask,
            "rollout_log_probs": logprobs,
            "response_length": len(response_ids),
            "reward": reward,
            "response": response_text,
        }
