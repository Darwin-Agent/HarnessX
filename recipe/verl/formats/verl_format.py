"""Convert annotated trajectories without fabricating behavior log probabilities."""

from __future__ import annotations

import math
from typing import Any


class VerlRLFormat:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def to_episode(self, traj: Any) -> dict:
        prompt_ids = []
        response_ids = []
        loss_mask = []
        logprobs = []
        for step in getattr(traj, "steps", []):
            annotation = step.token_annotation
            if annotation is None or not annotation.response_ids:
                continue
            if not prompt_ids:
                prompt_ids = list(annotation.prompt_ids or [])
            tokens = list(annotation.response_ids)
            masks = annotation.response_mask
            probabilities = annotation.response_logprobs
            if masks is None or probabilities is None or not (len(tokens) == len(masks) == len(probabilities)):
                raise ValueError("Annotated RL tokens require aligned masks and behavior log probabilities")
            for mask, probability in zip(masks, probabilities, strict=True):
                if mask not in (0, 1):
                    raise ValueError("RL loss masks must be binary")
                if mask and (probability is None or not math.isfinite(float(probability)) or float(probability) > 0):
                    raise ValueError("Trainable tokens require finite, non-positive behavior log probabilities")
                logprobs.append(float(probability) if mask else 0.0)
            response_ids.extend(tokens)
            loss_mask.extend(masks)
        steps = getattr(traj, "steps", [])
        reward = float(steps[-1].reward or 0.0) if steps else 0.0
        return {
            "tokens": prompt_ids + response_ids,
            "loss_mask": loss_mask,
            "rollout_log_probs": logprobs,
            "response_length": len(response_ids),
            "reward": reward,
            "response": self._tokenizer.decode(
                [token for token, mask in zip(response_ids, loss_mask, strict=True) if mask],
                skip_special_tokens=False,
            )
            if response_ids
            else "",
        }
