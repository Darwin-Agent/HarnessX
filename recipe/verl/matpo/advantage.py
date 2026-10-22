"""GRPO on valid parent episodes, with exact child advantage inheritance."""

from __future__ import annotations

import numpy as np
import torch


def compute_matpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index=None,
    non_tensor_batch=None,
    norm_adv_by_std_in_grpo=True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = non_tensor_batch or {}
    batch_size = token_level_rewards.shape[0]
    is_subagent = np.asarray(metadata.get("is_subagent", np.zeros(batch_size)), dtype=bool)
    is_pad = np.asarray(metadata.get("is_pad", np.zeros(batch_size)), dtype=bool)
    is_valid = response_mask.bool().any(dim=-1).detach().cpu().numpy() & ~is_pad
    parent_indices = np.flatnonzero(~is_subagent & is_valid)
    groups = np.asarray(index) if index is not None else np.arange(batch_size)
    scores = token_level_rewards.sum(dim=-1)
    scalar_advantages = torch.zeros_like(scores, dtype=torch.float32)
    for group in np.unique(groups[parent_indices]):
        members = parent_indices[groups[parent_indices] == group]
        group_scores = scores[members]
        if len(members) == 1:
            mean = torch.zeros_like(group_scores[0])
            deviation = torch.ones_like(group_scores[0])
        else:
            mean = group_scores.mean()
            deviation = group_scores.std()
        values = group_scores - mean
        if norm_adv_by_std_in_grpo:
            values = values / (deviation + 1e-6)
        scalar_advantages[members] = values

    children = np.flatnonzero(is_subagent & is_valid)
    if len(children):
        episode_ids = metadata.get("episode_id")
        if episode_ids is None:
            raise ValueError("MATPO children require explicit parent episode IDs")
        parent_lookup = {episode_ids[parent]: parent for parent in parent_indices}
        for child in children:
            parent = parent_lookup.get(episode_ids[child])
            if parent is None:
                raise ValueError("MATPO child refers to a missing or failed parent")
            scalar_advantages[child] = scalar_advantages[parent]

    advantages = scalar_advantages.unsqueeze(-1) * response_mask
    return advantages, advantages.clone()
