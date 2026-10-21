# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""MATPO advantage estimator — parent advantage inherited by subagents.

Integrated into GRPO: when is_subagent rows are detected, this function
is called instead of standard compute_grpo_outcome_advantage.

Handles padding rows (is_pad=True) by excluding them from group computation,
avoiding division-by-zero NaN in advantage normalization.
"""
from __future__ import annotations

import torch
import numpy as np


def compute_matpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray = None,
    non_tensor_batch: dict = None,
    batch: dict = None,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MATPO advantage: GRPO on parents, broadcast to children, skip padding.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) — grouping array (uid)
        non_tensor_batch: dict with 'is_subagent' and optionally 'is_pad'

    Returns:
        (advantages, returns) — both shape (bs, response_length)
    """
    epsilon = 1e-6

    bs = token_level_rewards.shape[0]
    _raw_sub = non_tensor_batch.get("is_subagent")
    _raw_pad = non_tensor_batch.get("is_pad")
    is_subagent = np.asarray(_raw_sub, dtype=bool) if _raw_sub is not None else np.zeros(bs, dtype=bool)
    is_pad = np.asarray(_raw_pad, dtype=bool) if _raw_pad is not None else np.zeros(bs, dtype=bool)

    if index is None:
        index = np.arange(bs)

    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # Valid parents: not subagent AND not padding
    valid_parent = (~is_subagent) & (~is_pad)
    parent_indices = np.where(valid_parent)[0]

    if len(parent_indices) == 0:
        return torch.zeros_like(response_mask, dtype=torch.float32), torch.zeros_like(response_mask, dtype=torch.float32)

    parent_scores = scores[parent_indices]
    parent_index = index[parent_indices]

    # Compute per-group mean/std for valid parents only
    unique_groups = np.unique(parent_index)
    group_mean = {}
    group_std = {}
    for g in unique_groups:
        g_mask = parent_index == g
        g_scores = parent_scores[g_mask]
        group_mean[g] = g_scores.mean().item()
        group_std[g] = g_scores.std().item() if g_mask.sum() > 1 else 0.0

    # Compute parent advantages (GRPO normalized)
    parent_advantages = torch.zeros(len(parent_indices), dtype=torch.float32, device=scores.device)
    for local_i, global_i in enumerate(parent_indices):
        g = index[global_i]
        std = group_std[g]
        if std > epsilon:
            parent_advantages[local_i] = (scores[global_i] - group_mean[g]) / (std + epsilon)
        else:
            parent_advantages[local_i] = scores[global_i] - group_mean[g]

    # Build per-uid advantage lookup (from parents)
    uid_to_adv: dict[Any, list[float]] = {}
    for local_i, global_i in enumerate(parent_indices):
        uid = index[global_i]
        uid_to_adv.setdefault(uid, []).append(parent_advantages[local_i].item())

    # Fill full advantage tensor
    advantages_scalar = torch.zeros(bs, dtype=torch.float32, device=scores.device)

    # Parents get their own advantage
    for local_i, global_i in enumerate(parent_indices):
        advantages_scalar[global_i] = parent_advantages[local_i]

    # Subagents inherit parent's advantage (mean if multiple parents in group)
    sub_indices = np.where(is_subagent & ~is_pad)[0]
    for global_i in sub_indices:
        uid = index[global_i]
        if uid in uid_to_adv:
            advantages_scalar[global_i] = sum(uid_to_adv[uid]) / len(uid_to_adv[uid])

    # Padding rows stay at 0 (no gradient contribution since response_mask=0 anyway)

    # Broadcast scalar advantage to token level
    advantages = advantages_scalar.unsqueeze(-1) * response_mask
    returns = advantages.clone()

    return advantages, returns
