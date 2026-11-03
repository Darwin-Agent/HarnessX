"""Explicit parent-child linkage for variable-size multi-agent batches."""

from __future__ import annotations

import numpy as np


def parent_row_indices(is_subagent, episode_ids, parent_count):
    flags = np.asarray(is_subagent, dtype=bool)
    if len(flags) != len(episode_ids) or int((~flags).sum()) != parent_count:
        raise ValueError("Rollout parent count does not match input batch")
    if flags[:parent_count].any() or not flags[parent_count:].all():
        raise ValueError("Rollout parents must precede children")
    parents = {}
    for index, episode_id in enumerate(episode_ids[:parent_count]):
        if episode_id is not None:
            if episode_id in parents:
                raise ValueError("Parent episode IDs must be unique")
            parents[episode_id] = index
    indices = list(range(parent_count))
    for episode_id in episode_ids[parent_count:]:
        if episode_id not in parents:
            raise ValueError("Subagent has no matching parent episode")
        indices.append(parents[episode_id])
    return np.asarray(indices, dtype=np.int64)


def order_rollout_rows(output, validation=False):
    flags = np.asarray(output.non_tensor_batch["is_subagent"], dtype=bool)
    parents = np.flatnonzero(~flags)
    indices = parents if validation else np.concatenate([parents, np.flatnonzero(flags)])
    return output.select_idxs(indices)


def expand_parent_batch(batch, output):
    indices = parent_row_indices(
        output.non_tensor_batch["is_subagent"], output.non_tensor_batch["episode_id"], len(batch)
    )
    return batch.select_idxs(indices)
