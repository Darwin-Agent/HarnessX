"""Pad multi-agent batches without introducing trainable dummy tokens."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


def pad_matpo_batch(data, size_divisor: int):
    from verl.protocol import DataProto

    if size_divisor <= 0 or not len(data):
        raise ValueError("Padding requires a nonempty batch and a positive divisor")
    pad_size = (-len(data)) % size_divisor
    if not pad_size:
        return data, 0
    data.non_tensor_batch.setdefault("is_pad", np.zeros(len(data), dtype=bool))
    padding = deepcopy(data.select_idxs(np.zeros(pad_size, dtype=np.int64)))
    for key in ("response_mask", "rm_scores", "token_level_scores", "token_level_rewards", "advantages", "returns"):
        if key in padding.batch:
            padding.batch[key].zero_()
    padding.non_tensor_batch["is_pad"] = np.ones(pad_size, dtype=bool)
    padding.meta_info = data.meta_info.copy()
    return DataProto.concat([data, padding]), pad_size
