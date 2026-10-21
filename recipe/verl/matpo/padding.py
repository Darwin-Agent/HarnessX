# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""MATPO padding — zero-mask dummy sequences for batch divisibility."""
from __future__ import annotations

import numpy as np
import torch


def pad_matpo_batch(data: "DataProto", size_divisor: int) -> tuple["DataProto", int]:
    """Pad DataProto with zero-mask dummy sequences to ensure divisibility.

    Unlike verl's default pad_dataproto_to_divisor (which duplicates real rows),
    MATPO pads with dummy rows that have response_mask=0 everywhere so they
    contribute zero gradient.

    Args:
        data: DataProto batch (may include both parent and subagent rows)
        size_divisor: target divisor (e.g. dp_size or num_workers)

    Returns:
        (padded_data, pad_size) — pad_size=0 means no padding needed
    """
    from verl.protocol import DataProto
    from tensordict import TensorDict

    current_size = len(data)
    if current_size % size_divisor == 0:
        return data, 0

    pad_size = size_divisor - (current_size % size_divisor)

    # Create zero-mask dummy tensors by cloning shape from first row
    dummy_batch = {}
    for key in data.batch.keys():
        tensor = data.batch[key]
        shape = list(tensor.shape)
        shape[0] = pad_size
        if key == "response_mask":
            dummy_batch[key] = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
        elif key == "rm_scores":
            dummy_batch[key] = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
        elif key in ("attention_mask",):
            dummy_batch[key] = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
        else:
            dummy_batch[key] = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)

    dummy_td = TensorDict(dummy_batch, batch_size=pad_size)

    # Create dummy non_tensor_batch
    dummy_non_tensor = {}
    for key, val in data.non_tensor_batch.items():
        if val.dtype == bool:
            dummy_non_tensor[key] = np.zeros(pad_size, dtype=bool)
        elif val.dtype == object:
            dummy_non_tensor[key] = np.empty(pad_size, dtype=object)
        elif np.issubdtype(val.dtype, np.integer):
            dummy_non_tensor[key] = np.zeros(pad_size, dtype=val.dtype)
        elif np.issubdtype(val.dtype, np.floating):
            dummy_non_tensor[key] = np.zeros(pad_size, dtype=val.dtype)
        else:
            dummy_non_tensor[key] = np.empty(pad_size, dtype=val.dtype)

    # Mark dummy rows
    if "is_subagent" in dummy_non_tensor:
        dummy_non_tensor["is_subagent"] = np.zeros(pad_size, dtype=bool)
    dummy_non_tensor["is_pad"] = np.ones(pad_size, dtype=bool)

    # Add is_pad to original data if not present
    if "is_pad" not in data.non_tensor_batch:
        data.non_tensor_batch["is_pad"] = np.zeros(current_size, dtype=bool)

    dummy_proto = DataProto(batch=dummy_td, non_tensor_batch=dummy_non_tensor)
    padded = DataProto.concat([data, dummy_proto])


    return padded, pad_size
