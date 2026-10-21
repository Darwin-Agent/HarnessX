# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Patch torch.nn.Parameter for HF transformers 5.x + verl FSDP compatibility.

Problem: transformers 5.x adds '_is_hf_initialized' to Parameter.__dict__.
verl's fsdp_utils passes __dict__ as kwargs to Parameter.__new__() which
rejects private attributes.

Fix: Filter kwargs starting with '_' in Parameter.__new__.
"""
import torch.nn as nn

_orig_param_new = nn.Parameter.__new__
_patched = False


def apply():
    global _patched
    if _patched:
        return
    _patched = True

    @staticmethod
    def _safe_param_new(cls, data=None, requires_grad=True, **kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        return _orig_param_new(cls, data, requires_grad, **clean_kwargs)

    nn.Parameter.__new__ = _safe_param_new
