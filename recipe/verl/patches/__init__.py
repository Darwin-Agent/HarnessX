# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Compatibility patches for verl recipe."""
from recipe.verl.patches.patch_parameter import apply as apply_parameter_patch
from recipe.verl.patches.patch_sglang_compat import apply as apply_sglang_patch


def apply_all():
    """Apply all verl compatibility patches."""
    apply_parameter_patch()
    apply_sglang_patch()
    # Disable flash_attn cross_entropy to fix ulysses SP shape mismatch
    # (verl issue #6284). Falls back to PyTorch native logsumexp.
    try:
        import verl.utils.torch_functional as tf
        tf.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False
    except (ImportError, AttributeError):
        pass
