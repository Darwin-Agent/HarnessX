# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Patch SGLang scheduler to load tokenizer even with skip_tokenizer_init=True.

This enables string-based stop (stop=["</tool_call>"]) in verl's hybrid mode.
Without this patch, req.tokenizer is None and _check_str_based_finish crashes.

The fix: after the original init_tokenizer() sets self.tokenizer=None,
we load a lightweight HuggingFace tokenizer (no model weights) so that
stop string detection works.
"""
import os


def apply():
    """Monkey-patch Scheduler.init_tokenizer to always load the tokenizer."""
    try:
        from sglang.srt.managers.scheduler import Scheduler
    except ImportError:
        return

    _original_init_tokenizer = Scheduler.init_tokenizer

    def _patched_init_tokenizer(self):
        _original_init_tokenizer(self)
        if self.tokenizer is None:
            model_path = os.environ.get("MODEL_DIR", "")
            if model_path:
                try:
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path, trust_remote_code=True
                    )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "patch_sglang_stop: failed to load tokenizer from %s: %s",
                        model_path, e
                    )

    Scheduler.init_tokenizer = _patched_init_tokenizer
