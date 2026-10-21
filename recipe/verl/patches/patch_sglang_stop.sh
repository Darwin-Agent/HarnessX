#!/bin/bash
# Patch SGLang source files to enable stop string detection in verl hybrid mode.
# Safe to run multiple times (checks if already patched).
#
# Patches:
# 1. scheduler.py: load tokenizer even with skip_tokenizer_init=True
# 2. sampling_params.py: fix stop_str_max_len to use char length * 2

set -e

SGLANG_ROOT=$(python3 -c "import sglang; import os; print(os.path.dirname(sglang.__file__))" 2>/dev/null)
if [ -z "$SGLANG_ROOT" ]; then
    echo "[patch_sglang_stop] sglang not found, skipping"
    exit 0
fi

SCHEDULER="$SGLANG_ROOT/srt/managers/scheduler.py"
SAMPLING="$SGLANG_ROOT/srt/sampling/sampling_params.py"

# --- Patch 1: scheduler.py ---
if grep -q "AutoTokenizer.from_pretrained" "$SCHEDULER" 2>/dev/null; then
    echo "[patch_sglang_stop] scheduler.py already patched"
else
    # Replace: self.tokenizer = self.processor = None
    # With: load tokenizer anyway
    python3 -c "
import re
with open('$SCHEDULER', 'r') as f:
    content = f.read()

old = '''        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None'''

new = '''        if server_args.skip_tokenizer_init:
            self.processor = None
            try:
                from transformers import AutoTokenizer
                import os
                _model_path = os.environ.get(\"MODEL_DIR\", server_args.tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    _model_path, trust_remote_code=True
                )
                if not hasattr(self.tokenizer, \"additional_stop_token_ids\"):
                    self.tokenizer.additional_stop_token_ids = None
            except Exception:
                self.tokenizer = None'''

if old in content:
    content = content.replace(old, new)
    with open('$SCHEDULER', 'w') as f:
        f.write(content)
    print('[patch_sglang_stop] scheduler.py patched OK')
else:
    print('[patch_sglang_stop] scheduler.py pattern not found (may already be patched)')
"
fi

# --- Patch 2: sampling_params.py ---
if grep -q "len(stop_str) \* 2" "$SAMPLING" 2>/dev/null; then
    echo "[patch_sglang_stop] sampling_params.py already patched"
else
    python3 -c "
with open('$SAMPLING', 'r') as f:
    content = f.read()

old = '''                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))'''

new = '''                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids), len(stop_str) * 2)'''

if old in content:
    content = content.replace(old, new)
    with open('$SAMPLING', 'w') as f:
        f.write(content)
    print('[patch_sglang_stop] sampling_params.py patched OK')
else:
    print('[patch_sglang_stop] sampling_params.py pattern not found (may already be patched)')
"
fi

# --- Patch 3: weight_sync/utils.py — ensure tensors on GPU ---
WEIGHT_SYNC="$SGLANG_ROOT/srt/weight_sync/utils.py"
if grep -q "\.cuda()" "$WEIGHT_SYNC" 2>/dev/null; then
    echo "[patch_sglang_stop] weight_sync/utils.py already patched"
else
    python3 -c "
with open('$WEIGHT_SYNC', 'r') as f:
    content = f.read()

old = '''    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor'''

new = '''    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    if tensor.device.type == \"cpu\":
        tensor = tensor.cuda()
    return tensor'''

if old in content:
    content = content.replace(old, new)
    with open('$WEIGHT_SYNC', 'w') as f:
        f.write(content)
    print('[patch_sglang_stop] weight_sync/utils.py patched OK')
else:
    print('[patch_sglang_stop] weight_sync/utils.py pattern not found')
"
fi
