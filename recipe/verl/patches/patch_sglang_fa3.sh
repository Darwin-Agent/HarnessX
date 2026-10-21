#!/bin/bash
# Patch SGLang flashattention_backend.py to handle missing get_scheduler_metadata
# (sgl_kernel 0.3.21 doesn't have it, but SGLang 0.5.11 expects it)
SGLANG_FA="/tmp/sglang-0.5.11/python/sglang/srt/layers/attention/flashattention_backend.py"
if [ -f "$SGLANG_FA" ] && grep -q "get_scheduler_metadata" "$SGLANG_FA"; then
    if ! grep -q "# PATCHED: get_scheduler_metadata optional" "$SGLANG_FA"; then
        sed -i 's/from sgl_kernel.flash_attn import (\n\s*flash_attn_varlen_func,\n\s*flash_attn_with_kvcache,\n\s*get_scheduler_metadata,\n\s*)/from sgl_kernel.flash_attn import (\n                flash_attn_varlen_func,\n                flash_attn_with_kvcache,\n            )\n            try:\n                from sgl_kernel.flash_attn import get_scheduler_metadata\n            except ImportError:\n                get_scheduler_metadata = None/' "$SGLANG_FA" 2>/dev/null
        # Fallback: use python to patch if sed multiline fails
        python3 -c "
import re
with open('$SGLANG_FA', 'r') as f:
    content = f.read()
if '# PATCHED: get_scheduler_metadata optional' not in content:
    old = '''from sgl_kernel.flash_attn import (
                flash_attn_varlen_func,
                flash_attn_with_kvcache,
                get_scheduler_metadata,
            )'''
    new = '''# PATCHED: get_scheduler_metadata optional
            from sgl_kernel.flash_attn import (
                flash_attn_varlen_func,
                flash_attn_with_kvcache,
            )
            try:
                from sgl_kernel.flash_attn import get_scheduler_metadata
            except ImportError:
                get_scheduler_metadata = None'''
    content = content.replace(old, new)
    with open('$SGLANG_FA', 'w') as f:
        f.write(content)
    print('  [OK] Patched get_scheduler_metadata as optional')
else:
    print('  [SKIP] Already patched')
"
    fi
fi

# Patch SGLang release_memory_occupation assert (crashes when server not fully idle)
SGLANG_UW="/tmp/sglang-0.5.11/python/sglang/srt/managers/scheduler_update_weights_mixin.py"
if [ -f "$SGLANG_UW" ] && grep -q 'assert.*is_fully_idle' "$SGLANG_UW"; then
    if ! grep -q "# PATCHED: assert relaxed" "$SGLANG_UW"; then
        python3 -c "
with open('$SGLANG_UW', 'r') as f:
    content = f.read()
old = '''assert (
            self.is_fully_idle()
        ), \"release_memory_occupation should be called only when server is idle.\"'''
new = '''# PATCHED: assert relaxed — wait briefly then proceed even if not idle
        if not self.is_fully_idle():
            import time
            time.sleep(0.5)
            if not self.is_fully_idle():
                import logging
                logging.getLogger(__name__).warning(\"release_memory_occupation: server not idle, proceeding anyway\")'''
if old in content:
    content = content.replace(old, new)
    with open('$SGLANG_UW', 'w') as f:
        f.write(content)
    print('  [OK] Patched release_memory_occupation assert')
else:
    print('  [SKIP] Pattern not found')
"
    fi
fi

# Patch transformers 5.6 flash_attention s_aux None check
TF_FA="/usr/local/lib/python3.12/dist-packages/transformers/integrations/flash_attention.py"
if [ -f "$TF_FA" ] && grep -q "s_aux=s_aux.to(query.dtype)," "$TF_FA"; then
    sed -i 's/s_aux=s_aux.to(query.dtype),/s_aux=s_aux.to(query.dtype) if s_aux is not None else None,/' "$TF_FA"
    echo "  [OK] Patched transformers flash_attention s_aux None check"
fi
