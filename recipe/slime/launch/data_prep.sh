#!/bin/bash
# data_prep.sh — Download datasets + models, convert torch_dist.
#
# Datasets (from retool README, using zhuzilin repos):
#   JoeYing/ReTool-SFT       → SFT training
#   zhuzilin/dapo-math-17k   → RL training
#   zhuzilin/aime-2024       → RL evaluation
#
# Models (downloaded to default HF cache):
#   Qwen/Qwen3-4B-Instruct-2507   → SFT base
#   font-info/qwen3-4b-sft         → RL base (pre-trained SFT weights)
#
# After downloading, converts both to Megatron torch_dist format.
#
# Usage:
#   bash recipe/slime/launch/data_prep.sh
#   bash recipe/slime/launch/data_prep.sh --skip-models   # data only
#   bash recipe/slime/launch/data_prep.sh --skip-data     # models only

set -e

SLIME_ROOT="${SLIME_ROOT:?SLIME_ROOT env var is required}"
HX_ROOT="${HX_ROOT:?HX_ROOT env var is required}"
MEGATRON_ROOT="${MEGATRON_ROOT:?MEGATRON_ROOT env var is required}"

# ── Unified data / model storage paths ───────────────────────────────────────
DATA_ROOT="${DATA_ROOT:?DATA_ROOT env var is required}"
export HF_HOME="${DATA_ROOT}/hf_cache"          # HuggingFace model cache
CKPT_ROOT="${DATA_ROOT}/harnessx_slime/ckpt"  # Megatron torch_dist checkpoints + training outputs
mkdir -p "${HF_HOME}" "${CKPT_ROOT}"

# ── High-speed download configuration ────────────────────────────────────────
# hf_transfer: multi-threaded concurrent downloads, 3-5x faster than default
export HF_HUB_ENABLE_HF_TRANSFER=0
# HF_ENDPOINT: uncomment to use mirror if direct HF connection is slow
export HF_ENDPOINT="https://hf-mirror.com"

SKIP_DATA=0
SKIP_MODELS=0
for arg in "$@"; do
    case "$arg" in
        --skip-data)   SKIP_DATA=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
    esac
done

# ── 1. Datasets ──────────────────────────────────────────────────────────────
if [ "$SKIP_DATA" -eq 0 ]; then
    echo "=== Preparing datasets ==="
    python3 -c "
import sys, logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
sys.path.insert(0, '$HX_ROOT')
from recipe.slime.math.data_prep import prepare_sft_data, prepare_rl_data, prepare_eval_data
prepare_sft_data()
prepare_rl_data()
prepare_eval_data()
print('All datasets ready.')
"
fi

# ── 2. Download models (default HF cache) ────────────────────────────────────
if [ "$SKIP_MODELS" -eq 0 ]; then
    echo "=== Downloading models to HF cache ==="
    python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
import os

print("Downloading Qwen/Qwen3-4B-Instruct-2507 ...")
base_path = snapshot_download("Qwen/Qwen3-4B-Instruct-2507")
print(f"  -> {base_path}")

print("Downloading font-info/qwen3-4b-sft ...")
sft_path = snapshot_download("font-info/qwen3-4b-sft")
print(f"  -> {sft_path}")

# Write paths to a file for the shell to read
with open("/tmp/slime_model_paths.sh", "w") as f:
    f.write(f'BASE_MODEL_PATH="{base_path}"\n')
    f.write(f'SFT_MODEL_PATH="{sft_path}"\n')

print("Model paths written to /tmp/slime_model_paths.sh")
PYEOF

    # ── 3. Convert to torch_dist ──────────────────────────────────────────────
    # shellcheck source=/dev/null
    source /tmp/slime_model_paths.sh
    source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

    echo "=== Converting Qwen3-4B-Instruct-2507 to torch_dist (SFT base) ==="
    cd "${SLIME_ROOT}"
    PYTHONPATH="${MEGATRON_ROOT}" python3 \
        "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${BASE_MODEL_PATH}" \
        --rotary-base 5000000 \
        --save "${CKPT_ROOT}/Qwen3-4B-Instruct-2507_torch_dist"

    echo "=== Converting qwen3-4b-sft to torch_dist (RL base) ==="
    PYTHONPATH="${MEGATRON_ROOT}" python3 \
        "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${SFT_MODEL_PATH}" \
        --rotary-base 5000000 \
        --save "${CKPT_ROOT}/qwen3-4b-sft_torch_dist"

    echo "=== torch_dist conversion complete ==="
fi

echo ""
echo "=== data_prep.sh done ==="
echo "Next steps:"
echo "  1. Run SFT:  bash ${HX_ROOT}/recipe/slime/launch/run_sft.sh"
echo "  2. Run RL:   bash ${HX_ROOT}/recipe/slime/launch/run_rl.sh"
