#!/bin/bash
# run_sft.sh — Phase 1: SFT training on ReTool-SFT dataset.
#
# Mirrors retool_qwen3_4b_sft.sh exactly.
# Changes from retool:
#   - HF checkpoint: resolved dynamically from ${DATA_ROOT}/hf_cache
#   - ref-load / save: under ${DATA_ROOT}/harnessx_slime/ckpt/
#   - prompt-data: ${HX_ROOT}/data/slime/retool/ReTool-SFT.parquet
#   - PYTHONPATH: includes ${HX_ROOT}
#
# Required env vars: HX_ROOT, DATA_ROOT, SLIME_ROOT, MEGATRON_ROOT
#
# Usage:
#   bash recipe/slime/launch/run_sft.sh

# Kill stale processes (identical to retool)
pkill -9 sglang 2>/dev/null; sleep 3
ray stop --force 2>/dev/null
pkill -9 ray 2>/dev/null; pkill -9 python 2>/dev/null; sleep 3
pkill -9 ray 2>/dev/null; pkill -9 python 2>/dev/null

set -ex

export PYTHONBUFFERED=16

SLIME_ROOT="${SLIME_ROOT:?SLIME_ROOT env var is required}"
HX_ROOT="${HX_ROOT:?HX_ROOT env var is required}"
MEGATRON_ROOT="${MEGATRON_ROOT:?MEGATRON_ROOT env var is required}"

# ── Unified data / model storage paths ───────────────────────────────────────
DATA_ROOT="${DATA_ROOT:?DATA_ROOT env var is required}"
export HF_HOME="${DATA_ROOT}/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT="https://hf-mirror.com"
CKPT_ROOT="${DATA_ROOT}/harnessx_slime/ckpt"

# NVLink detection (identical to retool)
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([[ "$NVLINK_COUNT" -gt 0 ]] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK"

# Resolve HF cache path for Qwen3-4B base model
BASE_MODEL_PATH=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-4B-Instruct-2507'))
")
echo "Base model: ${BASE_MODEL_PATH}"

# Model architecture args (identical to retool — qwen3-4B.sh)
source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

# ── Args (identical to retool_qwen3_4b_sft.sh) ───────────────────────────────

CKPT_ARGS=(
    --hf-checkpoint "${BASE_MODEL_PATH}"
    --ref-load      "${CKPT_ROOT}/Qwen3-4B-Instruct-2507_torch_dist"
    --save          "${CKPT_ROOT}/Qwen3-4B-Instruct-2507_sft/"
    --save-interval 1000
    --rotary-base   5000000
)

SFT_ARGS=(
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout
    --prompt-data   "${HX_ROOT}/data/slime/retool/ReTool-SFT.parquet"
    --input-key     messages
    --rollout-shuffle
    --num-epoch     3
    --rollout-batch-size 128
    --global-batch-size  128
    --loss-type          sft_loss
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
    --debug-train-only
)

PERF_ARGS=(
    --tensor-model-parallel-size  1
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size        1
    --expert-model-parallel-size   1
    --expert-tensor-parallel-size  1
    --recompute-granularity  full
    --recompute-method       uniform
    --recompute-num-layers   1
    --use-dynamic-batch-size
    --max-tokens-per-gpu     9216
)

OPTIMIZER_ARGS=(
    --optimizer         adam
    --lr                1e-5
    --lr-decay-style    cosine
    --min-lr            1e-6
    --lr-warmup-fraction 0.1
    --weight-decay      0.1
    --adam-beta1        0.9
    --adam-beta2        0.95
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project  slime-retool-harnessx
    --wandb-group    qwen3-4B-sft
    --wandb-key      "${WANDB_KEY}"
)

MISC_ARGS=(
    --attention-dropout      0.0
    --hidden-dropout         0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend      flash
)

# ── Launch Ray ────────────────────────────────────────────────────────────────
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"

LOCAL_IPS=$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//')
NO_PROXY_LIST="127.0.0.1,localhost,${MASTER_ADDR}${LOCAL_IPS:+,${LOCAL_IPS}}"

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus 8 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"working_dir\": \"${SLIME_ROOT}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SLIME_ROOT}:${HX_ROOT}\",
    \"HF_HOME\": \"${DATA_ROOT}/hf_cache\",
    \"HF_HUB_ENABLE_HF_TRANSFER\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"no_proxy\": \"${NO_PROXY_LIST}\",
    \"NO_PROXY\": \"${NO_PROXY_LIST}\"
  }
}"

cd "${SLIME_ROOT}"
ray job submit \
    --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
        --actor-num-nodes         1 \
        --actor-num-gpus-per-node 8 \
        "${MODEL_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${SFT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${MISC_ARGS[@]}"
