#!/bin/bash
# run_math_rl.sh — Phase 2: RL training with HarnessX tool execution.
#
# Aligned with OpenClaw-RL retool_qwen3_4b_rl.sh parameters.
# Changes from retool:
#   - CUSTOM_ARGS: generate_with_retool → recipe.slime.harness_rollout
#   - HF checkpoint: resolved dynamically from ${DATA_ROOT}/hf_cache
#   - ref-load / save: under ${DATA_ROOT}/harnessx_slime/ckpt/
#   - prompt-data / eval-data: ${HX_ROOT}/data/slime/retool/
#   - PYTHONPATH: includes ${HX_ROOT} so recipe.slime.* is importable
#
# Prerequisites:
#   1. data_prep.sh has been run (data files and torch_dist checkpoints exist)
#   2. run_sft.sh has been run, OR font-info/qwen3-4b-sft has been downloaded
#
# Required env vars: HX_ROOT, DATA_ROOT, SLIME_ROOT, MEGATRON_ROOT
#
# Usage:
#   bash recipe/slime/launch/run_rl.sh

# Kill stale processes (identical to retool)
pkill -9 sglang 2>/dev/null; sleep 3
ray stop --force 2>/dev/null
pkill -9 ray 2>/dev/null; pkill -9 python 2>/dev/null; sleep 3
pkill -9 ray 2>/dev/null; pkill -9 python 2>/dev/null

set -ex

# keep stdout/stderr unbuffered in ray jobs
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HARNESSX_SAMPLE_TIMEOUT=360
export KEEP_LAST_CKPTS=10

# default to 8 GPUs if not set by scheduler
NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}

# async mode usually runs actor/rollout on separate GPUs
if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

# Increase Ray heartbeat/health-check timeouts to reduce false node failures under heavy init.
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

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

# SFT HF model (font-info/qwen3-4b-sft), used for:
#   --hf-checkpoint : load architecture config and initial actor weights
# SFT torch_dist (converted by data_prep.sh), used for:
#   --ref-load      : KL divergence reference model
SFT_HF_PATH=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('font-info/qwen3-4b-sft', local_files_only=True))
")
echo "SFT HF model: ${SFT_HF_PATH}"

SFT_TORCH_DIST_PATH="${CKPT_ROOT}/qwen3-4b-sft_torch_dist"
echo "SFT torch_dist: ${SFT_TORCH_DIST_PATH}"

SAVE_CKPT="${CKPT_ROOT}/qwen3-4b-harnessx-rl/"
RESUME_LOAD=${RESUME_LOAD:-${SAVE_CKPT}}

# ── Persist wandb run id across restarts ─────────────────────────────────────
WANDB_RUN_ID_FILE="${SAVE_CKPT}/wandb_run_id.txt"
if [[ -f "${WANDB_RUN_ID_FILE}" ]]; then
    WANDB_RESUME_ID=$(cat "${WANDB_RUN_ID_FILE}")
    echo "Resuming wandb run: ${WANDB_RESUME_ID}"
else
    WANDB_RESUME_ID=""
fi

# Model architecture args (identical to retool)
source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

# ── Args (aligned with retool_qwen3_4b_rl.sh) ──────────────────────────────

CKPT_ARGS=(
    --hf-checkpoint "${SFT_HF_PATH}"
    --ref-load      "${SFT_TORCH_DIST_PATH}"
    --load          "${RESUME_LOAD}"
    --save          "${SAVE_CKPT}"
    --save-interval 80
    --rotary-base   5000000
)

ROLLOUT_ARGS=(
    --prompt-data            "${HX_ROOT}/data/slime/retool/dapo-math-17k.jsonl"
    --input-key              prompt
    --label-key              label
    --apply-chat-template
    --rollout-shuffle
    --reward-key             score
    --num-rollout            3000
    --rollout-batch-size     32
    --n-samples-per-prompt   8
    --rollout-max-response-len 8192
    --rollout-max-context-len  16384
    --rollout-temperature    1

    --num-steps-per-rollout  2
    --balance-data
)

# Eval disabled — remove comment and uncomment to re-enable
# EVAL_ARGS=(
#     --eval-interval          20
#     --eval-prompt-data aime  "${HX_ROOT}/data/slime/retool/aime-2024.jsonl"
#     --n-samples-per-eval-prompt 16
#     --eval-max-response-len  16384
#     --eval-max-context-len   32768
#     --eval-top-p             1
#     --eval-reward-key        is_correct
# )
EVAL_ARGS=()

PERF_ARGS=(
    --tensor-model-parallel-size  4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size        1
    --expert-model-parallel-size   1
    --expert-tensor-parallel-size  1

    --recompute-granularity  full
    --recompute-method       uniform
    --recompute-num-layers   1

    --use-dynamic-batch-size
    --max-tokens-per-gpu     16384
    --log-probs-chunk-size   1024
)

GRPO_ARGS=(
    --advantage-estimator    grpo
    --use-kl-loss
    --kl-loss-coef           0.01
    --kl-loss-type           k3
    --entropy-coef           0.00
    --eps-clip               0.2
    --eps-clip-high          0.28
)

OPTIMIZER_ARGS=(
    --optimizer              adam
    --lr                     1e-6
    --lr-decay-style         constant
    --weight-decay           0.1
    --adam-beta1             0.9
    --adam-beta2             0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project  slime-dapo-math-17k
    --wandb-group    qwen3-4B-test-math-17k
    --wandb-key      "${WANDB_KEY}"
    --log-passrate
    --log-reward-category exit_reason
    --disable-wandb-random-suffix
    ${WANDB_RESUME_ID:+--wandb-run-id "${WANDB_RESUME_ID}"}
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static  0.6
)

MISC_ARGS=(
    # default dropout in megatron is 0.1
    --attention-dropout      0.0
    --hidden-dropout         0.0
    # should be good for model performance
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    # need to comment this when using model with MLA
    --attention-backend      flash
)

# ── Key change: Harness.run()-based generate() + reward_func() ───────────────
CUSTOM_ARGS=(
    --custom-generate-function-path recipe.slime.harness_rollout.generate
    --custom-rm-path                recipe.slime.harness_rollout.reward_func
)

# ── Launch Ray ────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"

# Collect all local IPs so Ray workers can bypass proxy for intra-node traffic
# (SGLang servers bind to the actual host IP, not 127.0.0.1)
LOCAL_IPS=$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//')
NO_PROXY_LIST="127.0.0.1,localhost,${MASTER_ADDR}${LOCAL_IPS:+,${LOCAL_IPS}}"
echo "no_proxy for Ray workers: ${NO_PROXY_LIST}"

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus ${NUM_GPUS} \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SLIME_ROOT}:${HX_ROOT}\",
    \"HF_HOME\": \"${DATA_ROOT}/hf_cache\",
    \"HF_HUB_ENABLE_HF_TRANSFER\": \"1\",
    \"HF_ENDPOINT\": \"https://hf-mirror.com\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"no_proxy\": \"${NO_PROXY_LIST}\",
    \"NO_PROXY\": \"${NO_PROXY_LIST}\",
    \"KEEP_LAST_CKPTS\": \"${KEEP_LAST_CKPTS}\",
    \"HARNESSX_SAMPLE_TIMEOUT\": \"${HARNESSX_SAMPLE_TIMEOUT}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_ROOT}/train_async.py" \
        --actor-num-nodes         1 \
        --actor-num-gpus-per-node ${ACTOR_GPUS} \
        --rollout-num-gpus        ${ROLLOUT_GPUS} \
        "${MODEL_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${GRPO_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${EVAL_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${MISC_ARGS[@]}" \
        "${CUSTOM_ARGS[@]}"
