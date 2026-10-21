#!/bin/bash
# run_research_rl.sh — verl GRPO training with HarnessX agent loop
# Usage: bash recipe/verl/launch/run_research_rl.sh [hydra overrides...]
#
# 训练参数在 config.yaml 中配置，此脚本只负责环境变量和路径。
# 工具配置由 recipe.slime.research.config.cfg.to_env_dict() 注入。

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export HX_ROOT="${HX_ROOT:-${SCRIPT_ROOT}}"
export VERL_ROOT="${VERL_ROOT:-${HX_ROOT}/recipe/verl/_vendor}"
export DATA_DIR="${DATA_DIR:?Set DATA_DIR to your training data directory}"
export MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to your local model directory}"
export CKPT_DIR="${CKPT_DIR:-${DATA_DIR}/harnessx-verl-runs}"
export SCRIPT_DIR="${HX_ROOT}/recipe/verl/launch"
export LOG_DIR="${CKPT_DIR}/logs"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# Python — patch via usercustomize (loaded by all Python processes)
# ═══════════════════════════════════════════════════════════════════════════════
SGLANG_ROOT="${SGLANG_ROOT:-}"
SGLANG_PYTHON="${SGLANG_PYTHON:-}"
if [[ -n "$SGLANG_PYTHON" ]]; then
    export PYTHONPATH="${HX_ROOT}:${VERL_ROOT}:${SGLANG_PYTHON}:${PYTHONPATH:-}"
elif [[ -n "$SGLANG_ROOT" && -d "$SGLANG_ROOT/python" ]]; then
    export PYTHONPATH="${HX_ROOT}:${VERL_ROOT}:${SGLANG_ROOT}/python:${PYTHONPATH:-}"
else
    export PYTHONPATH="${HX_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"
fi
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ═══════════════════════════════════════════════════════════════════════════════
# SGLang / NCCL
# ═══════════════════════════════════════════════════════════════════════════════
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_ATTENTION_BACKEND=triton
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_TIMEOUT=3600

# ═══════════════════════════════════════════════════════════════════════════════
# Rollout Timeout (aligned with verl_harnessX/launch.sh)
# ═══════════════════════════════════════════════════════════════════════════════
export VERL_SEQ_TIMEOUT="${VERL_SEQ_TIMEOUT:-300}"
export VERL_ROLLOUT_TIMEOUT="${VERL_ROLLOUT_TIMEOUT:-1800}"
export VERL_TOOL_CONCURRENCY_PER_WORKER="${VERL_TOOL_CONCURRENCY_PER_WORKER:-18}"
export VERL_TOOL_CALL_TIMEOUT="${VERL_TOOL_CALL_TIMEOUT:-12}"

# ═══════════════════════════════════════════════════════════════════════════════
# HarnessX 工具配置 (via cfg.to_env_dict())
# ═══════════════════════════════════════════════════════════════════════════════
eval "$(python3 -c "
from recipe.verl.research.config import cfg
for k, v in cfg.to_env_dict().items():
    if v:
        print(f'export {k}=\"{v}\"')
")"
mkdir -p "${CKPT_DIR}/downloads"

# ═══════════════════════════════════════════════════════════════════════════════
# API Keys & Endpoints
# ═══════════════════════════════════════════════════════════════════════════════
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
export ANTHROPIC_API_BASE="${ANTHROPIC_API_BASE:-}"

export MCP_SEARCH_URL="${MCP_SEARCH_URL:-}"
export MCP_SEARCH_KEY="${MCP_SEARCH_KEY:-}"

# ═══════════════════════════════════════════════════════════════════════════════
# HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════
export HF_HOME="${DATA_DIR}/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT="${HF_ENDPOINT:-}"

# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3.5 patches
# ═══════════════════════════════════════════════════════════════════════════════
if [ -f "${HX_ROOT}/recipe/slime/patches/apply_all_patches.sh" ]; then
    bash "${HX_ROOT}/recipe/slime/patches/apply_all_patches.sh" 2>/dev/null || true
fi

# SGLang FA3 compat patch (sgl_kernel 0.3.21 missing get_scheduler_metadata)
bash "${HX_ROOT}/recipe/verl/patches/patch_sglang_fa3.sh" 2>/dev/null || true

# SGLang stop string patch (tokenizer + stop_str_max_len for verl hybrid mode)
bash "${HX_ROOT}/recipe/verl/patches/patch_sglang_stop.sh" 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════════════════════
# Optionally stop existing SGLang/Ray processes on a dedicated training node.
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "${STOP_EXISTING_SERVICES:-0}" == "1" ]]; then
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    pkill -9 -f "raylet|gcs_server" 2>/dev/null || true
    unset RAY_ADDRESS
    sleep 3
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════════
python3 "$SCRIPT_DIR/main.py" \
    --config-path="$SCRIPT_DIR" \
    --config-name=config \
    "$@" \
    2>&1 | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
