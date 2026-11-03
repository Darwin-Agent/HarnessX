#!/usr/bin/env bash
# Launch HarnessX GRPO without modifying installed dependencies.
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export HX_ROOT="${HX_ROOT:-${SCRIPT_ROOT}}"
export VERL_ROOT="${HX_ROOT}/.deps/verl"
export DATA_DIR="${DATA_DIR:?Set DATA_DIR to your training data directory}"
export MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to your local model directory}"
export CKPT_DIR="${CKPT_DIR:-${DATA_DIR}/harnessx-verl-runs}"
export SCRIPT_DIR="${HX_ROOT}/recipe/verl/launch"
export LOG_DIR="${CKPT_DIR}/logs"
export PYTHONPATH="${HX_ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HOME="${HF_HOME:-${DATA_DIR}/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export VERL_SEQ_TIMEOUT="${VERL_SEQ_TIMEOUT:-600}"
export VERL_ROLLOUT_TIMEOUT="${VERL_ROLLOUT_TIMEOUT:-1800}"

PYTHON="${PYTHON:-python3}"
"${PYTHON}" -m recipe.verl.bootstrap --check
"${PYTHON}" -m recipe.verl.launch.preflight
mkdir -p "${LOG_DIR}" "${CKPT_DIR}/downloads"

if [[ "${STOP_EXISTING_SERVICES:-0}" == "1" ]]; then
    ray stop --force
    unset RAY_ADDRESS
fi

"${PYTHON}" -m recipe.verl.launch.environment "${PYTHON}" "${SCRIPT_DIR}/main.py" \
    --config-path="${SCRIPT_DIR}" --config-name=config "$@" \
    2>&1 | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
