#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${GAIA_DATA_PATH:?Set GAIA_DATA_PATH to a local GAIA JSON file}"
: "${GAIA_MODEL:?Set GAIA_MODEL to the task-agent provider/model}"
: "${GAIA_META_MODEL:?Set GAIA_META_MODEL to the meta-agent provider/model}"
: "${GAIA_PROVIDER_ID:?Set GAIA_PROVIDER_ID for the selected provider}"
: "${DATA_DIR:?Set DATA_DIR to train.parquet and val.parquet directory}"
: "${MODEL_DIR:?Set MODEL_DIR to the local model directory}"
: "${CKPT_DIR:?Set CKPT_DIR to a writable checkpoint directory}"

[[ -f "$GAIA_DATA_PATH" ]] || { echo "GAIA data file not found" >&2; exit 1; }
[[ -f "$DATA_DIR/train.parquet" ]] || { echo "Training parquet not found" >&2; exit 1; }
[[ -f "$DATA_DIR/val.parquet" ]] || { echo "Validation parquet not found" >&2; exit 1; }
[[ -d "$MODEL_DIR" ]] || { echo "Model directory not found" >&2; exit 1; }
RUN_TAG="${RUN_TAG:-sequential-$(date +%Y%m%d-%H%M%S)}"

echo "[1/2] Running GAIA harness evolution (${RUN_TAG})"
python3 "$ROOT/recipe/evolution_pipeline/run_gaia.py" \
    --model "$GAIA_MODEL" \
    --meta-model "$GAIA_META_MODEL" \
    --provider-id "$GAIA_PROVIDER_ID" \
    --data-path "$GAIA_DATA_PATH" \
    --max-tasks "${GAIA_MAX_TASKS:-3}" \
    --num-rounds "${GAIA_NUM_ROUNDS:-2}" \
    --concurrency "${GAIA_CONCURRENCY:-1}" \
    --run-tag "$RUN_TAG"

echo "[2/2] Starting veRL GRPO model training"
echo "Prepare DATA_DIR Parquet files explicitly; GAIA trajectories are not auto-converted."
bash "$ROOT/recipe/verl_harnessX/run_train.sh" "$@"

echo "Sequential evolution completed. GAIA outputs: recipe/gaia_evolver/runs/${RUN_TAG}"
