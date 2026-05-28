#!/usr/bin/env bash
# run_aegis_telecom.sh — tau2 Aegis evolution pilot for telecom domain (base split, 114 tasks)

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$HOME/tau2-bench/data}"

DOMAIN=telecom
TASK_SPLIT=base
MAX_TASKS=0
NUM_ROUNDS=10
NUM_TRIALS=2
MAX_CONCURRENCY=16
MAX_SIM_STEPS=200
EVOLVE_COST=30.0

AGENT_MODEL="anthropic/mimo-v2.5"
AGENT_API_BASE="http://10.221.97.102:19010"

META_MODEL="anthropic/ppio/pa/claude-opus-4-6"
META_API_BASE="http://model.mify.ai.srv/anthropic"

OUTPUT_DIR=recipe/tau2_evolver/runs/aegis_telecom_full

echo "[aegis_telecom] Starting AEGIS pilot (${DOMAIN}, ${TASK_SPLIT} split, ${NUM_ROUNDS} rounds) ..."
mkdir -p "${OUTPUT_DIR}"

python -m recipe.tau2_evolver.run_meta_aegis \
    --domain          "${DOMAIN}"          \
    --task-split      "${TASK_SPLIT}"      \
    --base-config     recipe/tau2_evolver/configs/telecom_base \
    --output-dir      "${OUTPUT_DIR}"      \
    --agent-model     "${AGENT_MODEL}"     \
    --agent-api-base  "${AGENT_API_BASE}"  \
    --max-tasks       "${MAX_TASKS}"       \
    --num-rounds      "${NUM_ROUNDS}"      \
    --num-trials      "${NUM_TRIALS}"      \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --max-sim-steps   "${MAX_SIM_STEPS}"   \
    --evolve-cost     "${EVOLVE_COST}"     \
    --meta-model      "${META_MODEL}"      \
    --meta-api-base   "${META_API_BASE}"   \
    --meta-extended-thinking \
    2>&1 | tee "${OUTPUT_DIR}/run.log"

echo "[aegis_telecom] Done. Results: ${OUTPUT_DIR}/curves.json"
