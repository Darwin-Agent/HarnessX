#!/usr/bin/env bash
# run_aegis_airline.sh — tau2 Aegis evolution pilot for airline domain
#
# Runs the 6-stage AEGIS loop (Digester/Planner/Evolver/Critic + 5-gate pipeline)
# to automatically evolve the harness config for mimo-v2.5 on airline tasks.
#
# Usage (from HarnessX repo root):
#   ./recipe/tau2_evolver/run_aegis_airline.sh

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$HOME/tau2-bench/data}"

# ── Parameters ─────────────────────────────────────────────────────────────────
DOMAIN=airline
TASK_SPLIT=base
MAX_TASKS=20
NUM_ROUNDS=3
NUM_TRIALS=2
MAX_CONCURRENCY=16
MAX_SIM_STEPS=200
EVOLVE_COST=30.0

AGENT_MODEL="anthropic/mimo-v2.5"
AGENT_API_BASE="http://10.221.97.102:19010"

META_MODEL="anthropic/ppio/pa/claude-opus-4-6"
META_API_BASE="http://model.mify.ai.srv/anthropic"

OUTPUT_DIR=recipe/tau2_evolver/runs/aegis_airline_lite

# ── Run ────────────────────────────────────────────────────────────────────────
echo "[aegis_airline] Starting AEGIS pilot (${DOMAIN}, ${MAX_TASKS} tasks, ${NUM_ROUNDS} rounds) ..."
mkdir -p "${OUTPUT_DIR}"

python -m recipe.tau2_evolver.run_meta_aegis \
    --domain          "${DOMAIN}"          \
    --task-split      "${TASK_SPLIT}"      \
    --base-config     recipe/tau2_evolver/configs/airline_base \
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

echo "[aegis_airline] Done. Results: ${OUTPUT_DIR}/curves.json"
