#!/usr/bin/env bash
# Start evolution using r0-baseline as the warm-start R0 trajectories.
#
# Flow:
#   R0: read existing r0-baseline trajectories (no re-run)
#   evolve R0→R1: meta-agent produces new config
#   R1: run 16 tasks with the new config  ← validates improvement
#   (repeat for NUM_ROUNDS)
#
# Usage:
#   cd /root/evolution/HarnessX
#   bash recipe/tb2_evolver/scripts/start_evolve_r0baseline.sh
#
# Optional env overrides:
#   RUN_TAG=my-run-tag      (default: evolve_YYYYMMDD-HHMMSS)
#   NUM_ROUNDS=6            (default: 6)
#   CONCURRENT=10           (TB2 eval concurrency for rerun rounds)
#   RESUME=1                (resume an existing RUN_TAG from where it left off)
#   REQUIRE_EVIDENCE=0      (skip candidates.md requirement, default: 0 = skip)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load env vars from recipe/.env (META_MODEL, ANTHROPIC_API_KEY, etc.)
set -a
source "$PROJECT_ROOT/recipe/tb2_evolver/.env"
set +a

cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-evolve_to18k$(date +%Y%m%d-%H%M%S)}"
NUM_ROUNDS="${NUM_ROUNDS:-6}"
RESUME="${RESUME:-0}"
REQUIRE_EVIDENCE="${REQUIRE_EVIDENCE:-0}"

RESUME_FLAG=()
if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
    RESUME_FLAG=(--resume)
    echo "=== TB2 Evolution (RESUME mode) ==="
else
    echo "=== TB2 Evolution (rerun mode, r0-baseline warm-start) ==="
fi

EVIDENCE_FLAG=()
if [[ "$REQUIRE_EVIDENCE" != "1" && "$REQUIRE_EVIDENCE" != "true" ]]; then
    EVIDENCE_FLAG=(--no-require-evidence)
fi

echo "  run-tag:          $RUN_TAG"
echo "  r0-dir:           .benchmarks/tb2-baseline-results/r0-baseline  (R0 warm-start, no re-run)"
echo "  tasks:            recipe/tb2_evolver/tasks_sample16_seed42_act15.json"
echo "  num-rounds:       $NUM_ROUNDS"
echo "  trajectory-mode:  rerun  (new config is validated by running 16 tasks)"
echo "  concurrency:      ${CONCURRENT:-8}"
echo "  model:            ${META_MODEL:-<from .env>}"
echo "  evolve-cost-cap:  ${EVOLVE_COST_CAP_USD:-30.0} USD"
echo "  evolve-steps:     ${EVOLVE_MAX_STEPS:-500}"
echo "  evolve-wall-clock:${EVOLVE_WALL_CLOCK_S:-3600}s"
echo "  require-evidence: ${REQUIRE_EVIDENCE:-0}"
echo ""

exec python -m recipe.tb2_evolver.run \
  --r0-dir .benchmarks/tb2-baseline-results/r0-baseline \
  --tasks recipe/tb2_evolver/tasks_sample16_seed42_act15.json \
  --run-tag "$RUN_TAG" \
  --num-rounds "$NUM_ROUNDS" \
  --trajectory-mode rerun \
  --tb2-eval-script benchmarks/terminal_bench_2/scripts/eval_local_docker.sh \
  --tb2-eval-concurrent "${CONCURRENT:-8}" \
  "${RESUME_FLAG[@]}" \
  "${EVIDENCE_FLAG[@]}" \
  "$@"
