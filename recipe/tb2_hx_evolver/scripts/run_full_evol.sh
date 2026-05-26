#!/usr/bin/env bash
# Run multi-round TB2 harness evolution (full loop with eval).
#
# Usage:
#   bash recipe/tb2_hx_evolver/scripts/run_full_evol.sh
#   bash recipe/tb2_hx_evolver/scripts/run_full_evol.sh --env /path/to/.env
#   bash recipe/tb2_hx_evolver/scripts/run_full_evol.sh --config /path/to/harness_config.yaml
#   bash recipe/tb2_hx_evolver/scripts/run_full_evol.sh --resume --run-dir /path/to/existing-run
#
# All flags are forwarded to run_full_evol.py.
#
# Supported .env keys (set in --env file or as shell env vars):
#   EVOL_CONFIG           Initial harness config YAML
#   EVOL_RUN_DIR          Output directory (default: .benchmarks/evolve-runs/evolve-<ts>)
#   EVOL_R0_TRIALS        Existing trials dir for round-0 warm-start (skips first eval)
#   EVOL_TASKS            Task list JSON
#   EVOL_NUM_ROUNDS       Number of rounds (default: 5)
#   EVOL_CONCURRENT       Eval concurrency (default: 4)
#   EVOL_K                Rollouts per task (default: 3)
#   EVOL_DIGEST_MODEL     DigestAgent model (default: anthropic/claude-haiku-4-5-20251001)
#   EVOL_EVOLVE_MODEL     EvolveAgent model (default: anthropic/claude-sonnet-4-6)
#   EVOL_DIGEST_MAX_STEPS DigestAgent step budget (default: 200)
#   EVOL_EVOLVE_MAX_STEPS EvolveAgent step budget (default: 300)
#   EVOL_BASE_URL         API base URL for evol agents
#   EVOL_API_KEY          API key for evol agents
#   EVOL_PROVIDER_ID      Optional X-Model-Provider-Id header value

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Default .env path ─────────────────────────────────────────────────────────
# Look for .env next to this script, then at repo root.
DEFAULT_ENV=""
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  DEFAULT_ENV="$SCRIPT_DIR/.env"
elif [[ -f "$REPO_ROOT/.env" ]]; then
  DEFAULT_ENV="$REPO_ROOT/.env"
fi

# ── Forward all args to Python; inject --env default if not already supplied ─
ENV_ARGS=()
has_env_flag=false
for arg in "$@"; do
  [[ "$arg" == "--env" || "$arg" == --env=* ]] && has_env_flag=true
done
if ! $has_env_flag && [[ -n "$DEFAULT_ENV" ]]; then
  ENV_ARGS=(--env "$DEFAULT_ENV")
  echo "  [env] loading: $DEFAULT_ENV"
fi

cd "$REPO_ROOT"
exec python -m recipe.tb2_hx_evolver.run_full_evol "${ENV_ARGS[@]}" "$@"
