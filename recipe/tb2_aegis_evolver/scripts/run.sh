#!/usr/bin/env bash
# TB2 AEGIS evolution runner.
# Copy scripts/run.env.example → scripts/run.env, fill in your values, then run:
#   bash recipe/tb2_aegis_evolver/scripts/run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ENV_FILE="${AEGIS_ENV_FILE:-${SCRIPT_DIR}/.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: env file not found: ${ENV_FILE}" >&2
    echo "  Copy ${SCRIPT_DIR}/run.env.example → ${SCRIPT_DIR}/.env and fill in your values." >&2
    exit 1
fi

cd "${REPO_ROOT}"
exec python -m recipe.tb2_aegis_evolver.run --env "${ENV_FILE}" "$@"
