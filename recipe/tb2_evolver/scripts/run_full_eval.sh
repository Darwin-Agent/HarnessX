#!/usr/bin/env bash
# Run the full TB2 task set (tasks_all_tb2.json) with a given harness config.
#
# Usage:
#   bash recipe/tb2_evolver/scripts/run_full_eval.sh --config <path/to/config.yaml>
#   bash recipe/tb2_evolver/scripts/run_full_eval.sh --config runs/evolve_.../R4/config.yaml -n 4
#   bash recipe/tb2_evolver/scripts/run_full_eval.sh --config ... --job-name my-run --resume
#
# Required:
#   --config <path>   Path to evolved HarnessConfig YAML.
#
# Optional:
#   --tasks  <path>   Task list JSON (default: tasks_all_tb2.json next to this script).
#   -n <int>          Concurrency (default: 2).
#   --job-name <str>  Job name for output dir (default: derived from config path).
#   All other flags are forwarded to eval_local_docker.sh → tb2_eval.py.
#
# TB2_MODEL / TB2_API_BASE / TB2_API_KEY are loaded from .env if not already set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

EVAL_SCRIPT="$REPO_ROOT/benchmarks/terminal_bench_2/scripts/eval_local_docker.sh"
DEFAULT_ENV_FILE="$REPO_ROOT/recipe/tb2_evolver/.env"
DEFAULT_TASKS_JSON="$SCRIPT_DIR/../tasks_all_tb2.json"

# ── Parse script-level args ───────────────────────────────────────────────────
HARNESS_CONFIG=""
TASKS_JSON="$DEFAULT_TASKS_JSON"
ENV_FILE="$DEFAULT_ENV_FILE"
PASSTHROUGH=()
HAS_N=false
HAS_JOB=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      [[ $# -ge 2 ]] || { echo "ERROR: --env-file requires a path" >&2; exit 1; }
      ENV_FILE="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "ERROR: --config requires a path" >&2; exit 1; }
      HARNESS_CONFIG="$2"
      shift 2
      ;;
    --tasks)
      [[ $# -ge 2 ]] || { echo "ERROR: --tasks requires a path" >&2; exit 1; }
      TASKS_JSON="$2"
      shift 2
      ;;
    -n)
      HAS_N=true
      PASSTHROUGH+=("$1" "$2")
      shift 2
      ;;
    --job-name)
      HAS_JOB=true
      PASSTHROUGH+=("$1" "$2")
      shift 2
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

# ── Load env file so all vars (TB2_*, CONCURRENT, …) are available ────────────
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ -z "$HARNESS_CONFIG" ]]; then
  echo "ERROR: --config <path/to/config.yaml> is required" >&2
  echo "  Example: --config recipe/tb2_evolver/runs/evolve_to18k20260429-232938/R4/config.yaml" >&2
  exit 1
fi

HARNESS_CONFIG="$(realpath "$HARNESS_CONFIG")"
[[ -f "$HARNESS_CONFIG" ]] || { echo "ERROR: config not found: $HARNESS_CONFIG" >&2; exit 1; }
[[ -f "$TASKS_JSON" ]]     || { echo "ERROR: tasks file not found: $TASKS_JSON" >&2; exit 1; }
[[ -f "$EVAL_SCRIPT" ]]    || { echo "ERROR: eval script not found: $EVAL_SCRIPT" >&2; exit 1; }

# ── Default job-name: derived from config path (run_id + round) ───────────────
if ! $HAS_JOB; then
  _round="$(basename "$(dirname "$HARNESS_CONFIG")")"
  _run="$(basename "$(dirname "$(dirname "$HARNESS_CONFIG")")")"
  PASSTHROUGH+=(--job-name "${_run}-${_round,,}-full")
fi

$HAS_N || PASSTHROUGH+=(-n "${CONCURRENT:-2}")

# ── Print summary ─────────────────────────────────────────────────────────────
_ntasks="$(python3 -c "import json; print(len(json.load(open('$TASKS_JSON'))))" 2>/dev/null || echo '?')"
echo "========================================"
echo "  TB2 full eval"
echo "========================================"
echo "  Config   : $HARNESS_CONFIG"
echo "  Tasks    : $TASKS_JSON  ($_ntasks tasks)"
echo "  Model    : ${TB2_MODEL:-<not set>}"
echo "  API base : ${TB2_API_BASE:-<not set>}"
echo "  Extra args: ${PASSTHROUGH[*]}"
echo "========================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
exec bash "$EVAL_SCRIPT" \
  --harness-config "$HARNESS_CONFIG" \
  --tasks "$TASKS_JSON" \
  "${PASSTHROUGH[@]}"
