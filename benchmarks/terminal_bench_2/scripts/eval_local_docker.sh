#!/usr/bin/env bash
# Evaluate TB2 using local Docker (no OpenSandbox account needed).
# Task-agent model is configured via TB2_MODEL/TB2_API_BASE/TB2_API_KEY,
# keeping them separate from the meta-agent's ANTHROPIC_* credentials.
# Proxy is forwarded to Docker containers via HTTP_PROXY/HTTPS_PROXY/NO_PROXY.
#
# Required env vars:
#   TB2_API_KEY         — task-agent API key
#   TB2_API_BASE        — task-agent API base URL
#   TB2_MODEL           — task-agent model ID
#
# Optional env vars:
#   TB2_HARNESS_CONFIG         — path to evolved HarnessConfig YAML (default: builtin)
#   HTTP_PROXY / HTTPS_PROXY   — proxy forwarded into docker containers via env
#   NO_PROXY                   — hosts/IPs bypassing the proxy
#
# Flags handled by this script (removed before forwarding to tb2_eval.py):
#   --harness-config <path>  Path to evolved HarnessConfig YAML; sets TB2_HARNESS_CONFIG.
#                            Omit to use the built-in default config.
#   --tasks <json_file>      JSON file containing a list of task name strings.
#                            Each name is expanded to a -t argument for tb2_eval.py.
#
# All other flags (--job-name, --resume, -n, -t, -l, --max-steps, ...) are
# forwarded as-is to tb2_eval.py.
#
# Usage:
#   bash benchmarks/terminal_bench_2/scripts/eval_local_docker.sh
#   bash benchmarks/terminal_bench_2/scripts/eval_local_docker.sh --job-name my-run --resume
#   bash benchmarks/terminal_bench_2/scripts/eval_local_docker.sh -n 4 -t crack-7z-hash
#   bash benchmarks/terminal_bench_2/scripts/eval_local_docker.sh \
#     --harness-config /path/to/R1/config.yaml \
#     --tasks /path/to/tasks.json \
#     --job-name my-evolved-run -n 2
set -euo pipefail

: "${TB2_API_KEY:?TB2_API_KEY is not set}"
: "${TB2_API_BASE:?TB2_API_BASE is not set}"
: "${TB2_MODEL:?TB2_MODEL is not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Pre-process script-level flags ───────────────────────────────────────────
# --harness-config and --tasks are handled here; all other args pass through.
HARNESS_CONFIG="${TB2_HARNESS_CONFIG:-}"
TASKS_JSON=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --harness-config)
      [[ $# -ge 2 ]] || { echo "ERROR: --harness-config requires a path argument" >&2; exit 1; }
      HARNESS_CONFIG="$2"
      shift 2
      ;;
    --tasks)
      [[ $# -ge 2 ]] || { echo "ERROR: --tasks requires a JSON file argument" >&2; exit 1; }
      TASKS_JSON="$2"
      shift 2
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

# Export evolved harness config if given (agent reads TB2_HARNESS_CONFIG).
if [[ -n "$HARNESS_CONFIG" ]]; then
  [[ -f "$HARNESS_CONFIG" ]] || { echo "ERROR: harness config not found: $HARNESS_CONFIG" >&2; exit 1; }
  export TB2_HARNESS_CONFIG="$HARNESS_CONFIG"
  echo "Using harness config: $HARNESS_CONFIG"
fi

# Expand --tasks JSON file into individual -t <name> args.
TASK_ARGS=()
if [[ -n "$TASKS_JSON" ]]; then
  [[ -f "$TASKS_JSON" ]] || { echo "ERROR: tasks file not found: $TASKS_JSON" >&2; exit 1; }
  while IFS= read -r _name; do
    [[ -n "$_name" ]] && TASK_ARGS+=(-t "$_name")
  done < <(
    python - "$TASKS_JSON" <<'PYEOF'
import json, sys
data = json.loads(open(sys.argv[1], encoding="utf-8").read())
if not isinstance(data, list):
    raise SystemExit(f"ERROR: --tasks JSON must be a list, got {type(data).__name__}")
for name in data:
    name = str(name).strip()
    if name:
        print(name)
PYEOF
  )
  echo "Tasks from ${TASKS_JSON}: ${#TASK_ARGS[@]} task(s)"
fi

# ── Launch tb2_eval.py ────────────────────────────────────────────────────────
exec python "$SCRIPT_DIR/tb2_eval.py" \
  --env docker \
  -m "$TB2_MODEL" \
  -k "$TB2_API_KEY" \
  -b "$TB2_API_BASE" \
  "${TASK_ARGS[@]}" \
  "${PASSTHROUGH_ARGS[@]}"
