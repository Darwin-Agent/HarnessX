#!/usr/bin/env bash
# Evaluate TB2 using a self-hosted OpenSandbox server.
# Provides full control over networking, resource limits, and proxy settings.
#
# Required env vars:
#   ANTHROPIC_API_KEY   — model API key
#   ANTHROPIC_API_BASE  — model API base URL
#   OPENSANDBOX_URL     — OpenSandbox server URL (e.g. http://10.0.0.1:13081)
#
# Optional env vars:
#   MODEL                      — model ID passed to -m (default: claude-opus-4-6)
#   TB2_HARNESS_CONFIG         — path to evolved HarnessConfig YAML (default: builtin)
#   OPENSANDBOX_PROXY          — HTTP proxy injected into every sandbox exec() call
#   OPENSANDBOX_NO_PROXY       — hosts/IPs that bypass the proxy (comma-separated)
#   OPENSANDBOX_HEALTH_TIMEOUT — seconds to wait for /health before aborting (default: 120)
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
#   bash benchmarks/terminal_bench_2/scripts/eval_opensandbox.sh
#   bash benchmarks/terminal_bench_2/scripts/eval_opensandbox.sh --job-name my-run --resume
#   bash benchmarks/terminal_bench_2/scripts/eval_opensandbox.sh -n 4 -t crack-7z-hash
#   bash benchmarks/terminal_bench_2/scripts/eval_opensandbox.sh \
#     --harness-config /path/to/R1/config.yaml \
#     --tasks /path/to/tasks.json \
#     --job-name my-evolved-run -n 2
set -euo pipefail

: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is not set}"
: "${ANTHROPIC_API_BASE:?ANTHROPIC_API_BASE is not set}"
: "${OPENSANDBOX_URL:?OPENSANDBOX_URL is not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── OpenSandbox health check ──────────────────────────────────────────────────
_health_url="${OPENSANDBOX_URL%/}/health"
_timeout="${OPENSANDBOX_HEALTH_TIMEOUT:-120}"
_elapsed=0
_interval=5
echo "Checking OpenSandbox health: $_health_url (timeout: ${_timeout}s)"
while true; do
  _status=$(no_proxy='*' curl -s -o /dev/null -w "%{http_code}" --noproxy '*' \
            --connect-timeout 3 --max-time 5 "$_health_url" 2>/dev/null || echo "000")
  if [[ "$_status" == "200" ]]; then
    echo "OpenSandbox is healthy (${_elapsed}s)."
    break
  fi
  if (( _elapsed >= _timeout )); then
    echo "ERROR: OpenSandbox not healthy after ${_timeout}s (last HTTP status: $_status). Aborting." >&2
    exit 1
  fi
  echo "  waiting... ${_elapsed}s elapsed (HTTP $_status)"
  sleep "$_interval"
  _elapsed=$(( _elapsed + _interval ))
done
# ─────────────────────────────────────────────────────────────────────────────

PROXY_ARGS=()
if [[ -n "${OPENSANDBOX_PROXY:-}" ]]; then
  PROXY_ARGS=(--proxy-url "$OPENSANDBOX_PROXY")
fi

NO_PROXY_ARGS=()
if [[ -n "${OPENSANDBOX_NO_PROXY:-}" ]]; then
  NO_PROXY_ARGS=(--no-proxy "$OPENSANDBOX_NO_PROXY")
fi

MODEL="${MODEL:-claude-opus-4-6}"

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
  --env opensandbox \
  --sandbox-url "$OPENSANDBOX_URL" \
  "${PROXY_ARGS[@]}" \
  "${NO_PROXY_ARGS[@]}" \
  -m "$MODEL" \
  -k "$ANTHROPIC_API_KEY" \
  -b "$ANTHROPIC_API_BASE" \
  -n 1 \
  "${TASK_ARGS[@]}" \
  "${PASSTHROUGH_ARGS[@]}"
