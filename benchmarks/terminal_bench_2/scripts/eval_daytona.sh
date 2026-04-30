#!/usr/bin/env bash
# Evaluate TB2 using Daytona (non-internet tasks).
# Daytona Tier 1/2 blocks general outbound HTTPS — use for the 77 non-internet tasks.
#
# Required env vars:
#   ANTHROPIC_API_KEY   — model API key
#   ANTHROPIC_API_BASE  — model API base URL
#   DAYTONA_API_KEY     — Daytona workspace API key
#
# Optional env vars:
#   MODEL               — model ID passed to -m (default: claude-opus-4-6)
#
# Usage:
#   bash benchmarks/terminal_bench_2/scripts/eval_daytona.sh
#   bash benchmarks/terminal_bench_2/scripts/eval_daytona.sh --job-name my-run --resume
#   bash benchmarks/terminal_bench_2/scripts/eval_daytona.sh -n 4 -t crack-7z-hash
set -euo pipefail

: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is not set}"
: "${ANTHROPIC_API_BASE:?ANTHROPIC_API_BASE is not set}"
: "${DAYTONA_API_KEY:?DAYTONA_API_KEY is not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODEL:-claude-opus-4-6}"

exec python "$SCRIPT_DIR/tb2_eval.py" \
  --env daytona \
  -m "$MODEL" \
  -k "$ANTHROPIC_API_KEY" \
  -b "$ANTHROPIC_API_BASE" \
  -n 2 \
  "$@"
