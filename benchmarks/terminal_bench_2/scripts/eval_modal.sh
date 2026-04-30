#!/usr/bin/env bash
# Evaluate TB2 using Modal (internet tasks).
# Modal containers have outbound internet access — required for tasks that
# download models or datasets.
#
# Required env vars:
#   ANTHROPIC_API_KEY    — model API key
#   ANTHROPIC_API_BASE   — model API base URL
#   MODAL_TOKEN_ID       — Modal token ID (get from modal.com)
#   MODAL_TOKEN_SECRET   — Modal token secret
#
# Optional env vars:
#   MODEL                — model ID passed to -m (default: claude-opus-4-6)
#
# NOTE: do NOT run `modal token new` or rely on a pre-existing ~/.modal.toml —
# both trigger browser OAuth in headless environments. This script calls
# `modal token set` to write credentials programmatically before running.
#
# Usage:
#   bash benchmarks/terminal_bench_2/scripts/eval_modal.sh
#   bash benchmarks/terminal_bench_2/scripts/eval_modal.sh --job-name my-run --resume
#   bash benchmarks/terminal_bench_2/scripts/eval_modal.sh -t hf-model-inference
set -euo pipefail

: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is not set}"
: "${ANTHROPIC_API_BASE:?ANTHROPIC_API_BASE is not set}"
: "${MODAL_TOKEN_ID:?MODAL_TOKEN_ID is not set}"
: "${MODAL_TOKEN_SECRET:?MODAL_TOKEN_SECRET is not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Write credentials to ~/.modal.toml so the Modal SDK can authenticate.
# This avoids relying on a pre-existing profile or browser OAuth.
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"

MODEL="${MODEL:-claude-opus-4-6}"

exec python "$SCRIPT_DIR/tb2_eval.py" \
  --env modal \
  -m "$MODEL" \
  -k "$ANTHROPIC_API_KEY" \
  -b "$ANTHROPIC_API_BASE" \
  -n 1 \
  "$@"
