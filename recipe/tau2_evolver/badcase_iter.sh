#!/usr/bin/env bash
# badcase_iter.sh — iterative badcase-driven evolution
#
# Algorithm:
#   iter 1  — full task set: R0 (baseline sim) + evolve → R1 (C1 eval) → collect B1
#   iter k  — B_{k-1} only: R0 (load C_{k-1} report) + evolve → R1 (Ck eval) → Bk
#   stop    — Bk is empty, or iterations exceed --max-iter
#
# Usage (run from the HarnessX repo root):
#   ./recipe/tau2_evolver/badcase_iter.sh
#   ./recipe/tau2_evolver/badcase_iter.sh --run-prefix my_exp --max-iter 5
#
# Options:
#   --run-prefix  NAME    prefix for per-iteration run-tags; metadata written to
#                         runs/<NAME>_meta/  (default: badcase_iter)
#   --max-iter    N       hard upper bound on iterations (default: 10)
#   --base-config PATH    baseline harness_config.yaml for iter 1
#                         (default: benchmarks/tau2/harness_config_base.yaml)
#   --domain      NAME    tau2 domain (default: retail)
#   any other flag        forwarded to every run.py call

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

# Internal .srv endpoints must NOT go through the WSL2→Windows proxy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$HOME/tau2-bench/data}"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_PREFIX="badcase_iter"
MAX_ITER=10
BASE_CONFIG="benchmarks/tau2/harness_config_base.yaml"
DOMAIN="retail"

# Extra flags forwarded to every run.py call (override on the command line).
PASSTHROUGH=(
    --evolve-cost 200
    --max-concurrency 30
    --regression-tolerance 0.05
    --agent-temperature 0.3
    --user-temperature 0
)

# ── CLI parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-prefix)  RUN_PREFIX="$2";  shift 2 ;;
        --max-iter)    MAX_ITER="$2";    shift 2 ;;
        --base-config) BASE_CONFIG="$2"; shift 2 ;;
        --domain)      DOMAIN="$2";      shift 2 ;;
        *)             PASSTHROUGH+=("$1"); shift ;;
    esac
done

# ── Directories / logging ─────────────────────────────────────────────────────
RUNS_DIR="recipe/tau2_evolver/runs"
META_DIR="${RUNS_DIR}/${RUN_PREFIX}_meta"
mkdir -p "$META_DIR"

LOG_FILE="${META_DIR}/run.log"
SUMMARY_FILE="${META_DIR}/summary.md"
FIXES_FILE="${META_DIR}/fixes.md"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# ── Python helpers ────────────────────────────────────────────────────────────

# Extract comma-separated task IDs with reward < 1.0 from the last round of comparison.json.
bad_tasks_from() {
    python3 - "$1" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
records = data["rounds"][-1]["records"]
bad = sorted(
    (str(r["task_id"]) for r in records if float(r.get("reward", 0.0)) < 1.0),
    key=lambda x: int(x) if x.isdigit() else x,
)
print(",".join(bad))
PYEOF
}

# Extract avg_reward from the last round of comparison.json.
avg_reward_from() {
    python3 - "$1" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
records = data["rounds"][-1]["records"]
rewards = [float(r.get("reward", 0.0)) for r in records]
print(f"{sum(rewards)/len(rewards):.4f}" if rewards else "0.0000")
PYEOF
}

# Extract total task count from the last round of comparison.json.
total_tasks_from() {
    python3 - "$1" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
print(len(data["rounds"][-1]["records"]))
PYEOF
}

# Count elements in a comma-separated ID list (empty string → 0).
count_ids() {
    python3 - "$1" <<'PYEOF'
import sys
print(len([x for x in sys.argv[1].split(",") if x.strip()]))
PYEOF
}

# Set difference prev - cur, comma-separated, numerically sorted.
set_diff() {
    python3 - "$1" "$2" <<'PYEOF'
import sys
prev  = set(x for x in sys.argv[1].split(",") if x.strip())
cur   = set(x for x in sys.argv[2].split(",") if x.strip())
fixed = sorted(prev - cur, key=lambda x: int(x) if x.isdigit() else x)
print(",".join(fixed))
PYEOF
}

# ── Initialise output files ───────────────────────────────────────────────────
cat > "$SUMMARY_FILE" <<'EOF'
# Badcase Iteration Summary

| iter | run_tag | test set | avg_reward | badcases (Bk) | fixed this iter | fixed task IDs |
|------|---------|----------|-----------|--------------|----------------|----------------|
EOF

cat > "$FIXES_FILE" <<'EOF'
# Per-iteration Fix Details

EOF

# ════════════════════════════════════════════════════════════════════════════════
# ITER 1 — full task set: R0 (baseline sim) + evolve → R1 (C1 eval)
# ════════════════════════════════════════════════════════════════════════════════

ITER=1
RUN_TAG="${RUN_PREFIX}_iter${ITER}"

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "ITER ${ITER}: full-set evolution  run_tag=${RUN_TAG}"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

uv run --project ~/tau2-bench \
    python -m recipe.tau2_evolver.run \
    --domain      "$DOMAIN"      \
    --base-config "$BASE_CONFIG" \
    --num-rounds  2              \
    --run-tag     "$RUN_TAG"     \
    --clean                      \
    "${PASSTHROUGH[@]}"          \
    2>&1 | tee "${META_DIR}/iter${ITER}.log"

CMP="${RUNS_DIR}/${RUN_TAG}/comparison.json"
PREV_CONFIG="${RUNS_DIR}/${RUN_TAG}/R1/config.yaml"
PREV_REPORT="${RUNS_DIR}/${RUN_TAG}/R1/report.json"

AVG=$(avg_reward_from "$CMP")
PREV_BAD=$(bad_tasks_from "$CMP")
PREV_BAD_COUNT=$(count_ids "$PREV_BAD")
TOTAL=$(total_tasks_from "$CMP")
CUMULATIVE_FIXED=0

log "ITER ${ITER} done | avg_reward=${AVG} | badcases=${PREV_BAD_COUNT}/${TOTAL}"
[[ -n "$PREV_BAD" ]] && log "  B1: ${PREV_BAD}"

# No previous config to compare against in iter 1 — use dash placeholders.
echo "| iter${ITER} | \`${RUN_TAG}\` | ${TOTAL} | ${AVG} | ${PREV_BAD_COUNT} | — | — |" \
    >> "$SUMMARY_FILE"
printf "## Iter %d  (C1, full %d tasks)\n- avg_reward: %s\n- B1 badcases (%d): %s\n\n" \
    "$ITER" "$TOTAL" "$AVG" "$PREV_BAD_COUNT" "${PREV_BAD:-(none)}" >> "$FIXES_FILE"

# ════════════════════════════════════════════════════════════════════════════════
# ITER k≥2 — evolve on B_{k-1} only
#   R0:    load C_{k-1} report filtered to B_{k-1} (no re-simulation)
#   evolve: meta-agent reads B_{k-1} failure trajectories → authors Ck
#   R1:    simulate Ck on B_{k-1} → collect Bk
# ════════════════════════════════════════════════════════════════════════════════

ITER=2

while true; do

    # ── Termination ──────────────────────────────────────────────────────────
    if [[ -z "$PREV_BAD" || "$PREV_BAD_COUNT" -eq 0 ]]; then
        log "✓ All badcases resolved. Experiment finished after $((ITER - 1)) iterations."
        break
    fi
    if [[ "$ITER" -gt "$MAX_ITER" ]]; then
        log "⚠ Reached max-iter=${MAX_ITER}. Stopping with ${PREV_BAD_COUNT} badcases remaining."
        break
    fi

    RUN_TAG="${RUN_PREFIX}_iter${ITER}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "ITER ${ITER}: evolving on ${PREV_BAD_COUNT} badcases  run_tag=${RUN_TAG}"
    log "  base-config : ${PREV_CONFIG}"
    log "  from-report : ${PREV_REPORT}"
    log "  task-ids    : ${PREV_BAD}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    uv run --project ~/tau2-bench \
        python -m recipe.tau2_evolver.run \
        --domain      "$DOMAIN"       \
        --base-config "$PREV_CONFIG"  \
        --from-report "$PREV_REPORT"  \
        --task-ids    "$PREV_BAD"     \
        --num-rounds  2               \
        --run-tag     "$RUN_TAG"      \
        --clean                       \
        "${PASSTHROUGH[@]}"           \
        2>&1 | tee "${META_DIR}/iter${ITER}.log"

    CMP="${RUNS_DIR}/${RUN_TAG}/comparison.json"
    CUR_CONFIG="${RUNS_DIR}/${RUN_TAG}/R1/config.yaml"
    CUR_REPORT="${RUNS_DIR}/${RUN_TAG}/R1/report.json"

    AVG=$(avg_reward_from "$CMP")
    CUR_BAD=$(bad_tasks_from "$CMP")
    CUR_BAD_COUNT=$(count_ids "$CUR_BAD")
    FIXED=$(set_diff "$PREV_BAD" "$CUR_BAD")
    FIXED_COUNT=$(count_ids "$FIXED")
    CUMULATIVE_FIXED=$((CUMULATIVE_FIXED + FIXED_COUNT))

    log "ITER ${ITER} done | avg_reward=${AVG} | badcases=${CUR_BAD_COUNT} | fixed=${FIXED_COUNT} | cumulative=${CUMULATIVE_FIXED}"
    [[ -n "$FIXED" ]]   && log "  fixed tasks    : ${FIXED}"
    [[ -n "$CUR_BAD" ]] && log "  remaining badcases: ${CUR_BAD}"

    echo "| iter${ITER} | \`${RUN_TAG}\` | ${PREV_BAD_COUNT} | ${AVG} | ${CUR_BAD_COUNT} | ${FIXED_COUNT} | ${FIXED:-(none)} |" \
        >> "$SUMMARY_FILE"
    printf "## Iter %d  (C%d, %d tasks)\n- avg_reward: %s\n- fixed (%d): %s\n- remaining badcases (%d): %s\n\n" \
        "$ITER" "$ITER" "$PREV_BAD_COUNT" "$AVG" \
        "$FIXED_COUNT" "${FIXED:-(none)}" \
        "$CUR_BAD_COUNT" "${CUR_BAD:-(none)}" >> "$FIXES_FILE"

    # ── Advance state ─────────────────────────────────────────────────────────
    PREV_CONFIG="$CUR_CONFIG"
    PREV_REPORT="$CUR_REPORT"
    PREV_BAD="$CUR_BAD"
    PREV_BAD_COUNT="$CUR_BAD_COUNT"
    ITER=$((ITER + 1))

done

# ── Final summary ─────────────────────────────────────────────────────────────
{
    echo ""
    echo "**Total iterations**: $((ITER - 1))  |  **Cumulative tasks fixed**: ${CUMULATIVE_FIXED}"
} >> "$SUMMARY_FILE"

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Done | iterations: $((ITER - 1)) | cumulative fixed: ${CUMULATIVE_FIXED}"
log "Summary → ${SUMMARY_FILE}"
log "Fixes   → ${FIXES_FILE}"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cat "$SUMMARY_FILE"
echo ""
cat "$FIXES_FILE"
