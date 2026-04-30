# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT

# ── Model ─────────────────────────────────────────────────────────────────────
# Task-doing (inner) agent: Haiku — cheapest tier, adequate for GAIA tasks.
# Meta-agent: Opus — architectural reasoning over trajectories benefits from
# the stronger model and from extended thinking.
DEFAULT_MODEL = "anthropic/YOUR_PROVIDER/claude-sonnet-4-6"
DEFAULT_META_MODEL = "anthropic/YOUR_PROVIDER/claude-opus-4-6"
DEFAULT_PROVIDER_ID = "YOUR_PROVIDER_ID"

# ── Task budget ───────────────────────────────────────────────────────────────
MAX_TASKS = 6  # number of tasks per round
MAX_COST_USD = 5.0  # per-task cost cap
MAX_STEPS = 20  # per-task step cap (overrides benchmarks/gaia/task.py default)
DEFAULT_CONCURRENCY = 4  # max concurrent trajectories per round

# ── Gating ────────────────────────────────────────────────────────────────────
# Absolute passed-task count delta below which a pass_rate regression is
# treated as noise (no rollback). Complements --regression-tolerance: the
# gate reverts only when BOTH the score-based check fails AND the absolute
# pass-count delta meets this threshold.
PASS_COUNT_NOISE_THRESHOLD = 3

# ── Rounds ────────────────────────────────────────────────────────────────────
NUM_ROUNDS = 3  # R0 = baseline + reflect, R1..R(N-1) = evolved + reflect

# ── Evolve phase (per-round reflect) budgets ─────────────────────────────────
EVOLVE_COST_CAP_USD = 50.0
EVOLVE_MAX_STEPS = 200
EVOLVE_WALL_CLOCK_S = 10000
