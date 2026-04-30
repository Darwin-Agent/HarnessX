# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Default constants for the tau2 evolver."""

import os

# ── Benchmark ────────────────────────────────────────────────────────────────
DEFAULT_DOMAIN = "retail"
DEFAULT_TASK_SPLIT = "base"
MAX_TASKS: int | None = None  # None = all tasks in split
NUM_TRIALS = 1
MAX_SIM_STEPS = 200
MAX_CONCURRENCY = 30  # parallel simulations per round

# ── Models ───────────────────────────────────────────────────────────────────
DEFAULT_AGENT_MODEL = os.environ.get("TAU2_AGENT_MODEL", "anthropic/claude-sonnet-4-6")
DEFAULT_AGENT_API_BASE = os.environ.get("ANTHROPIC_BASE_URL", "")
DEFAULT_AGENT_EXTENDED_THINKING = False
DEFAULT_AGENT_THINKING_BUDGET = 62976
DEFAULT_AGENT_MAX_TOKENS = 32000

DEFAULT_USER_MODEL = os.environ.get("TAU2_USER_MODEL", "openai/gpt-4.1")
DEFAULT_USER_API_BASE = os.environ.get("OPENAI_API_BASE", "")

# The `anthropic/` prefix routes through AnthropicProvider so extended thinking
# kwargs are honoured in _make_provider.
DEFAULT_META_MODEL = os.environ.get("TAU2_META_MODEL", "anthropic/claude-opus-4-6")
DEFAULT_META_API_BASE = os.environ.get("ANTHROPIC_BASE_URL", "")

# ── Evolve loop ───────────────────────────────────────────────────────────────
NUM_ROUNDS = 3
EVOLVE_COST_CAP_USD = 50.0
EVOLVE_MAX_STEPS = 100
EVOLVE_WALL_CLOCK_S = 3600

REGRESSION_TOLERANCE = 0.02  # 2 pp
COST_WEIGHT = 0.0  # no cost penalty in gating by default
