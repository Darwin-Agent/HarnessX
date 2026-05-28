# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Default constants for the tau2 evolver."""

# ── Benchmark ────────────────────────────────────────────────────────────────
DEFAULT_DOMAIN = "retail"
DEFAULT_TASK_SPLIT = "base"
MAX_TASKS: int | None = None  # None = all tasks in split
NUM_TRIALS = 2
MAX_SIM_STEPS = 200
MAX_CONCURRENCY = 30  # parallel simulations per round

# ── Models ───────────────────────────────────────────────────────────────────
# Agent model: set via --model (Anthropic proxy) or --agent-model
# DEFAULT_AGENT_MODEL = "anthropic/tongyi/qwen3.5-27b"
# DEFAULT_AGENT_MODEL = "anthropic/xiaomi/mimo-v2.5"
# DEFAULT_AGENT_API_BASE = "http://model.mify.ai.srv/anthropic"
DEFAULT_AGENT_EXTENDED_THINKING = False
DEFAULT_AGENT_THINKING_BUDGET = 62976
DEFAULT_AGENT_MAX_TOKENS = 32000
DEFAULT_AGENT_MODEL = "anthropic/mimo-v2.5"
DEFAULT_AGENT_API_BASE = "http://10.221.97.102:19010"

# User simulator model: OpenAI-compatible endpoint
DEFAULT_USER_MODEL = "openai/azure_openai/gpt-5.2"
DEFAULT_USER_API_BASE = "http://model.mify.ai.srv/v1"
# DEFAULT_USER_MODEL = "openai//preset-models"
# DEFAULT_USER_API_BASE = "http://10.221.97.102:19000/v1"

# Meta-agent model (for evolve()).
# Mirrors recipe/gaia_evolver/defaults.py: the outer loop runs on a stronger
# tier than the inner agent. The `anthropic/` prefix routes through
# AnthropicProvider so extended thinking kwargs are honoured in _make_provider.
DEFAULT_META_MODEL = "anthropic/ppio/pa/claude-opus-4-6"
DEFAULT_META_API_BASE = "http://model.mify.ai.srv/anthropic"

# ── Evolve loop ───────────────────────────────────────────────────────────────
NUM_ROUNDS = 3
EVOLVE_COST_CAP_USD = 50.0
EVOLVE_MAX_STEPS = 100
EVOLVE_WALL_CLOCK_S = 3600

# Gating: accept evolved config only if reward does not drop more than
# REGRESSION_TOLERANCE below the historical best.
REGRESSION_TOLERANCE = 0.01  # 1 pp
COST_WEIGHT = 0.0  # no cost penalty in gating by default
