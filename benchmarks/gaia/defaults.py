"""Default values for GAIA benchmark recipe."""

# ── Model provider (default — Claude via proxy) ──────────────────────────────
DEFAULT_MODEL = "openai/pa/claude-sonnet-4-6"
DEFAULT_PROVIDER_ID = "YOUR_PROVIDER_ID"

# ── RunLoop ───────────────────────────────────────────────────────────────────
MAX_STEPS = 15  # max agent steps per task

# ── Budget ────────────────────────────────────────────────────────────────────
MAX_COST_USD = 2.0  # per-task cost cap (argparse default)
COST_GUARD_MAX_USD = 5.0  # CostGuardProcessor hard ceiling

# ── Loop detection ────────────────────────────────────────────────────────────
LOOP_WINDOW_SIZE = 8
LOOP_THRESHOLD = 3  # exact fingerprint repeat threshold

# ── Context assembly ──────────────────────────────────────────────────────────
TOKEN_BUDGET_RATIO = 0.6  # TokenBudgetProcessor ratio

# ── Misc processors ──────────────────────────────────────────────────────────
CHECKPOINT_EVERY_N = 3  # CheckpointProcessor interval

# ═══════════════════════════════════════════════════════════════════════════════
# GPT-5 preset — tuned for fair GPT-5 GAIA evaluation
# Target (test set, 2025-12-11): Avg=66.11%, L1=78.49%, L2=64.78%, L3=46.94%
# ═══════════════════════════════════════════════════════════════════════════════
GPT5_MODEL = "gpt-5"
GPT5_API_BASE = "https://your-api-base.example.com/v1"
GPT5_PROVIDER_ID = "azure_openai"
GPT5_MAX_STEPS = 40
GPT5_MAX_COST_USD = 15.0
GPT5_COST_GUARD_MAX_USD = 25.0
GPT5_TOKEN_BUDGET_RATIO = 0.85
GPT5_LOOP_WINDOW_SIZE = 10
GPT5_LOOP_THRESHOLD = 4
GPT5_CHECKPOINT_EVERY_N = 5

# Per-level step budgets (used when adaptive steps are enabled)
GPT5_STEPS_PER_LEVEL = {1: 30, 2: 40, 3: 60}

# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3.5-9B preset — small Qwen model with tool-call XML recovery
# Tighter budgets than QwQ-32B; recovery processor handles raw XML tool calls.
# ═══════════════════════════════════════════════════════════════════════════════
QWEN35_9B_MAX_STEPS = 12
QWEN35_9B_MAX_COST_USD = 5.0
QWEN35_9B_COST_GUARD_MAX_USD = 10.0
QWEN35_9B_TOKEN_BUDGET_RATIO = 0.80
QWEN35_9B_LOOP_WINDOW_SIZE = 8
QWEN35_9B_LOOP_THRESHOLD = 3
QWEN35_9B_CHECKPOINT_EVERY_N = 5
