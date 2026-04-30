# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
WORKSPACE_PATH = "/app"  # task files live at /app inside the container

# ── RunLoop ───────────────────────────────────────────────────────────────────
MAX_STEPS = 500  # effectively unlimited; Harbor task.toml controls wall-clock
REQUEST_TIMEOUT_SEC = 600  # per-LLM-call timeout (seconds); override via --ak request_timeout_sec
TOKEN_BUDGET = 10_000_000  # cumulative token cap across all steps

# ── Model provider ────────────────────────────────────────────────────────────
API_KEY_DEFAULT = "EMPTY"

# ── Sandbox / tool output ─────────────────────────────────────────────────────
OUTPUT_LIMIT = 8_000  # max chars captured per tool call stdout/stderr

# ── Context assembly ──────────────────────────────────────────────────────────
MEMORY_WINDOW = 200  # SlidingWindowMemory message count

# ── Batch evaluation (tb2_eval.py) ────────────────────────────────────────────
N_CONCURRENT = 4
JOBS_DIR = ".benchmarks/tb2"
DATASET = "terminal-bench@2.0"
