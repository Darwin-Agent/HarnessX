"""Default values for SWE-bench recipe."""

# ── Repo setup ───────────────────────────────────────────────────────────────
import os

DEFAULT_WORK_DIR = os.environ.get("SWEBENCH_REPOS_DIR", "/ls/data/lushuo/swebench_repos_work")
LOCAL_REPOS_DIR = os.environ.get("SWEBENCH_LOCAL_REPOS", "/ls/data/lushuo/swebench_repos")

# ── RunLoop ──────────────────────────────────────────────────────────────────
MAX_STEPS = 60

# ── Loop detection ───────────────────────────────────────────────────────────
LOOP_THRESHOLD = 8  # exact-fingerprint repetitions before abort
TOOL_NAME_THRESHOLD = 12  # same-tool-pattern repetitions before abort

# ── Context assembly ─────────────────────────────────────────────────────────
MEMORY_WINDOW = 50  # SlidingWindowMemory message count
TOKEN_BUDGET_RATIO = 0.8  # TokenBudgetProcessor ratio

# ── Batch evaluation ─────────────────────────────────────────────────────────
CONCURRENCY = 3
DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DATASET_SPLIT = "test"
