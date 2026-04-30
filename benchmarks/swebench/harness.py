"""
SWE-bench harness factory.

Separates harness construction from the batch runner so the same config can
be reused or tested independently.

Usage::

    from benchmarks.swebench.harness import make_swebench_harness
    from harnessx.core.model_config import ModelConfig

    harness_config = make_swebench_harness(repo_dir, logs_dir)
    model = ModelConfig(main=provider)
    harness = model.agentic(harness_config)
    await harness.run(task)
"""

from __future__ import annotations

from pathlib import Path

from harnessx.core.builder import HarnessBuilder
from harnessx.core.harness import HarnessConfig
from harnessx.processors.context.system_prompt import SystemPromptProcessor
from harnessx.processors.context.user_wrapper import UserWrapperProcessor
from harnessx.processors.memory.memory_retrieval import MemoryRetrievalProcessor
from harnessx.processors.control.loop_detection import LoopDetectionProcessor
from harnessx.processors.control.token_budget import TokenBudgetProcessor
from harnessx.processors.memory.strategies.sliding_window import SlidingWindowMemory
from harnessx.processors.context.strategies.system_prompt.base import (
    BaseSystemPromptBuilder,
)
from harnessx.tools.builtin import build_default_tools
from harnessx.tracing.journal import HarnessJournal

from .defaults import (
    LOOP_THRESHOLD,
    MAX_STEPS,
    MEMORY_WINDOW,
    TOKEN_BUDGET_RATIO,
    TOOL_NAME_THRESHOLD,
)
from .processors import SWEBenchWorkflowProcessor

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SWE_SYSTEM_PROMPT = """\
# System

You are an expert software engineer fixing a bug. Repository: {repo_dir}

## Environment
- Source checkout with PYTHONPATH={repo_dir}
- Do NOT run pip install, setup.py, or any install commands.
- Common test deps are available: pytest, asgiref, sqlparse.

## Test Execution (USE THIS TO VERIFY YOUR FIX!)

Run the failing tests BEFORE and AFTER your fix to confirm:
- **Django repos**: `cd {repo_dir} && PYTHONPATH=. python3 tests/runtests.py <test_module> --verbosity=1`
  Example: `PYTHONPATH=. python3 tests/runtests.py expressions.tests.FTimeDeltaTests.test_date_subtraction --verbosity=1`
- **pytest repos** (requests, flask, sympy, pylint, xarray, etc.):
  `cd {repo_dir} && PYTHONPATH=. python3 -m pytest <test_file>::<TestClass>::<test_method> -xvs 2>&1 | tail -30`
- If tests fail with ImportError/ModuleNotFoundError, skip test execution and just produce your patch.
- If tests pass BEFORE your fix, your understanding of the bug is wrong — re-read the issue.

## Strategy (you have ~60 steps max — use them wisely!)

Phase 1 — READ TESTS FIRST (1-3 steps):
- If test names/files are provided, use `Grep` or `Read` to find and READ the test function.
- Understand what input the test provides and what output it expects.
- This tells you exactly what the fix must achieve. DO NOT SKIP THIS STEP.

Phase 2 — LOCATE (2-4 steps):
- Extract key class/function/module names from the issue and test.
- Use `Grep` to search for those names. Use `Glob` to find relevant files.
- Do NOT browse directory trees. Go straight to the relevant code.

Phase 3 — UNDERSTAND & FIX (3-8 steps):
- Read ONLY the relevant function/method (use `Read` with offset/limit).
- Use `Edit` to make the minimal fix. Change only what is necessary.
- ALWAYS use `Read` to get the EXACT current text before editing — do NOT type from memory.
- After editing, verify syntax: `python3 -c "import ast; ast.parse(open('{repo_dir}/FILE').read())"`
- If Edit says "old_string not found", re-read the file and copy the exact text.

Phase 4 — VERIFY (2-4 steps):
- Run the failing tests to check if your fix works.
- If tests still fail, read the error, adjust your fix, and re-test.
- Also run a few of the PASS_TO_PASS tests to check for regressions.
- Iterate until the failing tests pass OR you run out of ideas.

Phase 5 — OUTPUT (1-2 steps):
- Run `cd {repo_dir} && git diff` as your FINAL action to output the patch.

## Rules

- Work ONLY within: {repo_dir}
- Make MINIMAL changes — fix the ROOT CAUSE, not symptoms
- Do NOT modify test files, do NOT add tests/comments/docstrings
- Do NOT refactor, rename, or make cosmetic changes
- Do NOT create new files
- Focus on the SINGLE most likely root cause
- If unsure, make your best guess — a partial fix is better than no fix
- Do NOT spend more than 8 steps exploring — start fixing by step 12
- If you reach step 40, STOP and immediately make your best fix + git diff
- ALWAYS end with `cd {repo_dir} && git diff` to produce the patch
"""


class SWEBenchSystemPromptBuilder(BaseSystemPromptBuilder):
    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir

    async def build(self, workspace=None) -> str:
        return SWE_SYSTEM_PROMPT.format(repo_dir=self.repo_dir)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_swebench_harness(
    repo_dir: str,
    logs_dir: str | Path,
    *,
    max_steps: int = MAX_STEPS,
    memory_window: int = MEMORY_WINDOW,
    token_budget_ratio: float = TOKEN_BUDGET_RATIO,
    loop_threshold: int = LOOP_THRESHOLD,
    tool_name_threshold: int = TOOL_NAME_THRESHOLD,
) -> HarnessConfig:
    """Build a HarnessConfig for SWE-bench.

    Composes:
    - ``SystemPromptProcessor``          — system prompt
    - ``SWEBenchWorkflowProcessor``     — nudges, anti-apology, forces git diff
    - ``LoopDetectionProcessor``        — detects repeated tool-call patterns

    Usage::

        model = ModelConfig(main=provider)
        harness = model.agentic(make_swebench_harness(repo_dir, logs_dir))

    Args:
        repo_dir:             Absolute path to the checked-out repository.
        logs_dir:             Directory for trajectory JSONL and run logs.
        max_steps:            Maximum agent steps per instance.
        memory_window:        SlidingWindowMemory message count.
        token_budget_ratio:   TokenBudgetProcessor ratio.
        loop_threshold:       Exact-fingerprint repetitions before abort.
        tool_name_threshold:  Same-tool-pattern repetitions before abort.

    Returns:
        A ready-to-use ``HarnessConfig`` — combine with ``ModelConfig`` via
        ``model.agentic(harness_config)``.
    """
    logs_dir = str(logs_dir)
    return (
        HarnessBuilder()
        .slot(
            tool_registry=build_default_tools(),
            tracer=HarnessJournal(
                export_jsonl=True,
                base_dir=logs_dir,
            ),
        )
        .add(SystemPromptProcessor(SWEBenchSystemPromptBuilder(repo_dir)))
        .add(MemoryRetrievalProcessor(SlidingWindowMemory(n=memory_window)))
        .add(TokenBudgetProcessor(ratio=token_budget_ratio))
        .add(UserWrapperProcessor())
        .add(SWEBenchWorkflowProcessor(max_steps=max_steps, repo_dir=repo_dir))
        .add(
            LoopDetectionProcessor(
                threshold=loop_threshold,
                name_warn_threshold=tool_name_threshold,
            )
        )
    ).build()
