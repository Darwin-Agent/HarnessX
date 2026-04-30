# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from harnessx import (
    BaseTask,
    HarnessResult,
    ModelConfig,
    Workspace,
    FullStateSnapshot,
    StatefulTrajectory,
)
from harnessx.providers.litellm_provider import LiteLLMProvider
from harnessx.tracing.journal import HarnessJournal

try:
    from ._utils import load_provider
except ImportError:
    from _utils import load_provider  # when run as a script directly


def _e2e_workspace() -> Workspace:
    """Shared e2e workspace auto-derived from HXE2E_TEST_HOME."""
    try:
        from ._utils import get_test_home
    except ImportError:
        from _utils import get_test_home  # when run as a script directly
    return Workspace(agent_id="harness_e2e", home=get_test_home(), mode="shared")


def _cli_config(provider: LiteLLMProvider):
    """Full CLI-equivalent config: personal assistant profile with workspace + provider."""
    import yaml as _yaml
    from harnessx.core.harness import HarnessConfig as _HC

    _examples = PROJECT_ROOT / "examples"
    raw = _yaml.safe_load((_examples / "assistant" / "harness_config.yaml").read_text()) or {}
    config = _HC(
        processors=raw.get("processors") or [],
        plugins=raw.get("plugins") or [],
    ).copy(
        workspace=_e2e_workspace(),
        tracer=HarnessJournal(silent=True),
    )
    return ModelConfig(main=provider, summarize=provider).agentic(config)


# ─── Test Result ──────────────────────────────────────────────────────────────


class E2EResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.errors: list[str] = []
        self.trajectory: StatefulTrajectory | None = None
        self.harness_result: HarnessResult | None = None
        self.checks: list[tuple[str, bool, str]] = []  # (check_name, passed, detail)

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.checks.append((name, condition, detail))
        if not condition:
            self.errors.append(f"FAIL [{name}]: {detail}")

    def finalize(self) -> None:
        self.passed = all(c[1] for c in self.checks)

    def report(self) -> str:
        lines = [f"\n{'=' * 60}", f"Test: {self.name}", f"{'=' * 60}"]
        for check_name, ok, detail in self.checks:
            status = "✓" if ok else "✗"
            lines.append(f"  {status} {check_name}: {detail}")
        lines.append(f"\n  Result: {'PASS' if self.passed else 'FAIL'}")
        if self.errors:
            lines.append("  Errors:")
            for err in self.errors:
                lines.append(f"    - {err}")
        return "\n".join(lines)


# ─── Workspace Validator ──────────────────────────────────────────────────────


def validate_workspace(result: E2EResult, workspace: Workspace) -> None:
    """Check workspace has skills/ directory and AGENTS.md after run."""
    # Skills go to AGENT_HOME/skills/ when home is set, otherwise workspace.root/skills/
    skills_base = workspace.home if workspace.home is not None else workspace.root
    skills_dir = skills_base / "skills"
    agents_file = workspace.root / "AGENTS.md"
    result.check(
        "workspace_skills_dir",
        skills_dir.is_dir(),
        f"skills/ at {skills_dir}: {'exists' if skills_dir.is_dir() else 'MISSING'}",
    )
    result.check(
        "workspace_agents_md",
        agents_file.is_file(),
        f"AGENTS.md at {agents_file}: {'exists' if agents_file.is_file() else 'MISSING'}",
    )


# ─── Trajectory Validators ────────────────────────────────────────────────────


def validate_trajectory(result: E2EResult, trajectory: StatefulTrajectory, min_steps: int = 1) -> None:
    """Validate trajectory completeness and correctness."""
    result.check("trajectory_exists", trajectory is not None, "trajectory object present")
    if trajectory is None:
        return

    result.check(
        "trajectory_has_steps",
        len(trajectory.steps) >= min_steps,
        f"steps={len(trajectory.steps)}, expected>={min_steps}",
    )

    for i, step in enumerate(trajectory.steps):
        result.check(
            f"step_{i}_has_snapshot",
            isinstance(step.state_snapshot, FullStateSnapshot),
            f"step {i} snapshot type: {type(step.state_snapshot).__name__}",
        )
        result.check(
            f"step_{i}_snapshot_has_messages",
            len(step.state_snapshot.messages) > 0,
            f"step {i} snapshot has {len(step.state_snapshot.messages)} messages",
        )
        result.check(
            f"step_{i}_has_delta",
            step.state_delta is not None,
            f"step {i} delta type: {type(step.state_delta).__name__}",
        )
        result.check(
            f"step_{i}_has_step_start_event",
            step.step_start_event is not None,
            f"step {i} step_start_event: {type(step.step_start_event).__name__ if step.step_start_event else 'None'}",
        )

    records = trajectory.to_training_records()
    result.check(
        "training_records_count",
        len(records) == len(trajectory.steps),
        f"records={len(records)}, steps={len(trajectory.steps)}",
    )

    if records:
        first_record = records[0]
        result.check(
            "training_record_has_messages",
            "messages" in first_record and len(first_record["messages"]) > 0,
            f"first record has {len(first_record.get('messages', []))} messages",
        )
        result.check(
            "training_record_has_metadata",
            "metadata" in first_record and "delta" in first_record.get("metadata", {}),
            "first record has metadata.delta",
        )
        result.check(
            "training_record_has_run_id",
            first_record.get("run_id") == trajectory.run_id,
            f"run_id={first_record.get('run_id')}",
        )
        msgs = first_record.get("messages", [])
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        if assistant_msgs:
            last_assistant = assistant_msgs[-1]
            if last_assistant.get("tool_calls"):
                tc = last_assistant["tool_calls"][0]
                result.check(
                    "tool_calls_openai_format",
                    "id" in tc and "type" in tc and "function" in tc,
                    f"tool_call keys: {list(tc.keys())}",
                )
                fn = tc.get("function", {})
                result.check(
                    "tool_calls_has_arguments_str",
                    isinstance(fn.get("arguments"), str),
                    f"arguments type: {type(fn.get('arguments')).__name__}",
                )


# ─── Scenario 1: Deep Research ───────────────────────────────────────────────


async def run_deep_research(provider: LiteLLMProvider) -> E2EResult:
    """
    Deep Research task: multi-step information synthesis.
    Uses full CLI stack (PersonalAssistantPreset) with web tools.
    """
    result = E2EResult("DeepResearch - Multi-Step Information Synthesis")

    try:
        ws = _e2e_workspace()
        harness = _cli_config(provider)

        task = BaseTask(
            description=(
                "Research and synthesize information about AI Agent frameworks. "
                "Please provide:\n"
                "1. What are the key components of an AI Agent framework?\n"
                "2. What is the ReAct (Reasoning + Acting) pattern?\n"
                "3. How does context management affect agent performance?\n"
                "4. Summarize your findings in 3-5 sentences."
            ),
            success_criteria="Comprehensive summary of AI Agent framework components provided",
            max_steps=10,
            token_budget=50_000,
            max_cost_usd=1.0,
        )

        harness_result = await harness.run(task)
        result.harness_result = harness_result
        result.trajectory = harness_result.trajectory

        result.check(
            "harness_returns_result",
            isinstance(harness_result, HarnessResult),
            f"type={type(harness_result).__name__}",
        )
        result.check(
            "task_completed",
            harness_result.task_end.exit_reason in ("done", "budget_exceeded"),
            f"exit_reason={harness_result.task_end.exit_reason}",
        )
        result.check(
            "final_output_comprehensive",
            len(harness_result.task_end.final_output) > 100,
            f"output_length={len(harness_result.task_end.final_output)}",
        )

        validate_trajectory(result, harness_result.trajectory, min_steps=1)
        validate_workspace(result, ws)

        if harness_result.trajectory.steps:
            records = harness_result.trajectory.to_training_records()
            result.check(
                "training_records_have_full_context",
                any(len(r.get("messages", [])) > 1 for r in records),
                f"max_messages_per_record={max(len(r.get('messages', [])) for r in records) if records else 0}",
            )

    except Exception as e:
        result.errors.append(f"Exception: {type(e).__name__}: {e}")
        result.errors.append(traceback.format_exc())

    result.finalize()
    return result


# ─── Scenario 2: Daily Conversation ──────────────────────────────────────────


async def run_daily_conversation(provider: LiteLLMProvider) -> E2EResult:
    """
    Daily conversation: casual question/answer (Xiaohongshu topic query).
    Uses full CLI stack (PersonalAssistantPreset).
    """
    result = E2EResult("DailyConversation - Topic Query (Xiaohongshu style)")

    try:
        ws = _e2e_workspace()
        harness = _cli_config(provider)

        task = BaseTask(
            description=(
                "Tell me about popular beauty and skincare trends on Chinese social media "
                "(like Xiaohongshu/Little Red Book) in 2024-2025. "
                "What are the top trending products and routines?"
            ),
            success_criteria="Provides informative summary of skincare trends",
            max_steps=8,
            token_budget=80_000,
            max_cost_usd=1.0,
        )

        harness_result = await harness.run(task)
        result.harness_result = harness_result
        result.trajectory = harness_result.trajectory

        result.check(
            "harness_returns_result",
            isinstance(harness_result, HarnessResult),
            f"type={type(harness_result).__name__}",
        )
        result.check(
            "conversation_completed",
            harness_result.task_end.exit_reason in ("done", "budget_exceeded"),
            f"exit_reason={harness_result.task_end.exit_reason}",
        )
        result.check(
            "response_nonempty",
            len(harness_result.task_end.final_output) > 50,
            f"output_length={len(harness_result.task_end.final_output)}",
        )

        validate_trajectory(result, harness_result.trajectory, min_steps=1)
        validate_workspace(result, ws)

        # Verify snapshot immutability (FullStateSnapshot is frozen=True dataclass)
        if harness_result.trajectory.steps:
            snap = harness_result.trajectory.steps[0].state_snapshot
            try:
                snap.cumulative_tokens = 999999  # type: ignore
                result.check(
                    "snapshot_is_immutable",
                    False,
                    "FullStateSnapshot should be frozen but direct assignment succeeded",
                )
            except Exception:
                result.check(
                    "snapshot_is_immutable",
                    True,
                    "FullStateSnapshot is frozen (direct assignment raises exception)",
                )

        result.check(
            "step_count_matches",
            harness_result.task_end.total_steps == len(harness_result.trajectory.steps),
            f"task_end.total_steps={harness_result.task_end.total_steps}, "
            f"trajectory.steps={len(harness_result.trajectory.steps)}",
        )

    except Exception as e:
        result.errors.append(f"Exception: {type(e).__name__}: {e}")
        result.errors.append(traceback.format_exc())

    result.finalize()
    return result


# ─── Trajectory Detail Reporter ───────────────────────────────────────────────


def print_trajectory_detail(name: str, trajectory: StatefulTrajectory | None) -> None:
    """Print detailed trajectory information for inspection."""
    if trajectory is None:
        print(f"\n[{name}] No trajectory")
        return

    print(f"\n{'─' * 60}")
    print(f"Trajectory Detail: {name}")
    print(f"  run_id: {trajectory.run_id}")
    print(f"  total_steps: {len(trajectory.steps)}")
    print(f"  total_reward: {trajectory.total_reward()}")

    for i, step in enumerate(trajectory.steps):
        print(f"\n  Step {step.step_id}:")
        snap = step.state_snapshot
        print(f"    state_snapshot.messages: {len(snap.messages)} messages")
        for msg in snap.messages:
            content_preview = msg.content[:80].replace("\n", " ") if msg.content else ""
            print(f"      [{msg.role}]: {content_preview}...")
        print(f"    state_snapshot.cumulative_tokens: {snap.cumulative_tokens}")
        print(f"    state_snapshot.cumulative_cost_usd: {snap.cumulative_cost_usd:.6f}")
        print(f"    state_delta: {len(step.state_delta.operations)} operations")
        if step.state_delta.operations:
            for op in step.state_delta.operations:
                print(f"      {op.operation} slot '{op.key}'")
        if step.action:
            content_preview = step.action.content[:80].replace("\n", " ") if step.action.content else ""
            print(f"    action.content: {content_preview}...")
            print(f"    action.tool_calls: {len(step.action.tool_calls)}")
        print(f"    observation: {len(step.observation)} tool results")
        for obs in step.observation:
            print(f"      tool={obs.tool_name}, error={obs.error}")
        print(f"    reward: {step.reward}")


# ─── Main ─────────────────────────────────────────────────────────────────────


async def main() -> tuple[int, int]:
    print("HarnessX End-to-End Test Suite")
    print("=" * 60)

    provider = load_provider()
    print(f"\nProvider: {provider.model}")
    if hasattr(provider, "kwargs") and provider.kwargs.get("api_base"):
        print(f"API base: {provider.kwargs['api_base']}")
    print(f"Workspace: {_e2e_workspace().root}")

    results = []
    scenarios = [
        ("DeepResearch", run_deep_research),
        ("DailyConversation", run_daily_conversation),
    ]

    _TIMEOUT = 300  # 5 minutes per scenario

    for scenario_name, scenario_fn in scenarios:
        print(f"\nRunning: {scenario_name}...")
        try:
            result = await asyncio.wait_for(scenario_fn(provider), timeout=_TIMEOUT)
            results.append(result)
        except asyncio.TimeoutError:
            r = E2EResult(scenario_name)
            r.errors.append(f"TIMEOUT: scenario exceeded {_TIMEOUT}s limit")
            r.finalize()
            results.append(r)
        except Exception as e:
            r = E2EResult(scenario_name)
            r.errors.append(f"Top-level exception: {e}")
            r.errors.append(traceback.format_exc())
            r.finalize()
            results.append(r)

    # Print trajectory details for inspection
    print("\n\n" + "=" * 60)
    print("TRAJECTORY INSPECTION")
    print("=" * 60)
    for result in results:
        if result.trajectory:
            print_trajectory_detail(result.name, result.trajectory)

    # Print test results
    print("\n\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for result in results:
        print(result.report())

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed}/{total} scenarios passed")
    print("=" * 60)

    return passed, total


if __name__ == "__main__":
    passed, total = asyncio.run(main())
    sys.exit(0 if passed == total else 1)


# ─── Pytest entry points ───────────────────────────────────────────────────────

import pytest


@pytest.mark.asyncio
async def test_deep_research():
    provider = load_provider()
    result = await run_deep_research(provider)
    assert result.passed, "\n".join(result.errors)


@pytest.mark.asyncio
async def test_daily_conversation():
    provider = load_provider()
    result = await run_daily_conversation(provider)
    assert result.passed, "\n".join(result.errors)
