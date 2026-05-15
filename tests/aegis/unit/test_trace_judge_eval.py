# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Unit tests for ``GAIAPipelineEvaluator.evaluate_with_trace_judge``.

The legacy ``evaluate_answer`` path relied on a regex over ``final_output``.
It false-negatives whenever the agent emits ``FINAL ANSWER: X`` on an
assistant turn that is not the last one (e.g. after a mid-trajectory
CommitNudge), because ``final_output`` ends up empty. The new path sends
the recent assistant turns + ground truth to an LLM judge.

These tests use a stub provider so they run without any real model call.
"""
from __future__ import annotations

import pytest

from benchmarks.gaia.evaluator import GAIAPipelineEvaluator
from harnessx.core.events import Message


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubProvider:
    """Echo-style judge: the test sets ``reply`` and every ``.complete``
    call returns it. Also records the prompt it received for inspection."""

    def __init__(self, reply: str = "PASS\ncorrect") -> None:
        self.reply = reply
        self.last_prompt: str = ""
        self.call_count: int = 0

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ARG002
        self.call_count += 1
        self.last_prompt = messages[-1].content if messages else ""
        return _StubResponse(self.reply)


@pytest.mark.asyncio
async def test_judge_primary_path_fires_when_provider_available():
    """With a judge provider, the LLM judge is the PRIMARY grader — not
    the string-match fallback."""
    stub = _StubProvider(reply="PASS\nanswer matches")
    ev = GAIAPipelineEvaluator(judge_provider=stub)

    msgs = [
        Message(role="user", content="what is 2+2?"),
        Message(role="assistant", content="Let me compute... FINAL ANSWER: 4"),
    ]
    result = await ev.evaluate_with_trace_judge(
        task_description="what is 2+2?",
        ground_truth="4",
        final_output="",  # deliberately empty to mimic the bug
        trajectory_messages=msgs,
    )
    assert result.passed is True
    assert stub.call_count == 1
    assert "GROUND TRUTH" in stub.last_prompt
    assert "4" in stub.last_prompt  # gt appears


@pytest.mark.asyncio
async def test_judge_sees_final_answer_on_earlier_turn():
    """The e4e91f1c regression case: FINAL ANSWER emitted at step 17, then
    a later assistant turn (or the final_output extractor) ends up empty.
    Judge should still see the commit and pass."""
    stub = _StubProvider(reply="PASS\nFINAL ANSWER cloak present")
    ev = GAIAPipelineEvaluator(judge_provider=stub)

    msgs = [
        Message(role="user", content="find the word"),
        Message(role="assistant", content="Let me search the database..."),
        Message(role="tool", content="result: cloak"),  # noqa: E501
        Message(role="assistant", content="FINAL ANSWER: cloak"),
        # Post-commit: agent turned once more to verify and produced empty.
        Message(role="assistant", content=""),
    ]
    result = await ev.evaluate_with_trace_judge(
        task_description="find the word",
        ground_truth="cloak",
        final_output="",  # empty despite valid commit earlier
        trajectory_messages=msgs,
    )
    assert result.passed is True
    # The judge's view should include "cloak" from the commit turn.
    assert "cloak" in stub.last_prompt


@pytest.mark.asyncio
async def test_judge_rejects_wrong_answer():
    stub = _StubProvider(reply="FAIL\nanswer says 5 but gt is 4")
    ev = GAIAPipelineEvaluator(judge_provider=stub)

    msgs = [
        Message(role="assistant", content="FINAL ANSWER: 5"),
    ]
    result = await ev.evaluate_with_trace_judge(
        task_description="what is 2+2?",
        ground_truth="4",
        final_output="FINAL ANSWER: 5",
        trajectory_messages=msgs,
    )
    assert result.passed is False


@pytest.mark.asyncio
async def test_falls_back_to_string_match_without_judge():
    """If no judge_provider is configured (e.g. unit tests that don't want
    to mock a model), the method must degrade to ``evaluate_answer``."""
    ev = GAIAPipelineEvaluator(judge_provider=None)

    msgs = [Message(role="assistant", content="FINAL ANSWER: 4")]
    result = await ev.evaluate_with_trace_judge(
        task_description="x",
        ground_truth="4",
        final_output="FINAL ANSWER: 4",
        trajectory_messages=msgs,
    )
    assert result.passed is True


@pytest.mark.asyncio
async def test_empty_trajectory_without_final_output():
    """Empty everywhere — must not raise, must grade FAIL."""
    stub = _StubProvider(reply="FAIL\nempty trace")
    ev = GAIAPipelineEvaluator(judge_provider=stub)
    result = await ev.evaluate_with_trace_judge(
        task_description="x",
        ground_truth="4",
        final_output="",
        trajectory_messages=[],
    )
    assert result.passed is False
    # Judge should NOT have been invoked on empty content — the method
    # short-circuits to FAIL directly.
    assert stub.call_count == 0


@pytest.mark.asyncio
async def test_recent_assistant_turns_capped():
    """The prompt must cap how many assistant turns it includes so a very
    long trajectory doesn't blow the judge's context."""
    stub = _StubProvider(reply="PASS")
    ev = GAIAPipelineEvaluator(judge_provider=stub)

    many_turns = [
        Message(role="assistant", content=f"turn_{i}")
        for i in range(20)
    ]
    # Inject the real answer at the SECOND-TO-LAST turn so the cap (5)
    # still catches it.
    many_turns.append(Message(role="assistant", content="FINAL ANSWER: 42"))
    many_turns.append(Message(role="assistant", content=""))

    await ev.evaluate_with_trace_judge(
        task_description="x",
        ground_truth="42",
        final_output="",
        trajectory_messages=many_turns,
    )
    # Only the last 5 assistant turns with content should appear in the
    # prompt. turn_0..turn_14 should NOT be there.
    assert "turn_0" not in stub.last_prompt
    assert "turn_14" not in stub.last_prompt
    # FINAL ANSWER turn (recent) SHOULD be there.
    assert "42" in stub.last_prompt


@pytest.mark.asyncio
async def test_judge_exception_falls_back_to_string_match():
    """If the judge call raises (network/timeout), fall back to string
    match rather than crashing the whole round."""

    class _FailingProvider:
        async def complete(self, messages, tools=None, **kwargs):  # noqa: ARG002
            raise RuntimeError("simulated network timeout")

    ev = GAIAPipelineEvaluator(judge_provider=_FailingProvider())
    msgs = [Message(role="assistant", content="FINAL ANSWER: 4")]
    result = await ev.evaluate_with_trace_judge(
        task_description="x",
        ground_truth="4",
        final_output="FINAL ANSWER: 4",
        trajectory_messages=msgs,
    )
    # String-match fallback returns True for exact match.
    assert result.passed is True
