# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Tests for harnessx.processors.evaluation.llm_judge.LLMJudgeProcessor."""

from __future__ import annotations

import pytest

from harnessx.core.events import Message, ModelResponseEvent, TaskEndEvent
from harnessx.processors.evaluation.llm_judge import (
    LLMJudgeProcessor,
    build_judge_prompt,
    default_answer_extractor,
    parse_judge_response,
    render_trajectory_summary,
)


def test_processor_class_exists():
    assert LLMJudgeProcessor is not None


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def test_extractor_finds_final_answer_marker():
    msgs = [
        _msg("user", "What is the capital of France?"),
        _msg("assistant", "Let me think...\n\nFINAL ANSWER: Paris"),
    ]
    assert default_answer_extractor(msgs) == "Paris"


def test_extractor_is_case_insensitive():
    msgs = [_msg("assistant", "final answer: 42")]
    assert default_answer_extractor(msgs) == "42"


def test_extractor_falls_back_to_answer_is():
    msgs = [_msg("assistant", "The answer is: Tokyo")]
    assert default_answer_extractor(msgs) == "Tokyo"


def test_extractor_falls_back_to_last_nonempty_line():
    msgs = [_msg("assistant", "Paragraph one.\n\nParagraph two final.")]
    assert default_answer_extractor(msgs) == "Paragraph two final."


def test_extractor_strips_markdown():
    msgs = [_msg("assistant", "FINAL ANSWER: **Paris**")]
    assert default_answer_extractor(msgs) == "Paris"


def test_extractor_returns_empty_on_no_assistant_message():
    msgs = [_msg("user", "hello")]
    assert default_answer_extractor(msgs) == ""


def test_extractor_prefers_final_answer_over_last_line():
    msgs = [_msg("assistant", "FINAL ANSWER: Paris\n\nAdditional commentary.")]
    assert default_answer_extractor(msgs) == "Paris"


def test_extractor_tier1_wins_across_message_boundaries():
    """FINAL ANSWER in an earlier assistant message beats 'answer is' in a later one."""
    msgs = [
        _msg("user", "Question"),
        _msg("assistant", "FINAL ANSWER: Paris"),
        _msg("user", "Are you sure?"),
        _msg("assistant", "The answer is: London"),
    ]
    # Tier 1 scans ALL assistant messages (reversed) before tier 2 runs.
    # Even though the later message has a tier-2 match, tier-1 match in the
    # earlier message should win because tier-1 is tried first for every message.
    assert default_answer_extractor(msgs) == "Paris"


def test_summary_renders_tool_trace_one_line_per_step():
    messages = [
        _msg("user", "Q"),
        _msg("assistant", "thinking"),
        _msg("tool", "web_search result: 450 chars of content here..."),
        _msg("assistant", "FINAL ANSWER: X"),
    ]
    tool_trace = [
        (1, "web_search", '{"query":"Paris"}', "ok", 450),
        (2, "web_search", '{"query":"France"}', "error: rate_limit", 0),
    ]
    out = render_trajectory_summary(
        tool_trace=tool_trace,
        final_messages=messages,
        budget_chars=4000,
    )
    assert "## Tool Trace" in out
    assert "Step 1:" in out
    assert "web_search" in out
    assert "rate_limit" in out
    assert "## Final Reasoning" in out
    assert "FINAL ANSWER: X" in out


def test_summary_folds_middle_when_over_budget():
    tool_trace = [(i, "tool_x", '{"i":' + str(i) + "}", "ok", 100) for i in range(1, 21)]
    messages = [_msg("assistant", "short reasoning")]
    out = render_trajectory_summary(
        tool_trace=tool_trace,
        final_messages=messages,
        budget_chars=500,  # intentionally small
    )
    assert "omitted" in out.lower() or "..." in out
    assert out.count("Step") < 20  # must have folded


def test_summary_handles_empty_trace():
    out = render_trajectory_summary(
        tool_trace=[],
        final_messages=[_msg("assistant", "direct answer")],
        budget_chars=1000,
    )
    assert "## Tool Trace" in out
    assert "(no tools called)" in out
    assert "direct answer" in out


def test_build_prompt_includes_all_sections():
    prompt = build_judge_prompt(
        task_description="What is the capital of France?",
        trajectory_summary="## Tool Trace\nStep 1: web_search(...) → ok, 100 chars\n\n## Final Reasoning\nThe answer is Paris.",
        extracted_answer="Paris",
    )
    assert "## Question" in prompt
    assert "What is the capital of France?" in prompt
    assert "## Tool Trace" in prompt
    assert "## Extracted Answer" in prompt
    assert "Paris" in prompt
    assert "ground truth" in prompt.lower()  # prompt must state this explicitly


def test_build_prompt_truncates_huge_task_description():
    huge_task = "A" * 5000
    prompt = build_judge_prompt(
        task_description=huge_task,
        trajectory_summary="x",
        extracted_answer="x",
    )
    assert len(prompt) < 10_000  # bounded


def test_parse_valid_verdict():
    response = '{"verdict":"plausible","confidence":0.8,"cause":"triangulated","missing":"","lesson":""}'
    v = parse_judge_response(response)
    assert v["verdict"] == "plausible"
    assert v["confidence"] == 0.8
    assert v["cause"] == "triangulated"
    assert v["missing"] == ""
    assert v["lesson"] == ""


def test_parse_tolerates_code_fences():
    response = '```json\n{"verdict":"refused","confidence":1.0,"cause":"c","missing":"m","lesson":""}\n```'
    v = parse_judge_response(response)
    assert v["verdict"] == "refused"


def test_parse_rejects_unknown_verdict():
    response = '{"verdict":"OTHER","confidence":0.5,"cause":"","missing":"","lesson":""}'
    with pytest.raises(ValueError):
        parse_judge_response(response)


def test_parse_rejects_malformed_json():
    with pytest.raises(ValueError):
        parse_judge_response("this is not json")


def test_parse_fills_missing_fields_with_empty_strings():
    response = '{"verdict":"plausible","confidence":0.5}'
    v = parse_judge_response(response)
    assert v["cause"] == ""
    assert v["missing"] == ""
    assert v["lesson"] == ""


def test_parse_clamps_confidence_to_unit_interval():
    response = '{"verdict":"plausible","confidence":2.5,"cause":"","missing":"","lesson":""}'
    v = parse_judge_response(response)
    assert 0.0 <= v["confidence"] <= 1.0


class _FakeProvider:
    """Records calls and returns canned responses."""

    def __init__(self, responses: "list[str]"):
        self._responses = list(responses)
        self.calls: "list[list[Message]]" = []

    async def complete(self, messages, tools, stream_callback=None):
        self.calls.append(messages)
        content = (
            self._responses.pop(0)
            if self._responses
            else '{"verdict":"no_answer","confidence":1.0,"cause":"","missing":"","lesson":""}'
        )
        return ModelResponseEvent(run_id="fake-run", step_id=0, content=content)

    def count_tokens(self, messages):
        return 0

    def annotate_trajectory(self, trajectory):
        pass


def _make_task_end_event(*, run_id="r1", final_output="Paris", task_description="What is the capital?", messages=None):
    return TaskEndEvent(
        run_id=run_id,
        step_id=1,
        final_output=final_output,
        exit_reason="done",
        total_steps=3,
        total_tokens=1000,
        final_messages=tuple(
            messages
            or [
                _msg("user", task_description),
                _msg("assistant", "FINAL ANSWER: Paris"),
            ]
        ),
        task_description=task_description,
    )


@pytest.mark.asyncio
async def test_processor_writes_verdict_to_sink():
    provider = _FakeProvider(
        responses=['{"verdict":"plausible","confidence":0.9,"cause":"direct answer","missing":"","lesson":""}']
    )
    sink: dict = {}
    proc = LLMJudgeProcessor(provider, verdict_sink=sink)
    event = _make_task_end_event(run_id="r-42")

    async for _ in proc.on_task_end(event):
        pass

    assert "r-42" in sink
    entry = sink["r-42"]
    assert entry["verdict"]["verdict"] == "plausible"
    assert entry["verdict"]["confidence"] == 0.9
    assert entry["extracted_answer"] == "Paris"


@pytest.mark.asyncio
async def test_get_verdict_returns_entry():
    provider = _FakeProvider(responses=['{"verdict":"plausible","confidence":0.7,"cause":"","missing":"","lesson":""}'])
    proc = LLMJudgeProcessor(provider)
    event = _make_task_end_event(run_id="r-99")

    async for _ in proc.on_task_end(event):
        pass

    entry = proc.get_verdict("r-99")
    assert entry is not None
    assert entry["verdict"]["verdict"] == "plausible"


@pytest.mark.asyncio
async def test_processor_yields_event_unchanged():
    provider = _FakeProvider(responses=['{"verdict":"plausible","confidence":0.5,"cause":"","missing":"","lesson":""}'])
    proc = LLMJudgeProcessor(provider)
    event = _make_task_end_event()

    yielded = []
    async for ev in proc.on_task_end(event):
        yielded.append(ev)

    assert len(yielded) == 1
    assert yielded[0] is event  # processor does not mutate event itself


import asyncio


@pytest.mark.asyncio
async def test_malformed_json_then_valid_succeeds():
    provider = _FakeProvider(
        responses=[
            "not json",
            '{"verdict":"plausible","confidence":0.7,"cause":"","missing":"","lesson":""}',
        ]
    )
    proc = LLMJudgeProcessor(provider)
    event = _make_task_end_event(run_id="r-retry")

    async for _ in proc.on_task_end(event):
        pass

    entry = proc.get_verdict("r-retry")
    assert entry["verdict"]["verdict"] == "plausible"
    assert len(provider.calls) == 2  # one retry


@pytest.mark.asyncio
async def test_malformed_twice_yields_judge_error():
    provider = _FakeProvider(responses=["not json", "still not json"])
    proc = LLMJudgeProcessor(provider)
    event = _make_task_end_event(run_id="r-fail")

    async for _ in proc.on_task_end(event):
        pass

    entry = proc.get_verdict("r-fail")
    assert entry["verdict"]["verdict"] == "judge_error"
    assert "JSON" in entry["verdict"]["cause"]


@pytest.mark.asyncio
async def test_provider_exception_yields_judge_error():
    class _RaisingProvider:
        async def complete(self, messages, tools, stream_callback=None):
            raise RuntimeError("network down")

        def count_tokens(self, messages):
            return 0

        def annotate_trajectory(self, trajectory):
            pass

    proc = LLMJudgeProcessor(_RaisingProvider())
    event = _make_task_end_event(run_id="r-exc")

    # Must not raise
    async for _ in proc.on_task_end(event):
        pass

    entry = proc.get_verdict("r-exc")
    assert entry["verdict"]["verdict"] == "judge_error"
    assert "RuntimeError" in entry["verdict"]["cause"]


@pytest.mark.asyncio
async def test_timeout_yields_judge_error():
    class _SlowProvider:
        async def complete(self, messages, tools, stream_callback=None):
            await asyncio.sleep(5.0)
            return ModelResponseEvent(content='{"verdict":"plausible"}', run_id="fake", step_id=0)

        def count_tokens(self, messages):
            return 0

        def annotate_trajectory(self, trajectory):
            pass

    proc = LLMJudgeProcessor(_SlowProvider(), timeout_s=0.05)
    event = _make_task_end_event(run_id="r-slow")

    async for _ in proc.on_task_end(event):
        pass

    entry = proc.get_verdict("r-slow")
    assert entry["verdict"]["verdict"] == "judge_error"
    assert "timeout" in entry["verdict"]["cause"].lower()


@pytest.mark.asyncio
async def test_empty_answer_skips_judge_call():
    provider = _FakeProvider(responses=[])  # no responses queued
    proc = LLMJudgeProcessor(provider)
    event = _make_task_end_event(
        run_id="r-empty",
        final_output="",
        messages=[_msg("user", "x"), _msg("assistant", "")],
    )

    async for _ in proc.on_task_end(event):
        pass

    entry = proc.get_verdict("r-empty")
    assert entry["verdict"]["verdict"] == "no_answer"
    assert len(provider.calls) == 0  # judge NOT called


@pytest.mark.asyncio
async def test_ground_truth_never_reaches_judge_prompt():
    """Hard guarantee: ground-truth-like data adjacent to the event must not
    reach the judge prompt.

    Places a magic marker in state_snapshot as a potential leak vector. The
    processor must not surface ANY part of state_snapshot except the
    messages-derived tool trace (which does not contain the marker). If a
    future refactor widens the processor's reads, this test catches it.
    """
    magic = "MAGICSTRING_GROUNDTRUTH_XYZ123"
    provider = _FakeProvider(responses=['{"verdict":"plausible","confidence":0.9,"cause":"","missing":"","lesson":""}'])
    proc = LLMJudgeProcessor(provider)

    # Event with a pretend ground-truth marker buried in state_snapshot.
    # The processor must only surface task_description and message-derived
    # content — not arbitrary snapshot fields.
    event = _make_task_end_event(
        run_id="r-magic",
        task_description="What is the capital of France?",
        messages=[
            _msg("user", "What is the capital of France?"),
            _msg("assistant", "FINAL ANSWER: Paris"),
        ],
    )
    # Simulate a foreign field being present — modeled as hidden ground truth.
    # state_snapshot is a mutable dict on the frozen dataclass.
    import dataclasses

    if event.state_snapshot is None:
        event = dataclasses.replace(event, state_snapshot={})
    event.state_snapshot["hidden_ground_truth"] = magic

    async for _ in proc.on_task_end(event):
        pass

    call_dump = ""
    for msgs in provider.calls:
        for m in msgs:
            call_dump += m.content if isinstance(m.content, str) else str(m.content)

    assert magic not in call_dump, "processor surfaced hidden state_snapshot field into judge prompt"


@pytest.mark.asyncio
async def test_processor_renders_tool_trace_from_snapshot():
    """Verify that state_snapshot["messages"] with role='tool' entries
    produces a non-empty tool trace in the judge prompt."""
    provider = _FakeProvider(responses=['{"verdict":"plausible","confidence":0.9,"cause":"","missing":"","lesson":""}'])
    proc = LLMJudgeProcessor(provider)

    event = _make_task_end_event(
        run_id="r-tooltrace",
        messages=[
            _msg("user", "Q"),
            _msg("assistant", "Let me search"),
            _msg("tool", "search result content..."),
            _msg("assistant", "FINAL ANSWER: Paris"),
        ],
    )
    # Populate state_snapshot with dict-form messages including one tool message
    import dataclasses

    event = dataclasses.replace(
        event,
        state_snapshot={
            "messages": [
                {"role": "user", "content": "Q", "name": ""},
                {"role": "assistant", "content": "Let me search", "name": ""},
                {"role": "tool", "content": "search result content...", "name": "web_search"},
                {"role": "assistant", "content": "FINAL ANSWER: Paris", "name": ""},
            ],
        },
    )

    async for _ in proc.on_task_end(event):
        pass

    # The judge prompt must include "Step 1:" and the tool name from the snapshot
    assert len(provider.calls) == 1
    prompt = provider.calls[0][0].content
    assert "Step 1:" in prompt
    assert "web_search" in prompt  # tool name preserved from snapshot


def test_processor_has_no_access_to_base_task():
    """Structural assertion: LLMJudgeProcessor.__init__ does not accept BaseTask.

    If a future refactor adds a task or final_answer parameter, this test will
    catch it at import time.
    """
    import inspect

    sig = inspect.signature(LLMJudgeProcessor.__init__)
    for name in sig.parameters:
        assert name not in ("task", "final_answer", "ground_truth"), (
            f"LLMJudgeProcessor must not accept {name!r}; ground truth isolation violated."
        )


def test_get_judge_provider_uses_litellm_for_non_anthropic_model():
    proc = LLMJudgeProcessor(judge_model="openai/gpt-4o-mini")
    provider = proc._get_judge_provider()
    assert type(provider).__name__ == "LiteLLMProvider"


def test_get_judge_provider_uses_anthropic_for_anthropic_prefixed_model():
    proc = LLMJudgeProcessor(judge_model="anthropic/claude-sonnet-4-5")
    provider = proc._get_judge_provider()
    assert type(provider).__name__ == "AnthropicProvider"
