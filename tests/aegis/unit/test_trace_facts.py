# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Layer A mechanical trace-fact extraction — unit tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harnessx.aegis.stages.trace_facts import (
    extract_trace_facts,
    _shares_substring,
    _classify_return,
    _content_to_text,
    _args_sha,
)


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n",
        encoding="utf-8",
    )


def _traj(path: Path, *, assistant_tool_calls=(), tool_results=(), final_assistant_text="done", exit_reason="done", passed=True, total_steps=None):
    """Build a minimal synthetic trajectory jsonl with one assistant-tool cycle per tool call."""
    events: list[dict] = [
        {"type": "session_start", "step": 0, "task": "t"},
        {"type": "system", "step": 0, "message": {"role": "system", "content": "sys"}},
        {"type": "raw_user", "step": 0, "message": {"role": "user", "content": "go"}},
    ]
    for i, (call, result) in enumerate(zip(assistant_tool_calls, tool_results)):
        events.append({
            "type": "raw_assistant",
            "step": i,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [call],
            },
        })
        if result is not None:
            events.append({
                "type": "raw_tool",
                "step": i,
                "message": {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": result,
                },
            })
    events.append({
        "type": "raw_assistant",
        "step": len(assistant_tool_calls),
        "message": {"role": "assistant", "content": final_assistant_text, "tool_calls": []},
    })
    events.append({
        "type": "episode_end",
        "step": len(assistant_tool_calls) + 1,
        "exit_reason": exit_reason,
        "total_steps": total_steps if total_steps is not None else len(assistant_tool_calls) + 1,
        "passed": passed,
    })
    _write_jsonl(path, events)


# ---------- helpers ----------

def test_shares_substring_basic():
    assert _shares_substring("abcdefghij" * 3, "xxxabcdefghijxxx", min_len=10)
    assert not _shares_substring("abcdefghij", "zzzzzzzzzzz", min_len=10)
    # Too short to check: returns False (not true; avoids false positives on tiny data)
    assert not _shares_substring("short", "short something else", min_len=20)


def test_classify_return_variants():
    assert _classify_return("", False) == "empty"
    assert _classify_return("Error: boom\nTraceback", False) == "error"
    assert _classify_return("normal text output of reasonable length", False) == "text"
    # Multimodal structured blocks → multimodal
    assert _classify_return("<image>\nextra", True) == "multimodal"
    # Short text with marker → multimodal_coerced (the render_pdf_page failure mode)
    assert _classify_return("[image displayed below]", False) == "multimodal_coerced"


def test_content_to_text_flattens_multimodal_blocks():
    blocks = [{"type": "image", "source": {...}}, {"type": "text", "text": "hello"}]
    text, structured = _content_to_text(blocks)
    assert "<image>" in text
    assert "hello" in text
    assert structured is True

    text, structured = _content_to_text("plain string")
    assert text == "plain string"
    assert structured is False


def test_args_sha_is_stable_and_short():
    s1 = _args_sha({"query": "foo", "k": 5})
    s2 = _args_sha({"k": 5, "query": "foo"})
    assert s1 == s2
    assert len(s1) == 8


# ---------- extract_trace_facts end-to-end ----------

def test_extracts_simple_trajectory(tmp_path):
    """Tool output contains a long phrase that the final answer also quotes
    verbatim — next_uses_result must detect this."""
    traj = tmp_path / "task_A_r0.jsonl"
    shared_phrase = "Oregon deposit is ten cents per eligible container"
    _traj(
        traj,
        assistant_tool_calls=[
            {"id": "c1", "name": "WebSearch", "input": {"query": "bottles"}},
        ],
        tool_results=[f"Wikipedia says: {shared_phrase} as of 2024."],
        final_assistant_text=f"Answer: {shared_phrase}.",
    )
    facts = extract_trace_facts("task_A", [traj])
    assert facts.rollouts == ["r0"]
    assert len(facts.tool_calls) == 1
    c = facts.tool_calls[0]
    assert c.tool == "WebSearch"
    assert c.return_type == "text"
    assert c.next_uses_result is True
    assert facts.exits[0].exit_reason == "done"


def test_detects_multimodal_silent_drop(tmp_path):
    """The `render_pdf_page` failure mode: tool returns short marker string."""
    traj = tmp_path / "task_pdf_r0.jsonl"
    _traj(
        traj,
        assistant_tool_calls=[
            {"id": "c1", "name": "render_pdf_page", "input": {"path": "f.pdf", "page": 3}},
        ],
        tool_results=["[image displayed below]"],
        final_assistant_text="I cannot determine the answer from the image.",
        exit_reason="done",
        passed=False,
    )
    facts = extract_trace_facts("task_pdf", [traj])
    assert facts.tool_calls[0].return_type == "multimodal_coerced"
    # Short return (< 20 chars for heuristic purposes — "[image displayed below]" is 23,
    # but more importantly, its content DOES NOT appear in the final assistant text).
    # next_uses_result should be False OR None (both are acceptable; False is the
    # strongest signal we can ask from a 23-char string).
    assert facts.tool_calls[0].next_uses_result in (False, None)


def test_detects_tool_effect_missing(tmp_path):
    """Tool returns plenty of text but next model step ignores it."""
    traj = tmp_path / "task_X_r0.jsonl"
    big_output = "The capital of Mars is Olympus Mons as reported by NASA in their 2024 survey document" * 3
    _traj(
        traj,
        assistant_tool_calls=[
            {"id": "c1", "name": "WebSearch", "input": {"q": "mars capital"}},
        ],
        tool_results=[big_output],
        final_assistant_text="I don't know the answer to this question.",
        passed=False,
    )
    facts = extract_trace_facts("task_X", [traj])
    c = facts.tool_calls[0]
    assert c.return_type == "text"
    assert c.return_len > 100
    # Tool output has no 20-char substring in the short "I don't know..." text.
    assert c.next_uses_result is False


def test_detects_repeated_calls(tmp_path):
    """Same (tool, args_sha) three times consecutively → one RepeatRun."""
    traj = tmp_path / "task_loop_r0.jsonl"
    same_input = {"path": "f.pdf", "page": 3}
    _traj(
        traj,
        assistant_tool_calls=[
            {"id": "c1", "name": "render_pdf_page", "input": same_input},
            {"id": "c2", "name": "render_pdf_page", "input": same_input},
            {"id": "c3", "name": "render_pdf_page", "input": same_input},
        ],
        tool_results=["[image displayed below]"] * 3,
        final_assistant_text="Giving up.",
        passed=False,
    )
    facts = extract_trace_facts("task_loop", [traj])
    assert len(facts.repeats) == 1
    r = facts.repeats[0]
    assert r.tool == "render_pdf_page"
    assert len(r.steps) == 3


def test_repeat_broken_by_different_tool(tmp_path):
    """A different tool between two same-sha calls breaks the run."""
    traj = tmp_path / "task_mix_r0.jsonl"
    same = {"q": "foo"}
    _traj(
        traj,
        assistant_tool_calls=[
            {"id": "c1", "name": "WebSearch", "input": same},
            {"id": "c2", "name": "Read", "input": {"path": "/tmp/a"}},
            {"id": "c3", "name": "WebSearch", "input": same},
        ],
        tool_results=["r1", "r2", "r3"],
    )
    facts = extract_trace_facts("task_mix", [traj])
    assert len(facts.repeats) == 0


def test_to_markdown_contains_required_sections(tmp_path):
    traj = tmp_path / "task_md_r0.jsonl"
    _traj(
        traj,
        assistant_tool_calls=[
            {"id": "c1", "name": "WebSearch", "input": {"q": "x"}},
        ],
        tool_results=["result text " * 10],
    )
    facts = extract_trace_facts("task_md", [traj])
    md = facts.to_markdown()
    assert "## Trace Facts (Layer A" in md
    assert "### Exits" in md
    assert "### Tool calls" in md
    assert "### Repeated tool calls" in md
    assert "### Tool calls whose output the next step did NOT reference" in md
    assert "r0" in md
    # Stable schema: columns are always named the same way
    assert "return_type" in md


def test_handles_multiple_rollouts(tmp_path):
    r0 = tmp_path / "task_M_r0.jsonl"
    r1 = tmp_path / "task_M_r1.jsonl"
    _traj(r0, assistant_tool_calls=[{"id": "a", "name": "T1", "input": {}}], tool_results=["x"], passed=True)
    _traj(r1, assistant_tool_calls=[{"id": "b", "name": "T2", "input": {}}], tool_results=["y"], passed=False)
    facts = extract_trace_facts("task_M", [r0, r1])
    assert set(facts.rollouts) == {"r0", "r1"}
    assert len(facts.exits) == 2
    passed_map = {e.rollout: e.passed for e in facts.exits}
    assert passed_map == {"r0": True, "r1": False}


def test_missing_tool_result_marked_missing(tmp_path):
    """An assistant tool_call with no matching raw_tool event."""
    traj = tmp_path / "task_orphan_r0.jsonl"
    events = [
        {"type": "session_start", "step": 0, "task": "t"},
        {"type": "raw_user", "step": 0, "message": {"role": "user", "content": "go"}},
        {"type": "raw_assistant", "step": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "nope", "name": "Broken", "input": {}}],
        }},
        {"type": "episode_end", "step": 1, "exit_reason": "error", "total_steps": 1, "passed": False},
    ]
    _write_jsonl(traj, events)
    facts = extract_trace_facts("task_orphan", [traj])
    assert len(facts.tool_calls) == 1
    assert facts.tool_calls[0].return_type == "missing"


def test_empty_trajectory(tmp_path):
    traj = tmp_path / "task_empty_r0.jsonl"
    traj.write_text("", encoding="utf-8")
    facts = extract_trace_facts("task_empty", [traj])
    assert facts.tool_calls == []
    md = facts.to_markdown()
    assert "## Trace Facts" in md  # still renders a valid block


def test_to_markdown_flags_next_uses_result_NO(tmp_path):
    """The tool_effect_missing case should show up in the 'NO' shortlist section."""
    traj = tmp_path / "task_drop_r0.jsonl"
    _traj(
        traj,
        assistant_tool_calls=[{"id": "c1", "name": "WebSearch", "input": {"q": "foo"}}],
        tool_results=["massive detailed unused output " * 10],
        final_assistant_text="stopping here without reference",
    )
    facts = extract_trace_facts("task_drop", [traj])
    md = facts.to_markdown()
    assert "**NO**" in md  # at least one row flagged
    assert "WebSearch" in md
