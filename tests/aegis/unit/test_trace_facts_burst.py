"""Regression: TraceFacts must surface tool_burst patterns.

Without burst detection, the existing ``RepeatRun`` only fires when args
are byte-identical between consecutive calls. The observed loop-trap
shape — same tool called many times with different args (e.g. 75 SmartFetch
calls iterating URLs) — slipped through, and the digester's evidence
ground was missing the strongest behavioural signal we could extract.
"""

from __future__ import annotations

import json
from pathlib import Path


from harnessx.aegis.stages.trace_facts import (
    extract_trace_facts,
)


def _write_traj(path: Path, calls: list[tuple[int, str, dict]]) -> None:
    """Write a minimal jsonl trajectory: one assistant event per (step, tool)
    call with paired tool result. ``calls`` is [(step, tool_name, args), ...].
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    next_id = 0
    for step, tool, args in calls:
        tcid = f"tc_{next_id}"
        next_id += 1
        events.append(
            {
                "type": "raw_assistant",
                "step": step,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": tcid, "name": tool, "input": args}],
                },
            }
        )
        events.append(
            {
                "type": "raw_tool",
                "step": step,
                "message": {
                    "role": "tool",
                    "tool_call_id": tcid,
                    "content": "ok",
                },
            }
        )
    events.append(
        {"type": "episode_end", "exit_reason": "done", "total_steps": max(c[0] for c in calls) + 1, "passed": True}
    )
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_no_burst_when_under_threshold(tmp_path: Path) -> None:
    """A healthy multi-tool trajectory with light per-tool counts: no burst."""
    p = tmp_path / "tid_r0.jsonl"
    _write_traj(
        p,
        [
            (0, "WebSearch", {"q": "x"}),
            (1, "WebFetch", {"url": "u1"}),
            (2, "Bash", {"cmd": "ls"}),
        ],
    )
    facts = extract_trace_facts("tid", [p])
    assert facts.bursts == []


def test_burst_total_threshold(tmp_path: Path) -> None:
    """20+ calls of the same tool in one rollout, even with different args
    and spread across many steps, is a burst."""
    p = tmp_path / "tid_r0.jsonl"
    calls = [(i // 4, "SmartFetch", {"url": f"u{i}"}) for i in range(25)]
    _write_traj(p, calls)
    facts = extract_trace_facts("tid", [p])
    assert any(b.tool == "SmartFetch" and b.total_calls == 25 for b in facts.bursts)


def test_burst_peak_threshold_one_step(tmp_path: Path) -> None:
    """10+ calls of the same tool concentrated in a single step (parallel
    tool calls within one assistant turn) is also a burst."""
    p = tmp_path / "tid_r0.jsonl"
    # 12 calls of WebFetch, all at step 1 (each is its own assistant event,
    # but shipped from the parallel-tool-calls pattern).
    calls = [(1, "WebFetch", {"url": f"u{i}"}) for i in range(12)]
    _write_traj(p, calls)
    facts = extract_trace_facts("tid", [p])
    assert facts.bursts, "expected a burst from 12 calls in one step"
    burst = facts.bursts[0]
    assert burst.max_calls_in_one_step == 12
    assert burst.peak_step == 1


def test_burst_severity_grades(tmp_path: Path) -> None:
    """severity=high when total>=30 OR peak>=15; medium otherwise."""
    p_med = tmp_path / "med_r0.jsonl"
    _write_traj(p_med, [(i // 5, "SmartFetch", {"u": f"u{i}"}) for i in range(22)])  # total 22, peak ~5
    facts_med = extract_trace_facts("tid", [p_med])
    assert any(b.severity == "medium" for b in facts_med.bursts)

    p_hi = tmp_path / "hi_r0.jsonl"
    _write_traj(p_hi, [(0, "SmartFetch", {"u": f"u{i}"}) for i in range(35)])  # total 35, peak 35
    facts_hi = extract_trace_facts("tid", [p_hi])
    assert any(b.severity == "high" for b in facts_hi.bursts)


def test_burst_renders_in_markdown(tmp_path: Path) -> None:
    """to_markdown surfaces the burst section so the digester template
    can teach the LLM to act on it."""
    p = tmp_path / "tid_r0.jsonl"
    _write_traj(p, [(0, "SmartFetch", {"u": f"u{i}"}) for i in range(25)])
    md = extract_trace_facts("tid", [p]).to_markdown()
    assert "Tool burst" in md
    assert "SmartFetch" in md
    assert "total=25" in md
