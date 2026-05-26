"""
Raw trace parser for HarnessJournal JSONL files.

Belongs to Layer 1: pure Python, deterministic, no LLM calls.
Produces RolloutData from trace JSONL files.

Two entry points:

  parse_one_rollout(trace_path)
      HarnessX native format: one run = one {run_id}_trace.jsonl + {run_id}.jsonl pair.

  parse_session_rollout(session_dir)
      Session-directory format: one run = one session directory that may contain multiple
      {run_id}_trace.jsonl + {run_id}.jsonl pairs (one per compression segment).
      All segments are merged into a single RolloutData in step order.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .schema import RolloutData

logger = logging.getLogger(__name__)



def to_unix_ts(ts) -> float | None:
    """Convert timestamp to Unix float seconds. Handles both float and ISO 8601 string."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        from datetime import datetime, timezone
        try:
            s = ts.rstrip("Z")
            dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None


def parse_one_rollout(trace_path: Path) -> RolloutData | None:
    """
    Parse one rollout from a HarnessJournal trace JSONL file and its paired
    main JSONL file.

    Trace file ({run_id}_trace.jsonl):
      step_start  -> token_count at each step
      tool_call   -> tool_name, tool_call_id
      tool_result -> tool_name, tool_call_id, error (None = success)
      step_end    -> cumulative_tokens, cumulative_cost_usd (fallback for truncated traces)
      segment_boundary -> compaction marker
      task_end    -> exit_reason, total_steps, total_tokens, total_cost_usd

    Main file ({run_id}.jsonl):
      episode_end     -> passed, reward
      raw_assistant / assistant messages -> tool_call inputs (for repeat detection)
    """
    main_path = trace_path.with_name(trace_path.name.replace("_trace.jsonl", ".jsonl"))

    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    step_start_tokens: dict[int, int] = {}
    step_start_timestamps: dict[int, float] = {}
    tool_call_timestamps: dict[str, float] = {}
    compaction_step_ids: list[int] = []
    compaction_reasons: dict[int, str] = {}
    memory_written_steps: list[int] = []
    processor_trigger_counts: dict[str, int] = {}
    task_description: str = ""
    exit_reason = "unknown"
    total_steps = 0
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0
    token_budget = 0
    task_end_error: str | None = None
    task_end_timestamp: float | None = None
    # Fallback accumulators from step_end (used when task_end is missing or reports 0 tokens)
    last_step_end_tokens: int = 0
    last_step_end_cost: float = 0.0
    last_step_end_step: int = -1
    last_step_end_timestamp: float | None = None

    try:
        with trace_path.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                et = rec.get("event_type")
                if et == "step_start":
                    step = rec["step"]
                    step_start_tokens[step] = rec.get("token_count", 0)
                    ts = to_unix_ts(rec.get("timestamp"))
                    if ts is not None:
                        step_start_timestamps[step] = ts
                    if token_budget == 0 and rec.get("token_budget"):
                        token_budget = int(rec["token_budget"])
                elif et == "tool_call":
                    cid = rec["tool_call_id"]
                    entry = {
                        "step_id": rec["step"],
                        "tool_name": rec["tool_name"],
                        "tool_call_id": cid,
                        "input": {},  # enriched from main JSONL below
                    }
                    ts = to_unix_ts(rec.get("timestamp"))
                    if ts is not None:
                        entry["timestamp"] = ts
                        tool_call_timestamps[cid] = ts
                    tool_calls.append(entry)
                elif et == "tool_result":
                    tool_results.append({
                        "step_id": rec["step"],
                        "tool_name": rec["tool_name"],
                        "tool_call_id": rec["tool_call_id"],
                        "error": rec.get("error"),       # None → success
                        "duration_ms": rec.get("duration_ms", 0),
                    })
                elif et == "step_end":
                    if rec.get("memory_written"):
                        memory_written_steps.append(rec["step"])
                    # Track cumulative stats as fallback for when task_end is missing
                    if rec.get("cumulative_tokens"):
                        last_step_end_tokens = int(rec["cumulative_tokens"])
                        last_step_end_step = rec["step"]
                    if rec.get("cumulative_cost_usd"):
                        last_step_end_cost = float(rec["cumulative_cost_usd"])
                    ts = to_unix_ts(rec.get("timestamp"))
                    if ts is not None:
                        last_step_end_timestamp = ts
                elif et == "segment_boundary":
                    step = rec["step"]
                    compaction_step_ids.append(step)
                    if rec.get("reason"):
                        compaction_reasons[step] = rec["reason"]
                elif et == "processor_trigger":
                    proc = rec.get("processor", "unknown")
                    processor_trigger_counts[proc] = processor_trigger_counts.get(proc, 0) + 1
                elif et == "task_end":
                    exit_reason = rec.get("exit_reason", "unknown")
                    total_steps = rec.get("total_steps", 0)
                    total_tokens = rec.get("total_tokens", 0)
                    total_input_tokens = rec.get("total_input_tokens", 0)
                    total_output_tokens = rec.get("total_output_tokens", 0)
                    total_cost_usd = rec.get("total_cost_usd", 0.0)
                    task_end_error = rec.get("error") or None
                    task_end_timestamp = to_unix_ts(rec.get("timestamp"))
    except Exception as exc:
        logger.warning("Failed to parse trace %s: %s", trace_path, exc)
        return None

    # Fallback for truncated traces (process killed before task_end was written).
    # step_end.cumulative_tokens/cost are updated at every step, so the last one
    # is the best approximation of total usage when task_end is absent or reports 0.
    if total_tokens == 0 and last_step_end_tokens > 0:
        total_tokens = last_step_end_tokens
        total_cost_usd = last_step_end_cost
        exit_reason = "agent_timeout"   # no task_end = process was killed
    if total_steps == 0 and last_step_end_step >= 0:
        total_steps = last_step_end_step + 1
    # Use last step_end timestamp for wall-clock calculation when task_end is missing
    if task_end_timestamp is None and last_step_end_timestamp is not None:
        task_end_timestamp = last_step_end_timestamp

    eval_passed = False
    eval_score = 0.0
    tool_input_by_id: dict[str, dict] = {}

    try:
        if main_path.exists():
            with main_path.open(encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    rec = json.loads(raw)
                    rec_type = rec.get("type")

                    if rec_type == "session_start" and not task_description:
                        task_description = str(rec.get("task") or "")[:300]

                    elif rec_type == "episode_end":
                        eval_passed = bool(rec.get("passed", False))
                        reward = rec.get("reward")
                        eval_score = float(reward) if reward is not None else (1.0 if eval_passed else 0.0)

                    # Extract tool inputs from assistant messages for repeat detection
                    elif rec_type in ("assistant", "raw_assistant"):
                        msg = rec.get("message", {})
                        if isinstance(msg, dict):
                            for tc in msg.get("tool_calls", []):
                                tc_id = tc.get("id") or tc.get("tool_call_id")
                                if tc_id and "input" in tc:
                                    tool_input_by_id[tc_id] = tc["input"] or {}
    except Exception as exc:
        logger.warning("Failed to parse main JSONL %s: %s", main_path, exc)

    # Enrich tool_calls with inputs from main JSONL
    for tc in tool_calls:
        tc["input"] = tool_input_by_id.get(tc["tool_call_id"], {})

    # Total wall-clock time: task_end.timestamp - earliest step_start.timestamp
    total_wall_clock_ms = 0.0
    if task_end_timestamp is not None and step_start_timestamps:
        first_ts = min(step_start_timestamps.values())
        if task_end_timestamp >= first_ts:
            total_wall_clock_ms = (task_end_timestamp - first_ts) * 1000.0

    # Partial trace: task_end reports more steps than we have step_start records for.
    # This happens when the trace is ring-buffer truncated (only last N steps recorded).
    is_partial_trace = total_steps > 0 and len(step_start_tokens) < total_steps

    return RolloutData(
        rollout_path=trace_path,
        task_description=task_description,
        is_partial_trace=is_partial_trace,
        exit_reason=exit_reason,
        eval_passed=eval_passed,
        eval_score=eval_score,
        total_steps=total_steps,
        total_tokens=total_tokens,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=total_cost_usd,
        token_budget=token_budget,
        task_end_error=task_end_error,
        total_wall_clock_ms=total_wall_clock_ms,
        tool_calls=tool_calls,
        tool_results=tool_results,
        step_start_tokens=step_start_tokens,
        step_start_timestamps=step_start_timestamps,
        tool_call_timestamps=tool_call_timestamps,
        compaction_step_ids=compaction_step_ids,
        compaction_reasons=compaction_reasons,
        memory_written_steps=memory_written_steps,
        processor_trigger_counts=processor_trigger_counts,
    )


def parse_session_rollout(session_dir: Path) -> RolloutData | None:
    """
    Parse one rollout from a session directory.

    A session may contain multiple {run_id}_trace.jsonl + {run_id}.jsonl pairs,
    one per compression segment. All segments are merged into a single RolloutData
    in step order (sorted by the first step_start.step in each trace file).

    The transition point between segments is recorded as a compaction event
    (the first step of each non-first segment), matching the semantic of a
    context compression that triggered a new run_id.

    episode_end and tool_call inputs are collected from ALL main .jsonl files.
    task_end is taken from the segment that contains it (the last one to run).
    """
    trace_files = sorted(session_dir.glob("*_trace.jsonl"))
    if not trace_files:
        return None

    # Sort segments by their first step number
    def _first_step(path: Path) -> int:
        try:
            with path.open(encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    rec = json.loads(raw)
                    if rec.get("event_type") == "step_start":
                        return rec["step"]
        except Exception:
            pass
        return 999999

    trace_files_ordered = sorted(trace_files, key=_first_step)

    # Accumulators — same as parse_one_rollout
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    step_start_tokens: dict[int, int] = {}
    step_start_timestamps: dict[int, float] = {}
    tool_call_timestamps: dict[str, float] = {}
    compaction_step_ids: list[int] = []
    compaction_reasons: dict[int, str] = {}
    memory_written_steps: list[int] = []
    processor_trigger_counts: dict[str, int] = {}
    task_description: str = ""
    exit_reason = "unknown"
    total_steps = 0
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0
    token_budget = 0
    task_end_error: str | None = None
    task_end_timestamp: float | None = None
    last_step_end_tokens: int = 0
    last_step_end_cost: float = 0.0
    last_step_end_step: int = -1
    last_step_end_timestamp: float | None = None

    for seg_idx, trace_path in enumerate(trace_files_ordered):
        seg_first_step: int | None = None
        try:
            with trace_path.open(encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    rec = json.loads(raw)
                    et = rec.get("event_type")
                    if et == "step_start":
                        step = rec["step"]
                        # Record segment boundary as compaction event (first step of non-first segment)
                        if seg_idx > 0 and seg_first_step is None:
                            # segment_boundary may already have recorded this step at end of previous segment;
                            # avoid double-counting.
                            if step not in compaction_step_ids:
                                compaction_step_ids.append(step)
                        if seg_first_step is None:
                            seg_first_step = step
                        step_start_tokens[step] = rec.get("token_count", 0)
                        ts = to_unix_ts(rec.get("timestamp"))
                        if ts is not None:
                            step_start_timestamps[step] = ts
                        if token_budget == 0 and rec.get("token_budget"):
                            token_budget = int(rec["token_budget"])
                    elif et == "tool_call":
                        cid = rec["tool_call_id"]
                        entry = {
                            "step_id": rec["step"],
                            "tool_name": rec["tool_name"],
                            "tool_call_id": cid,
                            "input": {},
                        }
                        ts = to_unix_ts(rec.get("timestamp"))
                        if ts is not None:
                            entry["timestamp"] = ts
                            tool_call_timestamps[cid] = ts
                        tool_calls.append(entry)
                    elif et == "tool_result":
                        tool_results.append({
                            "step_id": rec["step"],
                            "tool_name": rec["tool_name"],
                            "tool_call_id": rec["tool_call_id"],
                            "error": rec.get("error"),
                            "duration_ms": rec.get("duration_ms", 0),
                        })
                    elif et == "step_end":
                        if rec.get("memory_written"):
                            memory_written_steps.append(rec["step"])
                        if rec.get("cumulative_tokens"):
                            last_step_end_tokens = int(rec["cumulative_tokens"])
                            last_step_end_step = rec["step"]
                        if rec.get("cumulative_cost_usd"):
                            last_step_end_cost = float(rec["cumulative_cost_usd"])
                        ts = to_unix_ts(rec.get("timestamp"))
                        if ts is not None:
                            last_step_end_timestamp = ts
                    elif et == "segment_boundary":
                        # Explicit compaction within a segment (HarnessX native compaction)
                        step = rec["step"]
                        if step not in compaction_step_ids:
                            compaction_step_ids.append(step)
                        if rec.get("reason"):
                            compaction_reasons[step] = rec["reason"]
                    elif et == "processor_trigger":
                        proc = rec.get("processor", "unknown")
                        processor_trigger_counts[proc] = processor_trigger_counts.get(proc, 0) + 1
                    elif et == "task_end":
                        exit_reason = rec.get("exit_reason", "unknown")
                        total_steps = rec.get("total_steps", 0)
                        total_tokens = rec.get("total_tokens", 0)
                        total_input_tokens = rec.get("total_input_tokens", 0)
                        total_output_tokens = rec.get("total_output_tokens", 0)
                        total_cost_usd = rec.get("total_cost_usd", 0.0)
                        task_end_error = rec.get("error") or None
                        task_end_timestamp = to_unix_ts(rec.get("timestamp"))
        except Exception as exc:
            logger.warning("Failed to parse trace segment %s: %s", trace_path, exc)
            continue

    if not step_start_tokens and total_tokens == 0:
        logger.warning("No usable data found in session %s", session_dir)
        return None

    # Fallback for agent_timeout (process killed before task_end)
    if total_tokens == 0 and last_step_end_tokens > 0:
        total_tokens = last_step_end_tokens
        total_cost_usd = last_step_end_cost
        exit_reason = "agent_timeout"
    if total_steps == 0 and last_step_end_step >= 0:
        total_steps = last_step_end_step + 1
    if task_end_timestamp is None and last_step_end_timestamp is not None:
        task_end_timestamp = last_step_end_timestamp

    # Collect eval_passed and tool inputs from all main .jsonl files
    eval_passed = False
    eval_score = 0.0
    tool_input_by_id: dict[str, dict] = {}

    for trace_path in trace_files_ordered:
        main_path = trace_path.with_name(trace_path.name.replace("_trace.jsonl", ".jsonl"))
        if not main_path.exists():
            continue
        try:
            with main_path.open(encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    rec = json.loads(raw)
                    rec_type = rec.get("type")
                    if rec_type == "session_start" and not task_description:
                        task_description = str(rec.get("task") or "")[:300]
                    elif rec_type == "episode_end":
                        eval_passed = bool(rec.get("passed", False))
                        reward = rec.get("reward")
                        eval_score = float(reward) if reward is not None else (1.0 if eval_passed else 0.0)
                    elif rec_type in ("assistant", "raw_assistant"):
                        msg = rec.get("message", {})
                        if isinstance(msg, dict):
                            for tc in msg.get("tool_calls", []):
                                tc_id = tc.get("id") or tc.get("tool_call_id")
                                if tc_id and "input" in tc:
                                    tool_input_by_id[tc_id] = tc["input"] or {}
        except Exception as exc:
            logger.warning("Failed to parse main JSONL %s: %s", main_path, exc)

    for tc in tool_calls:
        tc["input"] = tool_input_by_id.get(tc["tool_call_id"], {})

    total_wall_clock_ms = 0.0
    if task_end_timestamp is not None and step_start_timestamps:
        first_ts = min(step_start_timestamps.values())
        if task_end_timestamp >= first_ts:
            total_wall_clock_ms = (task_end_timestamp - first_ts) * 1000.0

    # Partial trace: task_end reports more steps than we have step_start records for.
    is_partial_trace = total_steps > 0 and len(step_start_tokens) < total_steps

    return RolloutData(
        rollout_path=session_dir,
        task_description=task_description,
        is_partial_trace=is_partial_trace,
        exit_reason=exit_reason,
        eval_passed=eval_passed,
        eval_score=eval_score,
        total_steps=total_steps,
        total_tokens=total_tokens,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=total_cost_usd,
        token_budget=token_budget,
        task_end_error=task_end_error,
        total_wall_clock_ms=total_wall_clock_ms,
        tool_calls=tool_calls,
        tool_results=tool_results,
        step_start_tokens=step_start_tokens,
        step_start_timestamps=step_start_timestamps,
        tool_call_timestamps=tool_call_timestamps,
        compaction_step_ids=compaction_step_ids,
        compaction_reasons=compaction_reasons,
        memory_written_steps=memory_written_steps,
        processor_trigger_counts=processor_trigger_counts,
    )
