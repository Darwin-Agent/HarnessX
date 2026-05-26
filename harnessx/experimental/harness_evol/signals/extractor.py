from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable

from .schema import (
    CompactionEvent,
    FailedToolCall,
    HistorySignals,
    PassVsFailSignals,
    RepRolloutSignals,
    RepeatedSequence,
    RolloutData,
    SlowToolCall,
    TaskMeta,
    TaskOutcome,
    TaskSignals,
    TestFailureSummary,
    SLOW_TOOL_THRESHOLD_MS,
)
from .solvability import SolvabilityJournal

# user-supplied function: (task_id, rollout_path) -> float score
ScoreFn = Callable[[str, Path], float]

# dict[task_id, list[RolloutData]]
TaskRollouts = list[RolloutData]


class TrajectorySignalExtractor:
    """
    Layer 1: pure Python, deterministic, no LLM calls. < 1s per task.

    Each task may have k rollouts. extract() receives all rollouts for
    one task and merges them into a single TaskSignals.
    """

    def extract(
        self,
        task_id: str,
        rollouts: TaskRollouts,
        solvability_journal: SolvabilityJournal | None = None,
        score_fn: ScoreFn | None = None,
    ) -> TaskSignals:
        if not rollouts:
            return _empty_signals(task_id)

        # ── representative rollout selection ────────────────────────────────
        # 1. Prefer non-partial traces.
        # 2. Among eligible, pick the first failing rollout (behavior signals
        #    should reflect the failure mode); fallback to rollouts[0].
        non_partial = [r for r in rollouts if not r.is_partial_trace]
        candidates = non_partial if non_partial else rollouts
        failing = [r for r in candidates if not r.eval_passed]
        rep = failing[0] if failing else candidates[0]
        rep_path_str = str(rep.rollout_path)

        # ── tool call histogram (rep) ────────────────────────────────────────
        tool_call_histogram = dict(Counter(tc["tool_name"] for tc in rep.tool_calls))
        total_calls = len(rep.tool_calls)

        # ── tool error rate + per-tool error histogram (rep) ────────────────
        error_count = sum(1 for tr in rep.tool_results if tr["error"] is not None)
        tool_error_rate = error_count / total_calls if total_calls else 0.0
        tool_error_histogram = _compute_tool_error_histogram(rep.tool_results)

        # ── failed tool calls (rep) ──────────────────────────────────────────
        failed_calls = _extract_failed_calls(rep.tool_calls, rep.tool_results, rep_path_str)
        first_tool_error_step = failed_calls[0].step_id if failed_calls else None

        # ── repeated sequences (rep) ─────────────────────────────────────────
        repeated = _extract_repeated_sequences(rep.tool_calls)
        first_repeated_seq_step = repeated[0].first_step if repeated else None

        # ── tool timing (rep) ────────────────────────────────────────────────
        per_tool_avg_duration_ms, per_tool_p95_duration_ms = _compute_tool_duration_stats(rep.tool_results)
        _tc_input_by_id: dict[str, dict] = {
            tc["tool_call_id"]: (tc.get("input") or {}) for tc in rep.tool_calls
        }
        slow_tool_calls = [
            SlowToolCall(
                step_id=tr["step_id"],
                tool_name=tr["tool_name"],
                tool_input_summary=_summarize_tool_input(
                    tr["tool_name"], _tc_input_by_id.get(tr["tool_call_id"], {})
                ),
                duration_ms=tr["duration_ms"],
                followed_by_error=tr["error"] is not None,
                rollout_path=rep_path_str,
            )
            for tr in rep.tool_results
            if tr["duration_ms"] >= SLOW_TOOL_THRESHOLD_MS
        ]

        # ── model inference timing (rep) ─────────────────────────────────────
        avg_model_inference_ms, max_model_inference_ms, long_model_inference_count = (
            _compute_inference_times(
                rep.tool_calls, rep.step_start_timestamps, rep.tool_call_timestamps
            )
        )

        # ── step structure (rep) ─────────────────────────────────────────────
        token_budget_utilization = (
            rep.total_tokens / rep.token_budget if rep.token_budget > 0 else 0.0
        )
        effective_steps = max(rep.total_steps, len(rep.step_start_tokens))
        active_step_ratio, first_action_step, max_tool_calls_per_step = _compute_step_action_stats(
            rep.tool_calls, effective_steps
        )

        # ── behavior quality (rep) ───────────────────────────────────────────
        self_verify_rate = _compute_self_verify_rate(rep.tool_calls)
        error_category_counts = _compute_error_categories(rep.tool_results)

        # ── tool transition patterns (rep) ───────────────────────────────────
        tool_bigrams = _compute_tool_bigrams(rep.tool_calls)

        # ── compaction events (rep) ──────────────────────────────────────────
        compaction_events = _build_compaction_events(
            rep.compaction_step_ids, rep.step_start_tokens
        )
        steps_after_last_compaction = _compute_steps_after_last_compaction(
            rep.compaction_step_ids, effective_steps
        )
        pre_compaction_error_rate, post_compaction_error_rate = _compute_compaction_error_rates(
            rep.compaction_step_ids, rep.tool_calls, rep.tool_results
        )

        # ── processor health (rep) ───────────────────────────────────────────
        memory_active_ratio = (
            len(rep.memory_written_steps) / effective_steps if effective_steps > 0 else 0.0
        )

        # ── eval score ───────────────────────────────────────────────────────
        eval_score = score_fn(task_id, rep.rollout_path) if score_fn is not None else rep.eval_score

        # ── cross-rollout aggregates ─────────────────────────────────────────
        rollout_count = len(rollouts)
        partial_rollout_count = sum(1 for r in rollouts if r.is_partial_trace)
        passes = [r.eval_passed for r in rollouts]
        rollout_pass_rate = sum(passes) / rollout_count
        any_rollout_passed = any(passes)
        all_rollouts_passed = all(passes)

        exit_reason_counts: dict[str, int] = {}
        for r in rollouts:
            exit_reason_counts[r.exit_reason] = exit_reason_counts.get(r.exit_reason, 0) + 1

        per_rollout_compaction_counts = [len(r.compaction_step_ids) for r in rollouts]

        # ── passing rollout stats ────────────────────────────────────────────
        passing_rollouts = [r for r in rollouts if r.eval_passed]
        (
            passing_rollout_count,
            passing_tokens_min,
            passing_tokens_max,
            passing_tokens_mean,
            passing_tokens_cv,
            passing_steps_min,
            passing_steps_max,
            passing_steps_mean,
            passing_steps_cv,
        ) = _compute_passing_token_stats(passing_rollouts)

        # ── failing rollout stats ─────────────────────────────────────────────
        failing_rollout_count = len(failing)
        failing_tokens_mean = (
            sum(r.total_tokens for r in failing) / failing_rollout_count
            if failing_rollout_count else 0.0
        )
        _failing_steps = [
            r.total_steps if r.total_steps > 0 else len(r.step_start_tokens)
            for r in failing
        ]
        failing_steps_mean = sum(_failing_steps) / failing_rollout_count if failing_rollout_count else 0.0
        failing_steps_min = min(_failing_steps) if _failing_steps else 0
        failing_steps_max = max(_failing_steps) if _failing_steps else 0
        failing_steps_cv = _cv([float(s) for s in _failing_steps], failing_steps_mean)
        _failing_tokens = [r.total_tokens for r in failing]
        failing_tokens_min = min(_failing_tokens) if _failing_tokens else 0
        failing_tokens_max = max(_failing_tokens) if _failing_tokens else 0
        failing_tokens_cv = _cv([float(t) for t in _failing_tokens], failing_tokens_mean)

        # ── tool histograms aggregated across cohorts ─────────────────────────
        failing_tool_histogram = _aggregate_tool_histogram(failing)
        passing_tool_histogram = _aggregate_tool_histogram(passing_rollouts)

        # ── cross-rollout comparisons ─────────────────────────────────────────
        passing_wall_clock_mean, passing_wall_clock_cv, failing_wall_clock_mean = (
            _compute_cross_rollout_wall_clock(passing_rollouts, failing)
        )
        passing_error_rate_mean, failing_error_rate_mean = (
            _compute_cross_rollout_error_rates(passing_rollouts, failing)
        )
        passing_inference_ms_mean, failing_inference_ms_mean = (
            _compute_cross_rollout_inference_ms(passing_rollouts, failing)
        )

        # ── token breakdown (rep) ─────────────────────────────────────────────
        output_token_ratio = (
            rep.total_output_tokens / rep.total_tokens if rep.total_tokens > 0 else 0.0
        )

        # ── historical solvability ────────────────────────────────────────────
        rec = solvability_journal.get_record(task_id) if solvability_journal else None
        ever_passed = rec.ever_passed if rec else False
        ever_all_passed = rec.ever_all_passed if rec else False
        last_passed_round = rec.last_passed_round if rec else None
        consecutive_pass_rounds_before = rec.consecutive_pass_rounds_entering_this_round if rec else 0
        was_stable = rec.was_stable_entering_this_round if rec else False
        rounds_without_flip = rec.rounds_without_flip if rec else 0
        # Convert str keys (JSON-compat) back to int for HistorySignals consumers.
        pass_rate_history: dict[int, float] | None = (
            {int(k): v for k, v in rec.pass_rate_history.items()} if rec and rec.pass_rate_history else None
        )
        gap_type_history: dict[int, str] | None = (
            {int(k): v for k, v in rec.gap_type_history.items()} if rec and rec.gap_type_history else None
        )
        # hist_best_pass_rate is None on round 0 (no prior history).
        # After the journal update() runs for this round, hist_best_pass_rate already
        # reflects the CURRENT round — so we compare against the value BEFORE this
        # round's update, which is max(hist_best_pass_rate BEFORE - current rate).
        # However, update() is called before extract(), so we use rec.hist_best_pass_rate
        # which already includes the current round.  To detect regression we compare
        # against prior rounds only: use the pre-update value by subtracting if needed.
        # Simplest correct approach: rate_regressed = current < hist_best (which includes
        # current, so current == hist_best when it's a new high, never triggering regression).
        hist_best_pass_rate: float | None = rec.hist_best_pass_rate if rec else None
        rate_regressed = (
            hist_best_pass_rate is not None
            and rollout_pass_rate < hist_best_pass_rate
        )

        # ── regression comparison ─────────────────────────────────────────────
        hist_best_passing_rollout_paths: list[str] | None = None
        hist_best_passing_tokens: int | None = None
        hist_best_passing_steps: int | None = None
        hist_best_passing_tool_histogram: dict[str, int] | None = None
        current_vs_hist_token_delta: int | None = None
        current_vs_hist_step_delta: float | None = None
        if ever_all_passed and not all_rollouts_passed and rec and rec.best_passing_tokens:
            hist_best_passing_rollout_paths = list(rec.best_passing_rollout_paths)
            hist_best_passing_tokens = rec.best_passing_tokens[0]
            hist_best_passing_steps = rec.best_passing_steps[0]
            hist_best_passing_tool_histogram = dict(rec.best_passing_tool_histogram)
            rep_steps_eff = rep.total_steps if rep.total_steps > 0 else len(rep.step_start_tokens)
            current_vs_hist_token_delta = rep.total_tokens - rec.best_passing_tokens[0]
            current_vs_hist_step_delta = float(rep_steps_eff - rec.best_passing_steps[0])

        # ── fixability ────────────────────────────────────────────────────────
        mechanical_fixability, signal_desc = _estimate_fixability(
            rep.exit_reason,
            all_rollouts_passed,
            repeated,
            failed_calls,
            any_rollout_passed,
            rollout_pass_rate,
            ever_passed,
            ever_all_passed,
            was_stable,
            last_passed_round,
        )

        # ── failure pattern tags (for Layer 2 clustering) ─────────────────────
        failure_pattern_tags = _compute_failure_pattern_tags(
            exit_reason=rep.exit_reason,
            all_rollouts_passed=all_rollouts_passed,
            any_rollout_passed=any_rollout_passed,
            is_partial_trace=rep.is_partial_trace,
            repeated=repeated,
            failed_calls=failed_calls,
            slow_tool_calls=slow_tool_calls,
            tool_error_histogram=tool_error_histogram,
            pre_compaction_error_rate=pre_compaction_error_rate,
            post_compaction_error_rate=post_compaction_error_rate,
            ever_all_passed=ever_all_passed,
            was_stable=was_stable,
            ever_passed=ever_passed,
            rounds_without_flip=rounds_without_flip,
            passing_steps_cv=passing_steps_cv,
            rate_regressed=rate_regressed,
        )

        # ── feedback pass-through + cross-rollout test aggregation ───────────
        per_rollout_feedbacks = [r.eval_feedback for r in rollouts]
        test_failure_summary = _compute_test_failure_summary(per_rollout_feedbacks)

        return TaskSignals(
            meta=TaskMeta(
                task_id=task_id,
                task_description=rep.task_description,
                rep_rollout_path=rep_path_str,
                all_rollout_paths=[str(r.rollout_path) for r in rollouts],
                rollout_count=rollout_count,
                partial_rollout_count=partial_rollout_count,
                is_partial_trace=rep.is_partial_trace,
            ),
            outcome=TaskOutcome(
                exit_reason=rep.exit_reason,
                eval_passed=all_rollouts_passed,
                eval_score=eval_score,
                rollout_pass_rate=rollout_pass_rate,
                any_rollout_passed=any_rollout_passed,
                all_rollouts_passed=all_rollouts_passed,
                exit_reason_counts=exit_reason_counts,
                mechanical_fixability=mechanical_fixability,
                mechanical_fixability_signal=signal_desc,
                failure_pattern_tags=failure_pattern_tags,
            ),
            rep_rollout=RepRolloutSignals(
                total_steps=rep.total_steps,
                total_tokens=rep.total_tokens,
                total_input_tokens=rep.total_input_tokens,
                total_output_tokens=rep.total_output_tokens,
                output_token_ratio=output_token_ratio,
                total_cost_usd=rep.total_cost_usd,
                total_wall_clock_ms=rep.total_wall_clock_ms,
                token_budget_utilization=token_budget_utilization,
                task_end_error=rep.task_end_error,
                tool_call_histogram=tool_call_histogram,
                tool_error_histogram=tool_error_histogram,
                tool_error_rate=tool_error_rate,
                failed_tool_calls=failed_calls,
                slow_tool_calls=slow_tool_calls,
                repeated_sequences=repeated,
                first_tool_error_step=first_tool_error_step,
                first_repeated_seq_step=first_repeated_seq_step,
                tool_bigrams=tool_bigrams,
                per_tool_avg_duration_ms=per_tool_avg_duration_ms,
                per_tool_p95_duration_ms=per_tool_p95_duration_ms,
                avg_model_inference_ms=avg_model_inference_ms,
                max_model_inference_ms=max_model_inference_ms,
                long_model_inference_count=long_model_inference_count,
                active_step_ratio=active_step_ratio,
                first_action_step=first_action_step,
                max_tool_calls_per_step=max_tool_calls_per_step,
                self_verify_rate=self_verify_rate,
                error_category_counts=error_category_counts,
                compaction_events=compaction_events,
                compaction_reasons=rep.compaction_reasons,
                steps_after_last_compaction=steps_after_last_compaction,
                pre_compaction_error_rate=pre_compaction_error_rate,
                post_compaction_error_rate=post_compaction_error_rate,
                memory_active_ratio=memory_active_ratio,
                processor_trigger_counts=rep.processor_trigger_counts,
                eval_feedback=rep.eval_feedback,
            ),
            pass_vs_fail=PassVsFailSignals(
                failing_rollout_count=failing_rollout_count,
                passing_rollout_count=passing_rollout_count,
                failing_steps_min=failing_steps_min,
                failing_steps_max=failing_steps_max,
                failing_steps_mean=failing_steps_mean,
                passing_steps_min=passing_steps_min,
                passing_steps_max=passing_steps_max,
                passing_steps_mean=passing_steps_mean,
                passing_steps_cv=passing_steps_cv,
                failing_tokens_mean=failing_tokens_mean,
                failing_tokens_min=failing_tokens_min,
                failing_tokens_max=failing_tokens_max,
                failing_tokens_cv=failing_tokens_cv,
                failing_steps_cv=failing_steps_cv,
                passing_tokens_min=passing_tokens_min,
                passing_tokens_max=passing_tokens_max,
                passing_tokens_mean=passing_tokens_mean,
                passing_tokens_cv=passing_tokens_cv,
                failing_error_rate_mean=failing_error_rate_mean,
                passing_error_rate_mean=passing_error_rate_mean,
                failing_wall_clock_mean=failing_wall_clock_mean,
                passing_wall_clock_mean=passing_wall_clock_mean,
                passing_wall_clock_cv=passing_wall_clock_cv,
                failing_inference_ms_mean=failing_inference_ms_mean,
                passing_inference_ms_mean=passing_inference_ms_mean,
                failing_tool_histogram=failing_tool_histogram,
                passing_tool_histogram=passing_tool_histogram,
                per_rollout_compaction_counts=per_rollout_compaction_counts,
                per_rollout_feedbacks=per_rollout_feedbacks,
                test_failure_summary=test_failure_summary,
            ),
            history=HistorySignals(
                ever_passed=ever_passed,
                ever_all_passed=ever_all_passed,
                last_passed_round=last_passed_round,
                consecutive_pass_rounds_before=consecutive_pass_rounds_before,
                was_stable=was_stable,
                rounds_without_flip=rounds_without_flip,
                hist_best_pass_rate=hist_best_pass_rate,
                rate_regressed=rate_regressed,
                hist_best_passing_rollout_paths=hist_best_passing_rollout_paths,
                hist_best_passing_tokens=hist_best_passing_tokens,
                hist_best_passing_steps=hist_best_passing_steps,
                hist_best_passing_tool_histogram=hist_best_passing_tool_histogram,
                current_vs_hist_token_delta=current_vs_hist_token_delta,
                current_vs_hist_step_delta=current_vs_hist_step_delta,
                pass_rate_history=pass_rate_history,
                gap_type_history=gap_type_history,
            ),
        )

    def extract_batch(
        self,
        task_runs: dict[str, TaskRollouts],
        solvability_journal: SolvabilityJournal | None = None,
        score_fn: ScoreFn | None = None,
    ) -> dict[str, TaskSignals]:
        return {
            task_id: self.extract(task_id, rollouts, solvability_journal, score_fn)
            for task_id, rollouts in task_runs.items()
        }


# ── helpers ──────────────────────────────────────────────────────────────────


def _empty_signals(task_id: str) -> TaskSignals:
    return TaskSignals(
        meta=TaskMeta(
            task_id=task_id,
            task_description="",
            rep_rollout_path="",
            rollout_count=0,
            partial_rollout_count=0,
            is_partial_trace=False,
        ),
        outcome=TaskOutcome(
            exit_reason="error",
            eval_passed=False,
            eval_score=0.0,
            rollout_pass_rate=0.0,
            any_rollout_passed=False,
            all_rollouts_passed=False,
            exit_reason_counts={},
            mechanical_fixability="unclear",
            mechanical_fixability_signal="no rollout data",
            failure_pattern_tags=[],
        ),
        rep_rollout=RepRolloutSignals(
            total_steps=0,
            total_tokens=0,
            total_input_tokens=0,
            total_output_tokens=0,
            output_token_ratio=0.0,
            total_cost_usd=0.0,
            total_wall_clock_ms=0.0,
            token_budget_utilization=0.0,
            task_end_error=None,
            tool_call_histogram={},
            tool_error_histogram={},
            tool_error_rate=0.0,
            failed_tool_calls=[],
            slow_tool_calls=[],
            repeated_sequences=[],
            first_tool_error_step=None,
            first_repeated_seq_step=None,
            tool_bigrams=[],
            per_tool_avg_duration_ms={},
            per_tool_p95_duration_ms={},
            avg_model_inference_ms=0.0,
            max_model_inference_ms=0.0,
            long_model_inference_count=0,
            active_step_ratio=0.0,
            first_action_step=-1,
            max_tool_calls_per_step=0,
            self_verify_rate=0.0,
            error_category_counts={},
            compaction_events=[],
            compaction_reasons={},
            steps_after_last_compaction=0,
            pre_compaction_error_rate=0.0,
            post_compaction_error_rate=0.0,
            memory_active_ratio=0.0,
            processor_trigger_counts={},
        ),
        pass_vs_fail=PassVsFailSignals(
            failing_rollout_count=0,
            passing_rollout_count=0,
            failing_steps_min=0,
            failing_steps_max=0,
            failing_steps_mean=0.0,
            passing_steps_min=0,
            passing_steps_max=0,
            passing_steps_mean=0.0,
            passing_steps_cv=0.0,
            failing_tokens_mean=0.0,
            failing_tokens_min=0,
            failing_tokens_max=0,
            failing_tokens_cv=0.0,
            failing_steps_cv=0.0,
            passing_tokens_min=0,
            passing_tokens_max=0,
            passing_tokens_mean=0.0,
            passing_tokens_cv=0.0,
            failing_error_rate_mean=0.0,
            passing_error_rate_mean=0.0,
            failing_wall_clock_mean=0.0,
            passing_wall_clock_mean=0.0,
            passing_wall_clock_cv=0.0,
            failing_inference_ms_mean=0.0,
            passing_inference_ms_mean=0.0,
            failing_tool_histogram={},
            passing_tool_histogram={},
            per_rollout_compaction_counts=[],
        ),
        history=HistorySignals(
            ever_passed=False,
            ever_all_passed=False,
            last_passed_round=None,
            consecutive_pass_rounds_before=0,
            was_stable=False,
            rounds_without_flip=0,
            hist_best_pass_rate=None,
            rate_regressed=False,
            hist_best_passing_rollout_paths=None,
            hist_best_passing_tokens=None,
            hist_best_passing_steps=None,
            hist_best_passing_tool_histogram=None,
            current_vs_hist_token_delta=None,
            current_vs_hist_step_delta=None,
            pass_rate_history=None,
            gap_type_history=None,
        ),
    )


_UNSTABLE_PASS_CV_THRESHOLD: float = 0.3


def _compute_failure_pattern_tags(
    *,
    exit_reason: str,
    all_rollouts_passed: bool,
    any_rollout_passed: bool,
    is_partial_trace: bool,
    repeated: list[RepeatedSequence],
    failed_calls: list[FailedToolCall],
    slow_tool_calls: list[SlowToolCall],
    tool_error_histogram: dict[str, int],
    pre_compaction_error_rate: float,
    post_compaction_error_rate: float,
    ever_all_passed: bool,
    was_stable: bool,
    ever_passed: bool,
    rounds_without_flip: int,
    passing_steps_cv: float = 0.0,
    rate_regressed: bool = False,
) -> list[str]:
    """Compute clustering tags for Layer 2.  Multiple tags may apply."""
    tags: list[str] = []

    if all_rollouts_passed:
        tags.append("all_pass")
        # Passes but with high path variance → stabilisation target for evolver
        if passing_steps_cv > _UNSTABLE_PASS_CV_THRESHOLD:
            tags.append("unstable_pass")
        return tags

    # outcome tags
    if any_rollout_passed:
        tags.append("k_divergence")
    if exit_reason == "budget_exceeded":
        tags.append("budget_exhausted")
    elif exit_reason == "loop_detected":
        tags.append("loop_detected")
    elif exit_reason == "error":
        tags.append("error_exit")

    # behavior tags
    if repeated:
        tags.append("loop_in_tool_calls")
    unrecovered = [f for f in failed_calls if not f.recovered]
    if unrecovered:
        tags.append("unrecovered_tool_error")
    if slow_tool_calls:
        tags.append("slow_tools")

    # top error tool
    if tool_error_histogram:
        top_err_tool = max(tool_error_histogram, key=lambda t: tool_error_histogram[t])
        if top_err_tool == "Bash":
            tags.append("bash_errors")

    # compaction spike: post error rate exceeds pre by >10pp
    if post_compaction_error_rate > pre_compaction_error_rate + 0.10:
        tags.append("compaction_error_spike")

    # trace quality
    if is_partial_trace:
        tags.append("partial_trace")

    # history-based tags
    if ever_all_passed and was_stable and not any_rollout_passed:
        tags.append("dangerous_regression")
    elif ever_passed and not any_rollout_passed:
        tags.append("historical_regression")
    if rate_regressed:
        # Pass rate dropped below historical best (includes partial regression:
        # e.g. 2/3 → 1/3). Fires even when some rollouts still pass.
        tags.append("rate_regression")
    if rounds_without_flip >= 3:
        tags.append("chronic_failure")

    return tags


def _summarize_tool_input(tool_name: str, inp: dict, max_len: int = 200) -> str:
    """
    Return a compact, human-readable summary of tool input arguments.

    Tool-specific extraction:
      Bash       → inp["command"] (the shell command actually run)
      Write      → inp["file_path"] + first 60 chars of content
      Edit       → inp["file_path"]
      Read/Glob/Grep/NotebookEdit → inp["file_path"] or inp["pattern"]
      WebFetch/WebSearch → inp["url"] or inp["query"]
    Generic fallback → first non-empty key=value pair.
    """
    if not inp:
        return ""

    def _trunc(s: str) -> str:
        s = str(s).replace("\n", " ")
        return s[:max_len] + "…" if len(s) > max_len else s

    if tool_name == "Bash":
        cmd = inp.get("command") or inp.get("cmd") or ""
        return _trunc(cmd)
    if tool_name in ("Write", "Edit", "NotebookEdit"):
        path = inp.get("file_path") or inp.get("path") or ""
        if tool_name == "Write":
            content_preview = str(inp.get("content") or "")[:60].replace("\n", " ")
            return _trunc(f"{path} ← {content_preview}")
        return _trunc(str(path))
    if tool_name in ("Read", "Glob"):
        return _trunc(str(inp.get("file_path") or inp.get("pattern") or ""))
    if tool_name == "Grep":
        pattern = inp.get("pattern") or ""
        path = inp.get("path") or inp.get("file_path") or ""
        return _trunc(f"{pattern} in {path}" if path else str(pattern))
    if tool_name in ("WebFetch", "WebSearch"):
        return _trunc(str(inp.get("url") or inp.get("query") or ""))
    # generic fallback: first key=value
    for k, v in inp.items():
        val = str(v)
        if val:
            return _trunc(f"{k}={val}")
    return ""


def _extract_failed_calls(
    tool_calls: list[dict],
    tool_results: list[dict],
    rollout_path: str = "",
    recovery_window: int = 3,
) -> list[FailedToolCall]:
    """
    Return FailedToolCall for each tool call whose result was an error.

    A failed call is considered 'recovered' if a successful call of the
    same tool_name occurs within recovery_window steps after the failure.
    """
    error_by_id: dict[str, str] = {
        tr["tool_call_id"]: tr["error"]
        for tr in tool_results
        if tr["error"] is not None
    }
    if not error_by_id:
        return []

    success_calls: list[tuple[int, str]] = [
        (tc["step_id"], tc["tool_name"])
        for tc in tool_calls
        if tc["tool_call_id"] not in error_by_id
    ]

    failed: list[FailedToolCall] = []
    for tc in tool_calls:
        cid = tc["tool_call_id"]
        if cid not in error_by_id:
            continue
        step = tc["step_id"]
        name = tc["tool_name"]
        recovered = any(
            s_name == name and step < s_step <= step + recovery_window
            for s_step, s_name in success_calls
        )
        failed.append(FailedToolCall(
            step_id=step,
            tool_name=name,
            error_summary=error_by_id[cid][:200],
            recovered=recovered,
            rollout_path=rollout_path,
        ))
    return failed


_UUID_RE = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b')
_TS_RE = re.compile(r'\b\d{10,13}\b')


def _normalize_input(inp: dict) -> str:
    s = json.dumps(inp, sort_keys=True, ensure_ascii=False)
    s = _UUID_RE.sub("<uuid>", s)
    s = _TS_RE.sub("<ts>", s)
    return s


def _extract_repeated_sequences(tool_calls: list[dict]) -> list[RepeatedSequence]:
    repeated: list[RepeatedSequence] = []
    i = 0
    while i < len(tool_calls):
        tc = tool_calls[i]
        name = tc["tool_name"]
        norm = _normalize_input(tc.get("input") or {})
        count = 1
        j = i + 1
        while j < len(tool_calls):
            next_tc = tool_calls[j]
            if next_tc["tool_name"] == name and _normalize_input(next_tc.get("input") or {}) == norm:
                count += 1
                j += 1
            else:
                break
        if count >= 6:
            repeated.append(RepeatedSequence(
                tool_name=name,
                normalized_input=norm[:500],
                first_step=tc["step_id"],
                count=count,
            ))
        i = j if count >= 6 else i + 1
    return repeated


def _build_compaction_events(
    compaction_step_ids: list[int],
    step_start_tokens: dict[int, int],
) -> list[CompactionEvent]:
    events: list[CompactionEvent] = []
    for step in compaction_step_ids:
        tokens_before = step_start_tokens.get(step - 1, 0) if step > 0 else 0
        tokens_after = step_start_tokens.get(step, 0)
        ratio = (tokens_after / tokens_before) if tokens_before > 0 else 0.0
        events.append(CompactionEvent(
            step_id=step,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compression_ratio=ratio,
        ))
    return events


def _compute_steps_after_last_compaction(
    compaction_step_ids: list[int],
    total_steps: int,
) -> int:
    if not compaction_step_ids:
        return 0
    return max(0, total_steps - max(compaction_step_ids))


def _compute_tool_duration_stats(
    tool_results: list[dict],
) -> tuple[dict[str, float], dict[str, float]]:
    by_tool: dict[str, list[int]] = {}
    for tr in tool_results:
        ms = tr.get("duration_ms", 0)
        if ms > 0:
            by_tool.setdefault(tr["tool_name"], []).append(ms)

    avg: dict[str, float] = {}
    p95: dict[str, float] = {}
    for name, samples in by_tool.items():
        avg[name] = sum(samples) / len(samples)
        samples_sorted = sorted(samples)
        idx = min(len(samples_sorted) - 1, int(len(samples_sorted) * 0.95))
        p95[name] = float(samples_sorted[idx])
    return avg, p95


_LONG_INFERENCE_THRESHOLD_MS: float = 120_000.0


def _compute_inference_times(
    tool_calls: list[dict],
    step_start_timestamps: dict[int, float],
    tool_call_timestamps: dict[str, float],
) -> tuple[float, float, int]:
    if not step_start_timestamps or not tool_call_timestamps:
        return 0.0, 0.0, 0

    first_tc_ts: dict[int, float] = {}
    for tc in tool_calls:
        step = tc["step_id"]
        ts = tc.get("timestamp")
        if ts is None:
            continue
        if step not in first_tc_ts or ts < first_tc_ts[step]:
            first_tc_ts[step] = ts

    all_ms: list[float] = []
    for step, tc_ts in first_tc_ts.items():
        start_ts = step_start_timestamps.get(step)
        if start_ts is not None and tc_ts >= start_ts:
            all_ms.append((tc_ts - start_ts) * 1000.0)

    if not all_ms:
        return 0.0, 0.0, 0

    max_ms = max(all_ms)
    normal = [v for v in all_ms if v <= _LONG_INFERENCE_THRESHOLD_MS]
    long_count = len(all_ms) - len(normal)
    avg_ms = sum(normal) / len(normal) if normal else 0.0
    return avg_ms, max_ms, long_count


def _compute_step_action_stats(
    tool_calls: list[dict],
    total_steps: int,
) -> tuple[float, int, int]:
    if not tool_calls or total_steps == 0:
        return 0.0, -1, 0

    calls_per_step: dict[int, int] = {}
    for tc in tool_calls:
        step = tc["step_id"]
        calls_per_step[step] = calls_per_step.get(step, 0) + 1

    active_step_ratio = len(calls_per_step) / total_steps
    first_action_step = min(calls_per_step)
    max_tool_calls_per_step = max(calls_per_step.values())
    return active_step_ratio, first_action_step, max_tool_calls_per_step


_WRITE_TOOLS = frozenset({"Write", "Edit", "NotebookEdit"})
_VERIFY_TOOLS = frozenset({"Read", "Glob", "Grep"})


def _compute_self_verify_rate(tool_calls: list[dict], window: int = 3) -> float:
    write_indices = [i for i, tc in enumerate(tool_calls) if tc["tool_name"] in _WRITE_TOOLS]
    if not write_indices:
        return 0.0

    verified = 0
    for idx in write_indices:
        write_step = tool_calls[idx]["step_id"]
        for j in range(idx + 1, len(tool_calls)):
            next_tc = tool_calls[j]
            if next_tc["step_id"] > write_step + window:
                break
            if next_tc["tool_name"] in _VERIFY_TOOLS:
                verified += 1
                break

    return verified / len(write_indices)


_ERROR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("timeout",     re.compile(r"timeout|timed.?out", re.IGNORECASE)),
    ("not_found",   re.compile(r"no such file|not found|404|command not found", re.IGNORECASE)),
    ("permission",  re.compile(r"permission denied|403|access denied", re.IGNORECASE)),
    ("parse_error", re.compile(r"invalid json|syntax error|unexpected token|parse error", re.IGNORECASE)),
]


def _compute_error_categories(tool_results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tr in tool_results:
        err = tr.get("error")
        if err is None:
            continue
        matched = False
        for cat, pattern in _ERROR_PATTERNS:
            if pattern.search(err):
                counts[cat] = counts.get(cat, 0) + 1
                matched = True
                break
        if not matched:
            counts["other"] = counts.get("other", 0) + 1
    return counts


def _compute_tool_bigrams(
    tool_calls: list[dict],
    top_n: int = 5,
) -> list[tuple[str, str, int]]:
    if len(tool_calls) < 2:
        return []
    counter: Counter = Counter(
        (tool_calls[i]["tool_name"], tool_calls[i + 1]["tool_name"])
        for i in range(len(tool_calls) - 1)
    )
    return [(a, b, cnt) for (a, b), cnt in counter.most_common(top_n)]


def _aggregate_tool_histogram(rollouts: list[RolloutData]) -> dict[str, int]:
    agg: Counter = Counter()
    for r in rollouts:
        agg.update(tc["tool_name"] for tc in r.tool_calls)
    return dict(agg)


def _compute_passing_token_stats(
    passing_rollouts: list[RolloutData],
) -> tuple[int, int, int, float, float, int, int, float, float]:
    if not passing_rollouts:
        return 0, 0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0

    tokens = [r.total_tokens for r in passing_rollouts]
    steps = [r.total_steps if r.total_steps > 0 else len(r.step_start_tokens) for r in passing_rollouts]
    count = len(tokens)
    min_tok, max_tok = min(tokens), max(tokens)
    mean_tok = sum(tokens) / count
    min_steps, max_steps = min(steps), max(steps)
    mean_steps = sum(steps) / count

    def _cv(values: list, mean: float) -> float:
        if count < 2 or mean == 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / count
        return (variance ** 0.5) / mean

    return count, min_tok, max_tok, mean_tok, _cv(tokens, mean_tok), min_steps, max_steps, mean_steps, _cv(steps, mean_steps)


def _compute_tool_error_histogram(tool_results: list[dict]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for tr in tool_results:
        if tr.get("error") is not None:
            hist[tr["tool_name"]] = hist.get(tr["tool_name"], 0) + 1
    return hist


def _cv(values: list[float], mean: float) -> float:
    if len(values) < 2 or mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return (variance ** 0.5) / mean


def _compute_cross_rollout_wall_clock(
    passing_rollouts: list[RolloutData],
    failing_rollouts: list[RolloutData],
) -> tuple[float, float, float]:
    p_walls = [r.total_wall_clock_ms for r in passing_rollouts]
    f_walls = [r.total_wall_clock_ms for r in failing_rollouts]
    p_mean = sum(p_walls) / len(p_walls) if p_walls else 0.0
    f_mean = sum(f_walls) / len(f_walls) if f_walls else 0.0
    p_cv = _cv(p_walls, p_mean)
    return p_mean, p_cv, f_mean


def _compute_cross_rollout_error_rates(
    passing_rollouts: list[RolloutData],
    failing_rollouts: list[RolloutData],
) -> tuple[float, float]:
    def _rate(r: RolloutData) -> float:
        tc = len(r.tool_calls)
        err = sum(1 for tr in r.tool_results if tr["error"] is not None)
        return err / tc if tc > 0 else 0.0

    p_rates = [_rate(r) for r in passing_rollouts]
    f_rates = [_rate(r) for r in failing_rollouts]
    p_mean = sum(p_rates) / len(p_rates) if p_rates else 0.0
    f_mean = sum(f_rates) / len(f_rates) if f_rates else 0.0
    return p_mean, f_mean


def _compute_cross_rollout_inference_ms(
    passing_rollouts: list[RolloutData],
    failing_rollouts: list[RolloutData],
) -> tuple[float, float]:
    def _avg_inf(r: RolloutData) -> float:
        avg, _, _ = _compute_inference_times(
            r.tool_calls, r.step_start_timestamps, r.tool_call_timestamps
        )
        return avg

    p_vals = [v for r in passing_rollouts if (v := _avg_inf(r)) > 0]
    f_vals = [v for r in failing_rollouts if (v := _avg_inf(r)) > 0]
    p_mean = sum(p_vals) / len(p_vals) if p_vals else 0.0
    f_mean = sum(f_vals) / len(f_vals) if f_vals else 0.0
    return p_mean, f_mean


def _compute_compaction_error_rates(
    compaction_step_ids: list[int],
    tool_calls: list[dict],
    tool_results: list[dict],
) -> tuple[float, float]:
    if not compaction_step_ids or not tool_calls:
        return 0.0, 0.0

    last_compaction = max(compaction_step_ids)
    error_ids = {tr["tool_call_id"] for tr in tool_results if tr["error"] is not None}

    before = [tc for tc in tool_calls if tc["step_id"] < last_compaction]
    after  = [tc for tc in tool_calls if tc["step_id"] >= last_compaction]

    if not before or not after:
        return 0.0, 0.0

    rate_before = sum(1 for tc in before if tc["tool_call_id"] in error_ids) / len(before)
    rate_after  = sum(1 for tc in after  if tc["tool_call_id"] in error_ids) / len(after)
    return rate_before, rate_after


def _compute_test_failure_summary(feedbacks: list[dict | None]) -> list[TestFailureSummary]:
    """
    Aggregate structured test results across all rollout feedbacks.

    Supports two formats (tried in order):
      - CTRF:   feedback["results"]["tests"] → list of {name, status, trace?, message?}
      - Flat:   feedback["tests"]            → list of {name, status, trace?, message?}

    Silently skips feedbacks that match neither format (returns empty list for those).
    Returns sorted by failed_count desc so the most-broken tests appear first.
    """
    counts: dict[str, dict] = {}
    for fb in feedbacks:
        if fb is None:
            continue
        try:
            # CTRF format
            tests = (fb.get("results") or {}).get("tests") or []
            # Flat format fallback
            if not tests:
                tests = fb.get("tests") or []
        except AttributeError:
            continue
        for t in tests:
            name = t.get("name") or ""
            if not name:
                continue
            status = t.get("status", "")
            if name not in counts:
                counts[name] = {"passed": 0, "failed": 0, "trace": None}
            if status == "passed":
                counts[name]["passed"] += 1
            else:
                counts[name]["failed"] += 1
                if counts[name]["trace"] is None:
                    trace = t.get("trace") or t.get("message") or None
                    if trace:
                        counts[name]["trace"] = str(trace)[-300:]

    rollouts_tested = sum(1 for fb in feedbacks if fb is not None)
    result = [
        TestFailureSummary(
            test_name=name,
            passed_count=v["passed"],
            failed_count=v["failed"],
            rollouts_tested=rollouts_tested,
            sample_trace=v["trace"],
        )
        for name, v in counts.items()
    ]
    result.sort(key=lambda x: (-x.failed_count, x.test_name))
    return result


def _estimate_fixability(
    exit_reason: str,
    all_rollouts_passed: bool,
    repeated: list[RepeatedSequence],
    failed_calls: list[FailedToolCall],
    any_rollout_passed: bool,
    rollout_pass_rate: float,
    ever_passed_in_history: bool,
    ever_all_passed_in_history: bool,
    was_stable_before_this_round: bool,
    last_passed_round: int | None,
) -> tuple[str, str]:
    """
    Return (mechanical_fixability, signal_description).

    Priority (highest to lowest):
      B. any_rollout_passed=True      AND eval_passed=False -> level1_fixable (k>1 divergence)
      A2. ever_all_passed_in_history=True AND any_rollout_passed=False -> level1_fixable (regression)
      A1. ever_passed_in_history=True  AND any_rollout_passed=False -> level1_fixable (historical partial)
      C. repeated_sequences AND exit_reason in budget_exceeded/loop_detected -> level1_fixable
      D. unrecovered failed_tool_calls -> level1_fixable
      otherwise -> unclear
    """
    if all_rollouts_passed:
        return "unclear", "task passed — fixability not applicable"

    if any_rollout_passed:
        rate_pct = int(rollout_pass_rate * 100)
        return "level1_fixable", f"k>1 divergence: pass_rate={rate_pct}% — harness consistency needed"

    if ever_all_passed_in_history:
        stable_note = " ⚠ WAS STABLE" if was_stable_before_this_round else ""
        detail = f"last passed round {last_passed_round}" if last_passed_round is not None else "previously all-passed"
        return "level1_fixable", f"dangerous regression{stable_note}: {detail} — all rollouts now fail (was fully reliable)"

    if ever_passed_in_history:
        detail = f"last passed round {last_passed_round}" if last_passed_round is not None else "previously passed"
        return "level1_fixable", f"historical partial regression: {detail} — all current rollouts fail"

    if repeated and exit_reason in ("budget_exceeded", "loop_detected"):
        r = repeated[0]
        return "level1_fixable", f"loop: '{r.tool_name}' repeated x{r.count} until {exit_reason}"

    unrecovered = [f for f in failed_calls if not f.recovered]
    if unrecovered:
        names = ", ".join(f.tool_name for f in unrecovered[:3])
        return "level1_fixable", f"unrecovered tool error(s): {names}"

    return "unclear", "no deterministic signal found"
