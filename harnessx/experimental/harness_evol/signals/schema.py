from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FailedToolCall:
    step_id: int
    tool_name: str
    error_summary: str      # first 200 chars
    recovered: bool         # retried with same tool within recovery_window steps
    rollout_path: str = ""  # source rollout path for Layer 2 traceability


@dataclass
class RepeatedSequence:
    tool_name: str
    normalized_input: str   # uuid/timestamp-stripped representation
    first_step: int
    count: int              # consecutive repeat count


@dataclass
class CompactionEvent:
    step_id: int
    tokens_before: int
    tokens_after: int
    compression_ratio: float    # tokens_after / tokens_before; 0.0 if tokens_before == 0


@dataclass
class SlowToolCall:
    step_id: int
    tool_name: str
    tool_input_summary: str     # key args — e.g. Bash command, file path, glob pattern
    duration_ms: int
    followed_by_error: bool
    rollout_path: str = ""  # source rollout path for Layer 2 traceability


@dataclass
class RolloutData:
    """
    One rollout's parsed signals. Produced by parse_session_rollout();
    consumed by TrajectorySignalExtractor.

    tool_calls:   [{step_id, tool_name, tool_call_id, input, timestamp}]
    tool_results: [{step_id, tool_name, tool_call_id, error, duration_ms}]
    """
    rollout_path: Path
    task_description: str           # from session_start.task (first line of first JSONL)
    exit_reason: str
    eval_passed: bool
    eval_score: float
    total_steps: int
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    token_budget: int
    task_end_error: str | None
    total_wall_clock_ms: float
    tool_calls: list[dict]
    tool_results: list[dict]
    step_start_tokens: dict[int, int]
    step_start_timestamps: dict[int, float]
    tool_call_timestamps: dict[str, float]
    compaction_step_ids: list[int]
    compaction_reasons: dict[int, str]
    memory_written_steps: list[int]
    processor_trigger_counts: dict[str, int]
    is_partial_trace: bool = False
    # Benchmark-specific evaluation feedback injected by score_fn.
    # Stored as-is — Layer 1 does not parse or interpret this field.
    # Structure is benchmark-defined; may contain structured test results, judge text, etc.
    eval_feedback: dict | None = None


# ── TaskSignals sub-objects ───────────────────────────────────────────────────

@dataclass
class TaskMeta:
    """Identity, trace provenance, and quality flags."""
    task_id: str
    task_description: str       # first 300 chars from session_start.task
    rep_rollout_path: str       # which rollout is the "representative" for single-rollout signals
    all_rollout_paths: list[str] = field(default_factory=list)  # all k rollout paths for cross-rollout comparison
    rollout_count: int = 0
    partial_rollout_count: int = 0  # rollouts with ring-buffer truncation
    is_partial_trace: bool = False  # True when the rep rollout itself is partial


@dataclass
class TaskOutcome:
    """Pass/fail result, exit reason distribution, and fixability classification."""
    exit_reason: str                    # exit reason of the rep rollout
    eval_passed: bool                   # True only when ALL rollouts passed
    eval_score: float                   # eval score of the rep rollout
    rollout_pass_rate: float
    any_rollout_passed: bool
    all_rollouts_passed: bool
    exit_reason_counts: dict[str, int]  # distribution across all k rollouts
    mechanical_fixability: str          # "level1_fixable" | "unclear"
    mechanical_fixability_signal: str   # short description of the deciding signal
    # Computed tags for Layer 2 clustering. A task can have multiple tags.
    # Tags: all_pass | unstable_pass | k_divergence | budget_exhausted | loop_detected | error_exit
    #       loop_in_tool_calls | unrecovered_tool_error | compaction_error_spike
    #       bash_errors | slow_tools | partial_trace
    #       dangerous_regression | historical_regression | chronic_failure
    # unstable_pass: all_rollouts_passed=True but passing_steps_cv > 0.3 (pass but path varies)
    failure_pattern_tags: list[str]


@dataclass
class RepRolloutSignals:
    """
    Single-rollout signals from the representative rollout.

    Rep selection: first non-partial failing rollout among k; fallback to rollouts[0].
    ALL fields here describe ONLY that one rollout — not aggregates.
    Use pass_vs_fail for cross-rollout comparison.
    """
    total_steps: int
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    output_token_ratio: float           # total_output_tokens / total_tokens
    total_cost_usd: float
    total_wall_clock_ms: float
    token_budget_utilization: float     # total_tokens / token_budget
    task_end_error: str | None

    # Tool usage
    tool_call_histogram: dict[str, int]     # call counts from THIS rollout
    tool_error_histogram: dict[str, int]    # error counts per tool
    tool_error_rate: float
    failed_tool_calls: list[FailedToolCall]
    slow_tool_calls: list[SlowToolCall]
    repeated_sequences: list[RepeatedSequence]
    first_tool_error_step: int | None
    first_repeated_seq_step: int | None
    tool_bigrams: list[tuple[str, str, int]]  # top-5 consecutive (from, to, count) pairs

    # Timing
    per_tool_avg_duration_ms: dict[str, float]
    per_tool_p95_duration_ms: dict[str, float]
    avg_model_inference_ms: float
    max_model_inference_ms: float
    long_model_inference_count: int     # steps where inference took >120s (infra stall indicator)

    # Step structure
    active_step_ratio: float            # steps with ≥1 tool call / total_steps
    first_action_step: int              # step_id of first tool call (-1 if none)
    max_tool_calls_per_step: int
    self_verify_rate: float             # Write/Edit followed by Read/Glob/Grep within 3 steps
    error_category_counts: dict[str, int]   # timeout/not_found/permission/parse_error/other

    # Context / compaction
    compaction_events: list[CompactionEvent]
    compaction_reasons: dict[int, str]      # step_id -> segment_boundary reason
    steps_after_last_compaction: int
    # Raw evidence for Layer 2 to reason about compaction impact.
    # Higher post-compaction error rate may be harder steps, not context loss.
    pre_compaction_error_rate: float
    post_compaction_error_rate: float

    # Processor health
    memory_active_ratio: float
    processor_trigger_counts: dict[str, int]

    # Benchmark-specific evaluation feedback from the representative rollout.
    # Passed through opaquely from score_fn — Layer 1 does not parse this.
    # Layer 2 (DigestAgent) reads it for failure diagnosis.
    eval_feedback: dict | None = None


@dataclass
class TestFailureSummary:
    """
    Cross-rollout aggregation of a single test case from CTRF feedback.
    Derived from per_rollout_feedbacks by the Layer 1 extractor.
    Sorted by failed_count desc — most-broken tests come first.
    """
    test_name: str
    passed_count: int
    failed_count: int
    rollouts_tested: int            # number of rollouts that reported this test
    sample_trace: str | None        # first non-None failure trace, truncated to 300 chars


@dataclass
class PassVsFailSignals:
    """
    Intra-task comparison across k rollouts.

    Comparing the failing and passing cohorts identifies what separates them.
    All fields are 0/empty when the corresponding cohort is empty.

    Key comparisons for Layer 2:
      failing_error_rate_mean >> passing_error_rate_mean  → tool errors explain divergence
      failing_wall_clock_mean >> passing_wall_clock_mean  → timeout sensitivity
      Compare failing_tool_histogram vs passing_tool_histogram to find which tools differ
    """
    failing_rollout_count: int
    passing_rollout_count: int

    # Step distribution
    failing_steps_min: int
    failing_steps_max: int
    failing_steps_mean: float
    passing_steps_min: int
    passing_steps_max: int
    passing_steps_mean: float
    passing_steps_cv: float             # high → some rollouts took many more steps (detour)

    # Token distribution
    failing_tokens_mean: float
    passing_tokens_min: int
    passing_tokens_max: int
    passing_tokens_mean: float
    passing_tokens_cv: float            # high + steps_cv low → heavy context per step

    # Error rate comparison
    failing_error_rate_mean: float
    passing_error_rate_mean: float

    # Wall clock comparison
    failing_wall_clock_mean: float
    passing_wall_clock_mean: float
    passing_wall_clock_cv: float

    # Model inference time comparison
    failing_inference_ms_mean: float
    passing_inference_ms_mean: float

    # Tool usage comparison (BOTH aggregated across their respective cohorts)
    # failing_tool_histogram: aggregated across ALL failing rollouts (not just rep)
    # passing_tool_histogram: aggregated across ALL passing rollouts
    failing_tool_histogram: dict[str, int]
    passing_tool_histogram: dict[str, int]

    # Failing cohort distribution (symmetric with passing cohort).
    # All four are 0/0.0 when failing_rollout_count == 0.
    failing_tokens_min: int
    failing_tokens_max: int
    failing_tokens_cv: float            # high → some failing runs hit budget, others bail early
    failing_steps_cv: float             # high → failing rollouts diverge in how far they get

    # Compaction count per rollout (same order as input rollouts list)
    per_rollout_compaction_counts: list[int]

    # Benchmark-specific feedback per rollout (same order as input rollouts list).
    # None entry when score_fn did not provide feedback for that rollout.
    # Layer 2 can iterate these to find which specific tests failed across rollouts.
    per_rollout_feedbacks: list[dict | None] = field(default_factory=list)

    # Pre-aggregated cross-rollout test failure summary (derived from per_rollout_feedbacks).
    # Sorted by failed_count desc so the most-broken tests appear first.
    # Layer 2 agents can read this directly instead of re-aggregating per_rollout_feedbacks.
    test_failure_summary: list["TestFailureSummary"] = field(default_factory=list)


@dataclass
class HistorySignals:
    """
    Cross-round solvability from SolvabilityJournal.
    All fields default when no journal is provided.
    """
    ever_passed: bool               # any rollout passed in any prior round
    ever_all_passed: bool           # ALL rollouts passed in at least one prior round
    last_passed_round: int | None
    consecutive_pass_rounds_before: int
    was_stable: bool                # consecutive_pass_rounds_before >= 2
    rounds_without_flip: int        # rounds with zero passing rollouts

    # Rate-regression signals.
    # hist_best_pass_rate: best rollout_pass_rate seen across ALL prior rounds (0.0–1.0).
    # rate_regressed: True when current rollout_pass_rate < hist_best_pass_rate.
    # These fire even when some rollouts still pass (partial regression), unlike
    # the all-fail "partial_regression" category.
    hist_best_pass_rate: float | None           # None only on round 0 (no history yet)
    rate_regressed: bool                        # current pass rate dropped from historical best

    # Regression comparison.
    # Populated only when ever_all_passed=True AND all_rollouts_passed=False.
    # Lets Layer 2 quantify regression magnitude and read historical traces directly.
    hist_best_passing_rollout_paths: list[str] | None   # up to 2 paths, ascending by token count
    hist_best_passing_tokens: int | None                # best single value (for delta computation)
    hist_best_passing_steps: int | None
    hist_best_passing_tool_histogram: dict[str, int] | None
    current_vs_hist_token_delta: int | None     # > 0 means more tokens now
    current_vs_hist_step_delta: float | None    # > 0 means more steps now

    # Per-round trend data. None when no journal history available.
    # pass_rate_history: round_idx → pass_rate (0.0–1.0). Enables trend analysis
    #   (e.g. 100%→80%→60% gradual decline vs 100%→0% sudden drop).
    # gap_type_history: round_idx → DigestAgent's gap_type classification.
    #   Enables cross-round pattern evolution ("was stability, now model_gap").
    pass_rate_history: dict[int, float] | None
    gap_type_history: dict[int, str] | None


@dataclass
class TaskSignals:
    """
    All Layer 1 signals for one task (k rollouts merged).

    JSON structure (via dataclasses.asdict):
      meta         → identity, trace provenance
      outcome      → pass/fail, fixability, failure_pattern_tags
      rep_rollout  → single-rollout behavior signals (tool usage, timing, compaction)
      pass_vs_fail → intra-task pass/fail cohort comparison
      history      → cross-round solvability (from SolvabilityJournal)
    """
    meta: TaskMeta
    outcome: TaskOutcome
    rep_rollout: RepRolloutSignals
    pass_vs_fail: PassVsFailSignals
    history: HistorySignals


# Tool calls exceeding this threshold are reported as SlowToolCall.
SLOW_TOOL_THRESHOLD_MS: int = 10_000
