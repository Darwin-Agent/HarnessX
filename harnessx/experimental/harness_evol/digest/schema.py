from __future__ import annotations
from pydantic import BaseModel, Field


class PatternImprovability(BaseModel):
    gap_type: str               # stability|behavior|knowledge|reasoning|model_gap|unknown
    improvability_level: int    # 1|2|3|4
    tasks: list[str]
    count: int
    signal: str                 # one-line explanation for this classification
    intervention_hint: str | None = None  # recommended intervention (optional)
    trace_evidence: list[str] = Field(default_factory=list)
    # each entry < 200 chars, must cite step number, e.g. "step 12: Bash('rm -rf /var') -> deleted key file"


class SevereRegression(BaseModel):
    task_id: str
    consecutive_pass_rounds_before: int
    last_passed_round: int
    suspected_change_ids: list[str]     # changes from previous round's manifest suspected as root cause
    regression_was_predicted: bool      # whether this was listed in the previous round's at_risk
    trace_diff_hint: str                # key difference between this round's failure and last round's success


class DigestReport(BaseModel):
    round: int
    pass_rate: float
    total_tasks: int
    failed_tasks: int

    # merged from failure_taxonomy and by_pattern; key = pattern name
    patterns: dict[str, PatternImprovability]

    # derived from patterns by the orchestrator, not trusted from LLM output
    level_counts: dict[int, int] = Field(default_factory=dict)  # {1: N, 2: N, 3: N, 4: N}
    harness_fixable_ratio: float = 0.0                          # (level1 + level2) / failed_tasks

    severe_regressions: list[SevereRegression] = Field(default_factory=list)

    # routing — two independent signals, not mutually exclusive
    # needs_revert:       LLM-detected severe regression; EvolveAgent should rollback before any other work
    # has_search_targets: L1/2 fixable patterns exist → EvolveAgent searches existing processors,
    #                     decides internally whether to tune params or implement a new processor
    has_severe_regression: bool = False
    needs_revert: bool = False
    has_search_targets: bool = False
    priority_pattern: str | None = None
    rationale: str = ""
