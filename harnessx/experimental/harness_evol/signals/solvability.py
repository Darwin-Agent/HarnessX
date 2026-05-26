from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import json

from .schema import RolloutData


@dataclass
class TaskSolvabilityRecord:
    task_id: str
    ever_passed: bool = False           # any rollout passed in any round (partial or full)
    ever_all_passed: bool = False       # ALL rollouts passed in at least one round
    last_passed_round: int | None = None
    gap_type: str = "unknown"           # stability|behavior|knowledge|reasoning|model_gap|unknown
    improvability_level: int = 3        # 1|2|3|4
    rounds_without_flip: int = 0        # rounds with no rollout passing
    level2_intervention_tried: bool = False
    consecutive_pass_rounds: int = 0    # rounds with all rollouts passing (current live value)
    was_stable_entering_this_round: bool = False
    consecutive_pass_rounds_entering_this_round: int = 0  # snapshot before this round's update
    # Best rollout pass rate ever seen across all rounds (0.0–1.0).
    # Updated every round regardless of all_passed / any_passed.
    hist_best_pass_rate: float = 0.0
    # Top-2 historical all-passing rollouts (lowest token count), stored whenever all rollouts pass.
    # Sorted ascending by tokens. Used by Layer 1 extractor to expose paths to DigestAgent
    # for divergence analysis without re-reading JSONL from scratch.
    best_passing_rollout_paths: list[str] = field(default_factory=list)   # max 2 paths
    best_passing_tokens: list[int] = field(default_factory=list)           # max 2, parallel to paths
    best_passing_steps: list[int] = field(default_factory=list)            # max 2, parallel to paths
    best_passing_tool_histogram: dict = field(default_factory=dict)        # histogram of the best single rollout
    # Per-round pass rate history (str key for JSON compatibility): round_idx → pass_rate (0.0–1.0).
    # Enables DigestAgent to see the trend (80%→60%→40%) rather than just hist_best vs current.
    pass_rate_history: dict = field(default_factory=dict)
    # Per-round gap_type history: round_idx → gap_type string.
    # Enables cross-round gap evolution analysis ("was stability, now model_gap").
    gap_type_history: dict = field(default_factory=dict)


class SolvabilityJournal:
    """Per-task solvability history across rounds. Append-only, JSON-persisted."""

    def __init__(self) -> None:
        self._records: dict[str, TaskSolvabilityRecord] = {}

    def update(
        self,
        round_idx: int,
        task_results: dict[str, list[RolloutData]],
    ) -> None:
        """
        Three-way split per task:
          all_passed  -> consecutive_pass_rounds++, ever_passed=True
          any_passed  -> reset consecutive_pass_rounds (partial pass is not stable)
          none_passed -> rounds_without_flip++, reset consecutive_pass_rounds

        was_stable_entering_this_round is saved before modifying consecutive_pass_rounds.
        """
        for task_id, rollouts in task_results.items():
            rec = self._records.setdefault(task_id, TaskSolvabilityRecord(task_id=task_id))

            # snapshot stability state before this round's results are applied
            rec.was_stable_entering_this_round = rec.consecutive_pass_rounds >= 2
            rec.consecutive_pass_rounds_entering_this_round = rec.consecutive_pass_rounds

            if not rollouts:
                rec.rounds_without_flip += 1
                rec.consecutive_pass_rounds = 0
                continue

            passes = [r.eval_passed for r in rollouts]
            all_passed = all(passes)
            any_passed = any(passes)
            rollout_pass_rate = sum(passes) / len(passes) if passes else 0.0

            # Always update best-ever pass rate and per-round history regardless of outcome.
            rec.hist_best_pass_rate = max(rec.hist_best_pass_rate, rollout_pass_rate)
            rec.pass_rate_history[str(round_idx)] = rollout_pass_rate

            if all_passed:
                rec.ever_passed = True
                rec.ever_all_passed = True
                rec.last_passed_round = round_idx
                rec.consecutive_pass_rounds += 1
                rec.rounds_without_flip = 0
                # Update top-2 best historical rollouts (lowest token count, ascending).
                # Merge current round's rollouts into the stored list, keep best 2.
                candidates = [
                    {
                        "path": str(r.rollout_path) if r.rollout_path else "",
                        "tokens": r.total_tokens,
                        "steps": r.total_steps if r.total_steps > 0 else len(r.step_start_tokens),
                    }
                    for r in rollouts
                ]
                existing = [
                    {"path": p, "tokens": t, "steps": s}
                    for p, t, s in zip(
                        rec.best_passing_rollout_paths,
                        rec.best_passing_tokens,
                        rec.best_passing_steps,
                    )
                ]
                merged = sorted(existing + candidates, key=lambda x: x["tokens"])[:2]
                rec.best_passing_rollout_paths = [x["path"] for x in merged]
                rec.best_passing_tokens = [x["tokens"] for x in merged]
                rec.best_passing_steps = [x["steps"] for x in merged]
                # Tool histogram tracks the single best rollout only.
                best_r = min(rollouts, key=lambda r: r.total_tokens)
                if not rec.best_passing_tool_histogram or best_r.total_tokens <= (rec.best_passing_tokens[0] if rec.best_passing_tokens else 0):
                    rec.best_passing_tool_histogram = dict(
                        Counter(tc["tool_name"] for tc in best_r.tool_calls)
                    )
            elif any_passed:
                rec.ever_passed = True
                # ever_all_passed unchanged — partial pass does not qualify
                rec.last_passed_round = round_idx
                rec.consecutive_pass_rounds = 0  # partial pass is not stable
                rec.rounds_without_flip = 0
            else:
                rec.consecutive_pass_rounds = 0
                rec.rounds_without_flip += 1

    def get_record(self, task_id: str) -> TaskSolvabilityRecord | None:
        return self._records.get(task_id)

    def get_all(self) -> dict[str, TaskSolvabilityRecord]:
        return dict(self._records)

    def update_gap_type(self, task_id: str, gap_type: str, level: int, round_idx: int = -1) -> None:
        """Write back gap_type and improvability_level after DigestAgent Phase B analysis."""
        rec = self._records.setdefault(task_id, TaskSolvabilityRecord(task_id=task_id))
        rec.gap_type = gap_type
        rec.improvability_level = level
        if round_idx >= 0:
            rec.gap_type_history[str(round_idx)] = gap_type

    def mark_level2_tried(self, task_id: str) -> None:
        rec = self._records.setdefault(task_id, TaskSolvabilityRecord(task_id=task_id))
        rec.level2_intervention_tried = True

    def save(self, path: Path) -> None:
        from dataclasses import asdict
        path.write_text(json.dumps(
            {tid: asdict(r) for tid, r in self._records.items()},
            indent=2,
        ))

    @classmethod
    def load(cls, path: Path) -> "SolvabilityJournal":
        obj = cls()
        if path.exists():
            data = json.loads(path.read_text())
            for tid, d in data.items():
                obj._records[tid] = TaskSolvabilityRecord(**d)
        return obj
