"""
EvolOrchestrator — single-round evolution interface.

The user controls the outer loop. Each call to run_single() performs one
round of evolution: load trajectories -> extract signals -> digest -> evolve -> validate.

Usage:

    orchestrator = EvolOrchestrator(
        evolve_model=ModelConfig(main=LiteLLMProvider("claude-opus-4-6")),
        digest_model=ModelConfig(main=LiteLLMProvider("claude-haiku-4-5")),
    )

    # user controls when and how to loop
    output = await orchestrator.run_single(
        harness_config=Path("my_harness.yaml"),
        trajectories_dir=Path("runs/round_000/trajectories"),
        task_context=TaskContext(round_idx=0, task_ids=["task-a", "task-b"]),
        output_dir=Path("runs/round_000/evol_output"),
    )
    if output.accepted:
        current_config = output.new_config_path
"""
from __future__ import annotations
import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from harnessx.core.model_config import ModelConfig

from .digest.runner import run_digest_from_signals
from .digest.schema import DigestReport
from .evolve.parse import EvolveResult
from .evolve.runner import run_evolve
from .signals.extractor import TrajectorySignalExtractor
from .signals.parser import parse_one_rollout
from .signals.runner import ScoreFn, _build_signals_report, _build_summary
from .signals.schema import RolloutData, TaskSignals
from .signals.solvability import SolvabilityJournal

logger = logging.getLogger(__name__)


# ── input / output types ──────────────────────────────────────────────────────

@dataclass
class TaskContext:
    """Metadata about the task batch for this round."""
    round_idx: int
    task_ids: list[str]     # expected task IDs, must match subdirs in trajectories_dir


@dataclass
class EvolRoundOutput:
    """
    Output of one evolution round.

    output_dir layout:
      evol-workspace/target_config.yaml   evolved harness config (present only if accepted)
      evol-workspace/                     new processor .py files (Explore mode only)
      mechanical_signals.json             Layer 1 signal extraction results
      digest_report.json                  DigestAgent analysis
      change_manifest.json                EvolveAgent proposed changes (present if evolve ran)
      validation_report.json              audit log of policy checks (does not gate acceptance)
      solvability_journal.json            updated cross-round solvability state

    When digest.needs_revert is True, EvolveAgent still runs and performs a surgical
    rollback (reverting only suspected_change_ids using old_value from last_change_manifest).
    accepted is True whenever the agent writes target_config.yaml and calls submit_change_manifest.
    """
    output_dir: Path
    accepted: bool
    new_config_path: Path | None    # absolute path to target_config.yaml; None if agent failed or skipped
    digest: DigestReport


# ── EvolOrchestrator ──────────────────────────────────────────────────────────

class EvolOrchestrator:
    """
    Single-round evolution orchestrator.

    Instantiate once with model configs; call run_single() each round.
    SolvabilityJournal accumulates internally across run_single() calls.

    Different agents = different harness configs loaded from YAML:
      - DigestAgent: digest/default_config.yaml — workspace (with skills) injected per round
      - EvolveAgent: evolve/default_config.yaml — workspace injected per round
    """

    def __init__(
        self,
        evolve_model: ModelConfig,
        digest_model: ModelConfig,
        digest_config_yaml: Path | None = None,
        digest_skill_dirs: list[Path] | None = None,
        evolve_config_yaml: Path | None = None,
        evolve_skill_dirs: list[Path] | None = None,
        score_fn: ScoreFn | None = None,
        journal_path: Path | None = None,
        notebook_path: Path | None = None,
        digest_max_steps: int = 200,
        evolve_max_steps: int = 300,
    ) -> None:
        self._evolve_model = evolve_model
        self._evolve_config_yaml = evolve_config_yaml
        self._evolve_skill_dirs = evolve_skill_dirs
        self._evolve_max_steps = evolve_max_steps
        self._score_fn = score_fn

        self._digest_model = digest_model
        self._digest_config_yaml = digest_config_yaml
        self._digest_skill_dirs = digest_skill_dirs
        self._digest_max_steps = digest_max_steps

        self._journal_path = journal_path
        self._notebook_path = notebook_path  # session-level shared notebook; None = per-round default
        self._solvability = (
            SolvabilityJournal.load(journal_path)
            if journal_path is not None and journal_path.exists()
            else SolvabilityJournal()
        )
        self._extractor = TrajectorySignalExtractor()
        self._last_change_manifest: dict | None = None

    async def run_single(
        self,
        harness_config: Path | None,
        trajectories_dir: Path,
        task_context: TaskContext,
        output_dir: Path,
    ) -> EvolRoundOutput:
        """
        Run one round of evolution.

        trajectories_dir must contain one subdirectory per task_id,
        each holding k rollout result files in HarnessJournal format:
          {trajectories_dir}/{task_id}/rollout_*.jsonl

        All intermediate and final outputs are written to output_dir.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        round_idx = task_context.round_idx

        # step 1: load task results from disk
        task_runs = self._load_task_results(trajectories_dir, task_context.task_ids)

        # step 2: update SolvabilityJournal
        self._solvability.update(round_idx, task_runs)

        # step 3: Layer 1 — mechanical signal extraction
        signals: dict[str, TaskSignals] = self._extractor.extract_batch(
            task_runs, self._solvability, self._score_fn
        )
        (output_dir / "mechanical_signals.json").write_text(
            json.dumps(
                {k: dataclasses.asdict(v) for k, v in signals.items()},
                default=str, indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # write per-task JSON + summary + report to signals/ subdir
        signals_dir = output_dir / "signals"
        signals_dir.mkdir(exist_ok=True)
        for task_id, sig in signals.items():
            (signals_dir / f"{task_id}.json").write_text(
                json.dumps(dataclasses.asdict(sig), default=str, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        (signals_dir / "all_tasks_summary.json").write_text(
            json.dumps(_build_summary(signals, ""), default=str, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (signals_dir / "signals_report.md").write_text(
            _build_signals_report(signals, ""),
            encoding="utf-8",
        )

        # step 4: Layer 2 — DigestAgent
        digest = await run_digest_from_signals(
            signals,
            self._solvability,
            trajectories_dir,
            output_dir,
            digest_model=self._digest_model,
            config_yaml=self._digest_config_yaml,
            skill_dirs=self._digest_skill_dirs,
            round_idx=round_idx,
            notebook_path=self._notebook_path,
            max_steps=self._digest_max_steps,
        )

        # step 5: write gap_type back to SolvabilityJournal
        for pattern in digest.patterns.values():
            for task_id in pattern.tasks:
                self._solvability.update_gap_type(
                    task_id,
                    gap_type=pattern.gap_type,
                    level=pattern.improvability_level,
                    round_idx=round_idx,
                )

        # step 6: Layer 3 — EvolveAgent
        # Skip only when there is genuinely nothing to act on.
        accepted = False
        new_config_path: Path | None = None

        has_work = digest.needs_revert or digest.has_search_targets
        if not has_work:
            logger.info("Round %d: no actionable signals, skipping EvolveAgent.", round_idx)
        else:
            evolve_result = await run_evolve(
                digest,
                output_dir,
                output_dir / "digest_report.json",
                evolve_model=self._evolve_model,
                current_config_path=harness_config,
                last_change_manifest=self._last_change_manifest,
                config_yaml=self._evolve_config_yaml,
                skill_dirs=self._evolve_skill_dirs,
                notebook_path=self._notebook_path,
                max_steps=self._evolve_max_steps,
            )
            if evolve_result.ok:
                accepted = True
                new_config_path = evolve_result.new_config_path
                self._last_change_manifest = evolve_result.change_manifest
                logger.info("Round %d: accepted, new config: %s", round_idx, new_config_path)
            else:
                logger.warning("Round %d: agent did not produce a config.", round_idx)

        # step 7: persist SolvabilityJournal
        self._solvability.save(output_dir / "solvability_journal.json")
        if self._journal_path is not None:
            self._solvability.save(self._journal_path)

        return EvolRoundOutput(
            output_dir=output_dir,
            accepted=accepted,
            new_config_path=new_config_path,
            digest=digest,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load_task_results(
        self,
        trajectories_dir: Path,
        task_ids: list[str],
    ) -> dict[str, list[RolloutData]]:
        """
        Parse rollouts from HarnessJournal JSONL files.

        Expected layout (set HarnessJournal base_dir=trajectories_dir, session_id=task_id):
          {trajectories_dir}/{task_id}/{run_id}.jsonl
          {trajectories_dir}/{task_id}/{run_id}_trace.jsonl
        Multiple run_ids per task_id = multiple rollouts.
        """
        result: dict[str, list[RolloutData]] = {}
        for task_id in task_ids:
            task_dir = trajectories_dir / task_id
            rollouts: list[RolloutData] = []
            if task_dir.is_dir():
                for trace_file in sorted(task_dir.glob("*_trace.jsonl")):
                    rollout = parse_one_rollout(trace_file)
                    if rollout is not None:
                        rollouts.append(rollout)
            if not rollouts:
                logger.warning("No rollouts found for task %s in %s", task_id, task_dir)
            result[task_id] = rollouts
        return result
