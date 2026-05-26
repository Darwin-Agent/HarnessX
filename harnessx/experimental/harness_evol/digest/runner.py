"""
Layer 2 runners.

Two entry points:

  run_digest_from_signals(signals, solvability_journal, trajectories_dir, output_dir, ...)
      Pure L2 runner — takes pre-computed Layer 1 signals and runs DigestAgent.
      Used by EvolOrchestrator which manages its own L1 extraction pipeline.

  run_digest(source, output_dir, ...)
      L1 + L2 convenience wrapper — auto-discovers rollouts, extracts signals,
      then runs DigestAgent.  Used by test scripts and standalone callers.

Usage::

    # Standalone (L1 + L2)
    report = await run_digest(
        source="/data/.benchmarks/*/runs/*/",
        output_dir=Path("out/"),
        score_fn=my_score_fn,
        digest_model=ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6")),
    )

    # Pre-computed signals (L2 only)
    report = await run_digest_from_signals(
        signals=signals,
        solvability_journal=journal,
        trajectories_dir=Path("trajectories/"),
        output_dir=Path("out/"),
        digest_model=ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6")),
    )

Outputs written to *output_dir*:
  ``digest_report.json``    — final DigestReport as JSON
  ``digest-workspace/evolution_notebook.md`` — DigestAgent's working notes (created/updated in-place)
"""
from __future__ import annotations

import logging
from pathlib import Path

from harnessx.core.model_config import ModelConfig

from ..signals.runner import ScoreFn, extract_signals
from ..signals.schema import TaskSignals
from ..signals.solvability import SolvabilityJournal
from .harness import build_digest_config
from .parse import parse_digest_result, fallback_digest
from .schema import DigestReport
from .tasks import build_digest_task

logger = logging.getLogger(__name__)


async def run_digest_from_signals(
    signals: dict[str, TaskSignals],
    solvability_journal: SolvabilityJournal,
    trajectories_dir: Path,
    output_dir: Path,
    *,
    digest_model: ModelConfig,
    config_yaml: Path | None = None,
    skill_dirs: list[Path] | None = None,
    round_idx: int = 0,
    max_steps: int = 200,
    indent: int = 2,
    notebook_path: Path | None = None,
) -> DigestReport:
    """
    Run Layer 2 DigestAgent given pre-computed Layer 1 signals.

    Parameters
    ----------
    signals:
        Pre-computed Layer 1 signals, keyed by task_id.
        Empty dict returns a fallback report without running the agent.
    solvability_journal:
        Cross-round solvability state (already updated for this round).
    trajectories_dir:
        Directory the agent can use to read rollout files.
        May differ from output_dir when called from EvolOrchestrator.
    output_dir:
        Directory for agent workspace and digest_report.json output.
    digest_model:
        ModelConfig used to run the DigestAgent.
    config_yaml:
        Optional override for default_config.yaml.
    skill_dirs:
        Benchmark-specific skill directories to install into the workspace.
    round_idx:
        Round index for the solvability journal and digest task.
    max_steps:
        Step budget for the DigestAgent.
    indent:
        JSON pretty-print indentation.
    """
    if not signals:
        logger.warning("run_digest_from_signals: no signals, returning fallback report")
        report = fallback_digest(signals, round_idx)
        _write_report(report, output_dir, indent)
        return report

    try:
        cfg = await build_digest_config(
            output_dir,
            config_yaml=config_yaml,
            skill_dirs=skill_dirs,
            notebook_path=notebook_path,
        )
        harness = digest_model.agentic(cfg)
        task = build_digest_task(
            signals,
            solvability_journal,
            round_idx,
            trajectories_dir,
            max_steps=max_steps,
        )
        result = await harness.run(task)
        report = parse_digest_result(result, signals, round_idx=round_idx)
    except Exception as exc:
        logger.error("run_digest_from_signals: DigestAgent failed (%s), using fallback", exc)
        report = fallback_digest(signals, round_idx)

    report = report.model_copy(update={"round": round_idx})
    _write_report(report, output_dir, indent)
    return report


async def run_digest(
    source: str | list[dict],
    output_dir: Path | str,
    *,
    score_fn: ScoreFn | None = None,
    digest_model: ModelConfig,
    harness_config: Path | None = None,
    solvability_journal: SolvabilityJournal | None = None,
    round_idx: int = 0,
    max_steps: int = 200,
    skill_dirs: list[Path] | None = None,
    indent: int = 2,
) -> DigestReport:
    """
    Run Layer 1 extraction followed by Layer 2 DigestAgent in one call.

    Parameters
    ----------
    source:
        Same as ``extract_signals``: a glob pattern string or a pre-aggregated
        list of dicts ``[{"task_name": ..., "rollout_dirs": [...]}, ...]``.
    output_dir:
        Directory for all outputs (Layer 1 JSONs + digest_report.json +
        digest-workspace/evolution_notebook.md).  Created if it does not exist.
    score_fn:
        Optional benchmark-specific scoring override (same as ``extract_signals``).
    digest_model:
        ``ModelConfig`` used to run the DigestAgent.
    harness_config:
        Path to a custom harness YAML file.  When None, uses the built-in
        ``default_config.yaml`` (``digest/default_config.yaml``).
    solvability_journal:
        Pre-existing cross-round journal.  A fresh one is created if None.
    round_idx:
        Round index for the solvability journal and digest task.
    max_steps:
        Step budget for the DigestAgent (default 200).
    skill_dirs:
        Benchmark-specific skill directories to install into the agent workspace.
    indent:
        JSON pretty-print indentation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if solvability_journal is None:
        solvability_journal = SolvabilityJournal()

    signals = extract_signals(
        source,
        output_dir,
        score_fn=score_fn,
        solvability_journal=solvability_journal,
        round_idx=round_idx,
        indent=indent,
    )

    # trajectories co-located with output in the standalone runner
    return await run_digest_from_signals(
        signals,
        solvability_journal,
        output_dir,
        output_dir,
        digest_model=digest_model,
        config_yaml=harness_config,
        skill_dirs=skill_dirs,
        round_idx=round_idx,
        max_steps=max_steps,
        indent=indent,
    )


def _write_report(report: DigestReport, output_dir: Path, indent: int) -> None:
    out = output_dir / "digest_report.json"
    out.write_text(
        report.model_dump_json(indent=indent),
        encoding="utf-8",
    )
    logger.info("Wrote digest_report.json to %s", out)
