"""
Layer 3 EvolveAgent runner.

Entry point::

    from harnessx.experimental.harness_evol.evolve.runner import run_evolve

    result = await run_evolve(
        digest=digest_report,
        output_dir=Path("out/round_0/"),
        evolve_model=ModelConfig(main=LiteLLMProvider("claude-opus-4-6")),
        current_config_path=Path("harness_config.yaml"),
    )
    if result.ok:
        # result.new_config_path — accepted new config
        # result.change_manifest — evidence + change details

Caller prerequisites:
  - output_dir/digest_report.json must exist (written by run_digest_from_signals or run_digest).
  - EvolveAgent writes its outputs to output_dir/workspace/.

Outputs written to *output_dir*:
  ``change_manifest.json``   — EvolveAgent's proposed changes
  ``validation_report.json`` — audit log (policy checks; does not gate acceptance)
  ``evol-workspace/``        — new/modified harness config and processor files
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

from harnessx.core.model_config import ModelConfig

from ..digest.schema import DigestReport
from .harness import build_evolve_config
from .parse import EvolveResult, parse_evolve_result
from .tasks import build_evolve_task
from .validator import EvolEvolveValidator

logger = logging.getLogger(__name__)

_validator = EvolEvolveValidator()


async def run_evolve(
    digest: DigestReport,
    output_dir: Path,
    digest_report_path: Path,
    *,
    evolve_model: ModelConfig,
    current_config_path: Path | None = None,
    last_change_manifest: dict | None = None,
    config_yaml: Path | None = None,
    skill_dirs: list[Path] | None = None,
    indent: int = 2,
    notebook_path: Path | None = None,
    max_steps: int = 300,
) -> EvolveResult:
    """
    Run Layer 3 EvolveAgent given a DigestReport.

    The agent reads digest_report.json, decides what changes to make
    (revert / tune params / write new processor), and writes outputs to
    output_dir/evol-workspace/.  Validation is applied before returning.

    Parameters
    ----------
    digest:
        DigestReport from Layer 2. Routing flags (needs_revert, has_search_targets)
        must be set by the caller before calling.
    output_dir:
        Round output directory. Workspace and outputs are written here.
    digest_report_path:
        Path to digest_report.json written by Layer 2.  Passed to the agent
        so it can read the raw JSON directly.
    evolve_model:
        ModelConfig for the EvolveAgent.
    current_config_path:
        Path to the current harness config YAML.  None if no config exists yet.
    last_change_manifest:
        Change manifest from the previous round for regression root-cause analysis.
    config_yaml:
        Optional override for default_config.yaml (EvolveAgent harness config).
    indent:
        JSON pretty-print indentation.

    Returns
    -------
    EvolveResult
        ok=True if the agent wrote target_config.yaml and called submit_change_manifest.
        Policy checks in validation_report are audit-only and do not gate acceptance.
        new_config_path points to the produced config; change_manifest and
        validation_report carry the full evidence trail.
    """
    workspace_dir = output_dir / "evol-workspace"

    evolve_config = await build_evolve_config(
        output_dir, config_yaml, skill_dirs=skill_dirs, notebook_path=notebook_path,
        baseline_config_path=current_config_path,
        digest=digest,
    )
    evolve_harness = evolve_model.agentic(evolve_config)

    task = build_evolve_task(
        digest,
        digest_report_path=digest_report_path,
        workspace_dir=workspace_dir,
        current_config_path=current_config_path,
        last_change_manifest=last_change_manifest,
        max_steps=max_steps,
    )

    harness_result = await evolve_harness.run(task)
    evolve_result = parse_evolve_result(harness_result, output_dir)

    validation = _validator.run(
        evolve_result.new_config_path,
        evolve_result.change_manifest,
        digest,
        baseline_config_path=current_config_path,
        last_change_manifest=last_change_manifest,
    )

    (output_dir / "change_manifest.json").write_text(
        json.dumps(evolve_result.change_manifest, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation.to_dict(), indent=indent),
        encoding="utf-8",
    )

    accepted = evolve_result.ok  # agent completed normally and produced a config — always accept
    logger.info("run_evolve: %s. new_config=%s", "accepted" if accepted else "no config produced", evolve_result.new_config_path)

    return dataclasses.replace(
        evolve_result,
        ok=accepted,
        validation_report=validation.to_dict(),
    )
