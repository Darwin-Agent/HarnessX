"""
Build EvolveAgent HarnessConfig.

  build_evolve_config(workspace_dir, config_yaml, notebook_path, skill_dirs)
      Full build — creates workspace, initialises it, installs benchmark skills,
      and substitutes runtime parameters into WriteScopeGateProcessor and
      EvolutionNotebookProcessor.  Mirrors build_digest_config in digest/harness.py.

Workspace layout
----------------
  mode="shared": agent can access workspace_dir.parent (session output_dir).
  This lets it read digest_report.json and signal files from output_dir while
  WriteScopeGateProcessor restricts writes to:
    - workspace_dir/             (code changes)
    - evolution_notebook.md      (cross-round analysis notes)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from harnessx.core.harness import HarnessConfig
from harnessx.workspace.initializer import WorkspaceInitializer
from harnessx.workspace.skill_manager import SkillManager
from harnessx.workspace.workspace import Workspace

if TYPE_CHECKING:
    from harnessx.experimental.harness_evol.digest.schema import DigestReport

_DEFAULT_CONFIG = Path(__file__).parent / "default_config.yaml"
_PROMPTS_DIR = Path(__file__).parent / "prompts"

_WRITE_GATE = "harnessx.experimental.harness_evol.processors.write_scope_gate.WriteScopeGateProcessor"
_NOTEBOOK = "harnessx.experimental.harness_evol.processors.notebook.EvolutionNotebookProcessor"
_SELF_VALIDATION = "harnessx.experimental.harness_evol.processors.self_validation.SelfValidationProcessor"


def _make_pre_submission_validator(
    baseline_config_path: "Path | None",
    digest: "DigestReport | None",
):
    """
    Build a pre_submission_validator closure for SelfValidationProcessor.

    Runs the full EvolEvolveValidator policy suite (param_drift, diff_not_empty,
    evidence_completeness, model_gap_filter, new_processors_registered,
    pattern_coverage) at submit time so the agent gets immediate feedback on
    any policy violation and can fix it before the external validator runs.

    Returns None when both baseline_config_path and digest are unavailable
    (e.g. round 0 with no prior config), since most policy checks would be
    vacuous or incomplete.
    """
    if baseline_config_path is None and digest is None:
        return None

    async def _validate(target: "Path", manifest: dict) -> "str | None":
        from harnessx.experimental.harness_evol.evolve.validator import EvolEvolveValidator
        from harnessx.experimental.harness_evol.digest.schema import DigestReport as _DR

        # Use a minimal DigestReport when not provided so the validator
        # can still run the manifest-only checks.
        _digest = digest if digest is not None else _DR(
            round=0, pass_rate=0.0, total_tasks=0, failed_tasks=0, patterns={},
        )

        result = EvolEvolveValidator().run(
            new_config_path=target,
            change_manifest=manifest,
            digest_report=_digest,
            baseline_config_path=baseline_config_path,
        )
        failures = [
            r for r in result.reports
            if not r.ok and r.phase in ("validity", "policy")
        ]
        if not failures:
            return None
        lines = [f"  [{r.check}] {r.reason}" for r in failures]
        return "Policy checks failed:\n" + "\n".join(lines)

    return _validate


async def build_evolve_config(
    output_dir: Path,
    config_yaml: Path | None = None,
    notebook_path: Path | None = None,
    skill_dirs: list[Path] | None = None,
    baseline_config_path: Path | None = None,
    digest: "DigestReport | None" = None,
) -> HarnessConfig:
    """
    Build EvolveAgent HarnessConfig for one round.

    Args:
        output_dir:           Round output directory (e.g. session_dir/round_N/).
                              Workspace is created at output_dir/evol-workspace/.
        config_yaml:          Override default_config.yaml.
        notebook_path:        Shared evolution notebook (DigestAgent writes, EvolveAgent reads).
                              Pass a session-level path to persist notes across rounds.
                              Defaults to output_dir / "digest-workspace/evolution_notebook.md".
        skill_dirs:           Benchmark-specific skill directories to install into the workspace.
        baseline_config_path: Path to the baseline harness config YAML from the current round.
                              Used by pre_submission_validator for param-drift and diff checks.
        digest:               DigestReport from the current round.  Used by pre_submission_validator
                              for model_gap_filter and pattern_coverage checks.
    """
    workspace_dir = output_dir / "evol-workspace"
    notebook_path = notebook_path or (output_dir / "digest-workspace" / "evolution_notebook.md")

    # mode="home": agent home is set to notebook_path.parent so the workspace
    # boundary covers both output_dir (for reading digest_report.json / signals)
    # and the cross-round notebook (which may live at run_dir level, above output_dir).
    # WriteScopeGateProcessor (in YAML) restricts writes to workspace + notebook only.
    notebook_home = notebook_path.resolve().parent
    ws = Workspace(agent_id="evolve", root=workspace_dir, mode="home", home=str(notebook_home))

    await WorkspaceInitializer(prompts_root=_PROMPTS_DIR).initialize(ws, copy_skills=False)

    if skill_dirs:
        # agent_id="" → installs to ws_root/skills/ which matches workspace root + AGENTS.md discovery
        mgr = SkillManager(ws_root=workspace_dir, agent_id="")
        for sd in skill_dirs:
            mgr.install(str(sd.resolve()))

    base = HarnessConfig.from_yaml_file(config_yaml or _DEFAULT_CONFIG)

    # Substitute runtime values into the placeholder processor dicts from YAML.
    nb_str = str(notebook_path)
    ws_str = str(workspace_dir)
    target_config = str(workspace_dir / "target_config.yaml")
    updated = []
    for p in base.processors:
        target = p.get("_target_", "")
        if target == _WRITE_GATE:
            p = {**p, "allowed_roots": [ws_str], "allowed_files": [nb_str]}
        elif target == _NOTEBOOK:
            p = {**p, "notebook_path": nb_str}
        elif target == _SELF_VALIDATION:
            # Replace the dict with a live instance so we can pass the
            # config_validator callable (not YAML-serializable).
            # HarnessConfig.__post_init__ moves non-dict entries to _rt_procs.
            from harnessx.experimental.harness_evol.processors.self_validation import (
                SelfValidationProcessor,
            )
            from harnessx.experimental.harness_evol.evolve.config_validator import (
                validate_harness_config,
            )
            # Wrap validate_harness_config so Phase 1 auto-check also runs
            # param_diff against the baseline.  The raw function signature is
            # (path,); the closure closes over baseline_config_path so param
            # changes are reported early, before leakage/regression/review phases.
            _baseline = baseline_config_path
            async def _config_validator_with_baseline(path: Path) -> str | None:
                return await validate_harness_config(path, baseline_config_path=_baseline)

            p = SelfValidationProcessor(
                completion_tool=p.get("completion_tool", "submit_change_manifest"),
                required_files=[target_config],
                config_validator=_config_validator_with_baseline,
                pre_submission_validator=_make_pre_submission_validator(
                    baseline_config_path, digest
                ),
                max_interventions=p.get("max_interventions", 8),
            )
        updated.append(p)

    return base.copy(workspace=ws, processors=updated, init_workspace=False)
