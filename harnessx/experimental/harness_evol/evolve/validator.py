"""
EvolEvolveValidator — extends meta_harness.EvolveValidator with additional checks.

New checks:
  7.  evidence_completeness: every change must have all four evidence fields
  8.  model_gap_filter:      reject changes targeting Level 4 patterns
  9.  regression_spot_check: when a rollback change is present, spot-check severe_regressions
  10. no_evol_leakage:       reject configs that contain harness_evol.* processors (evolve-pipeline internals)
  11. diff_not_empty:        output config must differ from baseline on at least one processor or tool
  12. pattern_coverage:      at least one change must address a L1/L2 harness-fixable pattern's failing tasks
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..digest.schema import DigestReport, SevereRegression

logger = logging.getLogger(__name__)


@dataclass
class ValidateReport:
    phase: str      # validity | policy | advisory
    check: str
    ok: bool
    reason: str


@dataclass
class ValidationResult:
    reports: list[ValidateReport]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.reports if r.phase in ("validity", "policy"))

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reports": [vars(r) for r in self.reports]}


class EvolEvolveValidator:
    """Standalone validator with three checks: evidence completeness, model-gap filter, regression spot-check."""

    def run(
        self,
        new_config_path: Path | None,
        change_manifest: dict,
        digest_report: DigestReport,
        baseline_config_path: Path | None = None,
        last_change_manifest: dict | None = None,
    ) -> ValidationResult:
        reports: list[ValidateReport] = []

        # base check: config file must exist
        if new_config_path is None or not new_config_path.exists():
            reports.append(ValidateReport(
                phase="validity", check="config_exists", ok=False,
                reason=f"new config not found: {new_config_path}",
            ))
            return ValidationResult(reports=reports)

        reports.append(ValidateReport(
            phase="validity", check="config_exists", ok=True,
            reason=f"config exists: {new_config_path}",
        ))

        # config YAML parse + _target_ format check
        reports.append(self._check_config_yaml_valid(new_config_path))
        if not reports[-1].ok:
            return ValidationResult(reports=reports)

        # check 7: four-field evidence completeness
        reports.append(self._check_evidence_completeness(change_manifest))

        # check 8: Level 4 pattern filter
        reports.append(self._check_model_gap_filter(change_manifest, digest_report))

        # check 9: rollback change IDs must exist in last_change_manifest
        has_revert = any(
            c.get("type") == "rollback"
            for c in change_manifest.get("changes", [])
        )
        if has_revert:
            reports.append(self._check_rollback_ids_exist(change_manifest, last_change_manifest, digest_report))

        # check 10: no evolve-pipeline processor leakage into the output config
        reports.append(self._check_no_evol_leakage(new_config_path))

        # check 11: output config must differ from baseline on at least one processor or tool
        if baseline_config_path is not None:
            reports.append(self._check_diff_not_empty(baseline_config_path, new_config_path))

        # check 13: existing processor params must not silently drift from baseline
        if baseline_config_path is not None:
            reports.append(self._check_param_drift(baseline_config_path, new_config_path, change_manifest))

        # check 14: every new_processor in manifest must appear in target_config._target_
        reports.append(self._check_new_processors_registered(change_manifest, new_config_path))

        # check 12: at least one change must address a L1/L2 harness-fixable pattern
        reports.append(self._check_pattern_coverage(change_manifest, digest_report))

        return ValidationResult(reports=reports)

    def _check_evidence_completeness(self, manifest: dict) -> ValidateReport:
        """Every change must have all four fields: failure_evidence / root_cause / targeted_fix / predicted_impact."""
        required = ["failure_evidence", "root_cause", "targeted_fix", "predicted_impact"]
        for chg in manifest.get("changes", []):
            for field in required:
                if not chg.get(field):
                    return ValidateReport(
                        phase="policy", check="evidence_completeness", ok=False,
                        reason=f"change {chg.get('id', '?')} missing '{field}'",
                    )
        return ValidateReport(
            phase="policy", check="evidence_completeness", ok=True,
            reason="all changes have 4-field evidence",
        )

    def _check_model_gap_filter(
        self,
        manifest: dict,
        digest_report: DigestReport,
    ) -> ValidateReport:
        """Reject changes targeting Level 4 patterns (model gap — harness cannot fix capability limits)."""
        level4_patterns = {
            name for name, p in digest_report.patterns.items()
            if p.improvability_level == 4
        }
        for chg in manifest.get("changes", []):
            if chg.get("failure_pattern") in level4_patterns:
                return ValidateReport(
                    phase="policy", check="model_gap_filter", ok=False,
                    reason=(
                        f"change {chg.get('id', '?')} targets '{chg.get('failure_pattern')}' "
                        "which is Level 4 (model gap). Harness cannot fix model capability limits."
                    ),
                )
        return ValidateReport(
            phase="policy", check="model_gap_filter", ok=True,
            reason="no changes target Level 4 patterns",
        )

    def _check_no_evol_leakage(self, config_path: Path) -> ValidateReport:
        """Reject configs that contain harness_evol.* processors — those are evolve-pipeline internals."""
        _FORBIDDEN_PREFIX = "harnessx.experimental.harness_evol."
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError as e:
            return ValidateReport(
                phase="validity", check="no_evol_leakage", ok=False,
                reason=f"could not read config: {e}",
            )
        leaked = [
            line.strip()
            for line in text.splitlines()
            if _FORBIDDEN_PREFIX in line and "_target_:" in line
        ]
        if leaked:
            return ValidateReport(
                phase="policy", check="no_evol_leakage", ok=False,
                reason=(
                    f"output config contains {len(leaked)} evolve-pipeline processor(s) "
                    f"that must not appear in a target-agent config: {leaked[:3]}"
                ),
            )
        return ValidateReport(
            phase="policy", check="no_evol_leakage", ok=True,
            reason="no harness_evol.* processors in output config",
        )

    def _check_diff_not_empty(self, baseline: Path, new: Path) -> ValidateReport:
        """Output config must differ from baseline on processors or tool_registry."""
        try:
            base_cfg = yaml.safe_load(baseline.read_text(encoding="utf-8")) or {}
            new_cfg = yaml.safe_load(new.read_text(encoding="utf-8")) or {}
        except Exception as e:
            return ValidateReport(phase="validity", check="diff_not_empty", ok=False,
                reason=f"could not parse configs for diff: {e}")

        base_procs = base_cfg.get("processors", [])
        new_procs = new_cfg.get("processors", [])
        base_tools = base_cfg.get("tool_registry", {})
        new_tools = new_cfg.get("tool_registry", {})

        if base_procs == new_procs and base_tools == new_tools:
            return ValidateReport(phase="policy", check="diff_not_empty", ok=False,
                reason=(
                    "output config is identical to baseline (processors and tool_registry unchanged). "
                    "You must apply at least one meaningful change to address the identified patterns."
                ))
        changed = []
        if base_procs != new_procs:
            changed.append("processors")
        if base_tools != new_tools:
            changed.append("tool_registry")
        return ValidateReport(phase="policy", check="diff_not_empty", ok=True,
            reason=f"config differs from baseline on: {', '.join(changed)}")

    def _check_param_drift(self, baseline: Path, new: Path, manifest: dict) -> ValidateReport:
        """Existing processor params must not change unless declared as param_change in manifest.

        Compares each processor in the output config against the baseline by _target_.
        A processor present in both configs must have identical params (all keys except
        _target_ and _code_hash) unless the change_manifest explicitly declares a
        param_change entry for that processor's _target_.
        """
        _IGNORED_KEYS = {"_target_", "_code_hash"}
        try:
            base_cfg = yaml.safe_load(baseline.read_text(encoding="utf-8")) or {}
            new_cfg = yaml.safe_load(new.read_text(encoding="utf-8")) or {}
        except Exception as e:
            return ValidateReport(phase="validity", check="param_drift", ok=False,
                reason=f"could not parse configs for param-drift check: {e}")

        # Build target → params index for both configs
        def _index(processors: list) -> dict[str, dict]:
            idx: dict[str, dict] = {}
            for p in (processors or []):
                if not isinstance(p, dict):
                    continue
                t = p.get("_target_")
                if t:
                    idx[t] = {k: v for k, v in p.items() if k not in _IGNORED_KEYS}
            return idx

        base_idx = _index(base_cfg.get("processors", []))
        new_idx = _index(new_cfg.get("processors", []))

        # Collect declared param_change targets from manifest
        declared_param_changes: set[str] = set()
        for chg in manifest.get("changes", []):
            if chg.get("type") == "param_change":
                target = chg.get("target", "")
                if target:
                    declared_param_changes.add(target)

        drifted: list[str] = []
        for target, base_params in base_idx.items():
            if target not in new_idx:
                continue  # removed — covered by diff_not_empty
            new_params = new_idx[target]
            if base_params != new_params and target not in declared_param_changes:
                # Surface the first differing key for actionable feedback
                changed_keys = [
                    k for k in set(base_params) | set(new_params)
                    if base_params.get(k) != new_params.get(k)
                ]
                drifted.append(
                    f"{target}: params changed {changed_keys} "
                    f"(baseline={[base_params.get(k) for k in changed_keys]}, "
                    f"output={[new_params.get(k) for k in changed_keys]})"
                )

        if drifted:
            return ValidateReport(
                phase="policy", check="param_drift", ok=False,
                reason=(
                    f"{len(drifted)} existing processor(s) have silently drifted params "
                    f"not declared as param_change in change_manifest. "
                    f"Either revert to baseline params or add a param_change entry. "
                    f"Drifted: {drifted[:3]}"
                ),
            )
        return ValidateReport(phase="policy", check="param_drift", ok=True,
            reason="all existing processor params match baseline or are declared as param_change")

    def _check_new_processors_registered(self, manifest: dict, new: Path) -> ValidateReport:
        """Every new_processor declared in the manifest must appear as a _target_ in the config.

        EvolveAgent sometimes creates the processor file and writes the manifest but forgets
        to add the processor entry to target_config.yaml.  This check catches that omission.
        """
        new_proc_changes = [
            c for c in manifest.get("changes", [])
            if c.get("type") == "new_processor"
        ]
        if not new_proc_changes:
            return ValidateReport(phase="policy", check="new_processors_registered", ok=True,
                reason="no new_processor changes in manifest")

        try:
            cfg = yaml.safe_load(new.read_text(encoding="utf-8")) or {}
        except Exception as e:
            return ValidateReport(phase="validity", check="new_processors_registered", ok=False,
                reason=f"could not parse target config: {e}")

        registered = {p.get("_target_", "") for p in (cfg.get("processors") or []) if isinstance(p, dict)}

        def _target_matches(import_path: str, target: str) -> bool:
            # Direct match (module-path format).
            if import_path == target:
                return True
            # file:// format: "file:///abs/path/to/module.py::ClassName"
            # import_path is "module.ClassName" — extract class name and stem for comparison.
            if target.startswith("file://"):
                # target stem: "slow_tool_budget_guard", class: "SlowToolBudgetGuard"
                file_part = target.split("::")[-1]          # "SlowToolBudgetGuard"
                stem = target.split("/")[-1].split("::")[0].replace(".py", "")  # "slow_tool_budget_guard"
                # import_path "slow_tool_budget_guard.SlowToolBudgetGuard"
                if import_path == f"{stem}.{file_part}":
                    return True
            return False

        missing = []
        for chg in new_proc_changes:
            import_path = chg.get("import_path", "")
            if import_path and not any(_target_matches(import_path, t) for t in registered):
                missing.append(import_path)

        if missing:
            return ValidateReport(
                phase="policy", check="new_processors_registered", ok=False,
                reason=(
                    f"{len(missing)} new processor(s) declared in manifest but absent from "
                    f"target_config.yaml processors list. Add each as a processor entry with "
                    f"_target_: <import_path>. Missing: {missing}"
                ),
            )
        return ValidateReport(phase="policy", check="new_processors_registered", ok=True,
            reason=f"all {len(new_proc_changes)} new processor(s) present in target_config.yaml")

    def _check_pattern_coverage(self, manifest: dict, digest_report: DigestReport) -> ValidateReport:
        """At least one change must address a task that appears in an L1/L2 harness-fixable pattern.

        Collects the union of tasks across all L1/L2 patterns, then checks whether any
        change's failure_evidence.tasks overlaps with that set.
        """
        fixable_tasks: set[str] = set()
        for pattern in digest_report.patterns.values():
            if pattern.improvability_level <= 2:
                fixable_tasks.update(pattern.tasks)

        if not fixable_tasks:
            # No L1/L2 patterns — check if suspicious (changes present but patterns empty)
            changes = [c for c in manifest.get("changes", []) if c.get("type") != "rollback"]
            if changes and not digest_report.patterns:
                return ValidateReport(
                    phase="advisory", check="pattern_coverage", ok=True,
                    reason=(
                        "WARNING: DigestReport.patterns is empty (possible DigestAgent parse "
                        "failure) but EvolveAgent made changes anyway. Changes were based on "
                        "rationale text only — verify DigestAgent output "
                        "was correctly parsed before treating this round as authoritative."
                    ),
                )
            return ValidateReport(phase="policy", check="pattern_coverage", ok=True,
                reason="no L1/L2 harness-fixable patterns in digest; check vacuously passes")

        changes = [c for c in manifest.get("changes", []) if c.get("type") != "rollback"]
        if not changes:
            return ValidateReport(phase="policy", check="pattern_coverage", ok=False,
                reason=(
                    f"digest has {len(fixable_tasks)} task(s) in L1/L2 patterns "
                    "but change_manifest has no non-rollback changes."
                ))

        for chg in changes:
            evidence_tasks = set((chg.get("failure_evidence") or {}).get("tasks") or [])
            overlap = evidence_tasks & fixable_tasks
            if overlap:
                return ValidateReport(phase="policy", check="pattern_coverage", ok=True,
                    reason=(
                        f"change targeting '{chg.get('target', '?')}' covers "
                        f"{len(overlap)} L1/L2 task(s): {sorted(overlap)[:3]}"
                    ))

        return ValidateReport(phase="policy", check="pattern_coverage", ok=False,
            reason=(
                f"no change's failure_evidence.tasks overlaps with the "
                f"{len(fixable_tasks)} task(s) in L1/L2 patterns. "
                "Your changes do not address the harness-fixable failures identified by DigestAgent. "
                f"L1/L2 tasks include: {sorted(fixable_tasks)[:5]}"
            ))

    def _check_config_yaml_valid(self, config_path: Path) -> ValidateReport:
        """Config YAML must be parseable and all _target_ values must be non-empty valid-looking paths.

        Accepts dotted Python import paths (e.g. harnessx.processors.Foo) and
        file:// paths (e.g. file:///abs/path/module.py::ClassName).
        Rejects empty strings, bare class names without a module, or obviously malformed paths.
        """
        import re
        _DOTTED_PATH = re.compile(r'^[A-Za-z_][A-Za-z0-9_.]*\.[A-Za-z_][A-Za-z0-9_]*$')
        _FILE_PATH = re.compile(r'^file:///.+\.py::[A-Za-z_][A-Za-z0-9_]*$')

        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            return ValidateReport(
                phase="validity", check="config_yaml_valid", ok=False,
                reason=f"YAML parse error: {e}",
            )

        if cfg is None:
            return ValidateReport(
                phase="validity", check="config_yaml_valid", ok=False,
                reason="config YAML is empty",
            )

        bad_targets: list[str] = []
        for entry in cfg.get("processors") or []:
            if not isinstance(entry, dict):
                continue
            t = entry.get("_target_", "")
            if not t:
                bad_targets.append("(empty _target_)")
            elif not (_DOTTED_PATH.match(t) or _FILE_PATH.match(t)):
                bad_targets.append(t)

        if bad_targets:
            return ValidateReport(
                phase="validity", check="config_yaml_valid", ok=False,
                reason=(
                    f"{len(bad_targets)} processor(s) have invalid _target_ values "
                    f"(expected dotted.module.ClassName or file:///path.py::ClassName): "
                    f"{bad_targets[:5]}"
                ),
            )

        return ValidateReport(
            phase="validity", check="config_yaml_valid", ok=True,
            reason=f"YAML valid; {len(cfg.get('processors') or [])} processor(s) have well-formed _target_ values",
        )

    def _check_rollback_ids_exist(
        self,
        change_manifest: dict,
        last_change_manifest: dict | None,
        digest_report: DigestReport,
    ) -> ValidateReport:
        """When a rollback change is present, verify its suspected_change_ids exist in last_change_manifest.

        Two sources of IDs to check:
        1. change_manifest.changes[type=rollback] — may carry a 'reverted_change_ids' list.
        2. digest_report.severe_regressions[].suspected_change_ids.

        If last_change_manifest is None (no prior round), cannot verify — advisory only.
        """
        if last_change_manifest is None:
            return ValidateReport(
                phase="advisory", check="rollback_ids_exist", ok=True,
                reason="no last_change_manifest available — cannot verify rollback IDs (first round or manifest lost)",
            )

        known_ids: set[str] = {
            c.get("id", "")
            for c in last_change_manifest.get("changes", [])
            if c.get("id")
        }

        # Collect all suspected IDs referenced in this round's rollback
        claimed_ids: list[str] = []
        for chg in change_manifest.get("changes", []):
            if chg.get("type") == "rollback":
                claimed_ids.extend(chg.get("reverted_change_ids") or [])
        for reg in digest_report.severe_regressions:
            claimed_ids.extend(reg.suspected_change_ids or [])

        if not claimed_ids:
            return ValidateReport(
                phase="advisory", check="rollback_ids_exist", ok=True,
                reason="rollback present but no change IDs specified — surgical rollback cannot be verified",
            )

        missing = [cid for cid in claimed_ids if cid not in known_ids]
        if missing:
            return ValidateReport(
                phase="policy", check="rollback_ids_exist", ok=False,
                reason=(
                    f"{len(missing)} rollback target ID(s) not found in last_change_manifest "
                    f"(known IDs: {sorted(known_ids)[:5]}). "
                    f"Missing: {missing[:5]}. "
                    "Verify the correct change IDs from the previous round's change_manifest.json."
                ),
            )

        return ValidateReport(
            phase="policy", check="rollback_ids_exist", ok=True,
            reason=f"all {len(set(claimed_ids))} rollback target ID(s) confirmed in last_change_manifest",
        )
