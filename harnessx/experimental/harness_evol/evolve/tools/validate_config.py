"""
ValidateHarnessConfig — agent-callable tool for proactive config validation.

The EvolveAgent can call this at any point during its work to verify that
evol-workspace/target_config.yaml is correct before attempting to submit.
SelfValidationProcessor is only the final exit-gate fallback; this tool lets
the agent validate incrementally without waiting for an exit-intent intervention.

Usage (agent call):
    ValidateHarnessConfig(path="evol-workspace/target_config.yaml")

Returns "PASSED" on success or a human-readable error string describing the
issue so the agent can fix it immediately.
"""
from __future__ import annotations

from harnessx.tools.base import tool


@tool(
    description=(
        "Validate an evolved harness config YAML file. "
        "Checks YAML syntax, processor imports, instantiation, mock on_task_start, "
        "and (when baseline_config_path is provided) detects any processor param changes "
        "that are not yet declared in the change manifest. "
        "Returns 'PASSED' if the config is valid, or an error description to fix. "
        "Always pass baseline_config_path so param drift is caught before submission."
    )
)
async def ValidateHarnessConfig(
    path: str = "evol-workspace/target_config.yaml",
    baseline_config_path: str = "",
) -> str:
    from pathlib import Path
    from harnessx.experimental.harness_evol.evolve.config_validator import validate_harness_config

    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"

    baseline = Path(baseline_config_path) if baseline_config_path else None
    error = await validate_harness_config(p, baseline_config_path=baseline)
    if error is None:
        return "PASSED — config is valid."
    return f"FAILED:\n{error}"
