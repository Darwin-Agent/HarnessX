"""Extract submit_change_manifest parameters from HarnessResult and build EvolveResult."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from harnessx.core.harness import HarnessResult

logger = logging.getLogger(__name__)


@dataclass
class EvolveResult:
    new_config_path: Path | None    # path to the new harness config file
    change_manifest: dict           # change_manifest written by the meta-agent
    mode: str                       # search | explore | revert
    ok: bool                        # True if parsing succeeded (validation is done by orchestrator)
    validation_report: dict         # reserved for orchestrator-level validation results


def parse_evolve_result(
    result: HarnessResult,
    output_dir: Path,
) -> EvolveResult:
    """
    Extract outputs from an EvolveAgent HarnessResult.

    submit_change_manifest is a normal registered tool; its call lands in the
    trajectory steps.  The interrupted_at path is kept as a fallback in case
    a caller wraps the task with interrupt_on for other reasons.

    new_config_path is located by scanning output_dir/workspace/ for any
    .yaml file written by the agent (the agent is instructed to write the
    updated harness config there).
    """
    workspace_dir = output_dir / "evol-workspace"

    tool_input = _extract_stop_tool_input(result, "submit_change_manifest")
    if tool_input is None:
        logger.warning("EvolveAgent: submit_change_manifest not called, returning empty result")
        # Still scan workspace so validators can report the real state
        # (e.g. "config exists but submit not called" vs "config not found").
        return EvolveResult(
            new_config_path=_find_config_yaml(workspace_dir),
            change_manifest={},
            mode="search",
            ok=False,
            validation_report={"error": "submit_change_manifest not called"},
        )

    # Extract change_manifest from tool input.
    # The tool accepts either a pre-parsed dict or a JSON string.
    raw = tool_input.get("change_manifest") or tool_input.get("change_manifest_json", {})
    try:
        change_manifest = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("EvolveAgent: failed to parse change_manifest (%s)", e)
        change_manifest = {}
    if not isinstance(change_manifest, dict):
        logger.warning("EvolveAgent: change_manifest is %s, expected dict — resetting", type(change_manifest).__name__)
        change_manifest = {}

    mode = _infer_mode(change_manifest)
    new_config_path = _find_config_yaml(workspace_dir)

    return EvolveResult(
        new_config_path=new_config_path,
        change_manifest=change_manifest,
        mode=mode,
        ok=new_config_path is not None,
        validation_report={},
    )


# ── internal helpers ──────────────────────────────────────────────────────────

def _extract_stop_tool_input(result: HarnessResult, tool_name: str) -> dict | None:
    """
    Extract input parameters of an interrupt_on tool call.

    Checks result.interrupted_at first (primary path), then scans trajectory
    steps as a fallback for runs that ended normally.
    """
    if result.is_interrupted and result.interrupted_at is not None:
        if result.interrupted_at.name == tool_name:
            return result.interrupted_at.input

    for step in result.trajectory.steps:
        if step.action is None:
            continue
        for tc in step.action.tool_calls:
            if tc.name == tool_name:
                return tc.input
    return None


def _infer_mode(change_manifest: dict) -> str:
    """Infer evolution mode from the change manifest content."""
    changes = change_manifest.get("changes", [])
    if any(c.get("type") == "rollback" for c in changes):
        return "revert"
    if any(c.get("type") == "new_processor" for c in changes):
        return "explore"
    return "search"


def _find_config_yaml(workspace_dir: Path) -> Path | None:
    """
    Find the evolved target config YAML written by EvolveAgent in workspace_dir.

    Prefers 'target_config.yaml' (canonical output name). Falls back to
    'config.yaml' for backwards compatibility, then to the most recently
    modified .yaml file, explicitly excluding 'harness_config.yaml' which is
    always the agent's own pipeline snapshot written by HarnessX core.
    """
    if not workspace_dir.exists():
        return None

    for preferred in ("target_config.yaml", "config.yaml"):
        candidate = workspace_dir / preferred
        if candidate.is_file():
            return candidate

    yaml_files = sorted(
        (p for p in workspace_dir.glob("*.yaml") if p.name != "harness_config.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return yaml_files[0] if yaml_files else None
