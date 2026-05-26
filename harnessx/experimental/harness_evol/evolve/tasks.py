"""
Task builder for EvolveAgent.

Single task. The agent reads digest_report.json (Layer 2 output) directly,
decides what changes to make, and writes outputs to the workspace directory.

The agent is not pre-classified into Search / Explore / Revert modes — it reads
the DigestReport and makes that judgment itself based on the routing flags and
evidence present.

Static role instructions (Decision Rules, Constraints, Evidence Format) live in
prompts/AGENTS.md and are injected as the system prompt via DefaultSystemPromptBuilder.
This task description carries only the round-specific context (paths, stats, manifest).
"""
from __future__ import annotations

import json
from pathlib import Path

from harnessx.core.harness import BaseTask

from ..digest.schema import DigestReport

# harnessx/processors/ source root — agent reads this for implementation details
_HARNESSX_PROCESSORS_DIR = Path(__file__).parents[3] / "processors"


def build_evolve_task(
    digest: DigestReport,
    digest_report_path: Path,
    workspace_dir: Path,
    *,
    current_config_path: Path | None = None,
    last_change_manifest: dict | None = None,
    max_steps: int = 300,
) -> BaseTask:
    last_manifest_section = ""
    if last_change_manifest:
        manifest_json = json.dumps(last_change_manifest, indent=2, ensure_ascii=False)
        if len(manifest_json) > 2000:
            manifest_json = manifest_json[:2000] + "\n  … (truncated)"
        last_manifest_section = (
            "\n## Previous Round Change Manifest\n"
            "Changes applied last round — use for regression root-cause analysis:\n"
            "```json\n"
            + manifest_json
            + "\n```\n"
        )

    description = _EVOLVE_TEMPLATE.format(
        total_tasks=digest.total_tasks,
        pass_rate=f"{digest.pass_rate:.1%}",
        failed_tasks=digest.failed_tasks,
        digest_report_path=digest_report_path,
        current_config_path=current_config_path or "(not provided — write config from scratch)",
        harnessx_processors_dir=_HARNESSX_PROCESSORS_DIR,
        workspace_dir=workspace_dir,
        last_manifest_section=last_manifest_section,
    )
    return BaseTask(
        description=description,
        max_steps=max_steps,
    )


# ── template ───────────────────────────────────────────────────────────────────

_EVOLVE_TEMPLATE = """\
Round summary: {total_tasks} tasks, pass_rate={pass_rate}, {failed_tasks} failing.

## Inputs

**Digest report** (read this first — Layer 2 analysis of all trajectories):
`{digest_report_path}`

Key fields:
- `patterns` — failure patterns with `gap_type`, `improvability_level` (1–4), `intervention_hint`, `tasks`, `trace_evidence`
- `needs_revert` / `has_search_targets` — routing signals
- `severe_regressions` — tasks that regressed from stable; each entry has `suspected_change_ids` (which prior changes to roll back) and `trace_diff_hint`
- `priority_pattern` — DigestAgent's highest-confidence pattern name; address it first (see Decision Rule 5)
- `rationale` — DigestAgent's plain-text explanation of the routing decision; read when patterns alone are ambiguous

**Current harness config** (baseline to modify):
`{current_config_path}`

Read this file first. For every `_target_` entry in its `processors` list, resolve the
Python module path to a source file under `{harnessx_processors_dir}` and read it.
You must understand what each already-active processor does (detection signals, params,
thresholds) before deciding whether to tune, replace, or add one.

**harnessx processors source** (read to discover candidate processors not yet in config):
`{harnessx_processors_dir}`

Use Glob/Grep to explore. When you find a candidate, read its source and verify that its
detection logic precisely matches the failure pattern in the current environment — a processor
that exists is not automatically a fit (e.g. name-only detection vs. name+input hash).
{last_manifest_section}
## Output

Write all outputs to: `{workspace_dir}/`

**Required:** `target_config.yaml` — updated config with your changes applied.

**Optional** (only for genuine gaps with level 1/2 evidence):
- New processor `.py` file — must be a `MultiHookProcessor` subclass with ≥2 tunable params

## Workflow

1. Read `{digest_report_path}` and the current config.
2. Apply changes → write `{workspace_dir}/target_config.yaml`.
3. (If applicable) Write new processor `.py` files under `{workspace_dir}/processors/`.
4. Call `ValidateHarnessConfig(path="{workspace_dir}/target_config.yaml", baseline_config_path="{current_config_path}")` — fix any errors before proceeding. If PARAM DIFF DETECTED is reported, you must either revert the param or add a param_change entry to your manifest before submitting.
5. **Call `submit_change_manifest`** with the complete change manifest. This is the final mandatory step.
"""
