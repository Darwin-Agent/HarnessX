# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from harnessx.tools.base import Tool

if TYPE_CHECKING:
    from .engine import WorkflowEngine
    from .types import WorkflowStep

# ── JSON Schemas ──────────────────────────────────────────────────────────────

_STEP_SCHEMA: dict = {
    "type": "object",
    "required": ["id"],
    "additionalProperties": False,
    "properties": {
        "id": {
            "type": "string",
            "description": "Unique step identifier. Later steps reference this output as $id.",
        },
        "description": {
            "type": "string",
            "description": "Human-readable description of what this step does.",
        },
        "shell": {
            "type": "string",
            "description": (
                "Shell command to execute. "
                "Reference prior step outputs with $step_id (full output), "
                "$step_id.stdout, or $step_id.success."
            ),
        },
        "condition": {
            "type": "string",
            "description": (
                "Skip this step when the expression evaluates to false. "
                "Example: '$tests.success' or '$score.stdout > 0.7'."
            ),
        },
        "approval": {
            "type": "boolean",
            "description": (
                "If true, the workflow pauses before this step and waits for human approval. "
                "The tool returns a resume_token; call flow_resume() to continue or cancel."
            ),
        },
        "timeout": {
            "type": "number",
            "description": "Step timeout in seconds (default: 60).",
        },
    },
}

_FLOW_SCHEMA: dict = {
    "type": "object",
    "required": ["name", "steps"],
    "additionalProperties": False,
    "properties": {
        "name": {
            "type": "string",
            "description": "Workflow name, used in output headers.",
        },
        "steps": {
            "type": "array",
            "description": "Ordered list of shell steps to execute.",
            "items": _STEP_SCHEMA,
        },
    },
}

_FLOW_RESUME_SCHEMA: dict = {
    "type": "object",
    "required": ["token"],
    "additionalProperties": False,
    "properties": {
        "token": {
            "type": "string",
            "description": "The resume_token returned by a previous flow() call that paused for approval.",
        },
        "approved": {
            "type": "boolean",
            "description": "true to continue the workflow; false to cancel it. Defaults to true.",
        },
    },
}

_FLOW_DESCRIPTION = """\
Define and execute a structured multi-step shell pipeline.

Use this tool to declare the full data-collection plan upfront as a named
workflow, rather than calling Bash repeatedly and tracking outputs ad-hoc.
Each step's output is labelled with its id and can be referenced in later
steps — giving you a single, structured result to reason over.

Each step runs a shell command (field: "shell"). Additional step controls:
  • condition  — "$prev.success" to skip on failure
  • approval   — pause and require human review before this step
  • $step_id   — reference a prior step's output in the shell command

Reference prior step outputs:
  $step_id           — full stdout output
  $step_id.stdout    — same as above
  $step_id.success   — "true" or "false"

On approval gates: the tool returns a resume_token. Show the completed steps
to the user, then call flow_resume() to continue or cancel.

After flow() returns, use the labelled step outputs to perform your own
analysis and reasoning — do not call flow() again just for LLM analysis.
"""

_FLOW_RESUME_DESCRIPTION = """\
Resume or cancel a workflow that paused at an approval gate.

Call this after a flow() call returned status=pending_approval.
Pass approved=true to continue executing the remaining steps, or
approved=false to cancel the workflow.
"""

_FLOW_EXEC_SCHEMA: dict = {
    "type": "object",
    "required": ["name"],
    "additionalProperties": False,
    "properties": {
        "name": {
            "type": "string",
            "description": "Name of the stored workflow to execute (matches the YAML filename without .yaml).",
        },
        "params": {
            "type": "object",
            "description": (
                "Key/value pairs to substitute for $param_name references in step shell commands. "
                "Omitted params use their declared defaults."
            ),
            "additionalProperties": {"type": "string"},
        },
    },
}

_FLOW_EXEC_DESCRIPTION = """\
Execute a previously stored workflow procedure by name.

Loads the named workflow YAML from the workflow directory, substitutes any
provided params into shell commands, and executes all steps in order.

Use this instead of flow() when a reusable procedure has already been captured
for this type of task — it saves the tokens that would otherwise be spent on
multi-turn reasoning.

Params override the workflow's declared defaults; omit a param to use its
default value.
"""


# ── Tool factories ─────────────────────────────────────────────────────────────


def build_workflow_tools(engine: "WorkflowEngine", workflow_dir: str | None = None) -> list[Tool]:
    """Return [flow_tool, flow_resume_tool, flow_exec_tool] bound to *engine*."""

    async def flow(name: str, steps: list) -> str:
        parsed = _parse_steps(steps)
        result = await engine.run(name, parsed)
        return result.to_text()

    async def flow_resume(token: str, approved: bool = True) -> str:
        result = await engine.resume(token=token, approved=approved)
        return result.to_text()

    async def flow_exec(name: str, params: dict | None = None) -> str:
        if not workflow_dir:
            return "flow_exec: no workflow_dir configured in WorkflowPlugin."
        wf_path = Path(workflow_dir) / f"{name}.yaml"
        if not wf_path.exists():
            # Try scanning for case-insensitive match
            candidates = list(Path(workflow_dir).glob("*.yaml"))
            matches = [p for p in candidates if p.stem.lower() == name.lower()]
            if matches:
                wf_path = matches[0]
            else:
                available = [p.stem for p in candidates]
                return f"flow_exec: workflow '{name}' not found in {workflow_dir}.\nAvailable: {available or '(none)'}"
        try:
            from .types import WorkflowDef

            wf = WorkflowDef.from_yaml(str(wf_path))
        except Exception as e:
            return f"flow_exec: failed to load workflow '{name}': {e}"
        result = await engine.exec_workflow(wf, params)
        return result.to_text()

    return [
        Tool(
            name="flow",
            description=_FLOW_DESCRIPTION,
            input_schema=_FLOW_SCHEMA,
            fn=flow,
            tags=["workflow"],
        ),
        Tool(
            name="flow_resume",
            description=_FLOW_RESUME_DESCRIPTION,
            input_schema=_FLOW_RESUME_SCHEMA,
            fn=flow_resume,
            tags=["workflow"],
        ),
        Tool(
            name="flow_exec",
            description=_FLOW_EXEC_DESCRIPTION,
            input_schema=_FLOW_EXEC_SCHEMA,
            fn=flow_exec,
            tags=["workflow"],
        ),
    ]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _parse_steps(raw: list) -> list["WorkflowStep"]:
    from .types import WorkflowStep

    steps = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        steps.append(
            WorkflowStep(
                id=str(item.get("id", "step")),
                description=str(item.get("description", "")),
                shell=item.get("shell"),
                condition=item.get("condition"),
                approval=bool(item.get("approval", False)),
                timeout=float(item["timeout"]) if item.get("timeout") is not None else None,
            )
        )
    return steps
