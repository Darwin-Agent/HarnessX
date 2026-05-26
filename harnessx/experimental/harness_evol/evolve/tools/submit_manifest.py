"""
submit_change_manifest — completion tool for EvolveAgent.

The agent calls this tool as its final mandatory step to submit the
change manifest and signal task completion.

The run ends naturally after this tool executes: SelfValidationProcessor
sets _completion_called=True, so the next text-only model response is
allowed to exit without interception.  parse_evolve_result extracts the
manifest by scanning trajectory steps.
"""
from __future__ import annotations

from harnessx.tools.base import tool


_CHANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["param_change", "new_processor", "rollback", "remove_processor"],
            "description": "Kind of change being made.",
        },
        "target": {
            "type": "string",
            "description": (
                "Full _target_ import path of the processor being changed/added "
                "(e.g. harnessx.processors.control.loop_detection.LoopDetectionProcessor). "
                "Required for param_change and new_processor types."
            ),
        },
        "import_path": {
            "type": "string",
            "description": (
                "Full module import path for the new processor Python class. "
                "Required for new_processor type — must match _target_ in config."
            ),
        },
        "field": {"type": "string", "description": "Param name being changed (param_change only)."},
        "old_value": {"description": "Previous value (param_change only)."},
        "new_value": {"description": "New value (param_change only)."},
        "failure_pattern": {
            "type": "string",
            "description": (
                "Pattern ID from DigestReport this change addresses. "
                "Used by model_gap_filter to reject Level-4 changes. Omit for rollback."
            ),
        },
        "failure_evidence": {
            "type": "object",
            "description": "Structured evidence from task traces supporting this change.",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task IDs whose traces support this change.",
                },
                "trace_excerpts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task":        {"type": "string"},
                            "step":        {"type": "string"},
                            "observation": {"type": "string", "description": "Exact tool call or output (<300 chars)."},
                            "consequence": {"type": "string", "description": "What went wrong as a result."},
                        },
                        "required": ["task", "step", "observation", "consequence"],
                    },
                },
            },
            "required": ["tasks"],
        },
        "root_cause": {
            "type": "string",
            "description": "Why the failure happens. Required top-level key — do NOT nest inside failure_evidence.",
        },
        "targeted_fix": {
            "type": "string",
            "description": "How this change addresses the root cause. Required top-level key.",
        },
        "predicted_impact": {
            "type": "object",
            "properties": {
                "should_fix": {"type": "array", "items": {"type": "string"}},
                "at_risk":    {"type": "array", "items": {"type": "string"}},
            },
            "required": ["should_fix", "at_risk"],
        },
        "rationale": {"type": "string", "description": "Evidence-backed reason for this change."},
    },
    "required": ["type", "failure_evidence", "root_cause", "targeted_fix", "predicted_impact"],
}


@tool(
    description=(
        "Submit the change manifest to complete the evolution task. "
        "This is the FINAL mandatory step — call this after writing "
        "evol-workspace/target_config.yaml and passing all validation checks. "
        "Provide a complete change_manifest describing every change made."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "change_manifest": {
                "type": "object",
                "properties": {
                    "summary":          {"type": "string", "description": "One-line description of what changed and why."},
                    "mode":             {"type": "string", "enum": ["search", "explore", "revert"]},
                    "changes":          {"type": "array", "items": _CHANGE_SCHEMA},
                    "expected_impact":  {"type": "string"},
                    "risk":             {"type": "string", "enum": ["low", "medium", "high"]},
                    "patterns_addressed": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "mode", "changes", "expected_impact", "risk"],
            },
        },
        "required": ["change_manifest"],
    },
)
async def submit_change_manifest(
    change_manifest: dict,
) -> str:
    """Submit the structured change manifest."""
    # SelfValidationProcessor sets _completion_called=True on this call,
    # allowing the run to end naturally on the next text-only model response.
    return "submitted"
