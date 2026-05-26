"""
Stop tool for DigestAgent.

submit_digest_report is the single structured stop tool.  The agent explores
trajectories freely (using Read/Grep/Glob/Bash) and writes intermediate gap
classifications to the evolution notebook via Write.  When the full analysis is
complete it calls submit_digest_report once to terminate the run.

When a stop tool is called the run_loop terminates; the tool's input
parameters are the structured output, extracted by parse.py from the trajectory.
"""
from harnessx.tools.base import tool


@tool(
    name="submit_digest_report",
    description=(
        "Submit the complete digest report after analyzing all failure patterns. "
        "Call this exactly once when the full analysis is done."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "needs_revert": {
                "type": "boolean",
                "description": "True if severe regression detected; EvolveAgent should rollback first.",
            },
            "priority_pattern": {
                "type": "string",
                "description": "Key name of the highest-priority pattern in patterns; empty string if none.",
            },
            "rationale": {
                "type": "string",
                "description": "Routing decision rationale (<500 chars). Plain text only.",
            },
            "patterns": {
                "type": "object",
                "description": "Dict mapping pattern_name to its PatternImprovability fields.",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "gap_type": {
                            "type": "string",
                            "description": "stability|behavior|knowledge|reasoning|model_gap|unknown",
                        },
                        "improvability_level": {
                            "type": "integer",
                            "description": "1=mechanical fix, 2=processor tuning, 3=unknown, 4=model gap",
                        },
                        "tasks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Task IDs exhibiting this pattern.",
                        },
                        "count": {"type": "integer"},
                        "signal": {
                            "type": "string",
                            "description": "One-line explanation of the pattern.",
                        },
                        "intervention_hint": {
                            "type": "string",
                            "description": "Recommended processor intervention (optional).",
                        },
                        "trace_evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Step-level evidence, each <200 chars.",
                        },
                    },
                    "required": ["gap_type", "improvability_level", "tasks", "count", "signal"],
                },
            },
            "severe_regressions": {
                "type": "array",
                "description": (
                    "List of systematic regressions detected this round — tasks that were "
                    "stable in previous rounds but now all-fail due to an evolution-caused change. "
                    "Only populate when needs_revert=true. Leave empty [] otherwise."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID that regressed."},
                        "consecutive_pass_rounds_before": {
                            "type": "integer",
                            "description": "How many consecutive rounds this task passed before regression.",
                        },
                        "last_passed_round": {
                            "type": "integer",
                            "description": "Round number of the last successful pass.",
                        },
                        "suspected_change_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Change IDs from the previous round's manifest suspected as root cause.",
                        },
                        "regression_was_predicted": {
                            "type": "boolean",
                            "description": "True if this task appeared in the previous round's predicted_impact.at_risk.",
                        },
                        "trace_diff_hint": {
                            "type": "string",
                            "description": "Key difference between this round's failure and last round's success (<200 chars).",
                        },
                    },
                    "required": [
                        "task_id", "consecutive_pass_rounds_before", "last_passed_round",
                        "suspected_change_ids", "regression_was_predicted", "trace_diff_hint",
                    ],
                },
            },
        },
        "required": ["needs_revert", "priority_pattern", "rationale", "patterns"],
    },
)
def submit_digest_report(
    needs_revert: bool,
    priority_pattern: str,
    rationale: str,
    patterns: dict,
) -> dict:
    """Stop tool: submit the full digest report."""
    return {"status": "recorded"}
