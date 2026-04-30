# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowStep:
    """A single step in a workflow."""

    id: str
    description: str = ""
    shell: str | None = None  # shell command to execute
    condition: str | None = None  # skip step if falsy; may use $ref
    approval: bool = False  # pause and require human approval before running
    timeout: float | None = None  # override default timeout (seconds)


@dataclass
class StepResult:
    """Outcome of a single workflow step execution."""

    step_id: str
    success: bool
    output: Any = None  # parsed output (dict if schema, str otherwise)
    stdout: str = ""  # raw stdout for shell steps
    stderr: str = ""  # raw stderr for shell steps
    error: str | None = None
    duration_ms: float = 0.0
    skipped: bool = False


@dataclass
class WorkflowResult:
    """Aggregated result of a complete (or paused) workflow execution."""

    name: str
    success: bool
    status: str = "done"  # "done" | "pending_approval" | "cancelled"
    steps: list[StepResult] = field(default_factory=list)
    pending_step: str | None = None  # step id waiting for approval
    resume_token: str | None = None  # opaque token for flow_resume

    def to_text(self) -> str:
        lines: list[str] = []

        if self.status == "pending_approval":
            lines.append(f"workflow '{self.name}' — ⏸ PAUSED (approval required for step '{self.pending_step}')")
        elif self.status == "cancelled":
            lines.append(f"workflow '{self.name}' — ✗ CANCELLED")
        else:
            state = "✓ OK" if self.success else "✗ FAILED"
            lines.append(f"workflow '{self.name}' — {state}")

        for s in self.steps:
            if s.skipped:
                icon = "↷"
            elif s.success:
                icon = "✓"
            else:
                icon = "✗"
            lines.append(f"  [{icon}] {s.step_id}  ({s.duration_ms:.0f}ms)")
            if s.error:
                lines.append(f"       error: {s.error}")
            elif not s.skipped and s.output is not None:
                if isinstance(s.output, (dict, list)):
                    out_str = json.dumps(s.output, ensure_ascii=False)
                else:
                    out_str = str(s.output)
                if len(out_str) > 500:
                    out_str = out_str[:500] + "…"
                lines.append(f"       output: {out_str}")

        if self.status == "pending_approval":
            lines.append(
                f"\nApproval gate: review the steps above, then call:\n"
                f'  flow_resume(token="{self.resume_token}", approved=true)  — continue\n'
                f'  flow_resume(token="{self.resume_token}", approved=false) — cancel'
            )

        return "\n".join(lines)


# ── Stored workflow definition (YAML on disk) ─────────────────────────────────


@dataclass
class WorkflowParam:
    """A named parameter accepted by a workflow."""

    name: str
    description: str = ""
    default: str | None = None


@dataclass
class WorkflowDef:
    """A stored workflow procedure loaded from a YAML file.

    Corresponds to the YAML schema written by the extractor sub-harness::

        name: deploy-app
        description: Deploy a Python app to Kubernetes
        tags: [deploy, python, k8s]
        trigger_patterns: ["deploy * to production"]
        params:
          - name: app_name
          - name: namespace
            default: production
        steps:
          - id: tests
            shell: python -m pytest tests/ -q
          - id: deploy
            shell: kubectl apply -f k8s/ -n $namespace
        created: "2026-04-15"
        source_session: "sess_abc"
    """

    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    trigger_patterns: list[str] = field(default_factory=list)
    params: list[WorkflowParam] = field(default_factory=list)
    steps: list[WorkflowStep] = field(default_factory=list)
    created: str = ""
    source_session: str = ""

    # ── YAML I/O ──────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "WorkflowDef":
        """Load a WorkflowDef from a YAML file."""
        import yaml  # type: ignore[import]

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "WorkflowDef":
        params = [
            WorkflowParam(
                name=str(p.get("name", "")),
                description=str(p.get("description", "")),
                default=str(p["default"]) if p.get("default") is not None else None,
            )
            for p in (data.get("params") or [])
        ]
        steps = [
            WorkflowStep(
                id=str(s.get("id", "step")),
                description=str(s.get("description", "")),
                shell=s.get("shell"),
                condition=s.get("condition"),
                approval=bool(s.get("approval", False)),
                timeout=float(s["timeout"]) if s.get("timeout") is not None else None,
            )
            for s in (data.get("steps") or [])
        ]
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            tags=list(data.get("tags") or []),
            trigger_patterns=list(data.get("trigger_patterns") or []),
            params=params,
            steps=steps,
            created=str(data.get("created", "")),
            source_session=str(data.get("source_session", "")),
        )
