# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid

from .types import StepResult, WorkflowDef, WorkflowResult, WorkflowStep

# ── Context interpolation ─────────────────────────────────────────────────────

_REF_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)(?:\.([a-zA-Z_][a-zA-Z0-9_]*))?")


def _interpolate(text: str, ctx: dict[str, StepResult]) -> str:
    """Replace $step_id or $step_id.field references with actual values.

    Supported fields: stdout, output, success, error.
    Plain $step_id expands to the step's output.
    """

    def replace(m: re.Match) -> str:
        step_id = m.group(1)
        field = m.group(2)
        result = ctx.get(step_id)
        if result is None:
            return m.group(0)
        if field == "stdout":
            return result.stdout
        if field == "output":
            return _output_str(result)
        if field == "success":
            return "true" if result.success else "false"
        if field == "error":
            return result.error or ""
        return _output_str(result)

    return _REF_RE.sub(replace, text)


def _output_str(r: StepResult) -> str:
    if r.output is None:
        return r.stdout
    if isinstance(r.output, (dict, list)):
        return json.dumps(r.output, ensure_ascii=False)
    return str(r.output)


def _eval_condition(expr: str) -> bool:
    """Evaluate a condition string (after $ref interpolation) as a boolean."""
    lower = expr.strip().lower()
    if lower in ("false", "0", "", "none", "null"):
        return False
    if lower in ("true", "1"):
        return True
    try:
        return bool(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return bool(expr.strip())


def _apply_params(text: str, params: dict[str, str]) -> str:
    """Substitute $param_name references from a params dict.

    Unlike _interpolate (which uses StepResult objects), this operates on plain
    string values — used to expand workflow parameters before execution starts.
    """

    def replace(m: re.Match) -> str:
        key = m.group(1)
        field = m.group(2)
        if field is not None:
            return m.group(0)  # $step_id.field — leave for runtime interpolation
        return params.get(key, m.group(0))

    return _REF_RE.sub(replace, text)


# ── Engine ────────────────────────────────────────────────────────────────────

_DEFAULT_SHELL_TIMEOUT = 60.0  # seconds


class WorkflowEngine:
    """Execute a sequence of WorkflowSteps.

    Handles shell commands, $ref interpolation, condition checks, and approval
    gates.  All reasoning and analysis is left to the agent (Claude) — this
    engine only runs deterministic operations.

    Paused workflows are held in ``_paused`` until resumed or cancelled via
    ``resume()``.
    """

    def __init__(self) -> None:
        # token → (name, remaining_steps, accumulated_results, context)
        self._paused: dict[str, tuple[str, list[WorkflowStep], list[StepResult], dict[str, StepResult]]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, name: str, steps: list[WorkflowStep]) -> WorkflowResult:
        """Start a fresh workflow execution."""
        return await self._execute(name, steps, [], {})

    async def exec_workflow(
        self,
        wf: "WorkflowDef",
        params: dict[str, str] | None = None,
    ) -> WorkflowResult:
        """Execute a stored WorkflowDef, substituting *params* into step shells.

        Default param values from the workflow definition are used for any param
        not provided in *params*.
        """
        merged: dict[str, str] = {}
        for p in wf.params:
            if p.default is not None:
                merged[p.name] = p.default
        if params:
            merged.update(params)

        # Apply param substitution to step shells up-front.
        # Uses the same $ref regex so $param_name works naturally.
        substituted_steps = []
        for step in wf.steps:
            if step.shell and merged:
                new_shell = _apply_params(step.shell, merged)
            else:
                new_shell = step.shell
            substituted_steps.append(
                WorkflowStep(
                    id=step.id,
                    description=step.description,
                    shell=new_shell,
                    condition=step.condition,
                    approval=step.approval,
                    timeout=step.timeout,
                )
            )

        return await self._execute(wf.name, substituted_steps, [], {})

    async def resume(self, token: str, approved: bool) -> WorkflowResult:
        """Resume (or cancel) a paused workflow identified by *token*."""
        entry = self._paused.pop(token, None)
        if entry is None:
            return WorkflowResult(
                name="unknown",
                success=False,
                status="done",
                steps=[
                    StepResult(
                        step_id="flow_resume",
                        success=False,
                        error=f"Invalid or expired resume token: {token!r}",
                    )
                ],
            )

        name, remaining_steps, done_results, ctx = entry

        if not approved:
            return WorkflowResult(
                name=name,
                success=False,
                status="cancelled",
                steps=done_results,
            )

        return await self._execute(name, remaining_steps, done_results, ctx)

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _execute(
        self,
        name: str,
        steps: list[WorkflowStep],
        prior_results: list[StepResult],
        ctx: dict[str, StepResult],
    ) -> WorkflowResult:
        results = list(prior_results)

        for i, step in enumerate(steps):
            # ── Approval gate ─────────────────────────────────────────────
            if step.approval:
                token = uuid.uuid4().hex[:16]
                gate_result = StepResult(
                    step_id=step.id,
                    success=True,
                    skipped=True,
                    output="(awaiting approval)",
                )
                # Store steps[i+1:] so resuming does not re-trigger the gate
                self._paused[token] = (
                    name,
                    steps[i + 1 :],
                    results + [gate_result],
                    dict(ctx),
                )
                return WorkflowResult(
                    name=name,
                    success=False,
                    status="pending_approval",
                    steps=results + [gate_result],
                    pending_step=step.id,
                    resume_token=token,
                )

            # ── Condition check ───────────────────────────────────────────
            if step.condition is not None:
                interpolated = _interpolate(step.condition, ctx)
                if not _eval_condition(interpolated):
                    r = StepResult(
                        step_id=step.id,
                        success=True,
                        skipped=True,
                        output="(condition false — skipped)",
                    )
                    results.append(r)
                    ctx[step.id] = r
                    continue

            # ── Execute step ──────────────────────────────────────────────
            if step.shell is not None:
                result = await self._run_shell(step, ctx)
            else:
                result = StepResult(
                    step_id=step.id,
                    success=False,
                    error="Step has no 'shell' command — nothing to execute.",
                )

            results.append(result)
            ctx[step.id] = result

            if not result.success:
                break

        overall = all(r.success for r in results)
        return WorkflowResult(name=name, success=overall, status="done", steps=results)

    # ── Shell runner ──────────────────────────────────────────────────────────

    async def _run_shell(self, step: WorkflowStep, ctx: dict[str, StepResult]) -> StepResult:
        cmd = _interpolate(step.shell, ctx)  # type: ignore[arg-type]
        timeout = step.timeout or _DEFAULT_SHELL_TIMEOUT
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            elapsed = (time.monotonic() - t0) * 1000
            stdout = stdout_b.decode(errors="replace").strip()
            stderr = stderr_b.decode(errors="replace").strip()
            success = proc.returncode == 0
            return StepResult(
                step_id=step.id,
                success=success,
                output=stdout,
                stdout=stdout,
                stderr=stderr,
                error=None if success else f"exit code {proc.returncode}",
                duration_ms=elapsed,
            )
        except asyncio.TimeoutError:
            return StepResult(
                step_id=step.id,
                success=False,
                error=f"timeout after {timeout:.0f}s",
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return StepResult(
                step_id=step.id,
                success=False,
                error=str(exc),
                duration_ms=(time.monotonic() - t0) * 1000,
            )
