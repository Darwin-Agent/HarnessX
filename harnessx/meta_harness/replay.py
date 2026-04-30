"""Replay gate — synthetic smoke test only.

Runs after the meta-agent's ``end_turn`` to verify the evolved config
boots through the real run loop. One tiny synthetic task is executed;
any crash, upstream error, or ``exit_reason=error`` fails the gate.

Earlier versions also supported ``config_only`` and ``task_replay``
modes. ``config_only`` was redundant with ``canonicalize`` (both
check "cfg can bind"). ``task_replay`` re-ran sampled benchmark
tasks after the meta-agent finished — but the next benchmark round
runs the full task set anyway, so ``task_replay`` only gave a
slightly earlier preview at meaningful extra cost. Both modes have
been removed; ``run_replay_gate_strict`` now only executes the
synthetic smoke gate.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from ..core.harness import BaseTask, HarnessConfig
    from ..core.model_config import ModelConfig

logger = logging.getLogger(__name__)


# Retained for backward compatibility with recipes that still pass a
# ``task_loader`` to ``MetaAgent.evolve``. The argument is ignored now
# (synthetic-task replay needs no benchmark task loader), but keeping
# the alias avoids import errors in callers.
TaskLoader = Callable[[str], Awaitable["BaseTask"]]


@dataclass
class ReplayOutcome:
    """One task's replay result. ``ok=False`` means the gate should reject."""

    task_id: str
    ok: bool
    kind: str  # "ok" | "exception:<ExcType>" | "runtime_error" | "timeout" | "loader_failed"
    detail: str = ""
    exit_reason: str | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int | None = None
    elapsed_s: float | None = None


@dataclass
class ReplayReport:
    """Aggregate of multiple ReplayOutcomes. ``ok`` = all outcomes OK."""

    ok: bool
    outcomes: list[ReplayOutcome]
    skipped_reason: str | None = None  # non-None when the gate was skipped


async def run_synthetic_task_smoke_gate(
    harness_config: "HarnessConfig",
    model_config: "ModelConfig",
    *,
    max_steps: int = 2,
    max_cost_usd: float | None = 0.1,
    timeout_s: float = 20.0,
) -> ReplayReport:
    """Synthetic-task replay gate.

    Runs a tiny fixed BaseTask to validate end-to-end run-loop execution
    (processor hooks, tool wiring, provider call path) with minimal cost.
    """
    from ..core.harness import BaseTask

    t0 = time.time()
    harness = model_config.agentic(harness_config)
    task = BaseTask(
        description="Synthetic replay smoke check. Reply with exactly: OK",
        success_criteria="Returns a non-error completion",
        max_steps=max_steps,
        max_cost_usd=max_cost_usd,
    )
    try:
        result = await asyncio.wait_for(harness.run(task), timeout=timeout_s)
    except asyncio.TimeoutError:
        return ReplayReport(
            ok=False,
            outcomes=[
                ReplayOutcome(
                    task_id="__synthetic_smoke__",
                    ok=False,
                    kind="timeout",
                    detail=f"synthetic smoke exceeded {timeout_s:.0f}s",
                    elapsed_s=time.time() - t0,
                )
            ],
        )
    except Exception as exc:  # noqa: BLE001
        return ReplayReport(
            ok=False,
            outcomes=[
                ReplayOutcome(
                    task_id="__synthetic_smoke__",
                    ok=False,
                    kind=f"exception:{type(exc).__name__}",
                    detail=str(exc)[:500],
                    elapsed_s=time.time() - t0,
                )
            ],
        )

    exit_reason = getattr(result, "exit_reason", None)
    total_tokens = getattr(result, "total_tokens", None)
    total_cost = getattr(result, "total_cost_usd", None)
    total_steps = getattr(result, "total_steps", None)
    if exit_reason == "error":
        return ReplayReport(
            ok=False,
            outcomes=[
                ReplayOutcome(
                    task_id="__synthetic_smoke__",
                    ok=False,
                    kind="runtime_error",
                    detail=(getattr(result, "final_output", "") or "")[:500],
                    exit_reason=exit_reason,
                    total_tokens=total_tokens,
                    total_cost_usd=total_cost,
                    total_steps=total_steps,
                    elapsed_s=time.time() - t0,
                )
            ],
        )
    return ReplayReport(
        ok=True,
        outcomes=[
            ReplayOutcome(
                task_id="__synthetic_smoke__",
                ok=True,
                kind="ok_synthetic_smoke",
                exit_reason=exit_reason,
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                total_steps=total_steps,
                elapsed_s=time.time() - t0,
            )
        ],
    )


def render_report_md(report: ReplayReport, *, config_path: Path | None = None) -> str:
    """Render a ReplayReport as markdown for REPLAY.md / REPLAY_FAIL.md.

    Used for both the pass (REPLAY.md audit trail) and fail
    (REPLAY_FAIL.md evidence) paths — the title reflects the actual
    outcome so readers aren't misled by a pass log titled "failed".
    """
    title = "# Replay gate passed" if report.ok else "# Replay gate failed"
    lines: list[str] = [title, ""]
    if config_path:
        lines.append(f"Config under test: `{config_path}`")
        lines.append("")
    if report.skipped_reason:
        lines.append(f"**Skipped**: {report.skipped_reason}")
        lines.append("")
        return "\n".join(lines)
    for o in report.outcomes:
        marker = "✓" if o.ok else "✗"
        lines.append(f"## {marker} `{o.task_id}`")
        lines.append("")
        lines.append(f"- kind: `{o.kind}`")
        if o.exit_reason:
            lines.append(f"- exit_reason: `{o.exit_reason}`")
        if o.total_steps is not None:
            lines.append(f"- steps: {o.total_steps}")
        if o.total_tokens is not None:
            lines.append(f"- tokens: {o.total_tokens}")
        if o.total_cost_usd is not None:
            lines.append(f"- cost: ${o.total_cost_usd:.3f}")
        if o.elapsed_s is not None:
            lines.append(f"- elapsed: {o.elapsed_s:.1f}s")
        if o.detail:
            lines.append("")
            lines.append("```")
            lines.append(o.detail)
            lines.append("```")
        lines.append("")
    lines.append(
        "Replay uses the actual run loop as oracle — any crash, 400, "
        "assertion, or `exit_reason=error` fails the gate. Fix the "
        "failing component (tool return shape / processor hook / "
        "template reference) and re-verify."
    )
    return "\n".join(lines)


async def run_replay_gate_strict(
    cfg: "HarnessConfig",
    scratch_dir: Path,
    *,
    replay_model: "ModelConfig",
    max_steps: int | None = None,
    max_cost_usd: float | None = None,
    timeout_s: float = 20.0,
    # Kept for backward-compat with older callers (agent.py / recipes
    # that used to pass these for the removed ``task_replay`` mode).
    # All ignored — gate is now fixed at synthetic-task mode.
    trajectories_dir: Path | None = None,
    task_loader: "TaskLoader | None" = None,
    max_tasks: int = 0,
    predicted_task_ids: list[str] | None = None,
    regression_probe_tasks: int = 0,
    replay_mode: str = "synthetic_task",
) -> dict:
    """Strict-mode synthetic-task replay gate used by the evolve orchestrator.

    - Writes the audit (``scratch_dir/REPLAY.md``) on both pass and fail.
    - On failure, ALSO writes ``scratch_dir/REPLAY_FAIL.md`` with the
      same report and raises ``validate_workflow.StrictValidationError`` so
      the orchestrator can reject the round.
    - On pass, removes any stale ``REPLAY_FAIL.md`` from a prior round.

    Returns the pass dict (with ``ok=True``) — on fail it raises instead.
    The ``replay_mode`` argument is accepted for compatibility but only
    ``"synthetic_task"`` remains; other values log a warning and fall
    through to synthetic mode.
    """
    from .validate_workflow import StrictValidationError

    stale_fail = scratch_dir / "REPLAY_FAIL.md"

    if replay_mode != "synthetic_task":
        logger.warning(
            "[replay_gate] replay_mode=%r is no longer supported; running synthetic-task gate instead",
            replay_mode,
        )

    report = await run_synthetic_task_smoke_gate(
        cfg,
        replay_model,
        max_steps=max_steps if max_steps is not None else 2,
        max_cost_usd=max_cost_usd,
        timeout_s=min(timeout_s, 20.0),
    )

    audit = scratch_dir / "REPLAY.md"
    audit.write_text(
        render_report_md(report, config_path=scratch_dir.parent / "config.yaml"),
        encoding="utf-8",
    )

    if report.ok:
        logger.info(
            "[replay_gate] passed (%d task(s), %s) — %s",
            len(report.outcomes),
            report.skipped_reason or "all runs completed cleanly",
            audit,
        )
        if stale_fail.is_file():
            stale_fail.unlink()
        return {
            "ok": True,
            "runs": len(report.outcomes),
            "skipped_reason": report.skipped_reason,
            "artifact": str(audit),
        }

    findings = scratch_dir / "REPLAY_FAIL.md"
    findings.write_text(
        render_report_md(report, config_path=scratch_dir.parent / "config.yaml"),
        encoding="utf-8",
    )
    failing = [o for o in report.outcomes if not o.ok]
    first = failing[0] if failing else None
    summary = f"{len(failing)}/{len(report.outcomes)} replay(s) failed" + (
        f": first={first.task_id} ({first.kind})" if first else ""
    )
    raise StrictValidationError(
        kind="replay_gate",
        message=summary,
        findings_path=findings,
    )


__all__ = [
    "TaskLoader",
    "ReplayOutcome",
    "ReplayReport",
    "run_synthetic_task_smoke_gate",
    "run_replay_gate_strict",
    "render_report_md",
]
