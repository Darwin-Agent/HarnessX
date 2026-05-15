# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Stage 1 — dispatch Planner agent. It writes a single ``landscape.md``
synthesising the round's evidence. No briefs, no dispatch coupling — the
downstream Evolver reads the landscape + raw digests and decides itself
how many candidates to produce.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from harnessx import BaseTask
from harnessx.aegis.agents.planner import (
    build_planner_harness, PlannerInputs,
)


class BriefQuotaViolation(RuntimeError):
    """Deprecated — kept as a symbol so existing imports don't break.

    Previously raised when the Planner's briefs failed a hard-coded
    diversity / explorer / archive quota. The whole brief-dispatch model
    is gone (Planner writes landscape.md now), so this exception is never
    raised. Orchestrator code still catches it defensively.
    """


async def run_stage_1(
    *,
    round_n: int,
    overview_path: Path,
    journal_path: Path,
    archive_dir: Path,
    current_config_path: Path,
    landscape_path: Path,
    digests_dir: Path,
    reputation_summary: dict,
    model_config,
    max_cost_usd: float = 100.0,
    actionability_score: float = 1.0,
    run_root: Path | None = None,
    sessions_dir: Path | None = None,
) -> dict:
    landscape_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = PlannerInputs(
        round=round_n,
        overview_path=overview_path,
        journal_path=journal_path,
        archive_dir=archive_dir,
        current_config_path=current_config_path,
        landscape_path=landscape_path,
        digests_dir=digests_dir,
        reputation_summary=reputation_summary,
        max_cost_usd=max_cost_usd,
        run_root=run_root,
        sessions_dir=sessions_dir,
    )
    cfg = build_planner_harness(inputs)
    harness = model_config.agentic(cfg)
    result = await harness.run(BaseTask(
        description=(
            f"Synthesise round {round_n}'s evidence into `landscape.md`. "
            f"Call write_tool exactly once to save it. Final_output alone is "
            f"discarded."
        ),
        max_steps=200, max_cost_usd=max_cost_usd,
    ))
    import logging
    _log = logging.getLogger("aegis.stage_1")
    _log.info(
        "Planner R%d: exit=%s steps=%s cost=$%.3f tokens=%d budget=$%.2f landscape_written=%s",
        round_n,
        getattr(result, "exit_reason", "?"),
        len(getattr(result.trajectory, "steps", [])) if getattr(result, "trajectory", None) else "?",
        getattr(result, "total_cost_usd", 0.0) or 0.0,
        getattr(result, "total_tokens", 0) or 0,
        max_cost_usd,
        landscape_path.exists(),
    )

    landscape_written = landscape_path.exists()
    frontmatter: dict = {}
    if landscape_written:
        try:
            text = landscape_path.read_text(encoding="utf-8")
            m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
            if m:
                parsed = yaml.safe_load(m.group(1))
                if isinstance(parsed, dict):
                    frontmatter = parsed
        except (OSError, yaml.YAMLError):
            pass

    return {
        "landscape_written": landscape_written,
        "landscape_path": str(landscape_path),
        "frontmatter": frontmatter,
    }
