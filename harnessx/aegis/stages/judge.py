# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Stage 3 — dispatch Critic + orchestrate ask-more loop with Evolvers."""
from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable

from harnessx import BaseTask
from harnessx.aegis.agents.critic import (
    build_critic_harness, CriticInputs, parse_decision,
)


def make_evolver_runner(
    brief_paths_by_cid: dict[str, Path],
    evolver_harness_factory: Callable,
    model_config,
) -> Callable[[str, str], Awaitable[str]]:
    async def runner(cid: str, question: str) -> str:
        brief_path = brief_paths_by_cid.get(cid)
        if brief_path is None:
            return f"(no brief found for {cid}; refusing to answer)"
        cfg = evolver_harness_factory(cid, brief_path)
        harness = model_config.agentic(cfg)
        result = await harness.run(BaseTask(
            description=f"Answer Critic question about {cid}: {question}",
            max_steps=200, max_cost_usd=100.0,
        ))
        return result.final_output or "(no answer)"
    return runner


async def run_stage_3(
    *,
    round_n: int,
    candidates_dir: Path,
    verdicts_dir: Path,
    decision_path: Path,
    digests_dir: Path,
    trajectories_dir: Path,
    sessions_dir: Path,
    journal_path: Path,
    current_config_path: Path,
    evolver_runner: Callable[[str, str], Awaitable[str]],
    model_config,
    max_ask_more: int = 2,
    max_cost_usd: float = 100.0,
    meta_sessions_dir: Path | None = None,
) -> dict:
    verdicts_dir.mkdir(parents=True, exist_ok=True)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = CriticInputs(
        round=round_n,
        candidates_dir=candidates_dir, verdicts_dir=verdicts_dir,
        decision_path=decision_path, digests_dir=digests_dir,
        trajectories_dir=trajectories_dir, sessions_dir=sessions_dir,
        journal_path=journal_path, current_config_path=current_config_path,
        max_ask_more=max_ask_more,
        meta_sessions_dir=meta_sessions_dir,
    )
    cfg = build_critic_harness(inputs, evolver_runner)
    harness = model_config.agentic(cfg)
    result = await harness.run(BaseTask(
        description=(
            f"Critic Round {round_n}: judge all candidates. You MUST call "
            f"write_tool to save decision to {decision_path}. Final_output "
            f"alone will be discarded. Decision MUST start with YAML frontmatter "
            f"delimited by '---'."
        ),
        max_steps=200, max_cost_usd=max_cost_usd,
    ))
    import logging
    _log = logging.getLogger("aegis.stage_3")
    _log.info(
        "Critic R%d: exit=%s steps=%s cost=$%.3f tokens=%d budget=$%.2f decision_written=%s",
        round_n,
        getattr(result, "exit_reason", "?"),
        len(getattr(result.trajectory, "steps", [])) if getattr(result, "trajectory", None) else "?",
        getattr(result, "total_cost_usd", 0.0) or 0.0,
        getattr(result, "total_tokens", 0) or 0,
        max_cost_usd,
        decision_path.exists(),
    )
    # IV-4: validate each Critic verdict has citation anchors. Don't hard-fail
    # (the ship decision is already made), but surface broken verdicts so the
    # orchestrator/audit records them.
    from ..gates.structure import validate_critic_verdict
    broken_verdicts: list[tuple[str, str]] = []
    for v in sorted(verdicts_dir.glob("V-*.md")):
        vres = validate_critic_verdict(v.read_text(encoding="utf-8"))
        if not vres.ok:
            broken_verdicts.append((v.name, vres.reason))
            _log.warning("Verdict %s failed validation: %s", v.name, vres.reason)

    if not decision_path.exists():
        return {
            "decision": None, "critic_failed": True,
            "broken_verdict_count": len(broken_verdicts),
        }
    try:
        decision, body = parse_decision(decision_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        _log.warning("Critic R%d: decision.md parse failed: %s", round_n, exc)
        return {
            "decision": None, "critic_failed": True,
            "broken_verdict_count": len(broken_verdicts),
        }
    return {
        "decision": decision, "decision_body": body, "critic_failed": False,
        "broken_verdict_count": len(broken_verdicts),
    }
