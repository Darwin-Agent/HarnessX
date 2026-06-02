# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Stage 2 — single Evolver session producing K candidates.

The Evolver reads landscape.md + digests + trajectories + ledgers and
decides how many candidates (K ≥ 0) to write based on evidence. The
orchestrator enumerates whatever actually got written under
``candidates_dir`` and ``applied_root``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from harnessx import BaseTask
from harnessx.aegis.agents.evolver import (
    EvolverInputs,
    build_evolver_harness,
    parse_candidate_manifest,
)
from harnessx.aegis.apply import ApplyError, validate_applied_config

_log = logging.getLogger("aegis.stage_2")


async def run_stage_2(
    *,
    round_n: int,
    landscape_path: Path,
    current_config_path: Path,
    candidates_dir: Path,
    trajectories_dir: Path,
    digests_dir: Path,
    model_config,
    max_cost_usd: float = 100.0,
    sessions_dir: Path | None = None,
    benchmark_context: str = "",
) -> dict:
    candidates_dir.mkdir(parents=True, exist_ok=True)
    applied_root = candidates_dir.parent / "applied"
    applied_root.mkdir(parents=True, exist_ok=True)

    inputs = EvolverInputs(
        round=round_n,
        current_config_path=current_config_path,
        landscape_path=landscape_path,
        digests_dir=digests_dir,
        trajectories_dir=trajectories_dir,
        candidates_dir=candidates_dir,
        applied_root=applied_root,
        sessions_dir=sessions_dir,
        benchmark_context=benchmark_context,
    )
    cfg = build_evolver_harness(inputs)
    harness = model_config.agentic(cfg)
    try:
        result = await harness.run(
            BaseTask(
                description=(
                    f"Round {round_n} Evolver. Produce K candidate manifests + "
                    f"K applied scratch dirs under {candidates_dir} / {applied_root}. "
                    f"K is your choice based on evidence (K ≥ 0). Final_output alone "
                    f"is discarded — only files written via write_tool survive."
                ),
                max_steps=200,
                max_cost_usd=max_cost_usd,
            )
        )
        _log.info(
            "Evolver R%d: exit=%s steps=%s cost=$%.3f tokens=%d candidates=%d",
            round_n,
            getattr(result, "exit_reason", "?"),
            len(getattr(result.trajectory, "steps", [])) if getattr(result, "trajectory", None) else "?",
            getattr(result, "total_cost_usd", 0.0) or 0.0,
            getattr(result, "total_tokens", 0) or 0,
            len(list(candidates_dir.glob(f"C-R{round_n}-*.md"))),
        )
    except Exception as exc:
        _log.warning("Evolver R%d raised: %s", round_n, exc)
        return {
            "results": [],
            "ok_count": 0,
            "candidate_paths": [],
            "error": repr(exc),
        }

    # Enumerate whatever got written. Validate each candidate's manifest +
    # applied YAML. Drop candidates that fail validation from the shortlist
    # but preserve their files on disk so the Critic / debugger can see
    # what went wrong.
    results: list[tuple[str, bool, str]] = []
    surviving: list[Path] = []
    for candidate_path in sorted(candidates_dir.glob(f"C-R{round_n}-*.md")):
        cid = candidate_path.stem
        scratch_dir = applied_root / cid
        applied_cfg = scratch_dir / "config.yaml"
        if not applied_cfg.exists():
            results.append((cid, False, f"applied config missing: {applied_cfg}"))
            continue
        expected_bucket: str | None = None
        try:
            fm, _body = parse_candidate_manifest(candidate_path.read_text(encoding="utf-8"))
            if isinstance(fm, dict):
                expected_bucket = fm.get("bucket")
        except Exception as exc:
            results.append((cid, False, f"manifest parse failed: {exc}"))
            continue
        try:
            validate_applied_config(
                applied_cfg,
                expected_bucket=expected_bucket,
                scratch_dir=scratch_dir,
            )
        except ApplyError as exc:
            results.append((cid, False, f"apply_validation_failed: {exc}"))
            continue
        results.append((cid, True, ""))
        surviving.append(candidate_path)

    ok_count = sum(1 for _, ok, _ in results if ok)
    return {
        "results": results,
        "ok_count": ok_count,
        "candidate_paths": surviving,
    }
