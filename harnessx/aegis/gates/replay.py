# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Replay gate — delegates to meta_harness synthetic-task smoke test.

Fatal gate: if smoke fails, the new config crashes on real tasks.
The model used for replay is supplied by the caller (orchestrator) so this
gate does not depend on any specific provider.
"""
from __future__ import annotations

from pathlib import Path

from harnessx import HarnessConfig

from .structure import GateResult


async def _run_meta_replay(cfg: HarnessConfig, model_config) -> bool:
    from harnessx.meta_harness.replay import run_synthetic_task_smoke_gate
    report = await run_synthetic_task_smoke_gate(cfg, model_config)
    return bool(report.ok)


async def check_replay_smoke(
    candidate_config_path_or_cfg,
    *,
    model_config=None,
) -> GateResult:
    """Run the synthetic-task smoke gate.

    Accepts either a path (backward-compatible) or a pre-loaded and
    canonicalized ``HarnessConfig``. Stage 4 pre-loads once and threads
    the cfg through both gates to avoid re-parsing the YAML three times.
    """
    if model_config is None:
        return GateResult(ok=True, reason="model_config not provided; replay skipped")
    try:
        if isinstance(candidate_config_path_or_cfg, (str, Path)):
            cfg = HarnessConfig.from_yaml_file(
                str(candidate_config_path_or_cfg)
            ).canonicalize()
        else:
            cfg = candidate_config_path_or_cfg
        ok = await _run_meta_replay(cfg, model_config)
    except Exception as exc:
        return GateResult(ok=False, reason=f"replay smoke failed: {exc!r}")
    if not ok:
        return GateResult(ok=False, reason="replay smoke returned falsy")
    return GateResult(ok=True)
