# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Canonicalize gate — delegates to HarnessConfig.from_yaml_file + template
render. Fatal: if this fails the new config cannot run at all.
"""
from __future__ import annotations

from pathlib import Path

from harnessx import HarnessConfig

from .structure import GateResult


def check_canonicalize(candidate_config_path_or_cfg) -> GateResult:
    """Verify the applied config loads and canonicalizes.

    Accepts either a path (backward-compatible) or a pre-loaded
    ``HarnessConfig``. Stage 4 pre-loads once and passes the cfg through
    both gates to avoid re-parsing the YAML three times per candidate.
    """
    try:
        if isinstance(candidate_config_path_or_cfg, (str, Path)):
            cfg = HarnessConfig.from_yaml_file(str(candidate_config_path_or_cfg))
        else:
            cfg = candidate_config_path_or_cfg
        cfg.canonicalize()
    except Exception as exc:
        return GateResult(ok=False, reason=f"canonicalize failed: {exc!r}")
    return GateResult(ok=True)
