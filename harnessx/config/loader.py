# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def _build_processors_flat(procs_cfg: Any) -> list:
    """Build flat processor list from ``[{_target_: ...}, ...]`` config."""
    from harnessx.core.harness import HarnessConfig as _HC

    if isinstance(procs_cfg, DictConfig):
        procs_cfg = OmegaConf.to_container(procs_cfg, resolve=True)

    return _HC(processors=procs_cfg).processors
