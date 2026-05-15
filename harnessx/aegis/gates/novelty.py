# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Novelty gate — blocks candidates whose file_changes signature is in the
``refuted_signatures`` ledger.
"""
from __future__ import annotations

from .structure import GateResult


def check_novelty(signature: str, *, refuted_signatures: set[str]) -> GateResult:
    if signature in refuted_signatures:
        return GateResult(
            ok=False,
            reason=f"signature previously refuted: {signature[:16]}...",
        )
    return GateResult(ok=True)
