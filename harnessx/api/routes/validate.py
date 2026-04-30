# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import traceback

from fastapi import APIRouter
from pydantic import BaseModel

from harnessx.api.models import HarnessConfigPayload

router = APIRouter()


class ValidateRequest(BaseModel):
    harness_config: HarnessConfigPayload = HarnessConfigPayload()


class ValidateResponse(BaseModel):
    ok: bool
    error: str | None = None
    hint: str | None = None
    details: str | None = None


_HINTS: list[tuple[str, str]] = [
    (
        "ModuleNotFoundError",
        "Check that the module path in `_target_` is correct and the package is installed.",
    ),
    (
        "ImportError",
        "A `_target_` import failed — verify the module path and your Python environment.",
    ),
    (
        "AttributeError",
        "The class name after the last `.` in a `_target_` doesn't exist in that module.",
    ),
    (
        "TypeError",
        "A `_target_` class received unexpected or missing constructor arguments.",
    ),
    (
        "KeyError",
        "A config dict is missing a required key (possibly `_target_` itself).",
    ),
    ("ValueError", "A config value is invalid — check types and allowed values."),
]


def _make_hint(exc: Exception) -> str:
    exc_type = type(exc).__name__
    for prefix, advice in _HINTS:
        if exc_type == prefix or exc_type.startswith(prefix):
            return advice
    return "Review the harness config and ensure all `_target_` class paths are correct."


@router.post("/validate", response_model=ValidateResponse)
async def validate_harness_config(req: ValidateRequest) -> ValidateResponse:
    """Dry-run build: instantiate processors without starting a run."""
    try:
        from harnessx.core.harness import HarnessConfig as _HC
        from harnessx.api.routes.run import _sanitize_harness_config_payload

        _payload = _sanitize_harness_config_payload(req.harness_config.model_dump())
        _HC(processors=_payload.get("processors") or [], plugins=_payload.get("plugins") or [])
        return ValidateResponse(ok=True)
    except Exception as exc:  # noqa: BLE001
        return ValidateResponse(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            hint=_make_hint(exc),
            details=traceback.format_exc(),
        )
