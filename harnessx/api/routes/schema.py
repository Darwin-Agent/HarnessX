# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from fastapi import APIRouter
from harnessx.api.dimension_schema import DIMENSION_SCHEMA
from harnessx.api.custom_processors.direct_targets import build_custom_dimension

router = APIRouter()


@router.get("/schema")
def get_schema():
    """Return the full dimension schema that drives the Harness Lab card UI."""
    dims = list(DIMENSION_SCHEMA)
    custom_dim = build_custom_dimension()
    if custom_dim is not None:
        dims.append(custom_dim)
    return {"dimensions": dims}
