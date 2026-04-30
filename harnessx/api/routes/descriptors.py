# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

router = APIRouter()


@router.get("/descriptors/{key}/yaml")
def export_descriptor_yaml(key: str):
    """Download a named example's harness_config.yaml verbatim."""
    import pathlib

    _REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
    yaml_path = _REPO_ROOT / "examples" / key / "harness_config.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail=f"Example '{key}' not found")
    return Response(
        content=yaml_path.read_text(encoding="utf-8"),
        media_type="text/yaml; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{key}.yaml"'},
    )


@router.post("/descriptors/yaml")
async def import_descriptor_yaml(request: Request):
    """Parse a YAML body and return a validated JSON harness config.

    Accepts the processor-list format::

        processors:
          - type: system_prompt
          - type: loop_detection
          - _target_: harnessx.processors.memory.memory_retrieval.MemoryRetrievalProcessor
            memory:
              type: sliding_window
              n: 20

    Returns ``application/json`` — the validated dict that can be passed
    directly as ``harness_config`` in a run request.
    """
    import yaml as _yaml
    from harnessx.core.harness import HarnessConfig as _HC

    yaml_text = (await request.body()).decode("utf-8")
    if not yaml_text.strip():
        raise HTTPException(status_code=400, detail="Empty request body")

    try:
        raw = _yaml.safe_load(yaml_text) or {}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")

    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="YAML must be a mapping")

    # Dry-run: validate that all processors can be instantiated
    try:
        _yaml_str = _yaml.dump(raw)
        _HC.from_yaml(_yaml_str)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid harness config: {exc}")

    return {
        "processors": raw.get("processors", []),
        "plugins": raw.get("plugins"),
    }
