# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
import pathlib
import re
import yaml as _yaml
from fastapi import APIRouter
from harnessx.api.models import ExampleItem

router = APIRouter()

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"


def _default_example_workspace(key: str) -> dict[str, str]:
    safe_key = re.sub(r"[^a-zA-Z0-9_]+", "_", key).strip("_").lower() or "default"
    return {
        "agent_id": "lab_agent",
        "project": f"example_{safe_key}",
    }


def _scan_examples() -> list[ExampleItem]:
    items: list[ExampleItem] = []
    if not _EXAMPLES_DIR.is_dir():
        return items

    for subdir in sorted(_EXAMPLES_DIR.iterdir()):
        yaml_path = subdir / "harness_config.yaml"
        if not subdir.is_dir() or not yaml_path.exists():
            continue

        try:
            raw = _yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue

        if not isinstance(raw, dict):
            continue

        # Processor demos or internal examples can opt out of Lab listing.
        if raw.get("lab_visible") is False:
            continue

        key = subdir.name
        label = raw.get("label") or key.replace("_", " ").title()
        description = raw.get("description") or ""
        workspace = raw.get("workspace")
        default_ws = _default_example_workspace(key)
        if not isinstance(workspace, dict):
            workspace = default_ws
        else:
            workspace = {
                "agent_id": str(workspace.get("agent_id", "")).strip() or default_ws["agent_id"],
                "project": str(workspace.get("project", "")).strip() or default_ws["project"],
            }

        items.append(
            ExampleItem(
                key=key,
                label=label,
                description=description,
                harness_config={
                    "processors": raw.get("processors", []),
                    "plugins": raw.get("plugins"),
                },
                workspace=workspace,
            )
        )

    return items


@router.get("/examples", response_model=list[ExampleItem])
def list_examples():
    """Return all runnable examples from the examples/ directory."""
    return _scan_examples()
