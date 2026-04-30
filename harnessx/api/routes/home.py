# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


class AgentHarnessConfigResponse(BaseModel):
    agent_id: str
    project: str
    path: str
    exists: bool
    harness_config: dict
    used_default: bool = False
    persisted_default: bool = False


def _hxagent_default_harness_config_payload() -> dict:
    """Build the same default harness config as CLI _load_default()."""
    import yaml

    from harnessx.cli import _load_default

    cfg = _load_default()
    raw: dict = yaml.safe_load(cfg.to_yaml())
    raw["plugins"] = [
        "harnessx.plugins.dimensions.light_memory.LightMemoryPlugin",
    ]

    processors = raw.get("processors", [])
    if not isinstance(processors, list):
        processors = []
    plugins = raw.get("plugins")
    if plugins is not None and not isinstance(plugins, list):
        plugins = None

    from harnessx.api.routes.run import _sanitize_harness_config_payload

    payload = _sanitize_harness_config_payload({"processors": processors, "plugins": plugins})
    payload["mcp_config"] = {"source": "agent_home"}
    return payload


@router.get("/home")
async def get_home():
    """Return AGENT_HOME path, active defaults, and full agent/project tree."""
    from harnessx.home import (
        agent_home,
        default_agent_id,
        default_project,
        list_agents,
        list_projects,
    )

    home = agent_home()
    agent_id = default_agent_id()
    agents = list_agents()

    agents_tree = []
    for aid in agents:
        projects = list_projects(aid)
        agents_tree.append(
            {
                "id": aid,
                "projects": projects,
                "workspace_path": str(home / "workspaces" / aid),
                "memory_path": str(home / "workspaces" / aid / "memory"),
                "project_paths": {p: str(home / "workspaces" / aid / p) for p in projects},
            }
        )

    return {
        "home": str(home),
        "default_agent_id": agent_id,
        "default_project": default_project(),
        "agents_tree": agents_tree,
        "plugins_path": str(home / "plugins"),
        "skills_path": str(home / "skills"),
        "configs_path": str(home / "configs"),
    }


@router.get("/home/agents")
async def list_agents_route():
    """List all agent ids that have a workspace under AGENT_HOME."""
    from harnessx.home import list_agents

    return {"agents": list_agents()}


@router.get("/home/agents/{agent_id}/projects")
async def list_projects_route(agent_id: str):
    """List projects for a given agent."""
    from harnessx.home import list_projects

    return {"agent_id": agent_id, "projects": list_projects(agent_id)}


@router.get("/home/harness-config", response_model=AgentHarnessConfigResponse)
async def get_agent_harness_config(
    agent_id: str | None = Query(None),
    project: str | None = Query(None),
    workspace_base: str = Query("workspaces", description="Top-level workspace directory under AGENT_HOME"),
    persist_default: bool = Query(
        False,
        description="When true and hxagent config is missing, write default harness_config.yaml for this agent",
    ),
):
    """Return per-agent shared harness_config.yaml parsed as {processors, plugins}.

    This is used by Lab UI to keep "CLI Agent" in sync with the current
    AGENT_HOME agent context.

    ``project`` is accepted for UI context echoing but does not affect the
    resolved harness_config path.
    """
    import yaml as _yaml
    from harnessx.home import default_agent_id, default_project, agent_harness_config_path

    aid = (agent_id or default_agent_id()).strip()
    proj = (project or default_project()).strip()

    try:
        cfg_path = agent_harness_config_path(aid, workspace_base=workspace_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid agent id: {exc}")
    if not cfg_path.exists():
        # Only the built-in default agent gets CLI default fallback.
        if aid == "hxagent":
            payload = _hxagent_default_harness_config_payload()
            if persist_default:
                try:
                    body = "# HarnessX Harness Config (defaulted from cli._load_default)\n\n" + _yaml.safe_dump(
                        payload,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False,
                    )
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    cfg_path.write_text(body, encoding="utf-8")
                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to persist default config to {cfg_path}: {exc}",
                    )
            return AgentHarnessConfigResponse(
                agent_id=aid,
                project=proj,
                path=str(cfg_path),
                exists=cfg_path.exists(),
                harness_config=payload,
                used_default=True,
                persisted_default=bool(persist_default and cfg_path.exists()),
            )
        return AgentHarnessConfigResponse(
            agent_id=aid,
            project=proj,
            path=str(cfg_path),
            exists=False,
            harness_config={"processors": [], "plugins": None, "mcp_config": {"source": "agent_home"}},
        )

    try:
        raw = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse {cfg_path}: {exc}")

    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail=f"{cfg_path} must be a YAML mapping")

    processors = raw.get("processors", [])
    if not isinstance(processors, list):
        processors = []
    plugins = raw.get("plugins")
    if plugins is not None and not isinstance(plugins, list):
        plugins = None
    mcp_config = raw.get("mcp_config")
    if not isinstance(mcp_config, dict):
        mcp_config = {"source": "agent_home"}
    elif not isinstance(mcp_config.get("source"), str) or not mcp_config.get("source"):
        mcp_config = {**mcp_config, "source": "agent_home"}

    from harnessx.api.routes.run import _sanitize_harness_config_payload

    return AgentHarnessConfigResponse(
        agent_id=aid,
        project=proj,
        path=str(cfg_path),
        exists=True,
        harness_config=_sanitize_harness_config_payload(
            {"processors": processors, "plugins": plugins, "mcp_config": mcp_config}
        ),
        used_default=False,
        persisted_default=False,
    )


class HarnessConfigSaveRequest(BaseModel):
    processors: list = []
    plugins: list | None = None
    mcp_config: dict | None = None


@router.put("/home/harness-config", response_model=AgentHarnessConfigResponse)
async def save_agent_harness_config(
    body: HarnessConfigSaveRequest,
    agent_id: str | None = Query(None),
    project: str | None = Query(None),
    workspace_base: str = Query("workspaces"),
):
    """Write per-agent shared harness_config.yaml."""
    import yaml as _yaml
    from harnessx.home import default_agent_id, default_project, agent_harness_config_path

    aid = (agent_id or default_agent_id()).strip()
    proj = (project or default_project()).strip()

    try:
        cfg_path = agent_harness_config_path(aid, workspace_base=workspace_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid agent id: {exc}")
    payload = {"processors": body.processors}
    if body.plugins is not None:
        payload["plugins"] = body.plugins
    if body.mcp_config is not None:
        payload["mcp_config"] = body.mcp_config
    else:
        payload["mcp_config"] = {"source": "agent_home"}

    try:
        content = _yaml.dump(payload, default_flow_style=False, allow_unicode=True, sort_keys=False)
        cfg_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write {cfg_path}: {exc}")

    return AgentHarnessConfigResponse(
        agent_id=aid,
        project=proj,
        path=str(cfg_path),
        exists=True,
        harness_config=payload,
        used_default=False,
        persisted_default=False,
    )
