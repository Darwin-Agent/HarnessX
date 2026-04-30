# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["mcp"])


# Persistence file (primary: AGENT_HOME; fallback: ~/.harnessx for compatibility)
def _state_candidates() -> list[Path]:
    from harnessx.home import agent_home

    primary = agent_home() / "mcp_servers.json"
    compat = Path.home() / ".harnessx" / "mcp_servers.json"
    if compat == primary:
        return [primary]
    return [primary, compat]


def _state_file_for_read() -> Path:
    for p in _state_candidates():
        if p.exists():
            return p
    return _state_candidates()[0]


def _state_file_for_write() -> Path:
    return _state_candidates()[0]


# ── Models ────────────────────────────────────────────────────────────────────


class McpServerConfig(BaseModel):
    id: str = ""
    name: str
    transport: str = "stdio"  # "stdio" | "http"
    command: str = ""  # stdio: shell command
    url: str = ""  # http: endpoint URL
    env: dict[str, str] = {}
    enabled: bool = True


class McpToolInfo(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = {}


# ── Persistence helpers ───────────────────────────────────────────────────────


def _load() -> list[McpServerConfig]:
    state_file = _state_file_for_read()
    if not state_file.exists():
        return []
    try:
        raw = json.loads(state_file.read_text(encoding="utf-8"))
        return [McpServerConfig(**item) for item in raw]
    except Exception:
        return []


def _save(servers: list[McpServerConfig]) -> None:
    state_file = _state_file_for_write()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps([s.model_dump() for s in servers], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/mcp/servers", response_model=list[McpServerConfig])
async def list_servers() -> Any:
    """Return all configured MCP servers."""
    return _load()


@router.post("/mcp/servers", response_model=McpServerConfig, status_code=201)
async def add_server(cfg: McpServerConfig) -> Any:
    """Add a new MCP server configuration."""
    servers = _load()
    cfg.id = str(uuid.uuid4())
    servers.append(cfg)
    _save(servers)
    return cfg


@router.patch("/mcp/servers/{server_id}", response_model=McpServerConfig)
async def update_server(server_id: str, patch: dict[str, Any]) -> Any:  # type: ignore[type-arg]
    """Partial-update a MCP server config (e.g. toggle enabled)."""
    servers = _load()
    for i, s in enumerate(servers):
        if s.id == server_id:
            updated = s.model_copy(update=patch)
            servers[i] = updated
            _save(servers)
            return updated
    raise HTTPException(404, f"MCP server not found: {server_id}")


@router.delete("/mcp/servers/{server_id}", status_code=204)
async def delete_server(server_id: str) -> None:
    """Remove a MCP server config."""
    servers = _load()
    remaining = [s for s in servers if s.id != server_id]
    if len(remaining) == len(servers):
        raise HTTPException(404, f"MCP server not found: {server_id}")
    _save(remaining)


@router.post("/mcp/servers/{server_id}/tools", response_model=list[McpToolInfo])
async def preview_tools(server_id: str) -> Any:
    """Connect to the MCP server and return its exposed tools."""
    servers = _load()
    server = next((s for s in servers if s.id == server_id), None)
    if server is None:
        raise HTTPException(404, f"MCP server not found: {server_id}")

    try:
        from ...tools.mcp import MCPClient

        client = MCPClient(
            transport=server.transport,
            command=server.command,
            url=server.url,
            env=server.env or None,
        )
        await asyncio.wait_for(client.connect(), timeout=15)
        try:
            raw_tools = await asyncio.wait_for(client.list_tools(), timeout=10)
        finally:
            await client.disconnect()

        return [
            McpToolInfo(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in raw_tools
        ]
    except asyncio.TimeoutError:
        raise HTTPException(408, "MCP server connection timed out")
    except Exception as exc:
        raise HTTPException(502, f"Failed to connect to MCP server: {exc}")
