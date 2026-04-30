# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _mcp_servers_candidates() -> list[Path]:
    """AGENT_HOME-first + ~/.harnessx compatibility candidates for mcp_servers.json."""
    from harnessx.home import agent_home

    primary = agent_home() / "mcp_servers.json"
    compat = Path.home() / ".harnessx" / "mcp_servers.json"
    if compat == primary:
        return [primary]
    return [primary, compat]


def ensure_agent_home_mcp_servers_json() -> None:
    """Ensure AGENT_HOME/mcp_servers.json exists with an empty array payload."""
    primary = _mcp_servers_candidates()[0]
    if primary.exists():
        return
    try:
        primary.parent.mkdir(parents=True, exist_ok=True)
        primary.write_text("[]\n", encoding="utf-8")
    except Exception:
        pass


def normalise_mcp_config(raw: Any) -> dict[str, Any]:
    """Normalize harness-level mcp_config with defaults."""
    cfg = raw if isinstance(raw, dict) else {}
    source = str(cfg.get("source", "agent_home")).strip().lower() or "agent_home"
    if source not in {"agent_home", "file", "disabled", "inline"}:
        source = "agent_home"
    path = str(cfg.get("path", "")).strip() or None
    servers = cfg.get("servers")
    if not isinstance(servers, list):
        servers = None
    return {
        "source": source,
        "path": path,
        "servers": servers,
    }


def normalise_enabled_mcp_servers(raw_items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert persisted UI items into runtime server specs."""
    if not raw_items:
        return []

    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw_items):
        if item.get("enabled", True) is False:
            continue

        name = str(item.get("name", "")).strip() or f"mcp-{i + 1}"
        transport = str(item.get("transport", "")).strip().lower()
        if transport not in {"stdio", "http"}:
            transport = "http" if item.get("url") else "stdio"

        spec: dict[str, Any] = {"name": name, "transport": transport}

        if transport == "http":
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            spec["url"] = url
        else:
            command = str(item.get("command", "")).strip()
            if not command:
                continue
            spec["command"] = command

        env = item.get("env")
        if isinstance(env, dict) and env:
            spec["env"] = {str(k): str(v) for k, v in env.items()}

        out.append(spec)
    return out


def load_enabled_mcp_servers_from_file(path: Path) -> list[dict[str, Any]]:
    """Load and normalize enabled servers from a specific JSON file path."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            raw_items = [item for item in data if isinstance(item, dict)]
            return normalise_enabled_mcp_servers(raw_items)
    except Exception:
        pass
    return []


def load_enabled_mcp_servers(*, ensure_primary: bool = True) -> list[dict[str, Any]]:
    """Load enabled MCP server specs from shared persisted settings."""
    if ensure_primary:
        ensure_agent_home_mcp_servers_json()

    seen: set[Path] = set()
    for path in _mcp_servers_candidates():
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                raw_items = [item for item in data if isinstance(item, dict)]
                return normalise_enabled_mcp_servers(raw_items)
        except Exception:
            continue
    return []


def resolve_mcp_servers(
    mcp_config: Any,
    *,
    base_dir: str | Path | None = None,
    ensure_primary: bool = True,
) -> list[dict[str, Any]]:
    """Resolve MCP servers by harness-level mcp_config source."""
    cfg = normalise_mcp_config(mcp_config)
    source = cfg["source"]
    if source == "disabled":
        return []
    if source == "agent_home":
        return load_enabled_mcp_servers(ensure_primary=ensure_primary)
    if source == "inline":
        servers = cfg.get("servers")
        if isinstance(servers, list):
            raw_items = [s for s in servers if isinstance(s, dict)]
            return normalise_enabled_mcp_servers(raw_items)
        return []
    if source == "file":
        path_raw = cfg.get("path")
        if not path_raw:
            return []
        p = Path(path_raw).expanduser()
        if not p.is_absolute():
            root = Path(base_dir) if base_dir is not None else Path.cwd()
            p = (root / p).resolve()
        return load_enabled_mcp_servers_from_file(p)
    return load_enabled_mcp_servers(ensure_primary=ensure_primary)
