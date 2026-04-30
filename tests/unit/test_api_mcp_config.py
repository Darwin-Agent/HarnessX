# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

from harnessx.plugins.dimensions.mcp_runtime.utils import resolve_mcp_servers


class TestApiMcpConfig:
    def test_resolve_mcp_servers_agent_home_default(self, tmp_path, monkeypatch) -> None:
        home = tmp_path / "hx-home"
        monkeypatch.setenv("HARNESSX_HOME", str(home))
        home.mkdir(parents=True, exist_ok=True)
        (home / "mcp_servers.json").write_text(
            json.dumps(
                [
                    {
                        "name": "sqlite",
                        "transport": "stdio",
                        "command": "uvx mcp-server-sqlite --db ./data.db",
                        "enabled": True,
                    },
                    {
                        "name": "off",
                        "transport": "stdio",
                        "command": "echo no",
                        "enabled": False,
                    },
                ]
            ),
            encoding="utf-8",
        )

        resolved = resolve_mcp_servers(None, ensure_primary=True)
        assert [s["name"] for s in resolved] == ["sqlite"]

    def test_resolve_mcp_servers_file_sidecar(self, tmp_path) -> None:
        sidecar = tmp_path / "mcp_servers.json"
        sidecar.write_text(
            json.dumps(
                [
                    {
                        "name": "wiki",
                        "transport": "http",
                        "url": "http://127.0.0.1:9000/mcp",
                        "enabled": True,
                    }
                ]
            ),
            encoding="utf-8",
        )

        resolved = resolve_mcp_servers(
            {"source": "file", "path": "./mcp_servers.json"},
            base_dir=tmp_path,
            ensure_primary=True,
        )
        assert len(resolved) == 1
        assert resolved[0]["name"] == "wiki"
        assert resolved[0]["transport"] == "http"

    def test_resolve_mcp_servers_disabled(self, tmp_path, monkeypatch) -> None:
        home = tmp_path / "hx-home"
        monkeypatch.setenv("HARNESSX_HOME", str(home))
        home.mkdir(parents=True, exist_ok=True)
        (home / "mcp_servers.json").write_text(
            json.dumps(
                [
                    {
                        "name": "sqlite",
                        "transport": "stdio",
                        "command": "uvx mcp-server-sqlite --db ./data.db",
                        "enabled": True,
                    }
                ]
            ),
            encoding="utf-8",
        )

        resolved = resolve_mcp_servers({"source": "disabled"}, ensure_primary=True)
        assert resolved == []
