# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from argparse import Namespace

from harnessx.cli import _build_agent, _build_harness, _load_default
from harnessx.plugins.dimensions.mcp_runtime import McpRuntimePlugin
from harnessx.processors.tools.skill_loader import ProgressiveSkillLoader
from harnessx.tools.spawn_subagent import SPAWN_TOOL_NAME


def _groups_set(config) -> set[str]:
    from harnessx.core.harness import _instantiate_runtime

    groups: set[str] = set()
    for procs in _instantiate_runtime(config).processors.values():
        for p in procs:
            sg = getattr(p, "_singleton_group", getattr(type(p), "_singleton_group", None))
            if sg:
                groups.add(sg)
    return groups


def _build_args(**kwargs) -> Namespace:
    base = dict(
        strict=False,
        max_steps=30,
        verbose=False,
        print_mode=False,
        resume=None,
        command=None,
    )
    base.update(kwargs)
    return Namespace(**base)


class TestCliDefaultModes:
    def test_cli_default_standard_disables_self_verify(self) -> None:
        cfg = _load_default(strict=False)
        groups = _groups_set(cfg)
        assert "self_verify" not in groups

    def test_cli_default_strict_enables_self_verify(self) -> None:
        cfg = _load_default(strict=True)
        groups = _groups_set(cfg)
        assert "self_verify" in groups

    def test_cli_default_build_harness_includes_spawn_tool(self) -> None:
        from harnessx.core.harness import _build_tool_registry_from_config
        from harnessx.core.config_schema import ToolRegistryConfig

        cfg = _build_harness(_build_args())
        assert cfg.tool_registry is not None
        if isinstance(cfg.tool_registry, ToolRegistryConfig):
            rt = _build_tool_registry_from_config(cfg.tool_registry)
            assert SPAWN_TOOL_NAME in rt.list_names()
        else:
            assert SPAWN_TOOL_NAME in cfg.tool_registry.list_names()

    def test_cli_default_build_agent_returns_harness_with_spawn_tool(self) -> None:
        harness = _build_agent(_build_args())
        assert SPAWN_TOOL_NAME in harness._rt.tool_registry.list_names()

    def test_cli_default_build_config_resolves_enabled_mcp_servers(self, tmp_path, monkeypatch) -> None:
        home = tmp_path / "hx-home"
        monkeypatch.setenv("HARNESSX_HOME", str(home))
        home.mkdir(parents=True, exist_ok=True)
        (home / "mcp_servers.json").write_text(
            json.dumps(
                [
                    {
                        "id": "1",
                        "name": "sqlite",
                        "transport": "stdio",
                        "command": "uvx mcp-server-sqlite --db ./data.db",
                        "enabled": True,
                    },
                    {
                        "id": "2",
                        "name": "disabled",
                        "transport": "stdio",
                        "command": "echo no",
                        "enabled": False,
                    },
                ]
            ),
            encoding="utf-8",
        )

        cfg = _build_harness(_build_args())
        groups = _groups_set(cfg)
        assert "_mcp_runtime" in groups
        runtimes = [p for p in cfg.plugins if isinstance(p, McpRuntimePlugin)]
        assert runtimes
        names = {str(s.get("name", "")) for p in runtimes for s in p._load_servers() if isinstance(s, dict)}
        assert "sqlite" in names
        assert "disabled" not in names

    def test_cli_default_build_config_applies_persisted_enabled_tools(self, tmp_path, monkeypatch) -> None:
        home = tmp_path / "hx-home"
        monkeypatch.setenv("HARNESSX_HOME", str(home))
        home.mkdir(parents=True, exist_ok=True)
        (home / "mcp_servers.json").write_text("[]", encoding="utf-8")
        (home / "slot_config.json").write_text(
            json.dumps(
                {
                    "enabled_tools": ["Read"],
                    "enabled_skills": None,
                    "sandbox_type": "local",
                    "sandbox_url": None,
                }
            ),
            encoding="utf-8",
        )

        harness = _build_agent(_build_args())
        names = set(harness._rt.tool_registry.list_names())
        assert "Read" in names
        assert "Bash" not in names
        assert SPAWN_TOOL_NAME in names

    def test_cli_default_build_config_applies_persisted_enabled_skills(self, tmp_path, monkeypatch) -> None:
        home = tmp_path / "hx-home"
        monkeypatch.setenv("HARNESSX_HOME", str(home))
        home.mkdir(parents=True, exist_ok=True)
        (home / "mcp_servers.json").write_text("[]", encoding="utf-8")
        (home / "slot_config.json").write_text(
            json.dumps(
                {
                    "enabled_tools": None,
                    "enabled_skills": ["pptx"],
                    "sandbox_type": "local",
                    "sandbox_url": None,
                }
            ),
            encoding="utf-8",
        )

        cfg = _build_harness(_build_args())

        from harnessx.plugins.dimensions.skill_runtime import SkillRuntimePlugin
        from harnessx.core.harness import _instantiate_runtime

        loaders = [
            p
            for procs in _instantiate_runtime(cfg).processors.values()
            for p in procs
            if isinstance(p, ProgressiveSkillLoader)
        ]
        assert loaders
        assert all(p.enabled_skills == ["pptx"] for p in loaders)

        skill_plugins = [p for p in (cfg.plugins or []) if isinstance(p, SkillRuntimePlugin)]
        assert skill_plugins
        assert all(p._enabled_skills == ["pptx"] for p in skill_plugins)

    def test_cli_default_prefers_agent_harness_config(self, tmp_path, monkeypatch) -> None:
        home = tmp_path / "hx-home"
        monkeypatch.setenv("HARNESSX_HOME", str(home))
        home.mkdir(parents=True, exist_ok=True)
        (home / "mcp_servers.json").write_text("[]", encoding="utf-8")
        (home / "slot_config.json").write_text(
            json.dumps(
                {
                    "enabled_tools": None,
                    "enabled_skills": None,
                    "sandbox_type": "local",
                    "sandbox_url": None,
                }
            ),
            encoding="utf-8",
        )

        ws = home / "workspaces" / "hxagent"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "harness_config.yaml").write_text("processors: []\n", encoding="utf-8")

        cfg = _build_harness(_build_args())
        groups = _groups_set(cfg)
        assert "_mcp_runtime" in groups
        assert "_skill_runtime" in groups
        assert "_skill_sysprompt" in groups

    def test_cli_default_creates_empty_mcp_servers_file(self, tmp_path, monkeypatch) -> None:
        home = tmp_path / "hx-home"
        monkeypatch.setenv("HARNESSX_HOME", str(home))
        home.mkdir(parents=True, exist_ok=True)

        # Do not pre-create mcp_servers.json
        cfg = _build_harness(_build_args())
        assert cfg is not None

        mcp_file = home / "mcp_servers.json"
        assert mcp_file.exists()
        assert mcp_file.read_text(encoding="utf-8").strip().startswith("[")
