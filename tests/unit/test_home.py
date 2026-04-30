# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations


import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    """Provide a clean HARNESSX_HOME pointing to a temp dir for every test."""
    monkeypatch.setenv("HARNESSX_HOME", str(tmp_path / "oh"))
    # Clear agent / project overrides so defaults are used
    monkeypatch.delenv("HARNESSX_AGENT", raising=False)
    monkeypatch.delenv("HARNESSX_PROJECT", raising=False)
    # Force reimport so module-level cache is fresh
    import importlib
    import harnessx.home as home_mod

    importlib.reload(home_mod)
    yield home_mod


# ── agent_home ────────────────────────────────────────────────────────────────


class TestHome:
    def test_agent_home_created(self, clean_env, tmp_path):
        home = clean_env.agent_home()
        assert home.exists()
        assert home == tmp_path / "oh"

    def test_agent_home_env_override(self, monkeypatch, tmp_path):
        custom = tmp_path / "custom_home"
        monkeypatch.setenv("HARNESSX_HOME", str(custom))
        import importlib
        import harnessx.home as m

        importlib.reload(m)
        assert m.agent_home() == custom
        assert custom.exists()

    # ── default_agent_id / default_project ───────────────────────────────────────

    def test_default_agent_id_default(self, clean_env):
        assert clean_env.default_agent_id() == "hxagent"

    def test_default_agent_id_env(self, monkeypatch, clean_env):
        monkeypatch.setenv("HARNESSX_AGENT", "my-agent")
        assert clean_env.default_agent_id() == "my-agent"

    def test_default_project_default(self, clean_env):
        # Without HARNESSX_PROJECT, default_project() returns a CWD-derived name.
        result = clean_env.default_project()
        assert result  # non-empty
        assert result != "default"  # no longer the old static default
        # Must be a valid name (safe chars only)
        clean_env._validate_name(result, "project")  # raises ValueError if invalid

    def test_default_project_env(self, monkeypatch, clean_env):
        monkeypatch.setenv("HARNESSX_PROJECT", "proj-x")
        assert clean_env.default_project() == "proj-x"

    # ── plugins_dir / skills_dir ─────────────────────────────────────────────────

    def test_plugins_dir(self, clean_env, tmp_path):
        d = clean_env.plugins_dir()
        assert d.exists()
        assert d == tmp_path / "oh" / "plugins"

    def test_skills_dir(self, clean_env, tmp_path):
        d = clean_env.skills_dir()
        assert d.exists()
        assert d == tmp_path / "oh" / "skills"

    # ── agent_workspace_root ──────────────────────────────────────────────────────

    def test_agent_workspace_root_default(self, clean_env, tmp_path):
        ws = clean_env.agent_workspace_root()
        assert ws.exists()
        # Default agent is "hxagent"; project is CWD-derived
        expected_agent = clean_env.default_agent_id()  # "hxagent"
        expected_project = clean_env.default_project()  # CWD-derived
        assert ws == tmp_path / "oh" / "workspaces" / expected_agent / expected_project

    def test_agent_workspace_root_custom(self, clean_env, tmp_path):
        ws = clean_env.agent_workspace_root("my-agent", "project-a")
        assert ws.exists()
        assert ws == tmp_path / "oh" / "workspaces" / "my-agent" / "project-a"

    def test_agent_workspace_root_distinct(self, clean_env):
        ws1 = clean_env.agent_workspace_root("alice", "p1")
        ws2 = clean_env.agent_workspace_root("alice", "p2")
        ws3 = clean_env.agent_workspace_root("bob", "p1")
        assert ws1 != ws2
        assert ws1 != ws3

    # ── agent_memory_dir ──────────────────────────────────────────────────────────

    def test_agent_memory_dir_default(self, clean_env, tmp_path):
        mem = clean_env.agent_memory_dir()
        assert mem.exists()
        expected_agent = clean_env.default_agent_id()  # "hxagent"
        assert mem == tmp_path / "oh" / "workspaces" / expected_agent / "memory"

    def test_agent_memory_dir_agent(self, clean_env, tmp_path):
        mem = clean_env.agent_memory_dir("pa")
        assert mem.exists()
        assert mem == tmp_path / "oh" / "workspaces" / "pa" / "memory"

    # ── agent_harness_config_path ────────────────────────────────────────────────

    def test_agent_harness_config_path_default(self, clean_env, tmp_path):
        p = clean_env.agent_harness_config_path()
        expected_agent = clean_env.default_agent_id()  # "hxagent"
        assert p == tmp_path / "oh" / "workspaces" / expected_agent / "harness_config.yaml"
        assert p.parent.exists()

    def test_agent_harness_config_path_custom(self, clean_env, tmp_path):
        p = clean_env.agent_harness_config_path("alice")
        assert p == tmp_path / "oh" / "workspaces" / "alice" / "harness_config.yaml"
        assert p.parent.exists()

    # ── agent_config_path ─────────────────────────────────────────────────────────

    def test_agent_config_path(self, clean_env, tmp_path):
        p = clean_env.agent_config_path("assistant")
        assert p == tmp_path / "oh" / "configs" / "assistant.yaml"
        assert p.parent.exists()

    # ── list_agents / list_projects ───────────────────────────────────────────────

    def test_list_agents_empty(self, clean_env):
        assert clean_env.list_agents() == []

    def test_list_agents(self, clean_env):
        clean_env.agent_workspace_root("alice", "p1")
        clean_env.agent_workspace_root("bob", "p1")
        agents = clean_env.list_agents()
        assert "alice" in agents
        assert "bob" in agents

    def test_list_projects_empty(self, clean_env):
        assert clean_env.list_projects("nobody") == []

    def test_list_projects(self, clean_env):
        clean_env.agent_workspace_root("alice", "proj-a")
        clean_env.agent_workspace_root("alice", "proj-b")
        projects = clean_env.list_projects("alice")
        assert "proj-a" in projects
        assert "proj-b" in projects
        # "memory" should not appear in project listing
        clean_env.agent_memory_dir("alice")
        projects2 = clean_env.list_projects("alice")
        assert "memory" not in projects2

    # ── _validate_name ────────────────────────────────────────────────────────────

    def test_validate_name_ok(self, clean_env):
        clean_env._validate_name("my-agent_1.0", "test")  # should not raise

    @pytest.mark.parametrize("bad", [".", "..", "a/b", "a b", "a\x00b", "a@b"])
    def test_validate_name_bad(self, clean_env, bad):
        with pytest.raises(ValueError):
            clean_env._validate_name(bad, "test")

    def test_validate_name_empty(self, clean_env):
        with pytest.raises(ValueError):
            clean_env._validate_name("", "test")
