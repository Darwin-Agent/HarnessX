# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from harnessx.workspace.workspace import Workspace, WorkspaceEscapeError


# ── home field ────────────────────────────────────────────────────────────────


class TestWorkspaceHome:
    def test_workspace_home_none_by_default(self, tmp_path):
        ws = Workspace(agent_id="a", root=tmp_path / "root")
        assert ws.home is None

    def test_workspace_home_creates_dir(self, tmp_path):
        home = tmp_path / "agent_home"
        assert not home.exists()
        ws = Workspace(agent_id="a", root=tmp_path / "root", home=home)
        assert home.exists()
        assert ws.home == home

    def test_workspace_home_resolves_tilde(self, tmp_path, monkeypatch):
        """home accepts ~-prefixed paths and resolves them."""
        monkeypatch.setenv("HOME", str(tmp_path))
        ws = Workspace(agent_id="a", root=tmp_path / "root", home=Path("~/.oh"))
        assert ws.home == tmp_path / ".oh"

    def test_workspace_child_propagates_home(self, tmp_path):
        home = tmp_path / "agent_home"
        parent = Workspace(agent_id="parent", root=tmp_path / "ws", home=home)
        child = parent.child("child")
        assert child.home == home

    # ── project field ─────────────────────────────────────────────────────────────

    def test_workspace_project_default(self, tmp_path):
        ws = Workspace(agent_id="a", root=tmp_path / "root")
        assert ws.project == "default"

    def test_workspace_project_custom(self, tmp_path):
        ws = Workspace(agent_id="a", root=tmp_path / "root", project="my-proj")
        assert ws.project == "my-proj"

    # ── auto-derivation of root from home ────────────────────────────────────────

    def test_auto_derive_root_default_project(self, tmp_path):
        """When root is omitted and home is set, root is derived from AGENT_HOME layout."""
        home = tmp_path / "oh"
        ws = Workspace(agent_id="alice", home=home)
        assert ws.root == home / "workspaces" / "alice" / "default"
        assert ws.root.exists()

    def test_auto_derive_root_custom_project(self, tmp_path):
        home = tmp_path / "oh"
        ws = Workspace(agent_id="alice", project="coding", home=home)
        assert ws.root == home / "workspaces" / "alice" / "coding"
        assert ws.root.exists()

    def test_auto_derive_root_stores_project(self, tmp_path):
        """ws.project reflects the project used for derivation."""
        home = tmp_path / "oh"
        ws = Workspace(agent_id="bob", project="research", home=home)
        assert ws.project == "research"

    def test_explicit_root_overrides_derivation(self, tmp_path):
        """When root is explicitly given, home-based derivation is skipped."""
        home = tmp_path / "oh"
        custom_root = tmp_path / "custom"
        ws = Workspace(agent_id="alice", root=custom_root, project="coding", home=home)
        assert ws.root == custom_root

    def test_no_root_no_home_raises(self, tmp_path):
        """Omitting both root and home should raise ValueError."""
        with pytest.raises(ValueError, match="root.*home"):
            Workspace(agent_id="a")

    def test_auto_derive_invalid_agent_id_raises(self, tmp_path):
        home = tmp_path / "oh"
        with pytest.raises(ValueError):
            Workspace(agent_id="bad/name", home=home)

    def test_auto_derive_invalid_project_raises(self, tmp_path):
        home = tmp_path / "oh"
        with pytest.raises(ValueError):
            Workspace(agent_id="alice", project="bad name", home=home)

    # ── home mode path jail ───────────────────────────────────────────────────────

    def test_home_mode_allows_root_path(self, tmp_path):
        home = tmp_path / "oh"
        ws = Workspace(agent_id="a", root=home / "ws" / "a" / "p1", home=home, mode="home")
        resolved = ws.resolve("subdir/file.txt")
        assert resolved == ws.root / "subdir" / "file.txt"

    def test_home_mode_allows_sibling_within_home(self, tmp_path):
        home = tmp_path / "oh"
        ws = Workspace(agent_id="a", root=home / "ws" / "a" / "p1", home=home, mode="home")
        sibling = str(home / "ws" / "a" / "p2")
        resolved = ws.resolve(sibling)
        assert resolved == Path(sibling)

    def test_home_mode_allows_memory(self, tmp_path):
        home = tmp_path / "oh"
        mem = home / "workspaces" / "a" / "memory"
        mem.mkdir(parents=True, exist_ok=True)
        ws = Workspace(agent_id="a", root=home / "workspaces" / "a" / "p1", home=home, mode="home")
        resolved = ws.resolve(str(mem / "notes.md"))
        assert mem in resolved.parents or resolved.parent == mem

    def test_home_mode_blocks_outside_home(self, tmp_path):
        home = tmp_path / "oh"
        ws = Workspace(agent_id="a", root=home / "ws", home=home, mode="home")
        outside = str(tmp_path / "outside" / "secret.txt")
        with pytest.raises(WorkspaceEscapeError):
            ws.resolve(outside)

    def test_home_mode_falls_back_to_root_when_home_none(self, tmp_path):
        """If mode='home' but home is None, jail falls back to root."""
        ws = Workspace(agent_id="a", root=tmp_path / "root", mode="home")
        assert ws.home is None
        ws.resolve("ok.txt")
        with pytest.raises(WorkspaceEscapeError):
            ws.resolve(str(tmp_path / "outside"))

    def test_home_mode_auto_derived_root_path_jail(self, tmp_path):
        """Auto-derived workspace with mode='home' jails to AGENT_HOME."""
        home = tmp_path / "oh"
        ws = Workspace(agent_id="alice", project="proj", home=home, mode="home")
        # Inside AGENT_HOME — allowed
        ws.resolve(str(home / "plugins" / "myplugin"))
        # Outside AGENT_HOME — blocked
        with pytest.raises(WorkspaceEscapeError):
            ws.resolve(str(tmp_path / "escape"))

    # ── isolated mode still works ─────────────────────────────────────────────────

    def test_isolated_mode_still_works(self, tmp_path):
        ws = Workspace(agent_id="a", root=tmp_path / "root", mode="isolated")
        resolved = ws.resolve("file.txt")
        assert resolved == ws.root / "file.txt"

    def test_isolated_mode_blocks_escape(self, tmp_path):
        ws = Workspace(agent_id="a", root=tmp_path / "root", mode="isolated")
        with pytest.raises(WorkspaceEscapeError):
            ws.resolve("../../etc/passwd")

    # ── None mode: no jail ────────────────────────────────────────────────────────

    def test_no_mode_allows_any_path(self, tmp_path):
        ws = Workspace(agent_id="a", root=tmp_path / "root", mode=None)
        resolved = ws.resolve("/etc/passwd")
        assert resolved == Path("/etc/passwd")
