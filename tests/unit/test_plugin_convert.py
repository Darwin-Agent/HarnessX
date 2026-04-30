# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harnessx.plugins.convert import convert_claude_plugin
from harnessx.plugins.discovery import discover_plugins


# ── convert_claude_plugin ─────────────────────────────────────────────────────


class TestConvertClaudePlugin:
    def _make_claude_plugin(self, tmp_path: Path, name: str = "my-plugin") -> Path:
        src = tmp_path / "src" / name
        src.mkdir(parents=True)
        commands_dir = src / "commands"
        commands_dir.mkdir()
        (commands_dir / "recall.md").write_text("Recall: {{input}}", encoding="utf-8")
        manifest = {
            "name": name,
            "version": "0.1.0",
            "description": "A Claude Code plugin",
            "commands": [
                {
                    "name": "recall",
                    "description": "Recall something",
                    "prompt": "./commands/recall.md",
                }
            ],
        }
        (src / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
        return src

    def test_basic_conversion(self, tmp_path):
        src = self._make_claude_plugin(tmp_path)
        dst = tmp_path / "out"
        result = convert_claude_plugin(src, dst)

        assert result == dst
        assert (dst / "plugin.json").exists()

    def test_converted_manifest_preserves_standard_fields(self, tmp_path):
        src = self._make_claude_plugin(tmp_path, name="test-plugin")
        dst = tmp_path / "out"
        convert_claude_plugin(src, dst)

        with open(dst / "plugin.json") as f:
            manifest = json.load(f)

        assert manifest["name"] == "test-plugin"
        assert manifest["version"] == "0.1.0"
        assert manifest["description"] == "A Claude Code plugin"
        assert len(manifest["commands"]) == 1

    def test_converted_manifest_adds_oh_extensions(self, tmp_path):
        src = self._make_claude_plugin(tmp_path)
        dst = tmp_path / "out"
        convert_claude_plugin(src, dst)

        with open(dst / "plugin.json") as f:
            manifest = json.load(f)

        assert "processors" in manifest
        assert "tools" in manifest
        assert "slash_commands" in manifest

    def test_slash_commands_derived_from_commands(self, tmp_path):
        src = self._make_claude_plugin(tmp_path)
        dst = tmp_path / "out"
        convert_claude_plugin(src, dst)

        with open(dst / "plugin.json") as f:
            manifest = json.load(f)

        slash_cmds = manifest["slash_commands"]
        assert any(e.get("command") == "/recall" for e in slash_cmds)

    def test_processor_stub_created(self, tmp_path):
        src = self._make_claude_plugin(tmp_path)
        dst = tmp_path / "out"
        convert_claude_plugin(src, dst)

        assert (dst / "processors").is_dir()
        assert (dst / "processors" / "recall_processor.py").exists()

    def test_processor_stub_content(self, tmp_path):
        src = self._make_claude_plugin(tmp_path)
        dst = tmp_path / "out"
        convert_claude_plugin(src, dst)

        stub = (dst / "processors" / "recall_processor.py").read_text()
        assert "class RecallProcessor" in stub
        assert "on_task_start" in stub
        assert "MultiHookProcessor" in stub

    def test_commands_dir_copied(self, tmp_path):
        src = self._make_claude_plugin(tmp_path)
        dst = tmp_path / "out"
        convert_claude_plugin(src, dst)

        assert (dst / "commands" / "recall.md").exists()
        assert (dst / "commands" / "recall.md").read_text() == "Recall: {{input}}"

    def test_default_output_path(self, tmp_path):
        src = self._make_claude_plugin(tmp_path, name="my-plugin")
        result = convert_claude_plugin(src)
        assert result == src.parent / "my-plugin_oh"
        result.stat()  # verify it exists

    def test_src_missing_manifest_raises(self, tmp_path):
        src = tmp_path / "no-manifest"
        src.mkdir()
        with pytest.raises(FileNotFoundError, match="plugin.json"):
            convert_claude_plugin(src)

    def test_dst_exists_raises(self, tmp_path):
        src = self._make_claude_plugin(tmp_path)
        dst = tmp_path / "out"
        dst.mkdir()
        with pytest.raises(FileExistsError):
            convert_claude_plugin(src, dst)


# ── discover_plugins ──────────────────────────────────────────────────────────


class TestDiscoverPlugins:
    def _write_plugin(self, base: Path, name: str) -> Path:
        d = base / name
        d.mkdir(parents=True)
        manifest = {"name": name, "version": "0.1.0"}
        (d / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
        return d

    def test_discovers_plugins_in_dir(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        self._write_plugin(plugins_dir, "plugin-a")
        self._write_plugin(plugins_dir, "plugin-b")

        plugins = discover_plugins(extra_paths=[plugins_dir])
        names = [p.name for p in plugins]
        assert "plugin-a" in names
        assert "plugin-b" in names

    def test_workspace_plugins_discovered(self, tmp_path):
        ws = tmp_path / "workspace"
        plugins_dir = ws / ".harnessx" / "plugins"
        self._write_plugin(plugins_dir, "ws-plugin")

        plugins = discover_plugins(workspace_root=ws)
        assert any(p.name == "ws-plugin" for p in plugins)

    def test_deduplicated_by_name(self, tmp_path):
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        self._write_plugin(d1, "same-name")
        self._write_plugin(d2, "same-name")

        plugins = discover_plugins(extra_paths=[d1, d2])
        names = [p.name for p in plugins]
        assert names.count("same-name") == 1

    def test_empty_dir_returns_empty(self, tmp_path):
        empty_dir = tmp_path / "empty_plugins"
        empty_dir.mkdir()
        plugins = discover_plugins(extra_paths=[empty_dir], include_claude_plugins=False)
        assert plugins == []

    def test_nonexistent_dir_silently_ignored(self, tmp_path):
        non_existent = tmp_path / "does_not_exist"
        plugins = discover_plugins(extra_paths=[non_existent], include_claude_plugins=False)
        assert plugins == []

    def test_subdirs_without_manifest_ignored(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        # Directory without plugin.json
        (plugins_dir / "not-a-plugin").mkdir()
        # File (not a directory) — should be skipped
        (plugins_dir / "readme.txt").write_text("hi", encoding="utf-8")

        plugins = discover_plugins(extra_paths=[plugins_dir], include_claude_plugins=False)
        assert plugins == []
