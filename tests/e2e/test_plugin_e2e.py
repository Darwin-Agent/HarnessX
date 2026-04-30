# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ._utils import make_test_workspace  # noqa: F401
except ImportError:
    pass

# ── Inline mock provider (no network needed) ──────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / "tests"))
from fixtures.mock_provider import MockProvider
from fixtures.mock_tools import make_registry

from harnessx import BaseTask, HarnessConfig, ModelConfig
from harnessx.core.builder import HarnessBuilder
from harnessx.core.processor import MultiHookProcessor
from harnessx.plugins.convert import convert_claude_plugin
from harnessx.plugins.discovery import discover_plugins, discover_claude_plugins
from harnessx.plugins.loader import load_from_directory
from harnessx.plugins.registry import PluginRegistry
from harnessx.tracing.null_tracer import NullTracer


# ── Helpers: real Claude Code install paths ───────────────────────────────────

INSTALLED_PLUGINS_JSON = Path.home() / ".claude" / "plugins" / "installed_plugins.json"


def _get_installed_claude_plugin_dirs() -> list[Path]:
    """Return install paths from ~/.claude/plugins/installed_plugins.json."""
    if not INSTALLED_PLUGINS_JSON.exists():
        return []
    try:
        data = json.loads(INSTALLED_PLUGINS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []
    dirs = []
    for entries in data.get("plugins", {}).values():
        for entry in entries:
            p = entry.get("installPath")
            if p and Path(p).is_dir():
                dirs.append(Path(p))
    return dirs


_INSTALLED_PLUGIN_DIRS = _get_installed_claude_plugin_dirs()
_HAS_REAL_PLUGINS = len(_INSTALLED_PLUGIN_DIRS) > 0

requires_real_plugins = pytest.mark.skipif(
    not _HAS_REAL_PLUGINS,
    reason="No Claude Code plugins installed — run `claude plugin install <name>` first",
)


# ── Fixture: realistic Claude Code plugin ─────────────────────────────────────


def make_claude_code_plugin(base_dir: Path) -> Path:
    """Create a local Claude Code plugin that mirrors the official format.

    Mimics the structure of plugins from
    https://github.com/anthropics/claude-plugins-official :
      .claude-plugin/
        plugin.json    — manifest with name/version/description
      commands/
        enhance.md     — prompt template for the 'enhance' command
        summarise.md   — prompt template for the 'summarise' command
    """
    plugin_dir = base_dir / "context-enhancer"
    (plugin_dir / ".claude-plugin").mkdir(parents=True)
    (plugin_dir / "commands").mkdir(parents=True)

    # Prompt templates (realistic Claude Code plugin content with frontmatter)
    (plugin_dir / "commands" / "enhance.md").write_text(
        textwrap.dedent("""\
        ---
        description: "Restate the goal before answering"
        argument-hint: "your question here"
        ---
        You are a context-aware assistant.  Before answering, briefly restate
        the user's goal to confirm your understanding, then respond.

        User input: $ARGUMENTS
        """),
        encoding="utf-8",
    )
    (plugin_dir / "commands" / "summarise.md").write_text(
        textwrap.dedent("""\
        ---
        description: "Summarise input in one sentence"
        ---
        Summarise the following in one sentence: $ARGUMENTS
        """),
        encoding="utf-8",
    )

    # Manifest lives inside .claude-plugin/ (Claude Code native format)
    manifest = {
        "name": "context-enhancer",
        "version": "0.1.0",
        "description": "Adds context-awareness and summarisation commands to Claude",
    }
    (plugin_dir / ".claude-plugin" / "plugin.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return plugin_dir


def _write_real_processor(plugin_dir: Path) -> None:
    """Replace the generated stub with a working processor implementation."""
    proc_dir = plugin_dir / "processors"
    proc_dir.mkdir(exist_ok=True)

    (proc_dir / "__init__.py").write_text("", encoding="utf-8")
    (proc_dir / "enhance_processor.py").write_text(
        textwrap.dedent("""\
        \"\"\"Working implementation of the /enhance processor.\"\"\"
        from __future__ import annotations
        import dataclasses
        from typing import AsyncIterator
        from harnessx.core.processor import MultiHookProcessor
        from harnessx.core.events import TaskStartEvent

        PREAMBLE = "[context-enhancer] Before answering, briefly restate the user's goal."

        class EnhanceProcessor(MultiHookProcessor):
            _singleton_group = "context_enhancer_enhance"
            _order = 5

            def __init__(self):
                self.fired = False  # observable for test assertions

            async def on_task_start(self, event: TaskStartEvent) -> AsyncIterator:
                self.fired = True
                new_system = (event.system_prompt + "\\n\\n" + PREAMBLE).strip()
                yield dataclasses.replace(event, system_prompt=new_system)
        """),
        encoding="utf-8",
    )


def _patch_manifest_for_real_processor(plugin_dir: Path) -> None:
    """Update plugin.json to point at the real processor target."""
    manifest_path = plugin_dir / "plugin.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["processors"] = [{"target": "processors.enhance_processor.EnhanceProcessor"}]
    manifest["slash_commands"] = [
        {"command": "/enhance", "slot": "_context_enhancer_enhance"},
        {"command": "/summarise", "slot": "_context_enhancer_summarise"},
    ]

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_minimal_config(extra_processors=None):
    return HarnessConfig(
        tool_registry=make_registry(),
        tracer=NullTracer(),
        processors=extra_processors or {},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Part A: Local fixture tests (always run)
# ══════════════════════════════════════════════════════════════════════════════

# ── Test 1: Claude Code plugin (.claude-plugin/ layout) → convert ─────────────


def test_convert_claude_code_plugin_produces_valid_manifest(tmp_path):
    """Convert a Claude Code native plugin (.claude-plugin/ layout), verify manifest."""
    src = make_claude_code_plugin(tmp_path / "src")
    dst = tmp_path / "converted"

    result = convert_claude_plugin(src, dst)

    assert result == dst, "convert_claude_plugin should return the destination path"
    assert (dst / "plugin.json").exists()

    with open(dst / "plugin.json") as f:
        manifest = json.load(f)

    # Standard Claude Code fields preserved
    assert manifest["name"] == "context-enhancer"
    assert manifest["version"] == "0.1.0"
    assert len(manifest["commands"]) == 2
    # Commands collected from commands/*.md with frontmatter parsing
    cmd_names = [c["name"] for c in manifest["commands"]]
    assert "enhance" in cmd_names
    assert "summarise" in cmd_names
    # description from frontmatter
    enhance_cmd = next(c for c in manifest["commands"] if c["name"] == "enhance")
    assert enhance_cmd["description"] == "Restate the goal before answering"
    assert enhance_cmd.get("argument_hint") == "your question here"

    # HarnessX extension sections added
    assert "processors" in manifest
    assert "slash_commands" in manifest

    # Slash commands derived from commands
    slash_names = [e.get("command") for e in manifest["slash_commands"]]
    assert "/enhance" in slash_names
    assert "/summarise" in slash_names


def test_converted_plugin_has_processor_stubs(tmp_path):
    """Verify processor stub files are generated for each command."""
    src = make_claude_code_plugin(tmp_path / "src")
    dst = tmp_path / "converted"
    convert_claude_plugin(src, dst)

    assert (dst / "processors").is_dir()
    assert (dst / "processors" / "enhance_processor.py").exists()
    assert (dst / "processors" / "summarise_processor.py").exists()

    stub = (dst / "processors" / "enhance_processor.py").read_text()
    assert "class EnhanceProcessor" in stub
    assert "on_task_start" in stub
    assert "MultiHookProcessor" in stub


def test_converted_plugin_prompt_files_copied(tmp_path):
    """Verify command prompt .md files are copied to the output."""
    src = make_claude_code_plugin(tmp_path / "src")
    dst = tmp_path / "converted"
    convert_claude_plugin(src, dst)

    assert (dst / "commands" / "enhance.md").exists()
    assert (dst / "commands" / "summarise.md").exists()
    content = (dst / "commands" / "enhance.md").read_text()
    assert "$ARGUMENTS" in content


def test_converted_plugin_frontmatter_stripped_from_prompt(tmp_path):
    """Converted command prompt should have frontmatter stripped (only body)."""
    src = make_claude_code_plugin(tmp_path / "src")
    dst = tmp_path / "converted"
    convert_claude_plugin(src, dst)

    with open(dst / "plugin.json") as f:
        manifest = json.load(f)

    enhance_cmd = next(c for c in manifest["commands"] if c["name"] == "enhance")
    # Body should not start with ---
    assert not enhance_cmd["prompt"].startswith("---")
    assert "context-aware assistant" in enhance_cmd["prompt"]


# ── Test 2: load converted plugin ─────────────────────────────────────────────


def test_load_converted_plugin_without_processor_targets(tmp_path):
    """A converted plugin with stubs patched away loads cleanly."""
    src = make_claude_code_plugin(tmp_path / "src")
    dst = tmp_path / "converted"
    convert_claude_plugin(src, dst)

    # Stub targets are not importable — patch them away
    with open(dst / "plugin.json") as f:
        manifest = json.load(f)
    manifest["processors"] = []
    with open(dst / "plugin.json", "w") as f:
        json.dump(manifest, f, indent=2)

    plugin = load_from_directory(dst)

    assert plugin.name == "context-enhancer"
    assert plugin.version == "0.1.0"
    assert len(plugin.commands) == 2


def test_load_converted_plugin_with_real_processor(tmp_path):
    """After writing a real processor, the plugin loads it correctly."""
    src = make_claude_code_plugin(tmp_path / "src")
    dst = tmp_path / "converted"
    convert_claude_plugin(src, dst)

    # Replace stub with working implementation
    _write_real_processor(dst)
    _patch_manifest_for_real_processor(dst)

    sys.path.insert(0, str(dst))
    try:
        plugin = load_from_directory(dst)
    finally:
        sys.path.remove(str(dst))

    assert plugin.name == "context-enhancer"
    assert len(plugin.processors) == 1
    proc = plugin.processors[0]
    assert type(proc).__name__ == "EnhanceProcessor"
    assert isinstance(proc, MultiHookProcessor)


# ── Test 3: plugin processor modifies TaskStartEvent during run ────────────────


@pytest.mark.asyncio
async def test_plugin_processor_injects_system_prompt(tmp_path):
    """Plugin's EnhanceProcessor modifies system_prompt at task_start."""
    src = make_claude_code_plugin(tmp_path / "src")
    dst = tmp_path / "converted"
    convert_claude_plugin(src, dst)
    _write_real_processor(dst)
    _patch_manifest_for_real_processor(dst)

    sys.path.insert(0, str(dst))
    try:
        plugin = load_from_directory(dst)
    finally:
        sys.path.remove(str(dst))

    proc = plugin.processors[0]

    config = (
        HarnessBuilder()
        .slot(
            tool_registry=make_registry(),
            tracer=NullTracer(),
        )
        .plugin(plugin)
        .build()
    )

    harness = ModelConfig(main=MockProvider(responses=["Done."])).agentic(config)
    result = await harness.run(BaseTask("What is 2 + 2?"))

    assert proc.fired, "EnhanceProcessor.on_task_start should have been called"
    assert result.exit_reason in ("done", "budget_exceeded")


# ── Test 4: plugin discovery (workspace) ──────────────────────────────────────


def test_discover_plugin_in_workspace(tmp_path):
    """discover_plugins() finds a converted plugin in the workspace dir."""
    src = make_claude_code_plugin(tmp_path / "src")
    dst_base = tmp_path / "converted"
    convert_claude_plugin(src, dst_base)

    # Patch processor targets away (avoid import errors)
    with open(dst_base / "plugin.json") as f:
        manifest = json.load(f)
    manifest["processors"] = []
    with open(dst_base / "plugin.json", "w") as f:
        json.dump(manifest, f)

    # Install into workspace plugin directory
    ws_plugins_dir = tmp_path / "workspace" / ".harnessx" / "plugins"
    ws_plugins_dir.mkdir(parents=True)
    import shutil

    shutil.copytree(dst_base, ws_plugins_dir / "context-enhancer")

    plugins = discover_plugins(workspace_root=tmp_path / "workspace")
    names = [p.name for p in plugins]
    assert "context-enhancer" in names, f"Expected 'context-enhancer' in {names}"


# ── Test 5: registry + slash command dispatch ─────────────────────────────────


def test_registry_dispatches_slash_from_converted_plugin(tmp_path):
    """Slash commands from the converted plugin are dispatchable.

    /enhance has a prompt body → dispatched as prompt-injection command
    (stores _pending_command_prompt on harness, not a slot flag).
    """
    src = make_claude_code_plugin(tmp_path / "src")
    dst = tmp_path / "converted"
    convert_claude_plugin(src, dst)

    # Patch manifest: no processor targets
    with open(dst / "plugin.json") as f:
        manifest = json.load(f)
    manifest["processors"] = []
    # keep slash_commands and commands as converted (commands have prompts)
    with open(dst / "plugin.json", "w") as f:
        json.dump(manifest, f)

    plugin = load_from_directory(dst)

    reg = PluginRegistry()
    reg.register(plugin)

    harness_mock = type("FakeHarness", (), {})()

    handled = reg.dispatch_slash("/enhance topic here", "sid", harness_mock)
    assert handled is True

    # /enhance has a prompt → stored as _pending_command_prompt (prompt injection)
    pending_prompt = getattr(harness_mock, "_pending_command_prompt", None)
    assert pending_prompt is not None, (
        "Expected _pending_command_prompt to be set for /enhance (prompt-injection command)"
    )
    assert "topic here" in pending_prompt or "context-aware" in pending_prompt


# ══════════════════════════════════════════════════════════════════════════════
# Part B: Real installed Claude Code plugins
# ══════════════════════════════════════════════════════════════════════════════


@requires_real_plugins
def test_discover_real_claude_plugins():
    """discover_claude_plugins() finds all plugins from installed_plugins.json."""
    plugins = discover_claude_plugins()
    assert len(plugins) > 0, "Expected at least one plugin from ~/.claude/plugins/installed_plugins.json"
    for p in plugins:
        assert p.name, f"Plugin loaded from real install has empty name: {p}"


@requires_real_plugins
def test_real_claude_plugin_loads_name_and_commands():
    """Each installed Claude Code plugin loads with a non-empty name."""
    for plugin_dir in _INSTALLED_PLUGIN_DIRS:
        plugin = load_from_directory(plugin_dir)
        assert plugin.name, f"Plugin in {plugin_dir} has no name"
        # Commands should be a list (may be empty if no commands/ dir)
        assert isinstance(plugin.commands, list)


@requires_real_plugins
def test_real_claude_plugin_commands_have_prompts():
    """Commands loaded from real plugins have non-empty prompt bodies."""
    has_commands = False
    for plugin_dir in _INSTALLED_PLUGIN_DIRS:
        plugin = load_from_directory(plugin_dir)
        for cmd in plugin.commands:
            has_commands = True
            assert cmd.get("name"), f"Command missing name in {plugin_dir}: {cmd}"
            # Prompt should be present and non-trivial if the .md file has content
            assert "prompt" in cmd, f"Command missing 'prompt' field: {cmd}"
    if not has_commands:
        pytest.skip("Installed plugins have no commands/*.md files to check")


@requires_real_plugins
def test_convert_real_claude_plugin(tmp_path):
    """A real installed Claude Code plugin can be converted to HarnessX format."""
    # Use the first available plugin
    plugin_dir = _INSTALLED_PLUGIN_DIRS[0]

    dst = tmp_path / "converted"
    result = convert_claude_plugin(plugin_dir, dst)
    assert result == dst

    with open(dst / "plugin.json", encoding="utf-8") as f:
        manifest = json.load(f)

    # Original name is preserved
    original = load_from_directory(plugin_dir)
    assert manifest["name"] == original.name

    # HarnessX extension sections are present
    assert "processors" in manifest
    assert "slash_commands" in manifest

    # The converted directory is loadable
    conv_manifest = dict(manifest)
    conv_manifest["processors"] = []  # remove non-importable stubs
    with open(dst / "plugin.json", "w", encoding="utf-8") as f:
        json.dump(conv_manifest, f, indent=2)
    loaded = load_from_directory(dst)
    assert loaded.name == original.name


@requires_real_plugins
def test_discover_plugins_includes_claude_installs():
    """discover_plugins(include_claude_plugins=True) returns Claude installs."""
    plugins = discover_plugins(include_claude_plugins=True)
    names = {p.name for p in plugins}

    # All individually discovered plugins should appear in the combined list
    for individual in discover_claude_plugins():
        assert individual.name in names, f"discover_plugins() missed Claude plugin '{individual.name}'"


@requires_real_plugins
def test_ralph_loop_plugin_loaded_correctly():
    """Specifically tests ralph-loop (known to be installed) loads correctly."""
    ralph_dirs = [d for d in _INSTALLED_PLUGIN_DIRS if "ralph-loop" in str(d)]
    if not ralph_dirs:
        pytest.skip("ralph-loop not in installed plugins list")

    plugin = load_from_directory(ralph_dirs[0])
    assert plugin.name == "ralph-loop"
    assert plugin.version is not None and len(plugin.version) > 0

    # Should have commands from commands/*.md
    assert len(plugin.commands) > 0, "ralph-loop should have at least one command"
    cmd_names = [c["name"] for c in plugin.commands]
    assert "ralph-loop" in cmd_names, f"Expected 'ralph-loop' command, got {cmd_names}"

    # The ralph-loop command should have a non-trivial prompt
    ralph_cmd = next(c for c in plugin.commands if c["name"] == "ralph-loop")
    assert len(ralph_cmd["prompt"]) > 50, "ralph-loop command prompt is unexpectedly short"

    # allowed_tools should have been parsed from frontmatter
    assert "allowed_tools" in ralph_cmd, "ralph-loop command should have allowed_tools from frontmatter"


# ── Pytest entry point ────────────────────────────────────────────────────────


async def main() -> int:
    """Script-mode runner for quick manual verification."""
    import traceback as _tb

    print("E2E Plugin Test — Claude Code plugin → convert → install → run")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0
    tmp = Path("/tmp/oh_plugin_e2e_test")
    tmp.mkdir(exist_ok=True)

    sync_tests: list[tuple[str, object]] = [
        (
            "convert .claude-plugin/ → valid manifest",
            lambda: test_convert_claude_code_plugin_produces_valid_manifest(tmp / "t1"),
        ),
        (
            "convert → processor stubs",
            lambda: test_converted_plugin_has_processor_stubs(tmp / "t2"),
        ),
        (
            "convert → prompt files copied",
            lambda: test_converted_plugin_prompt_files_copied(tmp / "t3"),
        ),
        (
            "convert → frontmatter stripped from prompt",
            lambda: test_converted_plugin_frontmatter_stripped_from_prompt(tmp / "t4"),
        ),
        (
            "load converted (no targets)",
            lambda: test_load_converted_plugin_without_processor_targets(tmp / "t5"),
        ),
        (
            "load converted (real processor)",
            lambda: test_load_converted_plugin_with_real_processor(tmp / "t6"),
        ),
        (
            "discover plugin in workspace",
            lambda: test_discover_plugin_in_workspace(tmp / "t8"),
        ),
        (
            "slash dispatch from converted plugin",
            lambda: test_registry_dispatches_slash_from_converted_plugin(tmp / "t9"),
        ),
    ]

    async_tests: list[tuple[str, object]] = [
        (
            "processor modifies system_prompt",
            lambda: test_plugin_processor_injects_system_prompt(tmp / "t7"),
        ),
    ]

    real_tests: list[tuple[str, object]] = [
        ("discover real Claude plugins", lambda: discover_claude_plugins()),
        (
            "real plugin loads name+commands",
            lambda: test_real_claude_plugin_loads_name_and_commands(),
        ),
        (
            "real plugin commands have prompts",
            lambda: test_real_claude_plugin_commands_have_prompts(),
        ),
        (
            "convert real Claude plugin",
            lambda: test_convert_real_claude_plugin(tmp / "t_real_convert"),
        ),
        (
            "discover_plugins includes Claude installs",
            lambda: test_discover_plugins_includes_claude_installs(),
        ),
        (
            "ralph-loop plugin loaded correctly",
            lambda: test_ralph_loop_plugin_loaded_correctly(),
        ),
    ]

    for name, fn in sync_tests:
        (tmp / name.replace(" ", "_")[:20]).mkdir(parents=True, exist_ok=True)
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            _tb.print_exc()
            failed += 1

    for name, fn in async_tests:
        try:
            await fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            _tb.print_exc()
            failed += 1

    print()
    if _HAS_REAL_PLUGINS:
        print(f"Real Claude plugins found: {[str(d) for d in _INSTALLED_PLUGIN_DIRS]}")
        for name, fn in real_tests:
            try:
                result = fn()
                if asyncio.iscoroutine(result):
                    await result
                print(f"  ✓ [real] {name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ [real] {name}: {e}")
                _tb.print_exc()
                failed += 1
    else:
        print("  (skipping real plugin tests — no Claude Code plugins installed)")
        skipped += len(real_tests)

    print(f"\n{'PASS' if not failed else 'FAIL'}  ({passed} passed, {failed} failed, {skipped} skipped)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
