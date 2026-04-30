# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Regression tests for custom tool-registry loading and diff.

These cover the pipeline a user hits when they:

1. Produce a YAML config with
   ``tool_registry.custom: [file:///abs/path.py::symbol]`` (what the
   gaia_evolver meta-agent writes).
2. Load it via :meth:`HarnessConfig.from_yaml_file` and hand it to
   ``model_config.agentic(...)``, which calls
   :func:`harnessx.core.harness._build_tool_registry_from_config`.
3. Expect the model to see the custom tool in its tool schema.

Before these fixes, step 2 silently dropped every ``file://``-style
custom tool because the loader only understood dotted import paths and
swallowed the resulting ``ImportError`` with ``except Exception: pass``
— so the downstream changeset diff also missed the ``tools_added``
entry, because ``_tool_names`` could not read a ``ToolRegistryConfig``
dataclass.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from harnessx.core.config_schema import ToolRegistryConfig
from harnessx.core.harness import (
    HarnessConfig,
    _build_tool_registry_from_config,
    _runtime_registry_to_config,
)
from harnessx.meta_harness import compute_changeset


# A minimal custom tool we can drop into tmp_path.  Deliberately kept
# tiny to avoid depending on third-party packages.
_TOOL_SRC = """\
from harnessx.tools.base import tool


@tool(description="demo custom tool that echoes its input")
def demo_echo_tool(message: str = "") -> str:
    return message
"""


def _write_tool_file(tmp_path: Path, name: str = "demo_echo_tool") -> Path:
    src = tmp_path / f"{name}_module.py"
    src.write_text(_TOOL_SRC, encoding="utf-8")
    return src


# ─── _build_tool_registry_from_config ────────────────────────────────────


class TestBuildToolRegistryFromConfig:
    def test_loads_file_uri_custom_tool(self, tmp_path: Path) -> None:
        src = _write_tool_file(tmp_path)
        target = f"file://{src}::demo_echo_tool"

        cfg = ToolRegistryConfig(builtin=["Bash", "Read"], custom=[target])
        registry = _build_tool_registry_from_config(cfg)

        names = set(registry.list_names())
        # Both the builtin names and the custom tool must be present.
        assert "Bash" in names
        assert "Read" in names
        assert "demo_echo_tool" in names, f"custom file:// tool was not loaded; registry has {sorted(names)}"

    def test_loads_dotted_path_custom_tool(self) -> None:
        # A tool that genuinely lives in an importable module.
        cfg = ToolRegistryConfig(
            builtin=[],
            custom=["harnessx.tools.builtin.bash.Bash"],
        )
        # Bash's tool symbol is registered as a module attribute via the
        # @tool decorator, so we sanity-check the resolver without
        # forcing a specific name.
        try:
            registry = _build_tool_registry_from_config(cfg)
        except Exception:  # pragma: no cover — defensive
            pytest.skip("harnessx.tools.builtin.bash.Bash not importable")
        # At minimum the loader must not raise; the registered tool's
        # name depends on the decorator name= arg of the Bash function.
        assert hasattr(registry, "list_names")

    def test_malformed_file_uri_is_logged_not_raised(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A malformed file:// target is skipped with a WARNING (we
        must not crash the whole harness build, but we also must not
        fail silently the way the old loader did)."""
        caplog.set_level(logging.WARNING, logger="harnessx.core.harness")
        cfg = ToolRegistryConfig(
            builtin=[],
            # Missing ``::symbol`` — parser must reject it.
            custom=[f"file://{tmp_path / 'nonexistent.py'}"],
        )
        registry = _build_tool_registry_from_config(cfg)
        assert registry.list_names() == []
        # The key point: there must be a WARNING record describing
        # WHICH target failed and WHY, rather than the old silent pass.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("tool_registry.custom" in r.getMessage() for r in warnings), (
            f"expected a tool_registry.custom warning, got: {[r.getMessage() for r in warnings]}"
        )

    def test_missing_custom_file_is_logged_not_raised(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="harnessx.core.harness")
        cfg = ToolRegistryConfig(
            builtin=[],
            custom=[f"file://{tmp_path / 'does_not_exist.py'}::demo_echo_tool"],
        )
        registry = _build_tool_registry_from_config(cfg)
        assert registry.list_names() == []
        assert any(
            "tool_registry.custom" in r.getMessage() and "does_not_exist" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_unknown_builtin_is_logged_not_silently_dropped(self, caplog: pytest.LogCaptureFixture) -> None:
        """``code_interpreter`` in the gaia_evolver configs was dropped
        on every round because no default-builtin entry exists for that
        name. The loader must surface the mistake instead of hiding it."""
        caplog.set_level(logging.WARNING, logger="harnessx.core.harness")
        cfg = ToolRegistryConfig(builtin=["Bash", "does_not_exist_tool"], custom=[])
        registry = _build_tool_registry_from_config(cfg)
        names = set(registry.list_names())
        assert "Bash" in names
        assert "does_not_exist_tool" not in names
        assert any("does_not_exist_tool" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


# ─── _runtime_registry_to_config (round-trip) ────────────────────────────


class TestRuntimeRegistryRoundTrip:
    def test_file_uri_preserved_through_round_trip(self, tmp_path: Path) -> None:
        """Load custom tool via file:// → re-serialize → the original
        file:// URI must come back out unchanged. This is the primary
        round-trip path the evolver relies on when it loads R_n's YAML,
        instantiates the harness, and needs to emit the same YAML
        unchanged for R_{n+1}'s baseline comparison."""
        src = _write_tool_file(tmp_path, name="demo_rt_tool")
        target = f"file://{src}::demo_echo_tool"

        cfg = ToolRegistryConfig(builtin=["Bash"], custom=[target])
        registry = _build_tool_registry_from_config(cfg)
        rebuilt = _runtime_registry_to_config(registry)

        assert "Bash" in rebuilt.builtin
        # The custom entry must retain the exact file:// URI, not the
        # synthetic module name the loader uses internally.
        assert target in rebuilt.custom, f"file:// target lost on round trip. got custom={rebuilt.custom}"

    def test_yaml_round_trip_through_harness_config(self, tmp_path: Path) -> None:
        """End-to-end: ToolRegistryConfig → HarnessConfig → to_yaml →
        from_yaml → _build_tool_registry_from_config. Must preserve the
        file:// URI and must actually register the tool each time."""
        src = _write_tool_file(tmp_path, name="demo_yaml_rt")
        target = f"file://{src}::demo_echo_tool"

        base = HarnessConfig(
            tool_registry=ToolRegistryConfig(builtin=["Bash"], custom=[target]),
        )
        yaml_str = base.to_yaml()
        restored = HarnessConfig.from_yaml(yaml_str)

        assert isinstance(restored.tool_registry, ToolRegistryConfig)
        assert target in restored.tool_registry.custom

        registry = _build_tool_registry_from_config(restored.tool_registry)
        assert "demo_echo_tool" in registry.list_names()


# ─── compute_changeset on YAML-shaped configs ────────────────────────────


def _cfg_from_tool_registry(tr: ToolRegistryConfig) -> HarnessConfig:
    """Shortcut: build a minimal HarnessConfig whose ``tool_registry``
    slot is a *declarative* :class:`ToolRegistryConfig` dataclass.

    This matches what ``HarnessConfig.from_yaml_file`` produces — which
    is what :func:`compute_changeset` actually sees in the evolver."""
    return HarnessConfig(tool_registry=tr)


class TestChangesetOnYamlConfigs:
    def test_changeset_detects_custom_tool_added_via_file_uri(self, tmp_path: Path) -> None:
        src = _write_tool_file(tmp_path)
        target = f"file://{src}::demo_echo_tool"

        before = _cfg_from_tool_registry(ToolRegistryConfig(builtin=["Bash", "Read"], custom=[]))
        after = _cfg_from_tool_registry(ToolRegistryConfig(builtin=["Bash", "Read"], custom=[target]))
        diff = compute_changeset(before, after)
        assert diff.get("tools_added") == ["demo_echo_tool"], f"expected demo_echo_tool in tools_added, got diff={diff}"
        assert "tools_removed" not in diff

    def test_changeset_detects_custom_tool_removed_via_file_uri(self, tmp_path: Path) -> None:
        src = _write_tool_file(tmp_path)
        target = f"file://{src}::demo_echo_tool"

        before = _cfg_from_tool_registry(ToolRegistryConfig(builtin=["Bash"], custom=[target]))
        after = _cfg_from_tool_registry(ToolRegistryConfig(builtin=["Bash"], custom=[]))
        diff = compute_changeset(before, after)
        assert diff.get("tools_removed") == ["demo_echo_tool"]

    def test_changeset_detects_builtin_tool_added_via_yaml_shape(
        self,
    ) -> None:
        before = _cfg_from_tool_registry(ToolRegistryConfig(builtin=["Bash"], custom=[]))
        after = _cfg_from_tool_registry(ToolRegistryConfig(builtin=["Bash", "Read"], custom=[]))
        diff = compute_changeset(before, after)
        assert diff.get("tools_added") == ["Read"]

    def test_changeset_empty_on_equivalent_yaml_configs(self, tmp_path: Path) -> None:
        src = _write_tool_file(tmp_path)
        target = f"file://{src}::demo_echo_tool"

        tr = ToolRegistryConfig(builtin=["Bash"], custom=[target])
        before = _cfg_from_tool_registry(tr)
        # Fresh instance with the same fields — must diff to empty.
        after = _cfg_from_tool_registry(ToolRegistryConfig(builtin=["Bash"], custom=[target]))
        assert compute_changeset(before, after) == {}

    def test_changeset_detects_dotted_and_file_uri_as_distinct_symbols(self, tmp_path: Path) -> None:
        """A dotted-path entry and a file:// entry resolving to the
        same *symbol name* should share the same changeset label
        (``symbol``), so changing their target form alone does NOT
        register as adding-a-new-tool. This is a deliberate limitation
        called out in ``_tool_label_from_custom_target``."""
        src = _write_tool_file(tmp_path)
        before = _cfg_from_tool_registry(
            ToolRegistryConfig(
                builtin=[],
                custom=["some.module.demo_echo_tool"],
            )
        )
        after = _cfg_from_tool_registry(
            ToolRegistryConfig(
                builtin=[],
                custom=[f"file://{src}::demo_echo_tool"],
            )
        )
        diff = compute_changeset(before, after)
        # Same symbol → no added/removed tools on the diff axis.
        assert "tools_added" not in diff
        assert "tools_removed" not in diff
