# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from harnessx.api.custom_processors import direct_targets as reg
from harnessx.api.routes import schema as schema_route
from harnessx.core.harness import HarnessConfig


def _patch_registry_paths(monkeypatch, base: Path) -> None:
    monkeypatch.setattr(reg, "_BASE_DIR", base / "custom_processors")
    monkeypatch.setattr(reg, "_IMPORTED_DIR", (base / "custom_processors" / "imported"))


class TestCustomProcessorRegistry:
    def test_scan_text_detects_multihook_class(self) -> None:
        content = (
            "from harnessx.core.processor import MultiHookProcessor\n\nclass DemoProc(MultiHookProcessor):\n    pass\n"
        )
        out = reg.scan_text_for_processors("demo.py", content)
        assert len(out) == 1
        assert out[0]["class_name"] == "DemoProc"

    def test_import_and_dimension(self, monkeypatch, tmp_path: Path) -> None:
        _patch_registry_paths(monkeypatch, tmp_path)

        src = tmp_path / "demo_proc.py"
        src.write_text(
            (
                "from harnessx.core.processor import MultiHookProcessor\n"
                "class DemoImportedProc(MultiHookProcessor):\n"
                "    def __init__(self, level:int=1):\n"
                "        self.level = level\n"
            ),
            encoding="utf-8",
        )

        test_res = reg.test_processor_from_path(str(src), "DemoImportedProc")
        assert test_res["ok"] is True

        entry = reg.import_processor_from_path(str(src), "DemoImportedProc", label="Demo Imported")
        assert entry["target"].startswith("file://")
        assert "::DemoImportedProc" in entry["target"]

        listed = reg.list_custom_processors()
        assert any(e["target"] == entry["target"] for e in listed)

        dim = reg.build_custom_dimension()
        assert dim is not None
        assert dim["key"] == "custom_processors"
        assert len(dim["options"]) >= 1

    def test_schema_route_appends_custom_dimension(self, monkeypatch, tmp_path: Path) -> None:
        _patch_registry_paths(monkeypatch, tmp_path)

        src = tmp_path / "demo_schema_proc.py"
        src.write_text(
            (
                "from harnessx.core.processor import MultiHookProcessor\n"
                "class DemoSchemaProc(MultiHookProcessor):\n"
                "    pass\n"
            ),
            encoding="utf-8",
        )
        reg.import_processor_from_path(str(src), "DemoSchemaProc", label="Schema Proc")

        data = schema_route.get_schema()
        dims = data["dimensions"]
        assert any(d.get("key") == "custom_processors" for d in dims)

    def test_harness_config_supports_file_target_without_init_py(self, tmp_path: Path) -> None:
        src = tmp_path / "my_proc.py"
        src.write_text(
            (
                "from harnessx.core.processor import MultiHookProcessor\n"
                "class DemoFileProc(MultiHookProcessor):\n"
                "    def __init__(self, x:int=7):\n"
                "        self.x = x\n"
            ),
            encoding="utf-8",
        )

        target = reg.make_file_target(src, "DemoFileProc")
        cfg = HarnessConfig(processors=[{"_target_": target, "x": 11}])
        from harnessx.core.harness import _instantiate_runtime

        rt_procs = _instantiate_runtime(cfg).processors
        procs = rt_procs.get("*", [])
        assert len(procs) == 1
        proc = procs[0]
        assert type(proc).__name__ == "DemoFileProc"
        assert getattr(proc, "x", None) == 11
