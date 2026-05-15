# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""End-to-end test that `processors: [{_target_: file:///abs/path.py::Class}]`
in a YAML HarnessConfig round-trips and produces a runtime instance.

AEGIS Evolvers previously halluclinated that this form would raise
ModuleNotFoundError. This test is the single-pointer ground truth
referenced from harnessx/aegis/templates/{evolver,critic}.j2.
"""
from pathlib import Path
import tempfile


def test_file_uri_processor_yaml_roundtrip_to_runtime(tmp_path: Path) -> None:
    """Load a HarnessConfig YAML containing a file:// processor target,
    then canonicalize + _instantiate_runtime to get a live instance.
    Kwargs must be forwarded to __init__.
    """
    from harnessx.core.harness import HarnessConfig, _instantiate_runtime
    from harnessx.core.builder import HarnessBuilder

    proc_py = tmp_path / "evolved_proc.py"
    proc_py.write_text(
        (
            "from harnessx.core.processor import MultiHookProcessor\n"
            "class EvolvedProc(MultiHookProcessor):\n"
            "    def __init__(self, threshold: int = 42, label: str = 'default'):\n"
            "        self.threshold = threshold\n"
            "        self.label = label\n"
            "    async def on_before_model(self, event):\n"
            "        yield event\n"
        ),
        encoding="utf-8",
    )

    baseline_yaml = HarnessBuilder().build().to_yaml()
    yaml_with_proc = baseline_yaml.replace(
        "processors: []",
        (
            f"processors:\n"
            f"- _target_: file://{proc_py}::EvolvedProc\n"
            f"  threshold: 99\n"
            f"  label: 'from_yaml'\n"
        ),
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml_with_proc, encoding="utf-8")

    loaded = HarnessConfig.from_yaml_file(cfg_path)
    # processors list survives as dict-target form
    assert any(
        isinstance(p, dict) and p.get("_target_", "").startswith("file://")
        for p in (loaded.processors or [])
    )

    rt = _instantiate_runtime(loaded).processors
    procs = rt.get("*", [])
    assert len(procs) == 1, f"expected 1 runtime proc, got {len(procs)}"
    proc = procs[0]
    assert type(proc).__name__ == "EvolvedProc"
    assert proc.threshold == 99
    assert proc.label == "from_yaml"
