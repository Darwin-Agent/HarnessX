# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Multi-ship composer — merges bucket-disjoint shipped candidates.

Each candidate's applied YAML is a FULL config derived from the round's
base + its one-bucket change. When multiple ship, compose_shipped_configs
overlays each candidate's bucket-specific fields onto the base so all
changes survive.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from harnessx.aegis.compose import compose_shipped_configs


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _base_config() -> dict:
    return {
        "tool_registry": {
            "builtin": ["Read", "WebSearch"],
            "custom": [],
        },
        "processors": [
            {
                "_target_": "harnessx.processors.context.system_prompt.SystemPromptProcessor",
                "system_builder": {
                    "_target_": "harnessx.processors.context.strategies.system_prompt.template.TemplateSystemPromptBuilder",
                    "template_path": "/orig/template.j2",
                },
            },
            {
                "_target_": "harnessx.processors.memory.strategies.sliding_window.SlidingWindowMemory",
                "n": 20,
            },
        ],
    }


def test_compose_prompt_plus_tools_disjoint_stick(tmp_path):
    """Prompt candidate changes template_path; tools candidate adds a tool.
    Both changes must survive in the merged output."""
    base = tmp_path / "base.yaml"
    _write(base, _base_config())

    prompt_cfg = _base_config()
    prompt_cfg["processors"][0]["system_builder"]["template_path"] = str(
        tmp_path / "C-R1-01" / "new.j2"
    )
    prompt_path = tmp_path / "C-R1-01" / "config.yaml"
    _write(prompt_path, prompt_cfg)

    tools_cfg = _base_config()
    tools_cfg["tool_registry"]["custom"] = [
        f"file://{tmp_path}/C-R1-02/pdf_tool.py::pdf_tool"
    ]
    tools_path = tmp_path / "C-R1-02" / "config.yaml"
    _write(tools_path, tools_cfg)

    out = tmp_path / "merged.yaml"
    compose_shipped_configs(
        base, shipped=[
            ("C-R1-01", "prompt", prompt_path),
            ("C-R1-02", "tools", tools_path),
        ],
        output_path=out,
    )

    merged = yaml.safe_load(out.read_text())
    # Prompt change survives.
    assert merged["processors"][0]["system_builder"]["template_path"] == str(
        tmp_path / "C-R1-01" / "new.j2"
    )
    # Tools change survives.
    assert any(
        "pdf_tool" in str(entry)
        for entry in merged["tool_registry"]["custom"]
    )
    # Unchanged fields preserved.
    assert merged["processors"][1]["n"] == 20


def test_compose_config_changes_kwargs(tmp_path):
    """Config candidate tweaks an existing processor's kwargs."""
    base = tmp_path / "base.yaml"
    _write(base, _base_config())

    cfg = _base_config()
    cfg["processors"][1]["n"] = 50
    cand_path = tmp_path / "C-R1-01" / "config.yaml"
    _write(cand_path, cfg)

    out = tmp_path / "merged.yaml"
    compose_shipped_configs(
        base, shipped=[("C-R1-01", "config", cand_path)], output_path=out,
    )
    merged = yaml.safe_load(out.read_text())
    assert merged["processors"][1]["n"] == 50


def test_compose_processor_appends_new_entry(tmp_path):
    """Processor candidate adds a new processor; merged config gets it appended."""
    base = tmp_path / "base.yaml"
    _write(base, _base_config())

    cfg = _base_config()
    cfg["processors"].append({
        "_target_": "file:///abs/my_processor.py::MyProcessor",
        "threshold": 3,
    })
    cand_path = tmp_path / "C-R1-01" / "config.yaml"
    _write(cand_path, cfg)

    out = tmp_path / "merged.yaml"
    compose_shipped_configs(
        base, shipped=[("C-R1-01", "processor", cand_path)], output_path=out,
    )
    merged = yaml.safe_load(out.read_text())
    # Original processors still there.
    targets = [p["_target_"] for p in merged["processors"]]
    assert any("SystemPromptProcessor" in t for t in targets)
    # New processor appended.
    assert any("MyProcessor" in t for t in targets)


def test_compose_ignores_unknown_bucket(tmp_path):
    """Unknown bucket → skip, don't break the merge."""
    base = tmp_path / "base.yaml"
    _write(base, _base_config())
    cand_path = tmp_path / "C-R1-01" / "config.yaml"
    _write(cand_path, _base_config())
    out = tmp_path / "merged.yaml"
    compose_shipped_configs(
        base, shipped=[("C-R1-01", "xyz", cand_path)], output_path=out,
    )
    merged = yaml.safe_load(out.read_text())
    assert merged == _base_config()


def test_compose_empty_shipped_returns_base(tmp_path):
    base = tmp_path / "base.yaml"
    _write(base, _base_config())
    out = tmp_path / "merged.yaml"
    compose_shipped_configs(base, shipped=[], output_path=out)
    merged = yaml.safe_load(out.read_text())
    assert merged == _base_config()


def test_compose_processor_replaces_existing_entry(tmp_path):
    """A candidate that drops parent's v1 processor and adds v2 should
    produce a merged config with ONLY v2 — not v1 alongside v2.

    Regression test for the aegis_64_v3 R2 bug where BudgetAwareCommit v1 and
    v2 both landed in merged.yaml because _apply_processor was append-only.
    """
    base_cfg = _base_config()
    base_cfg["processors"].append({
        "_target_": "file:///abs/budget_commit.py::BudgetAwareCommitProcessor",
        "trigger_remaining": 2,
        "max_nudges_per_run": 1,
    })
    base = tmp_path / "base.yaml"
    _write(base, base_cfg)

    # Candidate's config: parent minus v1, plus v2.
    cand_cfg = _base_config()
    cand_cfg["processors"].append({
        "_target_": "file:///abs/budget_commit_v2.py::BudgetAwareCommitProcessorV2",
        "trigger_remainings": [8, 4, 2],
        "max_nudges_per_run": 2,
    })
    cand_path = tmp_path / "C-R2-01" / "config.yaml"
    _write(cand_path, cand_cfg)

    out = tmp_path / "merged.yaml"
    compose_shipped_configs(
        base, shipped=[("C-R2-01", "processor", cand_path)], output_path=out,
    )
    merged = yaml.safe_load(out.read_text())
    targets = [p["_target_"] for p in merged["processors"]]

    # v1 must be gone.
    assert not any("budget_commit.py::BudgetAwareCommitProcessor" in t for t in targets), (
        f"v1 should have been removed but is still present: {targets}"
    )
    # v2 must be present.
    assert any("budget_commit_v2.py::BudgetAwareCommitProcessorV2" in t for t in targets), (
        f"v2 should have been added but is missing: {targets}"
    )
    # Untouched processors still there.
    assert any("SystemPromptProcessor" in t for t in targets)
    assert any("SlidingWindowMemory" in t for t in targets)


def test_compose_tools_drops_removed_entry(tmp_path):
    """A tools candidate that drops a parent custom tool and adds a new one
    must produce merged.yaml with ONLY the new tool, not both."""
    base_cfg = _base_config()
    base_cfg["tool_registry"]["custom"] = [
        "file:///abs/old_tool.py::old_tool",
    ]
    base = tmp_path / "base.yaml"
    _write(base, base_cfg)

    cand_cfg = _base_config()
    cand_cfg["tool_registry"]["custom"] = [
        "file:///abs/new_tool.py::new_tool",
    ]
    cand_path = tmp_path / "C-R2-02" / "config.yaml"
    _write(cand_path, cand_cfg)

    out = tmp_path / "merged.yaml"
    compose_shipped_configs(
        base, shipped=[("C-R2-02", "tools", cand_path)], output_path=out,
    )
    merged = yaml.safe_load(out.read_text())
    custom = merged["tool_registry"]["custom"]
    assert "file:///abs/old_tool.py::old_tool" not in custom
    assert "file:///abs/new_tool.py::new_tool" in custom


def test_compose_multi_ship_parent_snapshot_isolates_diffs(tmp_path):
    """When a prompt candidate and a processor candidate ship together, each
    applier must diff against the ORIGINAL parent, not against base mutated
    by the previous applier. Otherwise the processor-bucket diff would see
    phantom removals from the prompt applier's changes."""
    base_cfg = _base_config()
    base_cfg["processors"].append({
        "_target_": "file:///abs/v1.py::V1",
        "threshold": 1,
    })
    base = tmp_path / "base.yaml"
    _write(base, base_cfg)

    # Prompt candidate: changes template_path, keeps all processors.
    prompt_cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy via yaml
    prompt_cfg["processors"][0]["system_builder"]["template_path"] = "/new.j2"
    prompt_path = tmp_path / "C-R2-01" / "config.yaml"
    _write(prompt_path, prompt_cfg)

    # Processor candidate: drops V1, adds V2. Uses parent as baseline.
    proc_cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    proc_cfg["processors"] = [
        p for p in proc_cfg["processors"]
        if p.get("_target_") != "file:///abs/v1.py::V1"
    ]
    proc_cfg["processors"].append({
        "_target_": "file:///abs/v2.py::V2",
        "threshold": 2,
    })
    proc_path = tmp_path / "C-R2-02" / "config.yaml"
    _write(proc_path, proc_cfg)

    out = tmp_path / "merged.yaml"
    compose_shipped_configs(
        base,
        shipped=[
            ("C-R2-01", "prompt", prompt_path),
            ("C-R2-02", "processor", proc_path),
        ],
        output_path=out,
    )
    merged = yaml.safe_load(out.read_text())
    targets = [p["_target_"] for p in merged["processors"]]

    # Prompt change survives.
    assert merged["processors"][0]["system_builder"]["template_path"] == "/new.j2"
    # Processor diff: V1 removed, V2 added.
    assert not any("v1.py::V1" in t for t in targets)
    assert any("v2.py::V2" in t for t in targets)
