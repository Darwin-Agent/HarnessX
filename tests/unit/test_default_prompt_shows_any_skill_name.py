# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Regression test: DefaultSystemPromptBuilder no longer hides meta-* skills.

Historical behavior filtered out any skill whose name started with "meta-"
because the old meta_harness used that prefix to mark skills meant only for
the plan agent. The new meta_harness treats skills uniformly — no more
prefix-based hiding. This test pins that behavior.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from harnessx.processors.context.strategies.system_prompt.default import (
    DefaultSystemPromptBuilder,
)
from harnessx.workspace.workspace import Workspace


def _make_skill(root: Path, name: str, description: str = "Test skill") -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nBody.\n",
        encoding="utf-8",
    )


def test_skill_named_with_meta_prefix_is_visible(tmp_path: Path) -> None:
    skills_dir = tmp_path / "workspace" / "skills"
    skills_dir.mkdir(parents=True)
    _make_skill(skills_dir, "meta-example", "A skill named with the meta prefix.")
    _make_skill(skills_dir, "normal-example", "A normal skill.")

    ws = Workspace(root=tmp_path / "workspace", agent_id="test", mode="shared")
    builder = DefaultSystemPromptBuilder(max_skills_shown=10)
    prompt = asyncio.run(builder.build(workspace=ws))

    assert "<available_skills>" in prompt
    assert "meta-example" in prompt, "meta-* skills should be visible in the prompt after the filter removal"
    assert "normal-example" in prompt


def test_extra_skills_dirs_also_show_meta_prefix(tmp_path: Path) -> None:
    extra = tmp_path / "extra_skills"
    extra.mkdir(parents=True)
    _make_skill(extra, "meta-outside", "A skill under extra_skills_dirs.")

    # No workspace-level skills dir — only extra_skills_dirs provides skills.
    ws_root = tmp_path / "workspace"
    ws_root.mkdir()
    ws = Workspace(root=ws_root, agent_id="test", mode="shared")

    builder = DefaultSystemPromptBuilder(
        max_skills_shown=10,
        extra_skills_dirs=[extra],
    )
    prompt = asyncio.run(builder.build(workspace=ws))

    assert "meta-outside" in prompt
