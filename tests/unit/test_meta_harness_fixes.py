# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Regression tests for the meta_harness code-review fixes.

Covers:

- compute_attribution ``absent`` handling when ``appeared_*`` sets are
  provided (C2).
- build_context excludes ``absent`` from lever precision and surfaces
  ``regressed_unpredicted`` as a side-effects column (M6).
- validators.novelty.check_novelty raises when the latest journal entry
  re-uses a reverted hypothesis_id (M3).
- changeset.compute_changeset produces a stable dict of tool /
  processor / template diffs (M4).
- replay._compare_exit_reason flags downgrades, accepts upgrades (C1).
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from pathlib import Path

import pytest

from harnessx.meta_harness import compute_changeset, journal
from harnessx.meta_harness.agent import MetaAgent
from harnessx.meta_harness.replay import run_synthetic_task_smoke_gate
from harnessx.meta_harness.validate_workflow import StrictValidationError, check_novelty


# ─── C2: compute_attribution absent handling ──────────────────────────────


def test_attribution_absent_when_task_did_not_run_in_either_round():
    """A predicted task that appears in neither round is ``absent``
    under strict mode — previously it silently became ``still_F`` and
    poisoned lever precision."""
    attr = journal.compute_attribution(
        predicted=["task_ghost"],
        passed_now=set(),
        passed_before=set(),
        appeared_now={"task_a"},
        appeared_before={"task_a"},
    )
    assert attr == {"task_ghost": "absent"}


def test_attribution_absent_when_task_skipped_one_round():
    """Tasks that ran only on one side carry no delta signal."""
    attr = journal.compute_attribution(
        predicted=["task_a"],
        passed_now={"task_a"},
        passed_before=set(),
        appeared_now={"task_a"},
        appeared_before=set(),  # task_a did NOT run in R0
    )
    assert attr == {"task_a": "absent"}


def test_attribution_legacy_mode_still_F_when_unknown():
    """Without ``appeared_*`` args the caller gets the old behaviour —
    predicted but unseen defaults to still_F so existing recipes that
    don't track appearance keep working."""
    attr = journal.compute_attribution(
        predicted=["task_a"],
        passed_now=set(),
        passed_before=set(),
    )
    assert attr == {"task_a": "still_F"}


# ─── M6: precision denominator excludes absent + side-effects column ─────


def test_scoreboard_excludes_absent_from_precision(tmp_path: Path):
    jp = tmp_path / "j.md"
    journal.append_entry(
        jp,
        journal.JournalEntrySpec(
            round=0,
            label="x",
            hypothesis_id="h",
            levers=["action"],
            predicted_affected=["task_a", "task_ghost"],
            prose="",
        ),
    )
    # task_a flipped; task_ghost absent. Precision should be 1/1, not 1/2.
    journal.fill_gating(
        jp,
        0,
        "accepted",
        {"task_a": "flipped", "task_ghost": "absent"},
    )
    ctx = journal.build_context(jp, current_round=1, output_path=tmp_path / "CTX.md")
    assert ctx is not None
    text = ctx.read_text(encoding="utf-8")
    assert "| action | 1 | 1 | 0 | 1/1 |" in text


def test_scoreboard_shows_side_effects_count(tmp_path: Path):
    jp = tmp_path / "j.md"
    journal.append_entry(
        jp,
        journal.JournalEntrySpec(
            round=0,
            label="x",
            hypothesis_id="h",
            levers=["action"],
            predicted_affected=["task_a"],
            prose="",
        ),
    )
    journal.fill_gating(
        jp,
        0,
        "accepted",
        {"task_a": "flipped"},
        extra_frontmatter={
            "regressed_unpredicted": ["task_broke_1", "task_broke_2"],
        },
    )
    ctx = journal.build_context(jp, current_round=1, output_path=tmp_path / "CTX.md")
    text = ctx.read_text(encoding="utf-8")
    # Side-effects column shows 2 unpredicted regressions AND the raw
    # prediction-hits denominator is inflated by those 2 side effects:
    # 1 flipped / (1 attributed + 2 side_effects) = 1/3. A round that
    # helped its predicted task but broke 2 others does NOT register as
    # 100% precision. The Beta-posterior column smooths 1/3 toward the
    # 0.5 prior: Beta(1+1, 1+2) mean = 2/5 = 0.40.
    assert "| action | 1 | 1 | 0 | 1/3 | 0.40 (n_eff=3.0) | 2 |" in text


# ─── M3: reverted hypothesis novelty block ───────────────────────────────


def _append(jp: Path, round_idx: int, hid: str) -> None:
    journal.append_entry(
        jp,
        journal.JournalEntrySpec(
            round=round_idx,
            label=f"r{round_idx}",
            hypothesis_id=hid,
            levers=["action"],
            predicted_affected=[],
            prose="",
        ),
    )


def test_novelty_passes_on_new_hypothesis(tmp_path: Path):
    jp = tmp_path / "j.md"
    _append(jp, 0, "h_old")
    journal.fill_gating(jp, 0, "reverted", {})
    _append(jp, 1, "h_new")  # different id
    # Should not raise.
    check_novelty(jp, tmp_path)


def test_novelty_blocks_reverted_reuse(tmp_path: Path):
    jp = tmp_path / "j.md"
    _append(jp, 0, "h_bad")
    journal.fill_gating(jp, 0, "reverted", {})
    _append(jp, 1, "h_bad")  # re-uses reverted id
    with pytest.raises(StrictValidationError) as exc_info:
        check_novelty(jp, tmp_path)
    assert exc_info.value.kind == "reverted_hypothesis_reused"
    assert (tmp_path / "NOVELTY_FAIL.md").is_file()


def test_novelty_noop_on_missing_memo(tmp_path: Path):
    """No journal = no possible reuse. Must not raise."""
    check_novelty(tmp_path / "does-not-exist.md", tmp_path)


# ─── M4: structured changeset diff ───────────────────────────────────────


def _cfg_with_tools(names: list[str]):
    """Build a minimal HarnessConfig with the given tool names."""
    from harnessx.core.builder import HarnessBuilder
    from harnessx.tools.base import tool as _tool_decorator
    from harnessx.tools.inmemory import InMemoryToolRegistry

    reg = InMemoryToolRegistry()
    for n in names:

        @_tool_decorator(name=n, description=f"tool {n}")
        async def _fn(x: str = "") -> str:
            return x

        reg.register(_fn)
    return HarnessBuilder().slot(tool_registry=reg).build()


def test_changeset_detects_added_tool():
    before = _cfg_with_tools(["Bash", "Read"])
    after = _cfg_with_tools(["Bash", "Read", "NewPdfParser"])
    diff = compute_changeset(before, after)
    assert diff == {"tools_added": ["NewPdfParser"]}


def test_changeset_detects_removed_tool():
    before = _cfg_with_tools(["Bash", "Read", "WebSearch"])
    after = _cfg_with_tools(["Bash", "Read"])
    diff = compute_changeset(before, after)
    assert diff == {"tools_removed": ["WebSearch"]}


def test_changeset_empty_on_equivalent_configs():
    before = _cfg_with_tools(["Bash", "Read"])
    after = _cfg_with_tools(["Bash", "Read"])
    assert compute_changeset(before, after) == {}


# ─── Synthetic replay smoke defaults and behavior ─────────────────────────


def test_meta_agent_default_replay_mode_is_synthetic_task():
    sig = inspect.signature(MetaAgent.evolve)
    assert sig.parameters["replay_mode"].default == "synthetic_task"


def test_task_brief_includes_global_optimization_constraint():
    agent = MetaAgent(inner_model=object())  # type: ignore[arg-type]
    brief = agent._render_task_brief(
        current_config_path=Path("/tmp/current.yaml"),
        trajectories_dir=Path("/tmp/traj"),
        output_dir=Path("/tmp/out"),
        context_path=None,
    )
    assert "## Global optimization constraint (Pareto-style)" in brief
    assert "expected_global_gain" in brief
    assert "regression_risk" in brief
    assert "cost_shift" in brief


@pytest.mark.asyncio
async def test_synthetic_smoke_passes_on_non_error_exit():
    class _Harness:
        async def run(self, task):  # noqa: ANN001
            return SimpleNamespace(
                exit_reason="done",
                total_steps=1,
                total_tokens=42,
                total_cost_usd=0.001,
                final_output="OK",
            )

    class _Model:
        def agentic(self, cfg):  # noqa: ANN001
            return _Harness()

    report = await run_synthetic_task_smoke_gate(object(), _Model(), timeout_s=0.5)
    assert report.ok is True
    assert report.outcomes and report.outcomes[0].kind == "ok_synthetic_smoke"


@pytest.mark.asyncio
async def test_synthetic_smoke_fails_on_exception():
    class _Harness:
        async def run(self, task):  # noqa: ANN001
            raise RuntimeError("boom")

    class _Model:
        def agentic(self, cfg):  # noqa: ANN001
            return _Harness()

    report = await run_synthetic_task_smoke_gate(object(), _Model(), timeout_s=0.5)
    assert report.ok is False
    assert report.outcomes and report.outcomes[0].kind == "exception:RuntimeError"


@pytest.mark.asyncio
async def test_synthetic_smoke_fails_on_timeout():
    class _Harness:
        async def run(self, task):  # noqa: ANN001
            await asyncio.sleep(0.05)
            return SimpleNamespace(exit_reason="done")

    class _Model:
        def agentic(self, cfg):  # noqa: ANN001
            return _Harness()

    report = await run_synthetic_task_smoke_gate(object(), _Model(), timeout_s=0.001)
    assert report.ok is False
    assert report.outcomes and report.outcomes[0].kind == "timeout"
