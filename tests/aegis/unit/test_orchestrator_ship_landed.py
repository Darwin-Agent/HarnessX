"""Regression: a multi-bucket candidate (`bucket: [prompt, processor]`)
must actually mutate merged.yaml. The previous code stringified the list
via ``str(...)``, producing ``"['prompt', 'processor']"`` which compose
silently dropped — the round then ran the parent's config under a fresh
hash and any pass-rate delta looked like variance instead of a missed ship.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harnessx.aegis.compose import compose_shipped_configs
from harnessx.aegis.orchestrator import (
    ShipNotLandedError,
    _assert_merged_differs_from_base,
)


def _write_base(path: Path) -> None:
    path.write_text(
        yaml.safe_dump({
            "tracer": {"_target_": "harnessx.tracing.journal.HarnessJournal",
                       "base_dir": "/tmp/r1/sessions"},
            "processors": [
                {"_target_": "harnessx.processors.context.system_prompt.SystemPromptProcessor",
                 "system_builder": {"_target_": "X.PlainMarkdownSystemPromptBuilder",
                                    "template_path": "/tmp/old_prompt.md"}},
            ],
        }),
        encoding="utf-8",
    )


def _write_candidate_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({
            "tracer": {"_target_": "harnessx.tracing.journal.HarnessJournal",
                       "base_dir": "/tmp/r2/sessions"},
            "processors": [
                {"_target_": "harnessx.processors.context.system_prompt.SystemPromptProcessor",
                 "system_builder": {"_target_": "X.PlainMarkdownSystemPromptBuilder",
                                    "template_path": "/tmp/new_prompt.md"}},
                {"_target_": "file:///tmp/truncator.py::ContentTruncatorProcessor"},
            ],
        }),
        encoding="utf-8",
    )


def test_list_bucket_lands_both_changes(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    cand = tmp_path / "applied" / "C-X" / "config.yaml"
    out = tmp_path / "merged.yaml"
    _write_base(base)
    _write_candidate_yaml(cand)

    compose_shipped_configs(
        base_config_path=base,
        shipped=[("C-X", ["prompt", "processor"], cand)],
        output_path=out,
    )
    merged = yaml.safe_load(out.read_text())
    targets = [p.get("_target_") for p in merged["processors"]]

    assert any("ContentTruncator" in t for t in targets), \
        "processor bucket applier did not run for list bucket"
    sb = next(p["system_builder"] for p in merged["processors"]
              if "system_prompt" in p["_target_"])
    assert sb["template_path"] == "/tmp/new_prompt.md", \
        "prompt bucket applier did not swap template_path"

    _assert_merged_differs_from_base(
        base_path=base, merged_path=out, shipped_cids=["C-X"],
    )


def test_stringified_list_bucket_is_caught_by_assertion(tmp_path: Path) -> None:
    """If a future regression re-introduces the str() coercion, the merged
    config will be byte-equal to the base and ``_assert_merged_differs_from_base``
    must raise rather than let it through silently."""
    base = tmp_path / "base.yaml"
    cand = tmp_path / "applied" / "C-X" / "config.yaml"
    out = tmp_path / "merged.yaml"
    _write_base(base)
    _write_candidate_yaml(cand)

    compose_shipped_configs(
        base_config_path=base,
        shipped=[("C-X", "['prompt', 'processor']", cand)],
        output_path=out,
    )

    with pytest.raises(ShipNotLandedError):
        _assert_merged_differs_from_base(
            base_path=base, merged_path=out, shipped_cids=["C-X"],
        )
