# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
import pathlib
import yaml as _yaml
import pytest

from harnessx.bundles import coding, reliability, window_mgmt
from harnessx.core.harness import HarnessConfig

ALL_KNOWN_GROUPS = {
    # Atomic context processors
    "context.system",
    "context.user_wrapper",
    "context.env",
    # Atomic memory processors
    "memory.retrieval",
    "memory.extraction",
    # Atomic tools processors
    "tools.filter",
    # Evaluation
    "evaluation",
    # Control
    "cost_guard",
    "loop_detection",
    "compaction",
    "token_budget",
    "tool_failure_guard",
    "parse_retry",
    "tool_call_correction",
    "repeated_edit_detector",
    "self_verify",
    "todo_check",
    "bg_install_guard",
    # Observability
    "otel",
    "checkpoint",
    # Tools
    "progressive_skill_loader",
    "tool_whitelist",
    "tools.schema_adapter",
    "sycophancy_detector",
}

_EXAMPLES = pathlib.Path(__file__).parent.parent.parent / "examples"


# ── Config fixtures ────────────────────────────────────────────────────────────


def _load_example(name: str):
    raw = _yaml.safe_load((_EXAMPLES / name / "harness_config.yaml").read_text()) or {}
    return HarnessConfig(
        processors=raw.get("processors") or [],
        plugins=raw.get("plugins") or [],
    )


MinimalConfig = _load_example("minimal")
AssistantConfig = _load_example("assistant")
ResearchConfig = _load_example("research")
CodingConfig = _load_example("coding")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _groups_set(config) -> set:
    """Return set of non-None singleton_groups present in the config."""
    from harnessx.core.harness import _instantiate_runtime

    result = set()
    for procs in _instantiate_runtime(config).processors.values():
        for p in procs:
            sg = getattr(type(p), "_singleton_group", None)
            if sg is not None:
                result.add(sg)
    return result


def _type_names(config) -> set:
    from harnessx.core.harness import _instantiate_runtime

    return {type(p).__name__ for procs in _instantiate_runtime(config).processors.values() for p in procs}


# ── MinimalConfig ─────────────────────────────────────────────────────────────


class TestDescriptorSnapshots:
    def test_minimal_has_no_processors(self):
        assert _type_names(MinimalConfig) == set()

    # ── AssistantConfig (from examples/assistant/harness_config.yaml) ─────────────

    def test_assistant_has_memory_and_observability(self):
        groups = _groups_set(AssistantConfig)
        assert "memory.retrieval" in groups
        assert "memory.extraction" in groups
        assert "otel" in groups

    def test_assistant_has_full_control_stack(self):
        groups = _groups_set(AssistantConfig)
        for g in (
            "loop_detection",
            "tool_call_correction",
            "parse_retry",
            "tool_failure_guard",
            "repeated_edit_detector",
        ):
            assert g in groups, f"Expected control group '{g}' in assistant config"

    # ── ResearchConfig (from examples/research/harness_config.yaml) ───────────────

    def test_research_has_evaluation(self):
        assert "evaluation" in _groups_set(ResearchConfig)

    def test_research_has_cost_guard(self):
        assert "cost_guard" in _groups_set(ResearchConfig)

    def test_research_no_reliability_guards(self):
        """Research config intentionally omits reliability stack."""
        groups = _groups_set(ResearchConfig)
        for g in ("loop_detection", "tool_call_correction", "parse_retry"):
            assert g not in groups, f"Unexpected control group '{g}' in research config"

    # ── CodingConfig (from examples/coding/harness_config.yaml) ───────────────────

    def test_coding_has_skill_loading(self):
        assert "progressive_skill_loader" in _groups_set(CodingConfig)

    def test_coding_has_full_control_stack(self):
        groups = _groups_set(CodingConfig)
        for g in (
            "loop_detection",
            "tool_call_correction",
            "parse_retry",
            "tool_failure_guard",
            "repeated_edit_detector",
        ):
            assert g in groups, f"Expected control group '{g}' in coding config"

    # ── coding bundle ─────────────────────────────────────────────────────────────

    def test_coding_bundle_groups(self):
        config = coding.build()
        expected = {
            "tool_call_correction",
            "context.env",
            "compaction",
            "parse_retry",
            "progressive_skill_loader",
            "todo_check",
            "loop_detection",
            "repeated_edit_detector",
            "tool_failure_guard",
            "self_verify",
        }
        assert _groups_set(config) == expected

    # ── reliability bundle ────────────────────────────────────────────────────────

    def test_reliability_bundle_groups(self):
        config = reliability.build()
        expected = {
            "loop_detection",
            "parse_retry",
            "tool_call_correction",
            "repeated_edit_detector",
            "self_verify",
            "todo_check",
        }
        assert _groups_set(config) == expected

    # ── window_mgmt bundle ────────────────────────────────────────────────────────

    def test_window_mgmt_bundle_groups(self):
        config = window_mgmt.build()
        expected = {"compaction", "tool_failure_guard"}
        assert _groups_set(config) == expected

    # ── Stability: no unexpected singleton_group in any config ────────────────────

    @pytest.mark.parametrize(
        "config",
        [
            MinimalConfig,
            AssistantConfig,
            ResearchConfig,
            CodingConfig,
        ],
        ids=["minimal", "assistant", "research", "coding"],
    )
    def test_config_only_uses_known_groups(self, config):
        unknown = _groups_set(config) - ALL_KNOWN_GROUPS
        assert not unknown, f"Unknown singleton_groups: {unknown}"
