# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Tests for HarnessConfig.canonicalize().

After the unified merge, HarnessConfig.processors is a flat list[dict]
(each entry is a ``_target_`` descriptor).  ``canonicalize()`` deduplicates
by ``repr()`` (order-preserving, first wins) and returns a new config.
"""

from __future__ import annotations

from harnessx.core.harness import HarnessConfig


# ── Empty / pass-through ──────────────────────────────────────────────────────


class TestCanonicalizeBasic:
    def test_empty_processors(self):
        cfg = HarnessConfig()
        out = cfg.canonicalize()
        assert out.processors == []

    def test_single_dict_preserved(self):
        proc = {"_target_": "harnessx.processors.control.loop_detection.LoopDetectionProcessor"}
        cfg = HarnessConfig(processors=[proc])
        out = cfg.canonicalize()
        assert out.processors == [proc]

    def test_returns_new_config_not_mutating_input(self):
        proc = {"_target_": "harnessx.processors.control.cost_guard.CostGuardProcessor"}
        original = HarnessConfig(processors=[proc, proc])
        before = list(original.processors)
        out = original.canonicalize()
        # original untouched
        assert list(original.processors) == before
        # output deduped
        assert len(out.processors) == 1


# ── Dedup by repr ─────────────────────────────────────────────────────────────


class TestDedup:
    def test_dedup_identical_dicts(self):
        proc = {"_target_": "some.Processor", "threshold": 5}
        cfg = HarnessConfig(processors=[proc, proc, proc])
        out = cfg.canonicalize()
        assert len(out.processors) == 1
        assert out.processors[0] == proc

    def test_no_dedup_different_dicts(self):
        a = {"_target_": "some.Processor", "threshold": 1}
        b = {"_target_": "some.Processor", "threshold": 2}
        cfg = HarnessConfig(processors=[a, b])
        out = cfg.canonicalize()
        assert len(out.processors) == 2

    def test_dedup_preserves_first_occurrence(self):
        a = {"_target_": "some.Processor", "value": 7}
        b = {"_target_": "some.Processor", "value": 7}
        cfg = HarnessConfig(processors=[a, b])
        out = cfg.canonicalize()
        assert len(out.processors) == 1
        assert out.processors[0] is a

    def test_different_targets_both_kept(self):
        a = {"_target_": "pkg.A"}
        b = {"_target_": "pkg.B"}
        cfg = HarnessConfig(processors=[a, b])
        out = cfg.canonicalize()
        assert out.processors == [a, b]


# ── Idempotency ───────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_canonicalize_twice_is_same(self):
        procs = [
            {"_target_": "pkg.A"},
            {"_target_": "pkg.B"},
            {"_target_": "pkg.A"},  # duplicate
        ]
        cfg = HarnessConfig(processors=procs)
        out1 = cfg.canonicalize()
        out2 = out1.canonicalize()
        assert out1.processors == out2.processors


# ── Tool registry pass-through ────────────────────────────────────────────────


class TestToolRegistry:
    def test_registry_passed_through(self):
        from harnessx.tools.inmemory import InMemoryToolRegistry

        reg = InMemoryToolRegistry()
        cfg = HarnessConfig(tool_registry=reg)
        out = cfg.canonicalize()
        assert out.tool_registry is reg
