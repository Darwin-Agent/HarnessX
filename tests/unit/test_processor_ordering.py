# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
import pytest

from harnessx.core.processor import MultiHookProcessor, PRE, NORMAL, POST
from harnessx.core.builder import HarnessBuilder, HarnessConflictError


# ── Fixtures ──────────────────────────────────────────────────────────────────


class _Base(MultiHookProcessor):
    """No-op base; subclasses in each test define _order/_after."""

    pass


def _proc_names(config, hook="*") -> list[str]:
    from harnessx.core.harness import _instantiate_runtime

    return [type(p).__name__ for p in _instantiate_runtime(config).processors.get(hook, [])]


# ── Phase constants ───────────────────────────────────────────────────────────


class TestProcessorOrdering:
    def test_phase_constants(self):
        assert PRE == 0
        assert NORMAL == 50
        assert POST == 100

    def test_phase_constants_fine_grained(self):
        """Fine-grained offsets remain expressible."""
        assert PRE + 5 == 5
        assert NORMAL - 10 == 40
        assert POST - 1 == 99

    # ── Basic _order sort ─────────────────────────────────────────────────────────

    def test_order_sort_ascending(self):
        class Early(_Base):
            _order = PRE

        class Late(_Base):
            _order = POST

        config = HarnessBuilder().add(Late()).add(Early()).build()
        assert _proc_names(config) == ["Early", "Late"]

    # ── _after within same order bucket ──────────────────────────────────────────

    def test_after_within_same_order(self):
        """B._after=['a'] — B must run after A even when registered first."""

        class A(_Base):
            _singleton_group = "a"
            _order = NORMAL

        class B(_Base):
            _singleton_group = "b"
            _order = NORMAL
            _after = ["a"]

        # Register B before A — _after should flip the order
        config = HarnessBuilder().add(B()).add(A()).build()
        names = _proc_names(config)
        assert names.index("A") < names.index("B")

    def test_after_chain_within_bucket(self):
        """C → B → A (all same order); registration order is reversed."""

        class A(_Base):
            _singleton_group = "a"
            _order = NORMAL

        class B(_Base):
            _singleton_group = "b"
            _order = NORMAL
            _after = ["a"]

        class C(_Base):
            _singleton_group = "c"
            _order = NORMAL
            _after = ["b"]

        config = HarnessBuilder().add(C()).add(B()).add(A()).build()
        names = _proc_names(config)
        assert names.index("A") < names.index("B") < names.index("C")

    # ── Cross-bucket _after ───────────────────────────────────────────────────────

    def test_cross_bucket_after_already_satisfied(self):
        """_after pointing to a lower-order processor is a no-op (already ordered correctly)."""

        class Early(_Base):
            _singleton_group = "early"
            _order = PRE

        class Late(_Base):
            _singleton_group = "late"
            _order = POST
            _after = ["early"]  # already satisfied by _order

        config = HarnessBuilder().add(Late()).add(Early()).build()
        names = _proc_names(config)
        assert names.index("Early") < names.index("Late")

    def test_cross_bucket_contradiction_raises(self):
        """_after pointing to a *higher* _order processor — impossible, must fail at build."""

        class Lo(_Base):
            _singleton_group = "lo"
            _order = PRE
            _after = ["hi"]  # hi has _order=POST — contradiction

        class Hi(_Base):
            _singleton_group = "hi"
            _order = POST

        with pytest.raises(HarnessConflictError, match="contradictory"):
            HarnessBuilder().add(Lo()).add(Hi()).build()

    # ── Cycle detection ───────────────────────────────────────────────────────────

    def test_cycle_raises(self):
        """A._after=['b'], B._after=['a'], same order — cycle, must fail at build."""

        class A(_Base):
            _singleton_group = "cyc_a"
            _order = NORMAL
            _after = ["cyc_b"]

        class B(_Base):
            _singleton_group = "cyc_b"
            _order = NORMAL
            _after = ["cyc_a"]

        with pytest.raises(HarnessConflictError, match="Cycle"):
            HarnessBuilder().add(A()).add(B()).build()

    # ── Soft dependency ───────────────────────────────────────────────────────────

    def test_soft_dep_unregistered_ignored(self):
        """_after referencing a group that isn't registered — silently ignored, no error."""

        class Solo(_Base):
            _singleton_group = "solo"
            _order = NORMAL
            _after = ["nonexistent_group"]

        config = HarnessBuilder().add(Solo()).build()
        assert _proc_names(config) == ["Solo"]

    def test_soft_dep_partial_registration(self):
        """_after=['x', 'missing']; 'x' is registered, 'missing' is not — only 'x' honoured."""

        class X(_Base):
            _singleton_group = "x"
            _order = NORMAL

        class Y(_Base):
            _singleton_group = "y"
            _order = NORMAL
            _after = ["x", "missing"]

        config = HarnessBuilder().add(Y()).add(X()).build()
        names = _proc_names(config)
        assert names.index("X") < names.index("Y")

    # ── _after stored on _ProcEntry ───────────────────────────────────────────────

    def test_after_stored_in_entry(self):
        """HarnessBuilder.add() reads _after from the processor class."""
        from harnessx.core.builder import _ProcEntry

        class P(_Base):
            _singleton_group = "p"
            _after = ["q", "r"]

        builder = HarnessBuilder().add(P())
        entry: _ProcEntry = builder._entries[0]
        assert entry.after == ("q", "r")

    def test_after_default_empty(self):
        """Processors without _after get an empty tuple in the entry."""
        from harnessx.core.builder import _ProcEntry

        class Q(_Base):
            _singleton_group = "q"

        builder = HarnessBuilder().add(Q())
        entry: _ProcEntry = builder._entries[0]
        assert entry.after == ()
