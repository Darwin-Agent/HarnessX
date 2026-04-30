# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Boundary tests — run_id transition rules (spec §B, 6 cases)."""

from __future__ import annotations

import dataclasses
import unittest.mock as mock

import pytest

from harnessx.core.events import (
    Message,
    SegmentBoundaryEvent,
    StepStartEvent,
    compute_history_hash,
    compute_windows,
    make_run_id,
)
from harnessx.core.processor import ContractViolationError, ProcessorChain
from harnessx.core.runloop import _enforce_boundary_invariant

RUN_ID = "test-run"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _u(t: str) -> Message:
    return Message(role="user", content=t)


def _a(t: str) -> Message:
    return Message(role="assistant", content=t)


def _s(t: str) -> Message:
    return Message(role="system", content=t)


def _tool(t: str, tcid: str = "tc1") -> Message:
    return Message(role="tool", content=t, tool_call_id=tcid)


def _ss(*msgs: Message) -> StepStartEvent:
    return StepStartEvent(run_id=RUN_ID, step_id=0, messages=tuple(msgs))


# ══════════════════════════════════════════════════════════════════════════════
# compute_windows correctness
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeWindows:
    def test_empty_messages(self):
        sys_w, hist_w, active_w = compute_windows(())
        assert sys_w == ()
        assert hist_w == ()
        assert active_w == ()

    def test_single_user(self):
        m = _u("hi")
        sys_w, hist_w, active_w = compute_windows((m,))
        assert sys_w == ()
        assert hist_w == ()
        assert active_w == (m,)

    def test_system_then_user(self):
        s, u = _s("sys"), _u("hi")
        sys_w, hist_w, active_w = compute_windows((s, u))
        assert sys_w == (s,)
        assert hist_w == ()
        assert active_w == (u,)

    def test_system_history_active_user(self):
        s, u1, a1, u2 = _s("sys"), _u("t1"), _a("r1"), _u("t2")
        sys_w, hist_w, active_w = compute_windows((s, u1, a1, u2))
        assert sys_w == (s,)
        assert hist_w == (u1, a1)
        assert active_w == (u2,)

    def test_last_not_user_includes_last_in_history(self):
        """When last message is not user, it falls inside history_window."""
        s, u1, a1 = _s("sys"), _u("t1"), _a("r1")
        sys_w, hist_w, active_w = compute_windows((s, u1, a1))
        assert sys_w == (s,)
        assert hist_w == (u1, a1)
        assert active_w == ()

    def test_no_system_history_starts_at_zero(self):
        """Without system, history_window starts at messages[0]."""
        u1, a1, u2 = _u("t1"), _a("r1"), _u("t2")
        sys_w, hist_w, active_w = compute_windows((u1, a1, u2))
        assert sys_w == ()
        assert hist_w == (u1, a1)  # everything before last user
        assert active_w == (u2,)

    def test_tool_messages_in_history(self):
        from harnessx.core.events import ToolCall

        tc = ToolCall(id="tc1", name="bash", input={})
        asst = Message(role="assistant", content="", tool_calls=(tc,))
        tool_result = _tool("output", "tc1")
        u = _u("next turn")
        _, hist_w, active_w = compute_windows((asst, tool_result, u))
        assert active_w == (u,)
        assert asst in hist_w
        assert tool_result in hist_w


# ══════════════════════════════════════════════════════════════════════════════
# B-1. step_start compaction triggers SegmentBoundaryEvent (via RunLoop logic)
# ══════════════════════════════════════════════════════════════════════════════


class TestAutoHistoryBoundary:
    """Simulate the RunLoop's hash-comparison logic.

    Tests the hash-based boundary detection in isolation — the same logic
    that run_loop uses to emit an automatic SegmentBoundaryEvent after
    step_start processors modify the history_window.
    """

    def _simulate_auto_boundary(
        self,
        before_msgs: tuple[Message, ...],
        after_msgs: tuple[Message, ...],
    ) -> bool:
        """Return True if RunLoop would emit an auto-boundary."""
        _, hist_before, _ = compute_windows(before_msgs)
        _, hist_after, _ = compute_windows(after_msgs)
        return compute_history_hash(hist_before) != compute_history_hash(hist_after)

    def test_compaction_emits_boundary(self):
        """B-1: dropping old messages changes hash → boundary triggered."""
        full = (_s("sys"), _u("old1"), _a("old2"), _u("old3"), _a("old4"), _u("current"))
        compact = (_s("sys"), _u("current"))  # compacted
        assert self._simulate_auto_boundary(full, compact) is True

    def test_no_change_no_boundary(self):
        """B-5 (stable): identical history → no boundary."""
        msgs = (_s("sys"), _u("t1"), _a("r1"), _u("t2"))
        assert self._simulate_auto_boundary(msgs, msgs) is False

    def test_reordering_triggers_boundary(self):
        """B-3: reordering history messages changes hash → boundary triggered."""
        m1, m2 = _u("first"), _a("second")
        before = (m1, m2, _u("current"))
        after = (m2, m1, _u("current"))  # swapped history
        assert (
            self._simulate_auto_boundary(before, after) is False or self._simulate_auto_boundary(before, after) is True
        )
        # More precisely: swapped history_window MUST yield different hash
        _, h_before, _ = compute_windows(before)
        _, h_after, _ = compute_windows(after)
        assert compute_history_hash(h_before) != compute_history_hash(h_after)

    def test_content_change_in_history_triggers_boundary(self):
        """B-5: historical message content changed → hash changes → boundary."""
        before = (_u("original"), _a("reply"), _u("current"))
        after = (_u("different content"), _a("reply"), _u("current"))
        assert self._simulate_auto_boundary(before, after) is True

    def test_only_active_user_change_no_boundary(self):
        """Modifying only active_user_window (wrapping) should NOT trigger boundary."""
        common = (_s("sys"), _u("old"), _a("reply"))
        before = common + (_u("plain question"),)
        after = common + (_u("wrapped: plain question"),)
        # history_window is identical; active_user_window differs → no boundary
        assert self._simulate_auto_boundary(before, after) is False


# ══════════════════════════════════════════════════════════════════════════════
# B-2. step_start: only last_msg modification → violation (strict mode)
# ══════════════════════════════════════════════════════════════════════════════


class TestStepStartLastMsgViolation:
    @pytest.fixture(autouse=True)
    def strict(self, monkeypatch):
        monkeypatch.setenv("HARNESSX_CONTRACT_MODE", "strict")

    @pytest.mark.asyncio
    async def test_only_last_msg_change_raises(self):
        """B-2: step_start processor changes only last message → ContractViolationError."""
        msgs = (_u("t1"), _a("r1"), _u("t2"))
        event = _ss(*msgs)

        class LastMutator:
            async def process(self, ev):
                lst = list(ev.messages)
                lst[-1] = _u("tampered")
                yield dataclasses.replace(ev, messages=tuple(lst))

        chain = ProcessorChain(LastMutator())
        with pytest.raises(ContractViolationError):
            async for _ in chain.process(event, hook="step_start"):
                pass

    @pytest.mark.asyncio
    async def test_system_change_in_step_start_raises(self):
        """step_start: changing system message → system_mutated violation."""
        msgs = (_s("original"), _u("t1"))
        event = _ss(*msgs)

        class SysMutator:
            async def process(self, ev):
                lst = list(ev.messages)
                lst[0] = _s("tampered")
                yield dataclasses.replace(ev, messages=tuple(lst))

        chain = ProcessorChain(SysMutator())
        with pytest.raises(ContractViolationError):
            async for _ in chain.process(event, hook="step_start"):
                pass


# ══════════════════════════════════════════════════════════════════════════════
# B-4. No-system scenario: history_window starts at messages[0]
# ══════════════════════════════════════════════════════════════════════════════


class TestNoSystemHistoryWindow:
    def test_hash_uses_messages_from_index_zero(self):
        """B-4: without system, history_window covers from messages[0]."""
        msgs = (_u("t1"), _a("r1"), _u("current"))
        _, hist, _ = compute_windows(msgs)
        assert hist == (_u("t1"), _a("r1"))

    def test_system_excluded_from_history_hash(self):
        """System message is in system_window, NOT history_window.

        Both with and without system, history_window contains the same messages
        (t1, r1) → same hash.  System presence/absence doesn't affect history_hash.
        """
        no_sys = (_u("t1"), _a("r1"), _u("current"))
        with_sys = (_s("sys"), _u("t1"), _a("r1"), _u("current"))
        _, h_no_sys, _ = compute_windows(no_sys)
        _, h_with_sys, _ = compute_windows(with_sys)
        # Both history_windows contain the same messages → equal hashes
        assert h_no_sys == (_u("t1"), _a("r1"))
        assert h_with_sys == (_u("t1"), _a("r1"))
        assert compute_history_hash(h_no_sys) == compute_history_hash(h_with_sys)


# ══════════════════════════════════════════════════════════════════════════════
# B-5. hash equality → no boundary; hash inequality → boundary
# ══════════════════════════════════════════════════════════════════════════════


class TestHistoryHashBoundaryRule:
    def test_same_history_same_hash(self):
        msgs = (_s("sys"), _u("t1"), _a("r1"), _u("t2"))
        _, hist, _ = compute_windows(msgs)
        assert compute_history_hash(hist) == compute_history_hash(hist)

    def test_different_content_different_hash(self):
        h1 = (_u("aaa"), _a("bbb"))
        h2 = (_u("aaa"), _a("ccc"))
        assert compute_history_hash(h1) != compute_history_hash(h2)

    def test_different_order_different_hash(self):
        m1, m2 = _u("first"), _a("second")
        assert compute_history_hash((m1, m2)) != compute_history_hash((m2, m1))

    def test_empty_vs_nonempty_hash(self):
        assert compute_history_hash(()) != compute_history_hash((_u("x"),))


# ══════════════════════════════════════════════════════════════════════════════


class TestHistoryHashNoIO:
    def test_hash_does_not_call_open(self):
        """B-6: compute_history_hash must not perform any file I/O."""
        msgs = (_u("t1"), _a("r1 with a very long content " * 100))
        _, hist, _ = compute_windows(msgs)

        with mock.patch("builtins.open", side_effect=AssertionError("I/O not allowed in hash")) as patched:
            result = compute_history_hash(hist)
        patched.assert_not_called()
        assert isinstance(result, str) and len(result) == 64  # SHA-256 hex

    def test_hash_does_not_call_read(self):
        """B-6: compute_history_hash must not read from any file path."""
        msgs = (_u("content"),)
        _, hist, _ = compute_windows(msgs)

        with mock.patch("pathlib.Path.read_text", side_effect=AssertionError("I/O not allowed")) as patched:
            compute_history_hash(hist)
        patched.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# Fix 2 — Post-hook invariant 7: processor must not trigger boundary when hash is unchanged
# ══════════════════════════════════════════════════════════════════════════════


class TestInvariant7SpuriousBoundary:
    """Invariant 7: processor emitted boundary but history_hash unchanged → violation."""

    def _hash(self, msgs):
        _, hist, _ = compute_windows(msgs)
        return compute_history_hash(hist)

    def _fake_boundary(self) -> SegmentBoundaryEvent:
        return SegmentBoundaryEvent(run_id=RUN_ID, step_id=0, reason="manual", new_run_id=make_run_id())

    def test_spurious_boundary_raises_in_strict_mode(self, monkeypatch):
        """hash equal + processor boundary → ContractViolationError in strict mode."""
        monkeypatch.setenv("HARNESSX_CONTRACT_MODE", "strict")
        msgs = (_s("sys"), _u("t1"), _a("r1"), _u("current"))
        h = self._hash(msgs)
        with pytest.raises(ContractViolationError, match="spurious"):
            _enforce_boundary_invariant([self._fake_boundary()], h, h, step_id=0)

    def test_spurious_boundary_warns_in_warn_mode(self, monkeypatch):
        """hash equal + processor boundary → warning logged in warn mode (no raise)."""
        monkeypatch.setenv("HARNESSX_CONTRACT_MODE", "warn")
        msgs = (_s("sys"), _u("t1"), _a("r1"), _u("current"))
        h = self._hash(msgs)
        import logging

        with mock.patch.object(logging.getLogger("harnessx.contract"), "warning") as warn:
            _enforce_boundary_invariant([self._fake_boundary()], h, h, step_id=5)
        warn.assert_called_once()
        assert "spurious" in warn.call_args.args[0].lower() or any(
            "spurious" in str(a).lower() for a in warn.call_args.args
        )

    def test_legitimate_boundary_no_violation(self, monkeypatch):
        """hash changed + processor boundary → no error (legitimate boundary)."""
        monkeypatch.setenv("HARNESSX_CONTRACT_MODE", "strict")
        before_msgs = (_s("sys"), _u("old1"), _a("old2"), _u("old3"), _a("old4"), _u("cur"))
        after_msgs = (_s("sys"), _u("cur"))  # compacted
        h_before = self._hash(before_msgs)
        h_after = self._hash(after_msgs)
        assert h_before != h_after
        _enforce_boundary_invariant([self._fake_boundary()], h_before, h_after, step_id=0)

    def test_no_boundary_no_violation(self, monkeypatch):
        """No processor boundary at all → _enforce_boundary_invariant is a no-op."""
        monkeypatch.setenv("HARNESSX_CONTRACT_MODE", "strict")
        msgs = (_u("t1"), _a("r1"), _u("current"))
        h = self._hash(msgs)
        _enforce_boundary_invariant([], h, h, step_id=0)  # must not raise

    def test_hash_change_without_boundary_no_violation(self, monkeypatch):
        """hash changed but no processor boundary (auto-boundary path) → no error."""
        monkeypatch.setenv("HARNESSX_CONTRACT_MODE", "strict")
        before = (_u("t1"), _a("r1"), _u("cur"))
        after = (_u("cur"),)
        h_b = self._hash(before)
        h_a = self._hash(after)
        _enforce_boundary_invariant([], h_b, h_a, step_id=0)  # must not raise
