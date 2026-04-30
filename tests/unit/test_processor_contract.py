# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Contract tests — hook-level message mutation contract (spec §A, 12 cases)."""

from __future__ import annotations

import dataclasses

import pytest

from harnessx.core.events import (
    BeforeModelEvent,
    Message,
    StepStartEvent,
    TaskStartEvent,
)
from harnessx.core.processor import (
    ContractViolationError,
    ProcessorChain,
    check_post_hook_invariants,
)

RUN_ID = "test-run"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _u(text: str) -> Message:
    return Message(role="user", content=text)


def _a(text: str) -> Message:
    return Message(role="assistant", content=text)


def _s(text: str) -> Message:
    return Message(role="system", content=text)


def _tool(content: str, tool_call_id: str = "tc1") -> Message:
    return Message(role="tool", content=content, tool_call_id=tool_call_id)


def _bm(*msgs: Message) -> BeforeModelEvent:
    return BeforeModelEvent(run_id=RUN_ID, step_id=0, messages=tuple(msgs))


def _ss(*msgs: Message) -> StepStartEvent:
    return StepStartEvent(run_id=RUN_ID, step_id=0, messages=tuple(msgs))


async def _run(event, hook: str, *processors) -> list:
    chain = ProcessorChain(*processors)
    return [e async for e in chain.process(event, hook=hook)]


# ── Fixture: enforce strict mode for all tests in this file ───────────────────


@pytest.fixture(autouse=True)
def strict_contract(monkeypatch):
    monkeypatch.setenv("HARNESSX_CONTRACT_MODE", "strict")


# ══════════════════════════════════════════════════════════════════════════════


class TestTaskStart:
    @pytest.mark.asyncio
    async def test_system_prompt_modification_allowed(self):
        """task_start processor can modify system_prompt without raising."""
        event = TaskStartEvent(run_id=RUN_ID, step_id=0, system_prompt="original")

        class SysModifier:
            async def process(self, ev):
                yield dataclasses.replace(ev, system_prompt="patched")

        results = await _run(event, "task_start", SysModifier())
        assert results[0].system_prompt == "patched"

    @pytest.mark.asyncio
    async def test_passthrough_allowed(self):
        """task_start: pure pass-through never raises."""
        event = TaskStartEvent(run_id=RUN_ID, step_id=0)

        class NoOp:
            async def process(self, ev):
                yield ev

        results = await _run(event, "task_start", NoOp())
        assert results


# ══════════════════════════════════════════════════════════════════════════════


class TestBeforeModelHistoryMutation:
    @pytest.mark.asyncio
    async def test_modifying_history_raises(self):
        """before_model: processor edits messages[1] (history_window) → violation."""
        msgs = (_s("sys"), _u("turn1"), _a("reply1"), _u("turn2"))
        event = _bm(*msgs)

        class HistoryMutator:
            async def process(self, ev):
                lst = list(ev.messages)
                lst[1] = _u("tampered history")
                yield dataclasses.replace(ev, messages=tuple(lst))

        with pytest.raises(ContractViolationError):
            await _run(event, "before_model", HistoryMutator())

    @pytest.mark.asyncio
    async def test_modifying_system_in_before_model_raises(self):
        """before_model: processor edits system message → window_out_of_scope violation."""
        msgs = (_s("sys"), _u("current"))
        event = _bm(*msgs)

        class SysMutator:
            async def process(self, ev):
                lst = list(ev.messages)
                lst[0] = _s("tampered sys")
                yield dataclasses.replace(ev, messages=tuple(lst))

        with pytest.raises(ContractViolationError):
            await _run(event, "before_model", SysMutator())


# ══════════════════════════════════════════════════════════════════════════════


class TestStepStartOnlyLastMsg:
    @pytest.mark.asyncio
    async def test_only_last_msg_modification_raises(self):
        """step_start: processor changes only last message without structural history change → violation."""
        msgs = (_u("turn1"), _a("reply1"), _u("turn2"))
        event = _ss(*msgs)

        class LastMsgMutator:
            async def process(self, ev):
                lst = list(ev.messages)
                lst[-1] = _u("tampered last only")
                yield dataclasses.replace(ev, messages=tuple(lst))

        with pytest.raises(ContractViolationError):
            await _run(event, "step_start", LastMsgMutator())

    @pytest.mark.asyncio
    async def test_structural_compaction_allowed(self):
        """step_start: structural compaction (remove messages) is allowed."""
        msgs = (_s("sys"), _u("old1"), _a("old2"), _u("current"))
        event = _ss(*msgs)

        class Compactor:
            async def process(self, ev):
                yield dataclasses.replace(ev, messages=(ev.messages[0], ev.messages[-1]))

        results = await _run(event, "step_start", Compactor())
        assert len(results[0].messages) == 2


# ══════════════════════════════════════════════════════════════════════════════


class TestNoLengthChangeHooks:
    def test_step_end_length_change_raises(self):
        """post-hook invariant: step_end chain must not change messages length."""
        initial = (_u("u"),)
        final = (_u("u"), _u("extra"))  # +1
        with pytest.raises(ContractViolationError):
            check_post_hook_invariants("step_end", initial, final, step_id=0)

    def test_task_end_length_change_raises(self):
        """post-hook invariant: task_end chain must not change messages length."""
        initial = (_u("u"),)
        final = ()
        with pytest.raises(ContractViolationError):
            check_post_hook_invariants("task_end", initial, final, step_id=0)

    def test_step_end_no_change_allowed(self):
        """post-hook invariant: step_end unchanged → no violation."""
        msgs = (_u("u"),)
        check_post_hook_invariants("step_end", msgs, msgs, step_id=0)  # must not raise


# ══════════════════════════════════════════════════════════════════════════════


class TestBeforeModelLastUser:
    @pytest.mark.asyncio
    async def test_last_user_length_increase_raises(self):
        """before_model: last=user, processor adds message → violation."""
        msgs = (_u("turn1"),)
        event = _bm(*msgs)

        class Adder:
            async def process(self, ev):
                yield dataclasses.replace(ev, messages=ev.messages + (_u("extra"),))

        with pytest.raises(ContractViolationError):
            await _run(event, "before_model", Adder())

    @pytest.mark.asyncio
    async def test_last_user_content_modification_allowed(self):
        """before_model: last=user, processor wraps content → no violation."""
        msgs = (_u("plain question"),)
        event = _bm(*msgs)

        class Wrapper:
            async def process(self, ev):
                lst = list(ev.messages)
                lst[-1] = _u("wrapped: " + lst[-1].content)
                yield dataclasses.replace(ev, messages=tuple(lst))

        results = await _run(event, "before_model", Wrapper())
        assert results[0].messages[-1].content == "wrapped: plain question"

    @pytest.mark.asyncio
    async def test_last_user_history_modification_raises(self):
        """before_model: last=user, processor also modifies prior message → violation."""
        msgs = (_u("old"), _a("reply"), _u("current"))
        event = _bm(*msgs)

        class HistoryToucher:
            async def process(self, ev):
                lst = list(ev.messages)
                lst[0] = _u("tampered old")  # touches history
                yield dataclasses.replace(ev, messages=tuple(lst))

        with pytest.raises(ContractViolationError):
            await _run(event, "before_model", HistoryToucher())


# ══════════════════════════════════════════════════════════════════════════════


class TestBeforeModelAppendUser:
    @pytest.mark.asyncio
    async def test_last_not_user_append_one_user_allowed(self):
        """before_model: last=assistant, processor appends one user → allowed."""
        msgs = (_u("turn1"), _a("reply1"))
        event = _bm(*msgs)

        class Injector:
            async def process(self, ev):
                yield dataclasses.replace(ev, messages=ev.messages + (_u("verify step"),))

        results = await _run(event, "before_model", Injector())
        assert len(results[0].messages) == 3
        assert results[0].messages[-1].role == "user"

    @pytest.mark.asyncio
    async def test_last_not_user_modification_without_addition_raises(self):
        """before_model: last=assistant, processor modifies content without adding user → violation."""
        msgs = (_u("turn1"), _a("reply1"))
        event = _bm(*msgs)

        class AsstMutator:
            async def process(self, ev):
                lst = list(ev.messages)
                lst[-1] = _a("tampered assistant")
                yield dataclasses.replace(ev, messages=tuple(lst))

        with pytest.raises(ContractViolationError):
            await _run(event, "before_model", AsstMutator())


# ══════════════════════════════════════════════════════════════════════════════
#    (via check_post_hook_invariants directly — those events lack messages field)
# ══════════════════════════════════════════════════════════════════════════════


class TestOtherHooksNoLengthChange:
    @pytest.mark.parametrize("hook", ["after_model", "before_tool", "after_tool", "task_start"])
    def test_length_change_in_non_authorized_hook_raises(self, hook: str):
        """Non-authorized hooks: any messages length change → invariant violation."""
        initial = (_u("m"),)
        final = (_u("m"), _u("extra"))
        with pytest.raises(ContractViolationError):
            check_post_hook_invariants(hook, initial, final, step_id=0)


# ══════════════════════════════════════════════════════════════════════════════


class TestBeforeModelDoubleInsert:
    @pytest.mark.asyncio
    async def test_two_processors_add_user_second_raises(self):
        """before_model: proc_A adds user, proc_B also tries → ContractViolationError."""
        msgs = (_u("turn1"), _a("reply1"))
        event = _bm(*msgs)

        class AddUser:
            async def process(self, ev):
                yield dataclasses.replace(ev, messages=ev.messages + (_u("inject"),))

        with pytest.raises(ContractViolationError):
            await _run(event, "before_model", AddUser(), AddUser())


# ══════════════════════════════════════════════════════════════════════════════


class TestBeforeModelNetLengthExceeded:
    @pytest.mark.asyncio
    async def test_single_processor_adds_two_raises(self):
        """before_model: one processor adds 2 messages → violation."""
        msgs = (_u("turn1"), _a("reply1"))
        event = _bm(*msgs)

        class AddTwo:
            async def process(self, ev):
                yield dataclasses.replace(ev, messages=ev.messages + (_u("a"), _u("b")))

        with pytest.raises(ContractViolationError):
            await _run(event, "before_model", AddTwo())

    def test_chain_net_over_one_via_invariant(self):
        """post-hook invariant: before_model net > +1 → ContractViolationError."""
        initial = (_u("t1"), _a("r1"))
        final = initial + (_u("x"), _u("y"))  # +2
        with pytest.raises(ContractViolationError):
            check_post_hook_invariants("before_model", initial, final, step_id=0)


# ══════════════════════════════════════════════════════════════════════════════


class TestBeforeModelEmptyMessages:
    @pytest.mark.asyncio
    async def test_empty_messages_fail_fast(self):
        """before_model: messages=() → ContractViolationError before processors run."""
        event = BeforeModelEvent(run_id=RUN_ID, step_id=0, messages=())

        class NoOp:
            async def process(self, ev):
                yield ev

        with pytest.raises(ContractViolationError, match="empty"):
            await _run(event, "before_model", NoOp())

    @pytest.mark.asyncio
    async def test_step_start_empty_messages_fail_fast(self):
        """step_start: messages=() → ContractViolationError before processors run."""
        event = StepStartEvent(run_id=RUN_ID, step_id=0, messages=())

        class NoOp:
            async def process(self, ev):
                yield ev

        with pytest.raises(ContractViolationError, match="empty"):
            await _run(event, "step_start", NoOp())


# ══════════════════════════════════════════════════════════════════════════════


class TestPostHookInvariantsSystem:
    def test_multiple_system_messages_raises(self):
        """post-hook: > 1 system message in final → violation."""
        msgs = (_s("s1"), _s("s2"), _u("u"))
        with pytest.raises(ContractViolationError, match="system"):
            check_post_hook_invariants("step_start", (), msgs, step_id=0)

    def test_system_not_at_position_zero_raises(self):
        """post-hook: system message not at index 0 → violation."""
        msgs = (_u("u"), _s("s"))
        with pytest.raises(ContractViolationError, match="position 0"):
            check_post_hook_invariants("step_start", (), msgs, step_id=0)

    def test_raw_effective_role_mismatch_raises(self):
        """post-hook: raw_track[i].role != effective_track[i].role → violation."""
        raw = (_u("u"),)
        eff = (_a("a"),)
        with pytest.raises(ContractViolationError, match="role mismatch"):
            check_post_hook_invariants("step_start", (), eff, step_id=0, raw_msgs=raw)

    def test_raw_effective_tool_call_id_mismatch_raises(self):
        """post-hook: tool_call_id mismatch between raw and effective → violation."""
        raw = (_tool("r", tool_call_id="tc-1"),)
        eff = (_tool("r", tool_call_id="tc-2"),)
        with pytest.raises(ContractViolationError, match="tool_call_id"):
            check_post_hook_invariants("step_start", (), eff, step_id=0, raw_msgs=raw)

    def test_raw_effective_length_mismatch_raises(self):
        """post-hook: len(raw_track) != len(effective_track) → violation."""
        raw = (_u("u"), _a("a"))
        eff = (_u("u"),)
        with pytest.raises(ContractViolationError, match="len"):
            check_post_hook_invariants("step_start", (), eff, step_id=0, raw_msgs=raw)
