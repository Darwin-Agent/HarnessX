# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from harnessx.core.events import Message
from harnessx.core.state import State


class TestState:
    def test_state_initial(self):
        state = State(run_id="r1")
        assert state.step == 0
        assert state.raw_messages == []
        assert state.messages == []
        assert not state.budget_exceeded()

    def test_state_add_message(self):
        state = State(run_id="r1")
        msg = Message(role="user", content="hello")
        state.add_message(msg)
        assert len(state.messages) == 1
        assert len(state.raw_messages) == 1
        assert state.messages[0].content == "hello"

    def test_state_budget_exceeded_steps(self):
        state = State(run_id="r1", max_steps=3)
        state.step = 3
        assert state.budget_exceeded()

    def test_state_budget_exceeded_cost(self):
        state = State(run_id="r1", max_cost_usd=1.0)
        state.cumulative_cost_usd = 1.5
        assert state.budget_exceeded()

    def test_state_slot_operations(self):
        state = State(run_id="r1")
        state.set_slot("plan", slot_type="text", content="Step 1, Step 2")
        slot = state.get_slot("plan")
        assert slot is not None
        assert slot.content == "Step 1, Step 2"
        state.delete_slot("plan")
        assert state.get_slot("plan") is None

    def test_state_snapshot_roundtrip(self):
        state = State(run_id="abc-123")
        state.add_raw_message(Message(role="user", content="hello"))
        state.add_raw_message(Message(role="assistant", content="world"))
        state.step = 3
        state.cumulative_tokens = 100
        snap = state.snapshot()
        restored = State.from_snapshot(snap)
        assert restored.run_id == "abc-123"
        assert restored.step == 3
        assert len(restored.raw_messages) == 2
        assert restored.raw_messages[0].content == "hello"
        # Invariant: raw_messages and messages are always in sync
        assert len(restored.messages) == len(restored.raw_messages)
