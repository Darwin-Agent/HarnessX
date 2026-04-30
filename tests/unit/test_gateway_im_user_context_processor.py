from __future__ import annotations

import pytest

from gateway.core.base_channel import ConversationContext, ConversationType, MessageEvent
from gateway.core.dispatch import _im_event_var
from gateway.core.processors.im_user_context import IMUserContextProcessor
from harnessx.core.events import BeforeModelEvent, Message
from harnessx.core.processor import ProcessorChain


def _mk_event() -> MessageEvent:
    conv = ConversationContext(type=ConversationType.DM, chat_id="chat-1", is_dm=True)
    return MessageEvent(
        text="hello",
        sender_id="u-1",
        sender_name="Alice",
        platform="feishu",
        message_id="m-1",
        conversation=conv,
    )


@pytest.mark.asyncio
async def test_im_user_context_noop_when_tail_is_not_user(monkeypatch):
    monkeypatch.setenv("HARNESSX_CONTRACT_MODE", "strict")
    token = _im_event_var.set(_mk_event())
    try:
        event = BeforeModelEvent(
            run_id="run-1",
            step_id=1,
            messages=(
                Message(role="user", content="question"),
                Message(role="assistant", content="answer"),
            ),
        )
        chain = ProcessorChain(IMUserContextProcessor())
        out = [e async for e in chain.process(event, hook="before_model")]
    finally:
        _im_event_var.reset(token)

    assert len(out) == 1
    assert out[0].messages == event.messages


@pytest.mark.asyncio
async def test_im_user_context_mutates_tail_user_only(monkeypatch):
    monkeypatch.setenv("HARNESSX_CONTRACT_MODE", "strict")
    token = _im_event_var.set(_mk_event())
    try:
        event = BeforeModelEvent(
            run_id="run-1",
            step_id=1,
            messages=(
                Message(role="assistant", content="history"),
                Message(role="user", content="current question"),
            ),
        )
        chain = ProcessorChain(IMUserContextProcessor())
        out = [e async for e in chain.process(event, hook="before_model")]
    finally:
        _im_event_var.reset(token)

    assert len(out) == 1
    assert out[0].messages[0] == event.messages[0]
    assert out[0].messages[-1].role == "user"
    assert "[Message Context]" in out[0].messages[-1].content
    assert "- Sender: Alice#u-1" in out[0].messages[-1].content
    assert "- Sender ID: u-1" in out[0].messages[-1].content
    assert "- Conversation Type: DM" in out[0].messages[-1].content
    assert "- Message Time:" in out[0].messages[-1].content
    assert out[0].messages[-1].content.endswith("current question")
