from __future__ import annotations

import pytest

pytest.importorskip("lark_oapi")

from gateway.channels.feishu.channel import FeishuChannel
from gateway.core.base_channel import ConversationContext, ConversationType, MessageEvent


class _DummyDispatcher:
    pass


def _event(conv_type: ConversationType, chat_id: str = "oc_group_1", sender_id: str = "ou_user_1") -> MessageEvent:
    return MessageEvent(
        text="hello",
        sender_id=sender_id,
        sender_name="tester",
        platform="feishu",
        message_id="om_xxx",
        conversation=ConversationContext(
            type=conv_type,
            chat_id=chat_id,
            group_id=chat_id if conv_type != ConversationType.DM else None,
            topic_id="thread_1" if conv_type == ConversationType.TOPIC else None,
            is_dm=conv_type == ConversationType.DM,
        ),
    )


def test_feishu_topic_and_group_share_session_in_shared_mode() -> None:
    channel = FeishuChannel(
        config={"app_id": "cli_x", "app_secret": "sec_x", "session_mode": "shared"},
        dispatcher=_DummyDispatcher(),
    )
    group_event = _event(ConversationType.GROUP)
    topic_event = _event(ConversationType.TOPIC)

    group_sid = channel.resolve_session_id(group_event)
    topic_sid = channel.resolve_session_id(topic_event)

    assert group_sid == "feishu-g-oc_group_1"
    assert topic_sid == group_sid


def test_feishu_topic_and_group_share_session_in_per_user_mode() -> None:
    channel = FeishuChannel(
        config={"app_id": "cli_x", "app_secret": "sec_x", "session_mode": "per_user"},
        dispatcher=_DummyDispatcher(),
    )
    group_event = _event(ConversationType.GROUP, sender_id="ou_user_1")
    topic_event = _event(ConversationType.TOPIC, sender_id="ou_user_1")
    other_user_topic = _event(ConversationType.TOPIC, sender_id="ou_user_2")

    group_sid = channel.resolve_session_id(group_event)
    topic_sid = channel.resolve_session_id(topic_event)
    other_user_sid = channel.resolve_session_id(other_user_topic)

    assert group_sid == "feishu-g-oc_group_1-u-ou_user_1"
    assert topic_sid == group_sid
    assert other_user_sid == "feishu-g-oc_group_1-u-ou_user_2"


def test_feishu_dm_session_unaffected() -> None:
    channel = FeishuChannel(
        config={"app_id": "cli_x", "app_secret": "sec_x", "session_mode": "shared"},
        dispatcher=_DummyDispatcher(),
    )
    dm_event = _event(ConversationType.DM, chat_id="oc_dm_1", sender_id="ou_user_dm")

    assert channel.resolve_session_id(dm_event) == "feishu-dm-ou_user_dm"
