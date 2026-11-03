import asyncio

from harnessx.core.events import BeforeModelEvent, Message, ToolSchema
from harnessx.core.runloop import _validate_messages
from harnessx.providers.sglang import _parse_qwen3_coder_tool_calls
from harnessx.tools.base import Tool, _execute_tool
from recipe.verl.processor.early_commit import EarlyCommitProcessor


def test_xml_string_parameters_keep_json_looking_text():
    schema = ToolSchema(
        name="Write",
        description="",
        input_schema={
            "type": "object",
            "properties": {"content": {"type": "string"}, "enabled": {"type": "boolean"}},
        },
    )
    calls = _parse_qwen3_coder_tool_calls(
        '<tool_call><function=Write><parameter=content>{"enabled":true}</parameter>'
        "<parameter=enabled>true</parameter></function></tool_call>",
        [schema],
    )
    assert calls[0].input == {"content": '{"enabled":true}', "enabled": True}


def test_tool_schema_validation_precedes_side_effecting_callable():
    calls = []
    tool = Tool(
        name="write",
        description="",
        input_schema={"type": "object", "properties": {"content": {"type": "string"}}},
        fn=lambda content: calls.append(content) or content,
    )
    result = asyncio.run(_execute_tool(tool, {"content": {"bad": True}}))
    assert result.error == "Parameter 'content' must be string"
    assert calls == []


def test_consecutive_user_messages_preserve_multimodal_content():
    messages = _validate_messages(
        [
            Message(role="user", content="FIRST_RESULT"),
            Message(role="user", content=[{"type": "image", "source": {"data": "x"}}]),
            Message(role="user", content="SECOND_RESULT"),
        ]
    )
    assert len(messages) == 1
    assert messages[0].content == [
        {"type": "text", "text": "FIRST_RESULT"},
        {"type": "image", "source": {"data": "x"}},
        {"type": "text", "text": "SECOND_RESULT"},
    ]


def test_early_commit_exposes_persistent_reminder():
    processor = EarlyCommitProcessor(max_steps=2, token_budget_ratio=1.0)
    processor._current_step = 1
    event = BeforeModelEvent(
        run_id="run",
        step_id=1,
        messages=(Message(role="assistant", content="<tool_call>"),),
    )
    result = asyncio.run(anext(processor.on_before_model(event)))
    assert result.persistent_messages
    assert result.persistent_messages[0] in result.messages
