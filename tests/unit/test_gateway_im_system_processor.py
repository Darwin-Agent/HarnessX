from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.core.dispatch import _im_channel_var
from gateway.core.processors.im_system import IMSystemProcessor
from harnessx.core.events import TaskStartEvent
from harnessx.core.processor import ProcessorChain


def _mk_channel() -> SimpleNamespace:
    return SimpleNamespace(
        name="feishu",
        display_name="Feishu",
        system_prompt=lambda: "Channel supplement.",
    )


@pytest.mark.asyncio
async def test_im_system_injects_runtime_context(tmp_path):
    channel = _mk_channel()
    token = _im_channel_var.set(channel)
    try:
        ws_home = tmp_path / "home"
        ws_root = ws_home / "im-workspaces" / "gateway"
        ws_root.mkdir(parents=True, exist_ok=True)
        event = TaskStartEvent(
            run_id="run-1",
            step_id=0,
            model="gpt-4o",
            session_id="sess-1",
            system_prompt="Base prompt.",
            workspace=SimpleNamespace(root=ws_root, home=ws_home, agent_id="gateway"),
        )
        chain = ProcessorChain(IMSystemProcessor())
        out = [e async for e in chain.process(event, hook="task_start")]
    finally:
        _im_channel_var.reset(token)

    assert len(out) == 1
    prompt = out[0].system_prompt
    assert "Base prompt." in prompt
    assert "You are a personal AI assistant on Feishu." in prompt
    assert "Platform: Feishu (feishu)" in prompt
    assert "Runtime context:" in prompt
    assert "- Agent ID: gateway" in prompt
    assert "- Session ID: sess-1" in prompt
    assert "- Model: gpt-4o" in prompt
    assert f"- Agent workspace path: {ws_root.resolve()}" in prompt
    assert f"- Skills path: {(ws_home / 'skills').resolve()}" in prompt
    assert "- Timezone:" in prompt
    assert "- Current date:" not in prompt
    assert "Channel supplement." in prompt


@pytest.mark.asyncio
async def test_im_system_noop_without_channel_context():
    event = TaskStartEvent(run_id="run-1", step_id=0, system_prompt="Base only")
    chain = ProcessorChain(IMSystemProcessor())
    out = [e async for e in chain.process(event, hook="task_start")]
    assert len(out) == 1
    assert out[0].system_prompt == "Base only"
