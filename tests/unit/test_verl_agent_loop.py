"""Test the adapter contract without importing a GPU/Ray training installation."""

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def adapter(monkeypatch):
    module = ModuleType("verl.experimental.agent_loop.agent_loop")

    class AgentLoopBase:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    @dataclass
    class AgentLoopOutput:
        prompt_ids: list
        response_ids: list
        response_mask: list
        response_logprobs: list
        num_turns: int
        metrics: dict
        reward_score: float
        extra_fields: dict = field(default_factory=dict)
        sub_outputs: list | None = None

    module.AgentLoopBase = AgentLoopBase
    module.AgentLoopOutput = AgentLoopOutput
    module.register = lambda name: lambda target: target
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = Path(__file__).resolve().parents[2] / "recipe" / "verl" / "agent_loop.py"
    spec = importlib.util.spec_from_file_location("harnessx_test_verl_adapter", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    loop = loaded.HarnessXAgentLoop(
        tokenizer=SimpleNamespace(eos_token_id=2, pad_token_id=0, bos_token_id=1),
        rollout_config=SimpleNamespace(response_length=8),
        server_manager=SimpleNamespace(_server_id_to_handle={"127.0.0.1:8000": None}),
    )
    return loaded, loop


@pytest.mark.parametrize(
    "episode",
    [
        None,
        {"status": "aborted"},
        {
            "status": "completed",
            "tokens": [1, 2],
            "response_length": 1,
            "loss_mask": [1],
            "rollout_log_probs": [None],
        },
    ],
)
async def test_failed_samples_are_never_trainable(adapter, monkeypatch, episode):
    module, loop = adapter

    async def generate(**kwargs):
        return episode

    monkeypatch.setattr(module, "generate", generate)
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "question"}])
    assert output.response_ids == [2]
    assert output.response_mask == [0]
    assert output.reward_score == 0.0
    assert output.extra_fields["rollout_failed"] is True


@pytest.mark.parametrize(
    "exception", [TypeError("bad provider keyword"), ImportError("missing dependency"), AttributeError("bad interface")]
)
async def test_configuration_errors_fail_fast(adapter, monkeypatch, exception):
    module, loop = adapter

    async def generate(**kwargs):
        raise exception

    monkeypatch.setattr(module, "generate", generate)
    with pytest.raises(type(exception)):
        await loop.run({}, raw_prompt=[])


async def test_runtime_failure_masks_sample(adapter, monkeypatch):
    module, loop = adapter

    async def generate(**kwargs):
        raise TimeoutError("service unavailable")

    monkeypatch.setattr(module, "generate", generate)
    output = await loop.run({}, raw_prompt=[])
    assert output.response_mask == [0]


@pytest.mark.parametrize(
    "mask, probabilities", [([1], []), ([1], [float("nan")]), ([1], [0.2]), ([0], [0.0]), ([2], [-0.2])]
)
def test_invalid_training_sequences_are_rejected(adapter, mask, probabilities):
    module, _ = adapter
    with pytest.raises(ValueError):
        module._validate_tokens([1], [2], mask, probabilities)
