"""CPU-only smoke tests using the real Harness and SGLang provider."""

from types import SimpleNamespace

import pytest

from harnessx.core.events import EvalResult, Message
from harnessx.providers.sglang import GenerationAborted, SGLangProvider
from harnessx.rl.task import NullPRM, RLTask
from harnessx.tools.base import Tool
from recipe.verl import harness_rollout
from recipe.verl.matpo import spawn_rl
from recipe.verl.spec import VerlConfigSpec


class CharacterTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    bos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(character) + 3 for character in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token - 3) for token in tokens if token >= 3)

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        text = "".join(f"<{message['role']}>{message.get('content') or ''}</{message['role']}>" for message in messages)
        if add_generation_prompt:
            text += "<assistant><think>\n"
        return self.encode(text) if tokenize else text


class AnswerEvaluator:
    def __init__(self, task):
        self.label = task.label

    async def evaluate(self, event):
        passed = self.label in str(event.final_output)
        return EvalResult(passed=passed, score=float(passed), reason="smoke", reward=float(passed))


@pytest.mark.parametrize("tool_format", ["json", "qwen3_coder"])
async def test_actual_harness_tool_turn_and_final_answer(monkeypatch, tool_format):
    tokenizer = CharacterTokenizer()
    calls = []
    spec = VerlConfigSpec(
        task_builder=SimpleNamespace(build=lambda sample: RLTask(description="Compute 6 * 7", label=sample.label)),
        tools=[Tool("calculate", "Return the answer", {"type": "object", "properties": {}}, lambda: "42")],
        evaluator_cls=AnswerEvaluator,
        prm=NullPRM(),
        max_steps=3,
        tool_call_format=tool_format,
        tool_result_role="user",
    )
    monkeypatch.setattr(harness_rollout, "load_harness_config", lambda sample: spec)
    tool_call = (
        '<tool_call>{"name":"calculate","arguments":{}}</tool_call>'
        if tool_format == "json"
        else "<tool_call><function=calculate>{}</function></tool_call>"
    )
    replies = iter([tool_call, "The answer is 42."])

    class Server:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            tokens = tokenizer.encode(next(replies))
            return SimpleNamespace(token_ids=tokens, log_probs=[-0.25] * len(tokens), stop_reason="stop")

    episode = await harness_rollout.generate(
        harness_rollout.VerlSample("Compute 6 * 7", label="42"),
        {"max_new_tokens": 256},
        "http://unused/generate",
        tokenizer,
        Server(),
    )
    assert episode is not None
    assert episode["status"] == "completed"
    assert episode["eval_result"].passed
    assert len(calls) == 2
    assert all(call["sampling_params"]["logprobs"] for call in calls)
    assert len(episode["loss_mask"]) == len(episode["rollout_log_probs"]) == episode["response_length"]
    assert 0 in episode["loss_mask"] and 1 in episode["loss_mask"]
    assert all(
        probability == -0.25 for probability, mask in zip(episode["rollout_log_probs"], episode["loss_mask"]) if mask
    )
    assert "<think>" not in tokenizer.decode(calls[0]["prompt_ids"])
    assert (await harness_rollout.reward_func(harness_rollout.VerlSample("", "42"), episode))["is_correct"] == 1


@pytest.mark.parametrize("probabilities", [None, [None], [float("nan")], [float("inf")], [0.1]])
async def test_provider_rejects_missing_or_invalid_training_logprobs(probabilities):
    async def post(url, payload):
        metadata = {"finish_reason": {"type": "stop"}}
        if probabilities is not None:
            metadata["output_token_logprobs"] = [(probability, 45, None) for probability in probabilities]
        return {"meta_info": metadata, "text": "*"}

    provider = SGLangProvider("unused", CharacterTokenizer(), {}, 4096, post_fn=post, require_logprobs=True)
    with pytest.raises(GenerationAborted):
        await provider.complete([Message("user", "test")], [])
    assert provider.aborted
    assert not provider.step_captures


async def test_inprocess_bridge_rejects_misaligned_logprobs():
    server = SimpleNamespace(generate=None)

    async def generate(**kwargs):
        return SimpleNamespace(token_ids=[1, 2], log_probs=[-0.2], stop_reason="stop")

    server.generate = generate
    with pytest.raises(ValueError, match="different token"):
        await harness_rollout.make_inprocess_post_fn(server)("unused", {"input_ids": [1], "return_logprob": True})


@pytest.mark.parametrize(
    "probabilities, token_ids",
    [([(-0.2, 45, None)], [99]), ([(-0.2, -1, None)], [-1]), ([("invalid", 45, None)], [45]), ([()], [45])],
)
async def test_provider_rejects_corrupted_token_probability_pairs(probabilities, token_ids):
    async def post(url, payload):
        return {"output_ids": token_ids, "meta_info": {"output_token_logprobs": probabilities}}

    provider = SGLangProvider("unused", CharacterTokenizer(), {}, 4096, post_fn=post, require_logprobs=True)
    with pytest.raises(GenerationAborted):
        await provider.complete([Message("user", "test")], [])
    assert provider.aborted and not provider.step_captures


async def test_zero_reward_is_not_automatically_correct(monkeypatch):
    episode = {"trajectory": None, "eval_result": EvalResult(False, 0.0, "wrong", 0.0)}
    spec = SimpleNamespace(prm=NullPRM(), reward_shaping=False, extra_reward_fn=None)
    monkeypatch.setattr(harness_rollout, "load_harness_config", lambda sample: spec)
    result = await harness_rollout.reward_func(harness_rollout.VerlSample(""), episode)
    assert result["is_correct"] == 0


@pytest.mark.parametrize("tool_format", ["json", "qwen3_coder"])
async def test_real_matpo_child_has_system_prompt_and_captured_probabilities(monkeypatch, tool_format):
    tokenizer = CharacterTokenizer()
    monkeypatch.setattr(spawn_rl, "_CHILD_CONFIGS", {})
    spawn_rl.register_child_config("helper", [], "You are a careful arithmetic helper.", "Solve: ", max_steps=2)

    async def ask_helper(task):
        return await spawn_rl.spawn_subagent_rl(task, "helper")

    tool = Tool(
        "ask_helper",
        "Ask the arithmetic helper",
        {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
        ask_helper,
    )
    spec = VerlConfigSpec(
        task_builder=SimpleNamespace(build=lambda sample: RLTask(description="Compute 6 * 7", label=sample.label)),
        tools=[tool],
        evaluator_cls=AnswerEvaluator,
        prm=NullPRM(),
        max_steps=3,
        tool_call_format=tool_format,
        tool_result_role="user",
    )
    monkeypatch.setattr(harness_rollout, "load_harness_config", lambda sample: spec)
    tool_call = (
        '<tool_call>{"name":"ask_helper","arguments":{"task":"6 * 7"}}</tool_call>'
        if tool_format == "json"
        else "<tool_call><function=ask_helper><parameter=task>6 * 7</parameter></function></tool_call>"
    )
    replies = iter([tool_call, "42", "The answer is 42."])
    prompts = []

    class Server:
        async def generate(self, **kwargs):
            prompts.append(tokenizer.decode(kwargs["prompt_ids"]))
            assert kwargs["sampling_params"]["logprobs"]
            tokens = tokenizer.encode(next(replies))
            return SimpleNamespace(token_ids=tokens, log_probs=[-0.25] * len(tokens), stop_reason="stop")

    episode = await harness_rollout.generate(
        harness_rollout.VerlSample("Compute 6 * 7", label="42"),
        {"max_new_tokens": 256},
        "http://unused/generate",
        tokenizer,
        Server(),
    )
    assert episode["eval_result"].passed
    assert len(prompts) == 3
    assert "<system>You are a careful arithmetic helper.</system>" in prompts[1]
    assert "Solve: 6 * 7" in prompts[1]
    assert all("<think>" not in prompt for prompt in prompts)
    assert len(episode["sub_episodes"]) == 1
    child = episode["sub_episodes"][0]
    assert tokenizer.decode(child["response_ids"]) == "42"
    assert child["loss_mask"] == [1, 1]
    assert child["rollout_log_probs"] == [-0.25, -0.25]
    assert 0 in episode["loss_mask"] and 1 in episode["loss_mask"]
