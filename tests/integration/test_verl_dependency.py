import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
TensorDict = pytest.importorskip("tensordict").TensorDict
DataProto = pytest.importorskip("verl.protocol").DataProto

from recipe.verl.matpo.batch import expand_parent_batch, order_rollout_rows
from recipe.verl.matpo.padding import pad_matpo_batch


def make_batch(episodes, flags):
    size = len(episodes)
    return DataProto(
        batch=TensorDict(
            {
                "response_mask": torch.ones(size, 2),
                "attention_mask": torch.ones(size, 4),
                "token_level_rewards": torch.ones(size, 2),
            },
            batch_size=[size],
        ),
        non_tensor_batch={
            "episode_id": np.asarray(episodes, dtype=object),
            "is_subagent": np.asarray(flags, dtype=bool),
        },
    )


def test_real_dataproto_padding_preserves_attention_but_masks_loss():
    original = make_batch(["a", "b", "c"], [False] * 3)
    padded, count = pad_matpo_batch(original, 8)
    assert count == 5 and len(padded) == 8
    assert padded.batch["attention_mask"].all()
    assert not padded.batch["response_mask"][3:].any()
    assert not padded.batch["token_level_rewards"][3:].any()
    assert padded.non_tensor_batch["is_pad"].tolist() == [False] * 3 + [True] * 5
    assert original.batch["response_mask"].all()
    assert original.batch["token_level_rewards"].all()


def test_worker_concatenation_order_and_exact_parent_expansion():
    first = make_batch(["a", "a"], [False, True])
    second = make_batch(["b", "b"], [False, True])
    output = order_rollout_rows(DataProto.concat([first, second]))
    assert output.non_tensor_batch["episode_id"].tolist() == ["a", "b", "a", "b"]
    parents = make_batch(["a", "b"], [False, False])
    parents.batch["token_level_rewards"][1] *= 2
    expanded = expand_parent_batch(parents, output)
    assert expanded.batch["token_level_rewards"][:, 0].tolist() == [1, 2, 1, 2]
    validation = order_rollout_rows(DataProto.concat([first, second]), validation=True)
    assert len(validation) == 2
    assert not validation.non_tensor_batch["is_subagent"].any()


@pytest.mark.parametrize("divisor", [0, -1])
def test_invalid_padding_divisor_is_rejected(divisor):
    with pytest.raises(ValueError):
        pad_matpo_batch(make_batch(["a"], [False]), divisor)


@pytest.mark.parametrize("mode", ["token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean"])
def test_fully_masked_microbatch_has_zero_loss_and_gradients(mode):
    from verl.trainer.ppo.core_algos import agg_loss

    losses = torch.tensor([[1.0, 2.0]], requires_grad=True)
    loss = agg_loss(losses, torch.zeros_like(losses), mode)
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert torch.equal(losses.grad, torch.zeros_like(losses))


@pytest.mark.parametrize("failed", [False, True])
async def test_adapter_outputs_validate_against_pinned_verl_schema(monkeypatch, failed):
    from recipe.verl import agent_loop
    from recipe.verl.research import rewards
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

    loop = agent_loop.HarnessXAgentLoop.__new__(agent_loop.HarnessXAgentLoop)
    loop.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0, bos_token_id=1)
    loop.response_length = 8
    loop._sglang_url = "http://unused/generate"
    loop.server_manager = SimpleNamespace(_server_id_to_handle={"127.0.0.1:8000": None})
    child = {"prompt_ids": [1], "response_ids": [3], "loss_mask": [1], "rollout_log_probs": [-0.25]}

    async def generate(**kwargs):
        if failed:
            return None
        return {
            "status": "completed",
            "tokens": [1, 3],
            "response_length": 1,
            "loss_mask": [1],
            "rollout_log_probs": [-0.25],
            "sub_episodes": [child],
        }

    monkeypatch.setattr(agent_loop, "generate", generate)
    monkeypatch.setattr(rewards, "research_extra_reward_fn", lambda *args: {"accuracy": 1.0})
    output = await loop.run({}, raw_prompt=[])
    assert isinstance(output, AgentLoopOutput)
    assert output.extra_fields["rollout_failed"] is failed
    assert output.response_mask == ([0] if failed else [1])
    if not failed:
        assert len(output.sub_outputs) == 1
        child_output = output.sub_outputs[0]
        assert isinstance(child_output, AgentLoopOutput)
        assert child_output.extra_fields["episode_id"] == output.extra_fields["episode_id"]
        assert child_output.response_logprobs == [-0.25]
