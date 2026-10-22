"""Behavior safety, shared GRPO routing, command construction and budget guards."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from harnessx.core.events import BeforeModelEvent, Message, StepStartEvent
from recipe.coevolution.behavior import baseline_behavior, resolve_behavior, route_harnesses, validate_behavior
from recipe.coevolution.run import Settings, model_digest
from recipe.coevolution.verl_backend import offline_training_command
from recipe.verl.processor.early_commit import EarlyCommitProcessor


def test_behavior_roundtrip_and_explicit_config_priority(tmp_path, monkeypatch):
    config = baseline_behavior('Use this exact evolved prompt: $HOME, quotes: "x".')
    path = tmp_path / "behavior.yaml"
    config.to_yaml_file(path)
    monkeypatch.setenv("HARNESSX_HARNESS_POOL", '["/not/a/config"]')
    restored = resolve_behavior(explicit=path)
    assert restored.processors == config.processors
    validate_behavior(restored)


@pytest.mark.parametrize("slot,value", [("plugins", [{"_target_": "fake.Reward"}]), ("processors", [])])
def test_behavior_cannot_override_runtime_reward(slot, value):
    config = baseline_behavior()
    setattr(config, slot, value)
    with pytest.raises(ValueError):
        validate_behavior(config)


@pytest.mark.parametrize("raw", ['{"bad": 1}', "[3]", '"path"'])
def test_invalid_pool_is_not_silently_ignored(raw, monkeypatch):
    monkeypatch.setenv("HARNESSX_HARNESS_POOL", raw)
    with pytest.raises(ValueError, match="JSON list"):
        resolve_behavior()


def test_harness_variants_share_original_prompt_uid(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setenv("HARNESSX_HARNESS_POOL", '["/accepted.yaml", "/incumbent.yaml"]')

    class Batch:
        non_tensor_batch = {"uid": np.array(["a"] * 4 + ["b"] * 4, dtype=object)}

        def __len__(self):
            return 8

    batch = Batch()
    route_harnesses(batch, 4)
    assert batch.non_tensor_batch["uid"].tolist() == ["a"] * 4 + ["b"] * 4
    assert batch.non_tensor_batch["harness_index"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    with pytest.raises(ValueError, match="multiple"):
        route_harnesses(batch, 3)
    route_harnesses(batch, 1, validation=True)
    assert batch.non_tensor_batch["harness_index"].tolist() == [0] * len(batch)


def test_offline_training_handoff_uses_buffer_only_argv(tmp_path):
    settings = Settings(
        backend="verl",
        output=str(tmp_path),
        steps=7,
        train_batch_size=3,
        buffer_capacity=256,
        gpus=4,
        learning_rate=2e-6,
    )
    model = tmp_path / 'model "$HOME"'
    buffer = tmp_path / 'buffer with spaces.jsonl'
    output = tmp_path / "round output"

    argv, environment = offline_training_command(settings, model, [buffer], output)

    assert "recipe.coevolution.offline_train" in argv
    assert "--buffer" in argv
    assert str(buffer) in argv
    assert str(model) in argv
    assert str(output) in argv
    assert argv[argv.index("--steps") + 1] == "7"
    assert argv[argv.index("--batch-size") + 1] == "3"
    assert argv[argv.index("--buffer-capacity") + 1] == "256"
    assert argv[argv.index("--learning-rate") + 1] == "2e-06"
    assert argv[argv.index("--nproc_per_node=4")] == "--nproc_per_node=4"
    assert environment["HX_ROOT"]


def test_model_directory_requires_real_weight_files(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="No exported model weights"):
        model_digest(tmp_path)


async def test_early_commit_ratio_actually_nudges_before_context_overflow():
    processor = EarlyCommitProcessor(max_steps=12, token_budget_ratio=0.5)
    event = StepStartEvent(run_id="smoke", step_id="step", context_window=10)
    assert [result async for result in processor.on_step_start(event)] == [event]
    before = BeforeModelEvent(run_id="smoke", step_id="step", messages=(Message(role="user", content="word " * 30),))
    output = [result async for result in processor.on_before_model(before)]
    assert "LAST CHANCE" in output[0].messages[-1].content


def test_per_run_behavior_keeps_the_original_evaluator_and_skill_runtime():
    from recipe.verl.harness import make_verl_harness
    from recipe.verl.registry import load_harness_config
    from recipe.verl.harness_rollout import VerlSample
    from recipe.verl.processor.evaluation import VerlEvaluationProcessor

    sample = VerlSample("question", "correct label")
    spec = load_harness_config(sample)
    config = make_verl_harness(spec, SimpleNamespace(), spec.task_builder.build(sample), baseline_behavior("Evolved"))
    assert len(config.processors) == 2
    assert sum(isinstance(processor, VerlEvaluationProcessor) for processor in config._rt_procs) == 1
    assert len(config._rt_procs) == 2


async def test_standalone_research_factory_honors_sample_system_prompt():
    from harnessx.processors.context.system_prompt import SystemPromptProcessor
    from recipe.verl.harness import make_verl_harness
    from recipe.verl.registry import load_harness_config
    from recipe.verl.harness_rollout import VerlSample

    sample = VerlSample(
        [{"role": "system", "content": "Custom task instructions"}, {"role": "user", "content": "Question"}]
    )
    spec = load_harness_config(sample)
    config = make_verl_harness(spec, SimpleNamespace(), spec.task_builder.build(sample))
    processor = next(processor for processor in config._rt_procs if isinstance(processor, SystemPromptProcessor))
    assert await processor.system_builder.build() == "Custom task instructions"


async def test_production_metaagent_receives_incumbent_and_current_task_model(tmp_path, monkeypatch):
    from harnessx.meta_harness import MetaAgent
    from recipe.coevolution.verl_backend import VerlBackend

    calls = {}

    async def evolve(self, **kwargs):
        calls["inner_model"] = self.inner_model
        calls.update(kwargs)
        output = kwargs["output_dir"] / "config.yaml"
        output.parent.mkdir()
        baseline_behavior().to_yaml_file(output)
        return output

    monkeypatch.setattr(MetaAgent, "evolve", evolve)
    model_config = tmp_path / "meta.yaml"
    model_config.write_text(
        "main:\n  _target_: harnessx.providers.litellm_provider.LiteLLMProvider\n  model: openai/test-meta-model\n"
    )
    settings = Settings(backend="verl", output=str(tmp_path), meta_model=str(model_config))
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "existing.example")
    from recipe.verl.research.config import cfg

    for name in cfg.to_env_dict():
        monkeypatch.setenv(name, "test-value")
    backend = VerlBackend(settings)
    backend.tokenizer = SimpleNamespace()
    trajectories = tmp_path / "trajectories"
    trajectories.mkdir()
    incumbent = tmp_path / "incumbent.yaml"
    baseline_behavior().to_yaml_file(incumbent)
    candidate = await backend.propose(tmp_path / "current-model", incumbent, trajectories, tmp_path / "candidate", 1)
    assert candidate.is_file()
    assert calls["current_config"] == incumbent
    assert calls["trajectories_dir"] == trajectories
    assert calls["inner_model"].main.model == "openai/test-meta-model"
    assert calls["replay_model"].main.url == backend.url + "/generate"
    assert calls["replay_model"].main.tokenizer is backend.tokenizer
    assert calls["replay_model"].main.rollout_max_context_len == 32768
    assert calls["replay_model"].main._tool_call_format == cfg.tool_call_format
    assert calls["replay_model"].main.tool_result_role == cfg.tool_result_role
    assert calls["replay_model"].main._strip_think_prefix is True


async def test_same_output_cannot_run_two_schedulers(tmp_path):
    import fcntl
    from recipe.coevolution.run import alternate

    settings = Settings(backend="cpu", output=str(tmp_path / "run"))
    with (tmp_path / ".run.coevolution.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another process"):
            await alternate(settings, SimpleNamespace())


def test_round_explores_before_aegis_then_trains_from_buffer(tmp_path):
    from contextlib import asynccontextmanager

    from recipe.coevolution.behavior import baseline_behavior
    from recipe.coevolution.run import alternate

    events = []
    explored_harnesses = []

    class OrderedBackend:
        async def initialize(self, root):
            model = root / "seed-model"
            model.mkdir()
            (model / "policy.pt").write_bytes(b"seed")
            return model

        @asynccontextmanager
        async def search(self, model, work):
            events.append("aegis_search_start")
            yield
            events.append("aegis_search_end")

        async def collect(self, model, pool, work, round_index):
            events.append("explore")
            assert len(pool) == 1
            explored_harnesses.append(pool[0].read_bytes())
            trajectories = work / "exploration"
            trajectories.mkdir()
            (trajectories / "episode.md").write_text("saved exploration")
            buffer = work / "buffer" / "trajectories.jsonl"
            buffer.parent.mkdir()
            buffer.write_text("saved records")
            return buffer

        async def propose(self, model, incumbent, trajectories, output, index):
            events.append("aegis_evolve")
            assert (trajectories / "episode.md").is_file()
            output.mkdir()
            proposal = baseline_behavior("Evolved policy")
            path = output / "config.yaml"
            proposal.to_yaml_file(path)
            return path

        async def train(self, model, harness, work):
            events.append("offline_train")
            assert (work / "buffer" / "trajectories.jsonl").is_file()
            updated = work / "model"
            updated.mkdir()
            (updated / "policy.pt").write_bytes(f"updated-{work.name}".encode())
            return updated

    state = asyncio.run(alternate(Settings(backend="cpu", output=str(tmp_path / "run"), rounds=2), OrderedBackend()))

    assert events.index("explore") < events.index("aegis_search_start")
    assert events.index("explore") < events.index("aegis_evolve") < events.index("offline_train")
    assert len(state["rounds"]) == 2
    assert state["rounds"][0]["harness_selection"] == "latest_proposal"
    assert state["rounds"][0]["input_harness"] == state["initial"]["harness"]
    from harnessx.core.harness import HarnessConfig

    latest = HarnessConfig.from_yaml_file(state["rounds"][0]["harness"])
    evolved = HarnessConfig.from_yaml_file(state["rounds"][0]["harness_proposal"])
    assert latest.processors == evolved.processors
    assert explored_harnesses[1] == Path(state["rounds"][0]["harness"]).read_bytes()


def test_buffer_allows_warmup_without_cross_harness_groups(tmp_path):
    import json

    from recipe.coevolution.buffer import load_records

    path = tmp_path / "trajectories.jsonl"
    path.write_text(json.dumps({
        "task_uid": "task-1",
        "harness_id": "harness-a",
        "round": 1,
        "source_model": "model-a",
        "reward": 1.0,
        "prompt_ids": [1],
        "response_ids": [2],
        "response_mask": [1],
        "old_logprobs": [-0.5],
    }) + "\n")

    assert load_records([path], allow_empty=True) == []


def test_buffer_allows_warmup_without_reward_variation(tmp_path):
    import json

    from recipe.coevolution.buffer import load_records

    path = tmp_path / "trajectories.jsonl"
    common = {
        "task_uid": "task-1",
        "round": 1,
        "source_model": "model-a",
        "reward": 1.0,
        "prompt_ids": [1],
        "response_ids": [2],
        "response_mask": [1],
        "old_logprobs": [-0.5],
    }
    rows = [{**common, "harness_id": harness} for harness in ("harness-a", "harness-b")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert load_records([path], allow_empty=True) == []
