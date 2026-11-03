"""Real CPU optimization and artifact handoff through the shared alternating loop."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from harnessx.core.harness import HarnessConfig
from recipe.coevolution.behavior import SYSTEM
from recipe.coevolution.run import Settings, alternate, digest, model_digest
from recipe.coevolution.smoke import CPUBackend, TinyPolicy
from recipe.coevolution.verify import verify


@pytest.fixture
def settings(tmp_path):
    return Settings(
        backend="cpu",
        output=str(tmp_path / "run"),
        rounds=2,
        steps=3,
        model_warmup_rounds=1,
    )


async def test_two_rounds_really_update_weights_and_feed_back_model_and_harness(settings):
    state = await alternate(settings, CPUBackend(settings))
    assert verify(Path(settings.output))["completed_rounds"] == 2
    initial, first, second = state["initial"], *state["rounds"]
    assert first["input_model"] == initial["model"]
    assert second["input_model"] == first["model"]
    assert second["input_model_hash"] == first["model_hash"]
    assert second["input_harness"] == first["harness"]
    assert first["model_hash"] == initial["model_hash"]
    assert second["model_hash"] != first["model_hash"]
    assert len(Path(first["buffer"]).read_text().splitlines()) == 2
    assert len(Path(second["buffer"]).read_text().splitlines()) == 2
    assert len({initial["harness_hash"], first["harness_hash"], second["harness_hash"]}) == 3
    for record in (first, second):
        work = Path(record["buffer"]).parent.parent
        loaded = json.loads((work / "loaded_model.json").read_text())
        assert loaded["weights"] == record["input_model_hash"]
        training = json.loads((work / "training.json").read_text())
        assert record["harness_selection"] == "latest_proposal"
        assert training["model_updated"] == record["model_updated"]
        if record["model_updated"]:
            assert training["accepted_harness"] == record["harness"]
            assert len(training["updates"]) == settings.steps
            before = TinyPolicy.load(record["input_model"]).weights.detach()
            after = TinyPolicy.load(record["model"]).weights.detach()
            assert torch.isfinite(after).all() and not torch.equal(before, after)
        else:
            assert record["model"] == record["input_model"]
    extended = replace(settings, rounds=3)
    resumed = await alternate(extended, CPUBackend(extended), resume=True)
    assert resumed["rounds"][:2] == state["rounds"]
    assert resumed["rounds"][2]["input_model"] == second["model"]
    assert verify(Path(settings.output))["completed_rounds"] == 3


async def test_latest_harness_is_carried_forward_without_score_gate(settings):
    settings = replace(settings, rounds=1)
    state = await alternate(settings, CPUBackend(settings))
    record = state["rounds"][0]
    assert record["harness_selection"] == "latest_proposal"
    assert record["harness_hash"] != state["initial"]["harness_hash"]
    assert record["model_hash"] == state["initial"]["model_hash"]


async def test_candidate_cannot_replace_reward_and_is_rejected(settings):
    class InvalidCandidateBackend(CPUBackend):
        async def propose(self, model, incumbent, trajectories, output, index):
            candidate = await super().propose(model, incumbent, trajectories, output, index)
            config = HarnessConfig.from_yaml_file(candidate)
            config.plugins = [{"_target_": "fake.RewardPlugin"}]
            config.to_yaml_file(candidate)
            return candidate

    settings = replace(settings, rounds=1)
    with pytest.raises(ValueError, match="AEGIS produced an invalid Harness"):
        await alternate(settings, InvalidCandidateBackend(settings))
    assert json.loads((Path(settings.output) / "state.json").read_text())["rounds"] == []


async def test_failed_training_is_not_committed_and_resume_uses_fresh_attempt(settings):
    class BrokenBackend(CPUBackend):
        async def train(self, *args):
            raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        await alternate(settings, BrokenBackend(settings))
    root = Path(settings.output)
    assert json.loads((root / "state.json").read_text())["rounds"] == []
    state = await alternate(settings, CPUBackend(settings), resume=True)
    assert "attempt_02" in Path(state["rounds"][0]["buffer"]).parent.parent.name
    assert (root / "round_001_attempt_01" / "search.json").is_file()


async def test_no_op_model_training_cannot_be_reported_as_coevolution(settings):
    class NoOpBackend(CPUBackend):
        async def train(self, model, harness, work):
            output = work / "model"
            output.mkdir()
            (output / "policy.pt").write_bytes((model / "policy.pt").read_bytes())
            return output

    with pytest.raises(ValueError, match="did not change"):
        await alternate(settings, NoOpBackend(settings))
    assert json.loads((Path(settings.output) / "state.json").read_text())["rounds"] == []


async def test_resume_rejects_changed_settings_and_tampered_weights(settings):
    state = await alternate(settings, CPUBackend(settings))
    changed = replace(settings, seed=7)
    with pytest.raises(ValueError, match="settings or input"):
        await alternate(changed, CPUBackend(changed), resume=True)
    latest = Path(state["rounds"][-1]["model"]) / "policy.pt"
    latest.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        await alternate(settings, CPUBackend(settings), resume=True)


async def test_existing_output_requires_explicit_resume(settings):
    await alternate(settings, CPUBackend(settings))
    with pytest.raises(FileExistsError, match="resume"):
        await alternate(settings, CPUBackend(settings))


async def test_evolved_prompt_is_used_by_actual_rollout(settings):
    backend = CPUBackend(settings)
    root = Path(settings.output)
    root.mkdir()
    model = await backend.initialize(root)
    from recipe.coevolution.behavior import baseline_behavior

    incumbent = root / "incumbent.yaml"
    baseline_behavior("Answer the question.").to_yaml_file(incumbent)
    candidate = await backend.propose(model, incumbent, root, root / "proposal", 1)
    assert await backend.evaluate(model, incumbent, root / "before") == 0.5
    assert await backend.evaluate(model, candidate, root / "after") == 1.0
    assert model_digest(model)
    config = HarnessConfig.from_yaml_file(candidate)
    prompt = next(processor for processor in config.processors if processor["_target_"] == SYSTEM)
    assert "Use parity" in prompt["system_builder"]["text"]
    assert digest(incumbent) != digest(candidate)
