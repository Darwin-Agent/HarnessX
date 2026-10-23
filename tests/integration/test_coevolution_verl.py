"""Exercise production veRL handoffs with real DataProto and command construction."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verl")

from recipe.coevolution.behavior import route_harnesses
from recipe.coevolution.run import Settings
from recipe.coevolution.verl_backend import offline_training_command
from verl import DataProto


def test_real_dataproto_routes_variants_with_shared_uids_and_parent_order(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setenv("HARNESSX_HARNESS_POOL", '["/latest.yaml", "/previous.yaml"]')
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.ones(2, 3, dtype=torch.long)},
        non_tensors={"uid": np.array(["prompt_a", "prompt_b"], dtype=object)},
    )
    repeated = batch.repeat(repeat_times=4, interleave=True)
    route_harnesses(repeated, 4)
    assert repeated.non_tensor_batch["uid"].tolist() == ["prompt_a"] * 4 + ["prompt_b"] * 4
    assert repeated.non_tensor_batch["harness_index"].tolist() == [0, 1, 0, 1] * 2


def test_model_update_command_reads_saved_buffer_without_rollout(tmp_path):
    settings = Settings(
        backend="verl",
        output=str(tmp_path / "run"),
        steps=3,
        train_batch_size=2,
        gpus=4,
    )
    model = tmp_path / 'model "$HOME"'
    buffer = tmp_path / "saved trajectories.jsonl"
    output = tmp_path / "round"

    command, environment = offline_training_command(settings, model, [buffer], output)

    assert "recipe.coevolution.offline_train" in command
    assert command[command.index("--model") + 1] == str(model)
    assert command[command.index("--output") + 1] == str(output)
    assert command[command.index("--buffer") + 1] == str(buffer)
    assert "sglang.launch_server" not in command
    assert "--train-file" not in command
    assert command[command.index("--steps") + 1] == "3"
    assert command[command.index("--batch-size") + 1] == "2"
    assert command[command.index("--nproc_per_node=4")] == "--nproc_per_node=4"
    assert environment["HX_ROOT"]
