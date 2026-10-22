import json

import pytest

from recipe.coevolution.buffer import load_records


def trajectory(task_uid, harness_id, reward, *, round_number=1, source_model="model-a"):
    return {
        "task_uid": task_uid,
        "harness_id": harness_id,
        "round": round_number,
        "source_model": source_model,
        "prompt_ids": [1, 2],
        "response_ids": [3, 4],
        "response_mask": [1, 1],
        "old_logprobs": [-0.5, -0.7],
        "reward": reward,
    }


def write_buffer(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_groups_are_scoped_to_collection_round_and_model(tmp_path):
    path = write_buffer(
        tmp_path / "buffer.jsonl",
        [
            trajectory("same-task", "harness-a", 1.0, round_number=1, source_model="model-a"),
            trajectory("same-task", "harness-b", 0.0, round_number=1, source_model="model-a"),
            trajectory("same-task", "harness-a", 0.0, round_number=2, source_model="model-b"),
            trajectory("same-task", "harness-b", 1.0, round_number=2, source_model="model-b"),
        ],
    )

    records = load_records([path], capacity=4)

    assert len(records) == 4
    assert {record["round"] for record in records} == {1, 2}
    assert all(abs(record["advantage"]) == pytest.approx(1.0) for record in records)


def test_capacity_keeps_complete_groups_without_exceeding_limit(tmp_path):
    path = write_buffer(
        tmp_path / "buffer.jsonl",
        [
            trajectory("task-1", "harness-a", 1.0),
            trajectory("task-1", "harness-b", 0.0),
            trajectory("task-2", "harness-a", 0.0),
            trajectory("task-2", "harness-b", 1.0),
            trajectory("task-3", "harness-a", 1.0),
            trajectory("task-3", "harness-b", 0.0),
        ],
    )

    records = load_records([path], capacity=4)

    assert {record["task_uid"] for record in records} == {"task-2", "task-3"}
    assert len(records) == 4


def test_rejects_multiple_rollouts_for_same_harness_in_one_group(tmp_path):
    path = write_buffer(
        tmp_path / "buffer.jsonl",
        [
            trajectory("task", "harness-a", 1.0),
            trajectory("task", "harness-a", 0.0),
        ],
    )

    with pytest.raises(ValueError, match="multiple rollouts"):
        load_records([path], capacity=2)


def test_rejects_misaligned_or_missing_behavior_logprobs(tmp_path):
    record = trajectory("task", "harness-a", 1.0)
    record["old_logprobs"] = [-0.5]
    path = write_buffer(tmp_path / "buffer.jsonl", [record])

    with pytest.raises(ValueError, match="token annotations"):
        load_records([path], capacity=2)


def test_offline_batch_masks_prompt_and_keeps_behavior_logprob_alignment():
    torch = pytest.importorskip("torch")
    from recipe.coevolution.offline_train import TrajectoryDataset, collate_records

    records = [
        {
            **trajectory("task", "harness-a", 1.0),
            "advantage": 0.5,
        }
    ]
    dataset = TrajectoryDataset(records)
    batch = collate_records([dataset[0]], pad_token_id=0)

    assert batch["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert batch["response_mask"].tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert torch.allclose(batch["old_logprobs"], torch.tensor([[0.0, 0.0, -0.5, -0.7]]))
    assert batch["response_mask"][:, 1:].sum().item() == 2
