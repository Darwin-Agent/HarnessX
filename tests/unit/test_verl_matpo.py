import numpy as np
import pytest

torch = pytest.importorskip("torch")

from recipe.verl.matpo.advantage import compute_matpo_advantage
from recipe.verl.matpo.batch import parent_row_indices


def test_children_inherit_exact_parent_not_group_mean():
    rewards = torch.tensor([[1.0], [3.0], [99.0], [99.0]])
    masks = torch.ones_like(rewards)
    metadata = {"is_subagent": [False, False, True, True], "episode_id": ["first", "second", "second", "first"]}
    advantages, returns = compute_matpo_advantage(
        rewards, masks, index=["group"] * 4, non_tensor_batch=metadata, norm_adv_by_std_in_grpo=False
    )
    assert advantages[:, 0].tolist() == [-1.0, 1.0, 1.0, -1.0]
    assert torch.equal(advantages, returns)


def test_failed_and_padding_rows_do_not_change_parent_statistics():
    rewards = torch.tensor([[1.0], [3.0], [999.0], [999.0], [0.0]])
    masks = torch.tensor([[1], [1], [0], [1], [1]])
    metadata = {
        "is_subagent": [False, False, False, False, True],
        "is_pad": [False, False, False, True, False],
        "episode_id": ["a", "b", "failed", "padding", "a"],
    }
    advantages, _ = compute_matpo_advantage(
        rewards, masks, index=["group"] * 5, non_tensor_batch=metadata, norm_adv_by_std_in_grpo=False
    )
    assert advantages[:, 0].tolist() == [-1.0, 1.0, 0.0, 0.0, -1.0]


def test_normalization_and_interleaved_loss_masks():
    rewards = torch.tensor([[1.0, 0, 0], [3.0, 0, 0], [0.0, 0, 0]])
    masks = torch.tensor([[1, 0, 1], [1, 1, 0], [1, 0, 1]])
    metadata = {"is_subagent": [False, False, True], "episode_id": ["a", "b", "a"]}
    advantages, _ = compute_matpo_advantage(rewards, masks, index=[0, 0, 0], non_tensor_batch=metadata)
    expected = 1 / (torch.tensor(2.0).sqrt() + 1e-6)
    assert torch.allclose(advantages[:, 0], torch.tensor([-expected, expected, -expected]))
    assert torch.equal(advantages[masks == 0], torch.zeros_like(advantages[masks == 0]))


def test_child_with_missing_or_failed_parent_is_rejected():
    with pytest.raises(ValueError, match="missing or failed parent"):
        compute_matpo_advantage(
            torch.zeros(2, 1),
            torch.tensor([[0], [1]]),
            non_tensor_batch={"is_subagent": [False, True], "episode_id": ["a", "a"]},
        )


def test_exact_parent_expansion_indices():
    indices = parent_row_indices([False, False, True, True, True], ["a", "b", "b", "a", "b"], 2)
    assert np.array_equal(indices, [0, 1, 1, 0, 1])


@pytest.mark.parametrize(
    "flags,episodes,count",
    [
        ([False, True], ["a", "missing"], 1),
        ([False, False], ["a", "a"], 2),
        ([True, False], ["a", "a"], 1),
        ([False], ["a"], 2),
    ],
)
def test_invalid_parent_mapping_is_rejected(flags, episodes, count):
    with pytest.raises(ValueError):
        parent_row_indices(flags, episodes, count)
