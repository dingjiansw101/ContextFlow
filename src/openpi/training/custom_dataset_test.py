import random

from openpi.training.custom_dataset import CustomLeRobotDataset


def test_custom_lerobot_dataset_seeded_episode_selection_is_deterministic():
    dataset = CustomLeRobotDataset.__new__(CustomLeRobotDataset)
    dataset.random_select = True
    dataset.seed_base = 123
    episodes = [0, 1, 2, 3]

    random.seed(1)
    first = dataset.select_incontext_episode(
        episodes,
        current_ep_idx=0,
        task_index=0,
        sample_index=17,
    )
    random.seed(999)
    second = dataset.select_incontext_episode(
        episodes,
        current_ep_idx=0,
        task_index=0,
        sample_index=17,
    )

    assert first == second


def test_custom_lerobot_dataset_non_random_selection_uses_first_episode():
    dataset = CustomLeRobotDataset.__new__(CustomLeRobotDataset)
    dataset.random_select = False
    dataset.seed_base = 123

    assert dataset.select_incontext_episode(
        [3, 4, 5],
        current_ep_idx=0,
        task_index=0,
        sample_index=17,
    ) == 3


def test_custom_lerobot_dataset_without_seed_base_uses_global_random():
    dataset = CustomLeRobotDataset.__new__(CustomLeRobotDataset)
    dataset.random_select = True
    dataset.seed_base = None
    episodes = [0, 1, 2, 3]

    random.seed(123)
    result = dataset.select_incontext_episode(
        episodes,
        current_ep_idx=0,
        task_index=0,
        sample_index=17,
    )

    assert result == random.Random(123).choice(episodes)
