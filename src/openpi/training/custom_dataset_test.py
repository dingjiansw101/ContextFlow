import random

from openpi.training.custom_dataset import CustomLeRobotDataset
from openpi.transforms import stable_sample_seed


def test_custom_lerobot_dataset_seeded_episode_selection_is_deterministic():
    dataset = CustomLeRobotDataset.__new__(CustomLeRobotDataset)
    dataset.random_select = True
    dataset.seed_base = 123
    dataset.demo_selection_seed_compat = False
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
    dataset.demo_selection_seed_compat = False

    assert dataset.select_incontext_episode(
        [3, 4, 5],
        current_ep_idx=0,
        task_index=0,
        sample_index=17,
    ) == 3


def test_custom_lerobot_dataset_seed_compat_reproduces_inject_demo_indexes_draw():
    """demo_selection_seed_compat must land on the episode InjectDemoIndexes would pick.

    This is the whole point of the flag: it lets a deterministic consistency check compare
    this loader against the legacy cache-based loader batch-for-batch. Without it the two
    seeded streams diverge (different domain tag, components, and draw call).
    """
    task_index, index, frame_index, episode_index = 3, 250, None, 7
    episodes = [0, 1, 2, 3, 4, 5, 6, 7]

    dataset = CustomLeRobotDataset.__new__(CustomLeRobotDataset)
    dataset.random_select = True
    dataset.seed_base = 12345
    dataset.demo_selection_seed_compat = True

    picked = dataset.select_incontext_episode(
        episodes,
        current_ep_idx=episode_index,
        task_index=task_index,
        sample_index=index,
        legacy_seed_components=(index, frame_index, episode_index),
    )

    # Exactly what InjectDemoIndexes does for the same sample (see transforms.py).
    rng = random.Random(
        stable_sample_seed(12345, "InjectDemoIndexes", task_index, index, frame_index, episode_index)
    )
    assert picked == rng.sample([int(ep) for ep in episodes], 1)[0]

    # And it must differ from the native stream, otherwise the flag would be redundant.
    dataset.demo_selection_seed_compat = False
    native = dataset.select_incontext_episode(
        episodes,
        current_ep_idx=episode_index,
        task_index=task_index,
        sample_index=index,
    )
    assert native != picked


def test_custom_lerobot_dataset_without_seed_base_uses_global_random():
    dataset = CustomLeRobotDataset.__new__(CustomLeRobotDataset)
    dataset.random_select = True
    dataset.seed_base = None
    dataset.demo_selection_seed_compat = False
    episodes = [0, 1, 2, 3]

    random.seed(123)
    result = dataset.select_incontext_episode(
        episodes,
        current_ep_idx=0,
        task_index=0,
        sample_index=17,
    )

    assert result == random.Random(123).choice(episodes)
