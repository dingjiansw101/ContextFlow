from collections import defaultdict
import json

from tqdm import tqdm

from openpi.training import config as _config
from openpi.training.data_loader import create_dataset
from openpi.training.data_loader import transform_dataset
from openpi.transforms import InjectDemoPrompt


def build_lookup_tables(hf_dataset):
    """
    Args:
        hf_dataset: A Hugging Face Dataset where 'task_index', 'episode_index', and 'index'
                    might be scalar tensors instead of plain Python ints.

    Returns:
        task_to_episode: dict[task_index, list of episode_index]
        episode_to_indexes: dict[episode_index, list of index_idx]
    """
    task_to_episode = defaultdict(set)
    episode_to_indexes = defaultdict(list)

    # Use tqdm to show a progress bar while iterating
    for i in tqdm(range(len(hf_dataset)), desc="Building lookup tables"):
        row = hf_dataset[i]
        # Convert to Python ints if they're 0-dim tensors
        task_idx = int(row["task_index"])  # e.g. tensor(3) -> 3
        episode_idx = int(row["episode_index"])
        index_idx = int(row["index"])

        task_to_episode[task_idx].add(episode_idx)
        episode_to_indexes[episode_idx].append(index_idx)

    # Convert sets to sorted lists
    for t in task_to_episode:
        task_to_episode[t] = sorted(task_to_episode[t])
    for e in episode_to_indexes:
        episode_to_indexes[e].sort()

    return dict(task_to_episode), dict(episode_to_indexes)


def save_lookup_tables(task_to_episode, episode_to_indexes, output_dir="."):
    """
    Saves the two lookup dictionaries as JSON files in 'output_dir'.

    Args:
        task_to_episode (dict): {task_index: [episode_index, ...]}
        episode_to_indexes (dict): {episode_index: [index_idx, ...]}
        output_dir (str): Directory path where the JSON files will be written.
    """
    # Ensure the output directory exists
    import os

    os.makedirs(output_dir, exist_ok=True)

    task_to_episode_path = os.path.join(output_dir, "task_to_episode.json")
    episode_to_indexes_path = os.path.join(output_dir, "episode_to_indexes.json")

    # Write 'task_to_episode' to JSON
    with open(task_to_episode_path, "w") as f:
        json.dump(task_to_episode, f, indent=2)

    # Write 'episode_to_indexes' to JSON
    with open(episode_to_indexes_path, "w") as f:
        json.dump(episode_to_indexes, f, indent=2)

    print(f"Wrote task_to_episode to {task_to_episode_path}")
    print(f"Wrote episode_to_indexes to {episode_to_indexes_path}")


def test_libero_dataset():
    config = _config.get_config("pi0_libero")

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_dataset(data_config, config.model)
    # import ipdb; ipdb.set_trace()
    hf_dataset = dataset._dataset.hf_dataset
    task_to_episode, episode_to_index = build_lookup_tables(hf_dataset)
    save_lookup_tables(task_to_episode, episode_to_index, "assets/pi0_libero/")

    inject_transform = InjectDemoPrompt(dataset)
    import ipdb

    ipdb.set_trace()

    for i in range(len(dataset)):
        print(dataset[i].keys())
        item = inject_transform(dataset[i])
        import ipdb

        ipdb.set_trace()
        # dict_keys(['image', 'wrist_image', 'state', 'actions',
        # 'timestamp', 'frame_index', 'episode_index', 'index',
        # 'task_index', 'actions_is_pad', 'prompt'])
        # The dataset sample orders are the exactly the same as shown in huggingface
        # TODO: write a transform to randomly read a demonstration according to tha task index
        # select the data with a specific episode index.
        # Then add the demonstration to the dataset
        # check the details of LeRobotDataset to see if we can use its functions

    # dataset = transform_dataset(dataset, data_config, skip_norm_stats=False)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=True)
    for i in range(len(dataset)):
        print(dataset[i].keys())
        # dict_keys(['state', 'image', 'image_mask', 'actions',
        # 'tokenized_prompt', 'tokenized_prompt_mask'])
        # import ipdb;
        # ipdb.set_trace()


if __name__ == "__main__":
    test_libero_dataset()
