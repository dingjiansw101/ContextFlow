"""
Minimal example script for converting Libero datasets to LeRobot format as separate datasets.

This script creates 4 separate LeRobot datasets (one for each Libero benchmark suite)
instead of combining them into a single dataset.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot_separate.py --data_dir /path/to/your/data

If you want to push your datasets to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot_separate.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting datasets will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

# Mapping from raw dataset names to output repository names
DATASET_MAPPING = {
    "libero_10_no_noops": "vo2yager/libero_10",
    "libero_goal_no_noops": "vo2yager/libero_goal",
    "libero_object_no_noops": "vo2yager/libero_object",
    "libero_spatial_no_noops": "vo2yager/libero_spatial",
}


def main(data_dir: str, *, push_to_hub: bool = False):
    # Process each dataset separately
    for raw_dataset_name, repo_name in DATASET_MAPPING.items():
        print(f"\n{'='*60}")
        print(f"Processing {raw_dataset_name} -> {repo_name}")
        print(f"{'='*60}\n")

        # Clean up any existing dataset in the output directory
        output_path = LEROBOT_HOME / repo_name
        if output_path.exists():
            print(f"Removing existing dataset at {output_path}")
            shutil.rmtree(output_path)

        # Create LeRobot dataset, define features to store
        # OpenPi assumes that proprio is stored in `state` and actions in `action`
        # LeRobot assumes that dtype of image data is `image`
        dataset = LeRobotDataset.create(
            repo_id=repo_name,
            robot_type="panda",
            fps=10,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

        # Load and process only this specific raw dataset
        print(f"Loading raw dataset: {raw_dataset_name}")
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")

        episode_count = 0
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                    }
                )
            dataset.save_episode(task=step["language_instruction"].decode())
            episode_count += 1

        print(f"Added {episode_count} episodes to {repo_name}")

        # Consolidate the dataset, skip computing stats since we will do that later
        print(f"Consolidating dataset: {repo_name}")
        dataset.consolidate(run_compute_stats=False)

        # Optionally push to the Hugging Face Hub
        if push_to_hub:
            print(f"Pushing {repo_name} to Hugging Face Hub")
            dataset.push_to_hub(
                tags=["libero", "panda", "rlds"],
                private=False,
                push_videos=True,
                license="apache-2.0",
            )

        print(f"✓ Completed processing {repo_name}")


if __name__ == "__main__":
    tyro.cli(main)
