"""
Convert raw LIBERO-90 HDF5 files directly into a LeRobot dataset.

The raw HDF5 files are the official LIBERO libero_90 demonstrations, obtained by running
``third_party/libero/benchmark_scripts/download_libero_datasets.py --datasets libero_100``
(libero_90 ships inside the libero_100 archive) and pointing ``--data-dir`` at the extracted
``libero_90`` directory.

Usage:
    PYTHONPATH=src uv run examples/libero/convert_libero90_hdf5_to_lerobot.py --data_dir /path/to/libero_90
    PYTHONPATH=src uv run examples/libero/convert_libero90_hdf5_to_lerobot.py --data_dir /path/to/libero_90 --push_to_hub
"""

from __future__ import annotations

from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

DEFAULT_DATA_DIR = Path("data/libero_90")
DEFAULT_REPO_ID = "vo2yager/libero_90"
IMAGE_SIZE = (256, 256)
IMAGE_WRITER_THREADS = 10
REQUIRED_OBS_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "ee_states", "gripper_states")


def build_task_name(file_path: Path) -> str:
    """Derive the task string from a raw LIBERO filename."""
    task_name = file_path.name.removesuffix("_demo.hdf5").replace("_", " ").strip()
    if not task_name:
        raise ValueError(f"Could not derive a task name from `{file_path}`.")
    return task_name


def resize_image(image: np.ndarray) -> np.ndarray:
    """Rotate 180 degrees and resize a LIBERO image to the LeRobot image resolution."""
    # Match the existing LIBERO eval preprocessing, which flips both camera images.
    rotated = np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[::-1, ::-1])
    resized = Image.fromarray(rotated).resize(IMAGE_SIZE, resample=Image.Resampling.LANCZOS)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def get_demo_keys(data_group: h5py.Group, file_path: Path) -> list[str]:
    """Get demo keys in numeric order."""
    demo_keys = [key for key in data_group if key.startswith("demo_")]
    if not demo_keys:
        raise ValueError(f"No `demo_*` groups found in `{file_path}`.")

    try:
        return sorted(demo_keys, key=lambda key: int(key.split("_", maxsplit=1)[1]))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected demo key format in `{file_path}`: {demo_keys}") from exc


def validate_demo_arrays(file_path: Path, demo_key: str, actions: np.ndarray, obs_group: h5py.Group) -> None:
    """Ensure a demo contains the fields and sequence lengths this converter expects."""
    missing_keys = [key for key in REQUIRED_OBS_KEYS if key not in obs_group]
    if missing_keys:
        raise KeyError(f"Missing observation keys in `{file_path}` {demo_key}: {missing_keys}")

    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected 7D actions in `{file_path}` {demo_key}, got shape {actions.shape}.")

    num_steps = actions.shape[0]
    if num_steps == 0:
        raise ValueError(f"`{file_path}` {demo_key} has zero timesteps.")

    image_shape = obs_group["agentview_rgb"].shape
    wrist_image_shape = obs_group["eye_in_hand_rgb"].shape
    ee_state_shape = obs_group["ee_states"].shape
    gripper_state_shape = obs_group["gripper_states"].shape

    lengths = {
        "actions": num_steps,
        "agentview_rgb": image_shape[0],
        "eye_in_hand_rgb": wrist_image_shape[0],
        "ee_states": ee_state_shape[0],
        "gripper_states": gripper_state_shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Mismatched sequence lengths in `{file_path}` {demo_key}: {lengths}")

    if len(image_shape) != 4 or image_shape[-1] != 3:
        raise ValueError(f"Expected RGB `agentview_rgb` frames in `{file_path}` {demo_key}, got {image_shape}.")
    if len(wrist_image_shape) != 4 or wrist_image_shape[-1] != 3:
        raise ValueError(f"Expected RGB `eye_in_hand_rgb` frames in `{file_path}` {demo_key}, got {wrist_image_shape}.")
    if len(ee_state_shape) != 2 or ee_state_shape[1] != 6:
        raise ValueError(f"Expected 6D `ee_states` in `{file_path}` {demo_key}, got {ee_state_shape}.")
    if len(gripper_state_shape) != 2 or gripper_state_shape[1] != 2:
        raise ValueError(f"Expected 2D `gripper_states` in `{file_path}` {demo_key}, got {gripper_state_shape}.")


def create_dataset(repo_id: str) -> LeRobotDataset:
    """Create the target LeRobot dataset."""
    return LeRobotDataset.create(
        repo_id=repo_id,
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
        # Use the LeRobot thread-only writer to avoid multiprocessing semaphores during conversion.
        image_writer_threads=IMAGE_WRITER_THREADS,
        image_writer_processes=0,
    )


def main(
    data_dir: Path = DEFAULT_DATA_DIR,
    repo_id: str = DEFAULT_REPO_ID,
    *,
    push_to_hub: bool = False,
) -> None:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Raw LIBERO-90 directory not found: `{data_dir}`")

    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in `{data_dir}`")

    output_path = LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = create_dataset(repo_id)
    total_episodes = 0
    total_frames = 0

    for file_path in tqdm(hdf5_files, desc="Converting LIBERO-90 files"):
        task_name = build_task_name(file_path)

        with h5py.File(file_path, "r") as handle:
            if "data" not in handle:
                raise KeyError(f"Missing top-level `data` group in `{file_path}`.")

            data_group = handle["data"]
            for demo_key in get_demo_keys(data_group, file_path):
                demo_group = data_group[demo_key]
                if "actions" not in demo_group or "obs" not in demo_group:
                    raise KeyError(f"Missing `actions` or `obs` in `{file_path}` {demo_key}.")

                actions = np.asarray(demo_group["actions"], dtype=np.float32)
                obs_group = demo_group["obs"]
                validate_demo_arrays(file_path, demo_key, actions, obs_group)

                ee_states = np.asarray(obs_group["ee_states"], dtype=np.float32)
                gripper_states = np.asarray(obs_group["gripper_states"], dtype=np.float32)
                agentview_images = np.asarray(obs_group["agentview_rgb"], dtype=np.uint8)
                wrist_images = np.asarray(obs_group["eye_in_hand_rgb"], dtype=np.uint8)

                for step_idx in range(actions.shape[0]):
                    dataset.add_frame(
                        {
                            "image": resize_image(agentview_images[step_idx]),
                            "wrist_image": resize_image(wrist_images[step_idx]),
                            "state": np.concatenate((ee_states[step_idx], gripper_states[step_idx]), axis=0).astype(
                                np.float32,
                                copy=False,
                            ),
                            "actions": actions[step_idx].astype(np.float32, copy=False),
                        }
                    )

                dataset.save_episode(task=task_name)
                total_episodes += 1
                total_frames += actions.shape[0]

    dataset.consolidate(run_compute_stats=False)

    print(
        f"Converted {len(hdf5_files)} files, {total_episodes} episodes, and {total_frames} frames "
        f"into `{output_path}`."
    )

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "raw-hdf5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
