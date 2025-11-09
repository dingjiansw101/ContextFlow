"""
Refactored script to convert (mobile) Aloha hdf5 data to the LeRobot dataset **v2.0** format **across multiple task folders**.

Each *first‑level* sub‑directory of ``--raw-root-dir`` is treated as a *task* and is expected to contain episode files with the pattern ``episode_*.hdf5``.  The script iterates over every detected task directory (or a user‑supplied subset) and appends the data to a single **LeRobotDataset**, tagging each episode with its task name so downstream code can filter by task.

Example usage (convert *all* tasks found under the root)**:**
    uv run convert_aloha_mobile_data_to_lerobot_multi_task.py \
        --raw-root-dir /path/to/raw/aloha \
        --repo-id <org>/<dataset-name>

Convert **only** tasks named “push” and “pick” and skip pushing to the Hub:
    uv run convert_aloha_mobile_data_to_lerobot_multi_task.py \
        --raw-root-dir /path/to/raw/aloha \
        --repo-id <org>/<dataset-name> \
        --tasks push pick \
        --no-push-to-hub
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Literal, Sequence

import h5py
import numpy as np
import torch
import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw

###############################################################################
# Generic dataset‑wide configuration
###############################################################################

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """Low‑level writer configuration shared across all tasks."""

    use_videos: bool = True  # store frames as videos instead of individual images
    tolerance_s: float = 1e-4
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None  # e.g. "ffmpeg" | "nvenc"


DEFAULT_DATASET_CONFIG = DatasetConfig()

###############################################################################
# Utility helpers (unchanged from the original script apart from minor clean‑ups)
###############################################################################

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """Initialise a blank LeRobotDataset on disk, deleting any existing copy."""

    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    bases = [
        "linear",
        "angular",
    ]
    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors) + 2,),
            "names": [motors + bases],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    dataset_path = LEROBOT_HOME / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_dataset_flags(sample_episode: Path) -> tuple[bool, bool]:
    """Inspect a representative episode to know whether velocity / effort are logged."""

    with h5py.File(sample_episode, "r") as ep:
        return "/observations/qvel" in ep, "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: Sequence[str]) -> dict[str, np.ndarray]:
    imgs_per_cam: dict[str, np.ndarray] = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4
        if uncompressed:
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            imgs_array = [
                cv2.imdecode(data, 1) for data in ep[f"/observations/images/{camera}"]
            ]
            imgs_array = np.asarray(imgs_array)
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def smooth_base_action(base_action: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.convolve(base_action[:, i], np.ones(5) / 5, mode="same")
            for i in range(base_action.shape[1])
        ],
        axis=-1,
    ).astype(np.float32)


def preprocess_base_action(base_action: np.ndarray) -> np.ndarray:
    return smooth_base_action(base_action)


def postprocess_base_action(base_action: torch.Tensor) -> torch.Tensor:
    linear_vel, angular_vel = base_action
    return torch.tensor([linear_vel * 1.0, angular_vel * 1.0])


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
    dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        base_action_np = preprocess_base_action(ep["/base_action"][:])
        base_action = torch.from_numpy(base_action_np)
        action = torch.cat([action, base_action], -1)

        velocity = torch.from_numpy(ep["/observations/qvel"][:]) if "/observations/qvel" in ep else None
        effort = torch.from_numpy(ep["/observations/effort"][:]) if "/observations/effort" in ep else None

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: Sequence[Path],
    task: str,
):
    """Stream episode frames into *dataset*, saving after each episode."""

    for ep_path in tqdm.tqdm(hdf5_files, desc=f"Task '{task}'", leave=False):
        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame: dict[str, torch.Tensor | np.ndarray] = {
                "observation.state": state[i],
                "action": action[i],
            }
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]
            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            dataset.add_frame(frame)

        dataset.save_episode(task=task)

###############################################################################
# Top‑level multi‑task conversion entry point
###############################################################################

def port_aloha(
    raw_root_dir: Path,
    repo_id: str,
    *,
    raw_repo_id: str | None = None,
    tasks: list[str] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Convert **all** (or a subset of) task directories found under *raw_root_dir*.

    Parameters
    ----------
    raw_root_dir
        Directory whose *immediate* sub‑directories correspond to distinct tasks.
    repo_id
        Target Hub repository ID.
    tasks
        Optional list of task directory names to include. If *None*, include all
        sub‑directories.
    """

    # ------------------------------------------------------------------
    # 1) Locate raw data (download if necessary)
    # ------------------------------------------------------------------
    if not raw_root_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_root_dir does not exist")
        download_raw(raw_root_dir, repo_id=raw_repo_id)

    # ------------------------------------------------------------------
    # 2) Determine which task folders to process
    # ------------------------------------------------------------------
    all_task_dirs = [d for d in raw_root_dir.iterdir() if d.is_dir()]
    if tasks is not None:
        unknown = set(tasks) - {d.name for d in all_task_dirs}
        if unknown:
            raise FileNotFoundError(
                f"Requested tasks {sorted(unknown)} not found under {raw_root_dir}"
            )
        task_dirs = [d for d in all_task_dirs if d.name in tasks]
    else:
        task_dirs = all_task_dirs

    if not task_dirs:
        raise RuntimeError(f"No valid task directories found in {raw_root_dir}")

    # ------------------------------------------------------------------
    # 3) Create an empty dataset (we inspect the first episode we find to
    #    decide on velocity/effort availability)
    # ------------------------------------------------------------------
    sample_episode = next((task_dirs[0]).glob("episode_*.hdf5"))
    if sample_episode is None:
        raise RuntimeError(
            f"No episodes found in first task folder '{task_dirs[0].name}' – aborting."
        )
    has_vel, has_eff = get_dataset_flags(sample_episode)

    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_velocity=has_vel,
        has_effort=has_eff,
        dataset_config=dataset_config,
    )

    # ------------------------------------------------------------------
    # 4) Loop through requested tasks and populate the dataset
    # ------------------------------------------------------------------
    for task_dir in tqdm.tqdm(task_dirs, desc="Tasks"):
        hdf5_files = sorted(task_dir.glob("episode_*.hdf5"))
        if not hdf5_files:
            tqdm.tqdm.write(f"⚠️  No episodes found for task '{task_dir.name}', skipping.")
            continue
        populate_dataset(dataset, hdf5_files, task=task_dir.name)

    # ------------------------------------------------------------------
    # 5) Finalise & optionally upload
    # ------------------------------------------------------------------
    dataset.consolidate()
    if push_to_hub:
        dataset.push_to_hub()

###############################################################################
# Command‑line interface
###############################################################################

if __name__ == "__main__":
    tyro.cli(port_aloha)
