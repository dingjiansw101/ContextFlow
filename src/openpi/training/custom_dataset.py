"""Custom dataset classes that extend LeRobotDataset functionality.

This module contains custom dataset implementations that inherit from
lerobot.common.datasets.lerobot_dataset.LeRobotDataset and add or modify
functionality for specific use cases.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, SupportsIndex
from typing import Callable
import torch
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class CustomLeRobotDataset(LeRobotDataset):
    """Custom dataset that extends LeRobotDataset with modified data loading.

    This class inherits from LeRobotDataset and allows you to customize how
    individual samples are loaded and processed. Override the __getitem__ method
    to implement your custom data loading logic.

    Example usage:
        dataset = CustomLeRobotDataset(
            repo_id="your/dataset",
            episodes=[0, 1, 2],  # Optional: specific episodes to load
            delta_timestamps={"action": [0.0, 0.1, 0.2]},  # Optional: action horizon
            local_files_only=True,  # Optional: don't download from hub
        )

    Args:
        Same as LeRobotDataset parent class.
    """
    def __init__(
        self,
        repo_id: str,
        root: str | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
        num_current_frames: int = 1,
        num_sample_frames: int = 2,
        num_sample_actions: int = 32,
        task_to_episode_path: str | None = "metadata/libero/task_to_episode.json",
    ):
        """
        CustomLeRobotDataset extends LeRobotDataset to load both sequences and in-context demonstrations.

        Args:
            repo_id: Dataset repository id.
            root, episodes, image_transforms, delta_timestamps, tolerance_s, download_videos, local_files_only, video_backend: Same as LeRobotDataset.
            num_current_frames (int): Number of consecutive frames for current frames sequence.
            num_sample_frames (int): Number of frames for in-context demonstration.
            num_sample_actions (int): Number of actions for in-context demonstration.
            task_to_episode_path (str): Path to task_to_episode.json mapping file.
        """

        # Initialize parent - all LeRobotDataset code, including file loading and indexing
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )
        self.num_current_frames = num_current_frames
        self.num_sample_frames = num_sample_frames
        self.num_sample_actions = num_sample_actions
        self.action_horizon = len(delta_timestamps["actions"])

        # Load task-to-episode and episode-to-indexes mappings
        self.task_to_episode = {}

        assert task_to_episode_path is not None, "task_to_episode_path is not set"
        with Path(task_to_episode_path).open("r") as f:
            task_to_episode_str = json.load(f)
        self.task_to_episode = {int(k): v for k, v in task_to_episode_str.items()}

    def __getitem__(self, idx: SupportsIndex) -> Dict[str, Any]:
        """Get a single sample from the dataset with custom processing.
        Return:
        (1) num_current_frames consecutive frames from the dataset, each frame at time step t includes: 
            (a) image, state, and action at time step t.
            (b) state, and action in [t, t + h - 1], h is the action horizon.
        make sure the num_current_frames + h time steps are consecutive in the same episode.
        if t + h - 1 is greater than the episode length, how to handle the situation? 
        Check the original implementation of __getitem__ method of LeRobotDataset.
        (2) m subsampled frames as a in-context demonstration, it's from another episode but same task:
            (a) each time step t includes image, state, and action at time step t.
        m is the number of subsampled frames for an in-context demonstration episode.
        n and m are hyperparameters, they are set in the initialization of class, you can set them in the config file.

        To read multiple frames from the dataset, you can use huggingface's dataset API to read the dataset:
        e.g, .select()  We want to use select function to read a sequence of frames at the same time.

        TODO: check how is the LeRobotDataset used in create_incontext_data_loader of data_loader.py 
        what are the transforms applied to the LeRobotDataset?

        If we use the CustomLeRobotDataset, we will need to write a new create_data_loader_incontextv2 function in data_loader.py     
        The transform AddImagePromptTransform, AddStatesActionsPromptTransform, AddCurrentFramesSequenceTransform are not needed anymore with CustomLeRobotDataset.
        Make sure the CustomLeRobotDataset is compatible with the create_data_loader_incontextv2 function, and the other transforms.
        Make sure we can get same data with: (1) CustomLeRobotDataset + create_data_loader_incontextv2, and (2) LeRobotDataset + create_incontext_data_loader.
     
        """
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        # import ipdb; ipdb.set_trace()
        query_indices = None
        if self.delta_indices is not None:
            current_ep_idx = self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
            query_indices, padding = self._get_query_indices(idx, current_ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # Load in-context demonstration from another episode with the same task
        task_index = int(item["task_index"])
        incontext_demo = self.load_incontext_demonstration(current_ep_idx, task_index)
        item.update(incontext_demo)
        # TODO: check if transforms are applied to the in-context demonstration
        # TODO: load current frames sequence
        # import ipdb; ipdb.set_trace()
        return item

    def load_incontext_demonstration(self, current_ep_idx: int, task_index: int) -> Dict[str, Any]:
        """Load in-context demonstration from another episode with the same task.

        Args:
            current_ep_idx: The episode index of the current frame
            task_index: The task index to match

        Returns:
            Dictionary containing sampled frames from another episode with the same task
        """
        # Get all episodes for this task
        episodes_for_task = self.task_to_episode.get(task_index, [])

        # Filter out the current episode to get a different one
        other_episodes = [ep for ep in episodes_for_task if ep != current_ep_idx]

        # Randomly select another episode
        selected_ep_idx = random.choice(other_episodes)

        episode_idx = selected_ep_idx if self.episodes is None else self.episodes.index(selected_ep_idx)
        # get the frame indices for the episode

        ep_start = self.episode_data_index["from"][episode_idx]
        ep_end = self.episode_data_index["to"][episode_idx]

        # Uniformly sample m frames between ep_start and ep_end
        frame_indices = list(range(ep_start, ep_end))
        num_frames = len(frame_indices)

        assert self.num_sample_frames <= num_frames, f"num_sample_frames ({self.num_sample_frames}) must be less than or equal to num_frames ({num_frames})"
        # Use evenly-spaced sampling when we have enough frames
        positions = np.linspace(0, num_frames - 1, num=self.num_sample_frames, dtype=int)
        sampled_indices = [frame_indices[p] for p in positions]

        # Load the frames using HuggingFace dataset API
        sampled_frames = self.hf_dataset.select(sampled_indices)

        data = {}
        data["dem_prompt_images"] = {}
        data["dem_prompt_images"]["image"] = torch.stack(sampled_frames["image"])
        data["dem_prompt_images"]["wrist_image"] = torch.stack(sampled_frames["wrist_image"])

        assert self.num_sample_actions <= num_frames, f"num_sample_actions ({self.num_sample_actions}) must be less than or equal to num_frames ({num_frames})"
        positions_actions = np.linspace(0, num_frames - 1, num=self.num_sample_actions, dtype=int)
        sampled_indices_actions = [frame_indices[p] for p in positions_actions]
        sampled_frames_actions = self.hf_dataset.select(sampled_indices_actions)
        data["dem_prompt_states"] = torch.stack(sampled_frames_actions["state"])
        data["dem_prompt_actions"] = torch.stack(sampled_frames_actions["actions"])

        # Add selected_episode for compatibility with ObservationIncontext
        data["selected_episode"] = np.array([selected_ep_idx], dtype=np.int32)

        return data
