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
        random_select: bool = True,
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
            random_select (bool): If True, randomly select demo episodes; if False, use deterministic selection (first episode).
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
        self.random_select = random_select

        # Load task-to-episode and episode-to-indexes mappings
        self.task_to_episode = {}

        assert task_to_episode_path is not None, "task_to_episode_path is not set"
        with Path(task_to_episode_path).open("r") as f:
            task_to_episode_str = json.load(f)
        self.task_to_episode = {int(k): v for k, v in task_to_episode_str.items()}

    def __getitem__(self, idx: SupportsIndex) -> Dict[str, Any]:
        """Get a single sample from the dataset with custom processing.
        """
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
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

        return item

    def load_incontext_demonstration(self, current_ep_idx: int, task_index: int) -> Dict[str, Any]:
        """Load in-context demonstration from another episode with the same task.

        Args:
            current_ep_idx: The episode index of the current frame
            task_index: The task index to match

        Returns:
            Dictionary containing sampled frames from another episode with the same task
        """
        # Mirror InjectDemoIndexes selection so comparison tests match the baseline loader.
        # TODO: do we need to exclude the current episode?
        other_episodes = [int(ep) for ep in self.task_to_episode.get(task_index, [])]
        if not other_episodes:
            raise ValueError(f"No episodes available for task {task_index}")

        # print(f"other_episodes: {other_episodes}")
        if self.random_select:
            selected_ep_idx = random.choice(other_episodes)
        else:
            selected_ep_idx = other_episodes[0]

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

        num_actions_to_sample = min(self.num_sample_actions, num_frames)
        positions_actions = np.linspace(0, num_frames - 1, num=num_actions_to_sample, dtype=int)
        sampled_indices_actions = [frame_indices[p] for p in positions_actions]
        sampled_frames_actions = self.hf_dataset.select(sampled_indices_actions)
        dem_prompt_states = torch.stack(sampled_frames_actions["state"])
        dem_prompt_actions = torch.stack(sampled_frames_actions["actions"])
        # TODO: review the implementation of padding
        if num_actions_to_sample < self.num_sample_actions:
            pad = self.num_sample_actions - num_actions_to_sample
            repeat_shape = (pad,) + (1,) * (dem_prompt_states.dim() - 1)
            states_pad = dem_prompt_states[-1:].repeat(repeat_shape)
            actions_pad = dem_prompt_actions[-1:].repeat(repeat_shape)
            dem_prompt_states = torch.cat([dem_prompt_states, states_pad], dim=0)
            dem_prompt_actions = torch.cat([dem_prompt_actions, actions_pad], dim=0)
        data["dem_prompt_states"] = dem_prompt_states
        data["dem_prompt_actions"] = dem_prompt_actions

        # Add selected_episode for compatibility with ObservationIncontext
        data["selected_episode"] = np.array([selected_ep_idx], dtype=np.int32)

        return data


class CustomLeRobotDatasetv2(CustomLeRobotDataset):
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
        random_select: bool = True,
        current_frame_sample_mode: str = "random",
        seed: int | None = None,
        multiple_current_frames: bool = True,
        future_state_downsample: int = 5,
        use_future_states: bool = False,
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
            random_select (bool): If True, randomly select demo episodes; if False, use deterministic selection (first episode).
            seed (int | None): Random seed for reproducibility. If None, RNG is unseeded (default behavior).
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
            num_current_frames=num_current_frames,
            num_sample_frames=num_sample_frames,
            num_sample_actions=num_sample_actions,
            task_to_episode_path=task_to_episode_path,
            random_select=random_select,
        )
        self.current_frame_sample_mode = current_frame_sample_mode
        self.multiple_current_frames = multiple_current_frames
        self.future_state_downsample = future_state_downsample
        self.use_future_states = use_future_states
        # Initialize RNG for random frame sampling
        self._rng = np.random.default_rng(seed)
        # Cache actions and states in memory for fast access (avoids slow HF dataset queries)
        print("[Cache] Loading actions and states into memory...")
        self.cached_arrays = {
            'actions': np.array(self.hf_dataset['actions']),
            'state': np.array(self.hf_dataset['state'])
        }
        total_mb = sum(arr.nbytes for arr in self.cached_arrays.values()) / 1024 / 1024
        print(f"[Cache] Loaded {self.cached_arrays['actions'].shape[0]} frames "
              f"(actions: {self.cached_arrays['actions'].shape}, "
              f"state: {self.cached_arrays['state'].shape}, "
              f"total: {total_mb:.2f} MB)")

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Override parent to use cached arrays for fast indexing.

        Uses numpy array indexing for cached keys (much faster than HF dataset queries).
        Falls back to parent's HF dataset query for non-cached keys.
        """
        result = {}
        for key, q_idx in query_indices.items():
            if key in self.cached_arrays:
                # Use fast numpy indexing for cached arrays
                result[key] = torch.tensor(self.cached_arrays[key][q_idx])
            else:
                # Fall back to parent's HF dataset query for non-cached keys
                result[key] = torch.stack(self.hf_dataset.select(q_idx)[key])
        return result

    def _pick_indices_random(
        self, n_total: int, anchor_local_idx: int | None, rng: np.random.Generator
    ) -> list[int]:
        """Randomly sample frame indices from episode, optionally keeping anchor frame.

        Adapted from AddCurrentFramesSequenceTransform._pick_indices_random in transforms.py.

        Args:
            n_total: Total number of frames in the episode
            anchor_local_idx: Local index of the anchor (current) frame, or None
            rng: Random number generator instance

        Returns:
            List of local frame indices (sorted in temporal order)
        """
        if self.num_current_frames > n_total:
            raise ValueError(
                f"num_current_frames={self.num_current_frames} > episode length n_total={n_total}"
            )

        # Always keep anchor when sampling randomly (matching keep_anchor_when_random=True)
        if anchor_local_idx is not None and 0 <= anchor_local_idx < n_total:
            # Fix anchor, then randomly sample (num_current_frames - 1) from remaining frames
            rest = np.delete(np.arange(n_total), anchor_local_idx)
            k = self.num_current_frames - 1
            choose = rng.choice(rest, size=k, replace=False)
            loc = np.concatenate([[anchor_local_idx], choose])
        else:
            # Fallback to simple random sampling
            loc = rng.choice(n_total, size=self.num_current_frames, replace=False)

        # Sort to maintain temporal order
        loc = np.sort(loc)
        return [int(i) for i in loc]

    def __getitem__(self, idx: SupportsIndex) -> Dict[str, Any]:

        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
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


        if self.use_future_states and self.multiple_current_frames:
            raise NotImplementedError("use_future_states with multiple current frames is not implemented yet")
       
        # Load current frames sequence (if num_current_frames > 1)
        if self.multiple_current_frames:
            assert self.num_current_frames > 1, "num_current_frames must be greater than 1"
           # Get episode frame boundaries
            ep_start = self.episode_data_index["from"][current_ep_idx]
            ep_end = self.episode_data_index["to"][current_ep_idx]

            # Create frame list for episode (global indices)
            frame_list = list(range(ep_start, ep_end))

            # Sample frame indices based on current_frame_sample_mode
            # TODO: simplify the pick_indices_random function. 
            # (1) why not directly operate on global indices?
            # (2) anchor idex may not be necessary
            if self.current_frame_sample_mode == "random":
                sampled_local_indices = self._pick_indices_random(
                    n_total=len(frame_list),
                    anchor_local_idx=item['frame_index'],
                    rng=self._rng
                )
            else:
                # Default to uniform spacing
                raise ValueError(f"Unsupported current_frame_sample_mode: {self.current_frame_sample_mode}")

            # Convert local indices to global indices
            sampled_global_indices = [frame_list[i] for i in sampled_local_indices]
            # Fetch sampled frames using HuggingFace dataset API
            sampled_items = self.hf_dataset.select(sampled_global_indices)

            # Extract images, states, and actions for each sampled frame
            # Optimized: Batch query all unique action indices to eliminate redundant queries
            assert self.delta_indices is not None, "delta_indices must be set"

            # Step 1: Collect all frame query info and unique action indices
            frame_action_indices = []  # Store action indices for each frame
            unique_action_indices = []  # Ordered list of unique action indices
            action_idx_to_batch_pos = {}  # Maps action index -> position in unique list

            for frame_idx in sampled_global_indices:
                frame_query_indices, frame_padding = self._get_query_indices(frame_idx, current_ep_idx)
                action_indices = frame_query_indices['actions']
                frame_action_indices.append(action_indices)

                # Collect unique indices
                for idx in action_indices:
                    if idx not in action_idx_to_batch_pos:
                        action_idx_to_batch_pos[idx] = len(unique_action_indices)
                        unique_action_indices.append(idx)

            # Step 2: Batch query all unique action indices at once
            batch_query_result = self._query_hf_dataset({'actions': unique_action_indices})
            all_actions_batch = batch_query_result['actions']

            # Step 3: Map results back to each frame
            actions_list = []
            for action_indices in frame_action_indices:
                # Get positions of this frame's actions in the batch result
                positions = [action_idx_to_batch_pos[idx] for idx in action_indices]
                # Index into batch result to get this frame's action sequence
                frame_actions = all_actions_batch[positions]
                actions_list.append(frame_actions)

            # print("idx: ", idx)
            
            item["current_images_seq"] = {}
            item["current_images_seq"]["image"] = torch.stack(sampled_items["image"])
            item["current_images_seq"]["wrist_image"] = torch.stack(sampled_items["wrist_image"])
            item["current_state_seq"] = torch.stack(sampled_items["state"])
            item["actions_seq"] = torch.stack(actions_list)

        if self.use_future_states:
            # Get the indices of future states by downsampling action indices
            # Action indices represent future timesteps from the current frame
            assert query_indices is not None and 'actions' in query_indices, \
                "Future states require action indices from delta_indices"

            action_indices = query_indices['actions']
            # Downsample: take every Nth action index as a state checkpoint
            future_state_indices = action_indices[::self.future_state_downsample]

            # Query future states from the dataset using cached arrays for fast access
            future_states_result = self._query_hf_dataset({'state': future_state_indices})
            item['future_states'] = future_states_result['state']

        return item
