"""Custom dataset classes that extend LeRobotDataset functionality.

This module contains custom dataset implementations that inherit from
lerobot.common.datasets.lerobot_dataset.LeRobotDataset and add or modify
functionality for specific use cases.
"""

import random
from typing import Any, Callable, Dict, SupportsIndex

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import openpi.transforms as _transforms
from openpi.training import lookup_tables


def _identity_hf_transform(items_dict):
    return items_dict


def _list_column_to_numpy(chunked_array, dtype=np.float32) -> np.ndarray:
    """Materialize a pyarrow ChunkedArray of fixed-length lists into a (N, D) numpy array.

    Used to pre-cache small numeric columns (state, raw actions) so demo sampling
    does not pay the per-row PIL→tensor decode that HuggingFace's set_transform
    applies to every column on every dataset access.
    """
    combined = chunked_array.combine_chunks()
    n_rows = len(combined)
    if n_rows == 0:
        return np.zeros((0, 0), dtype=dtype)
    # FixedSizeListArray exposes list_size; ListArray exposes offsets.
    inner_len = getattr(combined.type, "list_size", -1)
    if inner_len < 0:
        offsets = np.asarray(combined.offsets)
        inner_lens = np.diff(offsets)
        if not np.all(inner_lens == inner_lens[0]):
            raise ValueError("Expected fixed-length inner lists; got varying lengths.")
        inner_len = int(inner_lens[0])
    flat = np.asarray(combined.values.to_numpy(zero_copy_only=False), dtype=dtype)
    return flat.reshape(n_rows, int(inner_len))


def _index_tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.int64, copy=True)
    return np.asarray(value, dtype=np.int64).copy()


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
        random_select: bool = True,
        seed_base: int | None = None,
        state_key: str = "state",
        actions_key: str = "actions",
        demo_image_keys: dict[str, str] | None = None,
        demo_selection_seed_compat: bool = False,
    ):
        """
        CustomLeRobotDataset extends LeRobotDataset to load both sequences and in-context demonstrations.

        Args:
            repo_id: Dataset repository id.
            root, episodes, image_transforms, delta_timestamps, tolerance_s, download_videos, local_files_only, video_backend: Same as LeRobotDataset.
            num_current_frames (int): Number of consecutive frames for current frames sequence.
            num_sample_frames (int): Number of frames for in-context demonstration.
            num_sample_actions (int): Number of actions for in-context demonstration.
            random_select (bool): If True, randomly select demo episodes; if False, use deterministic selection (first episode).
            seed_base (int | None): Optional base seed for deterministic random demo selection.
            state_key (str): Dataset column holding the proprioceptive state (defaults to "state").
            actions_key (str): Dataset column holding raw actions (defaults to "actions").
            demo_image_keys (dict[str, str] | None): Mapping from output name in dem_prompt_images to the
                dataset image column. Defaults to the LIBERO layout
                {"image": "image", "wrist_image": "wrist_image"}.
            demo_selection_seed_compat (bool): When True and seed_base is set, reproduce the exact seeded
                episode selection of the legacy InjectDemoIndexes transform (same stable_sample_seed
                component stream and the same rng.sample draw). Lets deterministic consistency checks
                compare this loader against the legacy cache-based loader batch-for-batch. Has no effect
                when seed_base is None.
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
        self.state_key = state_key
        self.actions_key = actions_key
        self.demo_image_keys = demo_image_keys or {"image": "image", "wrist_image": "wrist_image"}
        self.demo_selection_seed_compat = demo_selection_seed_compat
        self.action_horizon = len(delta_timestamps[actions_key])
        self.random_select = random_select
        self.seed_base = seed_base

        # Derive the task-to-episode mapping from the dataset metadata that
        # super().__init__ already loaded, rather than a precomputed JSON that could
        # go stale against it. Restricted to self.episodes so demo candidates always
        # resolve through self._episode_id_to_idx below; for the task-level splits the
        # configs actually use, every episode of a kept task is kept, so this drops
        # only tasks that have no frames in the dataset and is otherwise a no-op.
        # Stored as a tuple of ints per task so random.choice / [0] indexing avoid
        # rebuilding lists on every call.
        self.task_to_episode = {
            task: tuple(eps)
            for task, eps in lookup_tables.build_task_to_episode(self.meta, self.episodes).items()
        }

        # Pre-cache small numeric columns so demo-state/action sampling does not
        # pay the per-row PIL decode cost from hf_transform_to_torch.
        # state/raw-actions are O(state_dim) per frame — tiny in aggregate.
        self._states_np = _list_column_to_numpy(self.hf_dataset.data[self.state_key], dtype=np.float32)
        self._raw_actions_np = _list_column_to_numpy(self.hf_dataset.data[self.actions_key], dtype=np.float32)
        # Raw image view for prompt frames. LeRobot's default transform converts
        # PIL images to float32 torch tensors, then the LIBERO transform converts
        # them back to uint8; this view avoids that round-trip for demo images.
        self._hf_dataset_raw = self.hf_dataset.with_transform(_identity_hf_transform)

        # Pre-compute O(1) episode-id → position lookup. Replaces self.episodes.index(...)
        # which is O(len(self.episodes)) and called twice per __getitem__.
        if self.episodes is not None:
            self._episode_id_to_idx = {int(ep): i for i, ep in enumerate(self.episodes)}
        else:
            self._episode_id_to_idx = None

        # Episode bounds as plain numpy ints for cheap arithmetic.
        self._ep_starts_np = _index_tensor_to_numpy(self.episode_data_index["from"])
        self._ep_ends_np = _index_tensor_to_numpy(self.episode_data_index["to"])

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Fast path for action/state chunk reads.

        The parent's implementation calls ``self.hf_dataset.select(q_idx)[key]``,
        which constructs a fresh ``Dataset`` and triggers ``hf_transform_to_torch``
        over every column — including the (~100 wasted) PIL image decodes for the
        action-chunk rows we only need numeric data from. Read from the cached
        numpy arrays instead; fall back to the parent for any other key.
        """
        out: dict = {}
        fallback: dict[str, list[int]] = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            if key == self.actions_key and self._raw_actions_np is not None:
                out[key] = torch.from_numpy(
                    self._raw_actions_np[np.asarray(q_idx, dtype=np.int64)]
                )
            elif key == self.state_key and self._states_np is not None:
                out[key] = torch.from_numpy(
                    self._states_np[np.asarray(q_idx, dtype=np.int64)]
                )
            else:
                fallback[key] = q_idx
        if fallback:
            out.update(super()._query_hf_dataset(fallback))
        return out

    def __getitem__(self, idx: SupportsIndex) -> Dict[str, Any]:
        """Get a single sample from the dataset with custom processing."""
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        if self._episode_id_to_idx is not None:
            current_ep_idx = self._episode_id_to_idx[ep_idx]
        else:
            current_ep_idx = ep_idx
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, current_ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # Load in-context demonstration from another episode with the same task
        task_index = int(item["task_index"])
        # Seed components mirroring what the legacy repack forwarded to InjectDemoIndexes
        # (the legacy repack forwarded "index" and "episode_index" but not "frame_index").
        legacy_seed_components = (item.get("index"), None, item.get("episode_index"))
        incontext_demo = self.load_incontext_demonstration(
            current_ep_idx, task_index, sample_index=idx, legacy_seed_components=legacy_seed_components
        )
        item.update(incontext_demo)

        return item

    def select_incontext_episode(
        self,
        other_episodes: list[int],
        *,
        current_ep_idx: int,
        task_index: int,
        sample_index: SupportsIndex | None,
        legacy_seed_components: tuple | None = None,
    ) -> int:
        if not self.random_select:
            return other_episodes[0]
        if self.seed_base is None:
            return random.choice(other_episodes)

        if self.demo_selection_seed_compat and legacy_seed_components is not None:
            # Reproduce InjectDemoIndexes' seeded draw exactly: same component stream
            # (task_index, index, frame_index, episode_index) and the same
            # rng.sample(candidates, 1) call, so a deterministic consistency check can
            # compare this loader against the legacy cache-based loader batch-for-batch.
            index_val, frame_index_val, episode_index_val = legacy_seed_components
            rng = random.Random(
                _transforms.stable_sample_seed(
                    self.seed_base,
                    "InjectDemoIndexes",
                    task_index,
                    index_val,
                    frame_index_val,
                    episode_index_val,
                )
            )
            candidates = [int(ep) for ep in other_episodes]
            return rng.sample(candidates, 1)[0]

        rng = random.Random(
            _transforms.stable_sample_seed(
                self.seed_base,
                "CustomLeRobotDataset",
                task_index,
                current_ep_idx,
                sample_index,
            )
        )
        return other_episodes[rng.randrange(len(other_episodes))]

    def load_incontext_demonstration(
        self,
        current_ep_idx: int,
        task_index: int,
        sample_index: SupportsIndex | None = None,
        legacy_seed_components: tuple | None = None,
    ) -> Dict[str, Any]:
        """Load in-context demonstration from another episode with the same task.

        Args:
            current_ep_idx: The episode index of the current frame
            task_index: The task index to match

        Returns:
            Dictionary containing sampled frames from another episode with the same task
        """
        # Mirror InjectDemoIndexes selection so comparison tests match the baseline loader.
        # TODO: do we need to exclude the current episode?
        other_episodes = self.task_to_episode.get(task_index)
        if not other_episodes:
            raise ValueError(f"No episodes available for task {task_index}")

        # print(f"other_episodes: {other_episodes}")
        selected_ep_idx = self.select_incontext_episode(
            other_episodes,
            current_ep_idx=current_ep_idx,
            task_index=task_index,
            sample_index=sample_index,
            legacy_seed_components=legacy_seed_components,
        )

        if self._episode_id_to_idx is not None:
            episode_idx = self._episode_id_to_idx[selected_ep_idx]
        else:
            episode_idx = selected_ep_idx

        ep_start = int(self._ep_starts_np[episode_idx])
        ep_end = int(self._ep_ends_np[episode_idx])
        num_frames = ep_end - ep_start

        assert self.num_sample_frames <= num_frames, (
            f"num_sample_frames ({self.num_sample_frames}) must be less than or equal to num_frames ({num_frames})"
        )
        # Frame positions for images. Pay PIL decode only on the kept frames.
        frame_positions = np.linspace(0, num_frames - 1, num=self.num_sample_frames, dtype=int)
        sampled_indices = (ep_start + frame_positions).tolist()
        # Direct indexing is cheaper than .select() and the raw view avoids
        # PIL -> float tensor -> uint8 conversion churn for demo images.
        sampled_frames = self._hf_dataset_raw[sampled_indices]

        data = {
            "dem_prompt_images": {
                out_name: np.asarray(sampled_frames[column], dtype=np.uint8)
                for out_name, column in self.demo_image_keys.items()
            }
        }

        # State/action positions. Pulled from the pre-cached numpy arrays so we
        # never trigger PIL→tensor on the (discarded) image columns of these rows.
        num_actions_to_sample = min(self.num_sample_actions, num_frames)
        action_positions = np.linspace(0, num_frames - 1, num=num_actions_to_sample, dtype=int)
        action_abs_indices = ep_start + action_positions
        # Fancy indexing already returns a fresh C-contiguous array, so the
        # explicit .copy() the parent version used was redundant.
        dem_prompt_states = torch.from_numpy(self._states_np[action_abs_indices])
        dem_prompt_actions = torch.from_numpy(self._raw_actions_np[action_abs_indices])
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
