from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
import random
import re
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools
from tqdm import tqdm

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize
from openpi.training import lookup_tables

# from openpi.training.data_loader import Dataset

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


def stable_sample_seed(seed_base: int, *components: Any) -> int:
    """Create a deterministic seed from a base seed and sample-specific values."""
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(str(int(seed_base)).encode("utf-8"))
    for component in components:
        hasher.update(b"\0")
        hasher.update(_seed_component_bytes(component))
    return int.from_bytes(hasher.digest(), "little")


def _seed_component_bytes(component: Any) -> bytes:
    if component is None:
        return b"<none>"
    if isinstance(component, bytes):
        return component
    if isinstance(component, str):
        return component.encode("utf-8")
    try:
        array = np.asarray(component)
    except Exception:
        return repr(component).encode("utf-8")
    if array.dtype == object:
        return repr(component).encode("utf-8")
    return f"{array.shape}:{array.dtype}:".encode() + array.tobytes()


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDemoIndexes(DataTransformFn):
    """
    Selects up to ``sample_episodes`` episodes for the current task and
    uniformly samples up to ``sample_frames`` frame indexes *per episode*.

    On output the ``data`` dict contains:
    ``selected_episode`` : List[int]
    ``dem_prompt_indexes`` : List[List[int]]  (parallel to selected_episode)
    """

    repo_id: str = ""
    local_files_only: bool = False

    task_to_episode: dict[int, list[int]] = dataclasses.field(init=False)
    episode_to_indexes: dict[int, list[int]] = dataclasses.field(init=False)

    sample_frames: int = 2
    random_select: bool = True
    sample_episodes: int = 1
    train_episode_index_list: list[int] | None = None
    seed_base: int | None = None

    def __post_init__(self):
        if not self.repo_id:
            raise ValueError("InjectDemoIndexes requires repo_id to build its lookup tables")

        # Both tables are derived from the LeRobot metadata rather than read from
        # precomputed JSON. When a train/test split is active, build_episode_to_indexes
        # reindexes through LeRobot's own get_episode_data_index, which is how the
        # dataset itself lays out frames — the previous hand-rolled reindex assumed the
        # kept-episode list was sorted and silently produced wrong offsets otherwise.
        task_to_episode, episode_to_indexes = lookup_tables.lookup_tables_for_repo(
            self.repo_id,
            episodes=self.train_episode_index_list,
            local_files_only=self.local_files_only,
        )

        # Store these dictionaries on the frozen dataclass
        object.__setattr__(self, "task_to_episode", task_to_episode)
        object.__setattr__(self, "episode_to_indexes", episode_to_indexes)

        # XJ: Initialize inference cache
        object.__setattr__(
            self,
            "_cache",
            {
                "task_index": None,  # type: Optional[int]
                "selected_episode": None,  # type: Optional[np.ndarray]
                "dem_prompt_indexes": None,  # type: Optional[List[List[int]]]
            },
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Example transform: randomly select a demonstration as prompt for the data,
        storing both the chosen indexes and the retrieved items.
        """
        # 1) Randomly pick an episode for this task
        task_index = int(data["task_index"])
        episodes_for_task = self.task_to_episode.get(task_index, [])

        split = data.get("split", "train")

        # === XJ: Inference cache hit ===
        if self._cache["task_index"] == task_index and split == "test":
            data["selected_episode"] = self._cache["selected_episode"]
            data["dem_prompt_indexes"] = self._cache["dem_prompt_indexes"]
            return data

        # === Otherwise: generate prompt ===
        episodes_for_task = [int(ep) for ep in episodes_for_task]

        episode_lengths = {ep: len(self.episode_to_indexes.get(ep, [])) for ep in episodes_for_task}
        valid_candidates = [ep for ep in episodes_for_task if episode_lengths[ep] >= self.sample_frames]
        fallback_candidates = [ep for ep in episodes_for_task if episode_lengths[ep] > 0]

        candidates = valid_candidates if valid_candidates else fallback_candidates
        if not candidates:
            raise ValueError(f"InjectDemoIndexes: no prompt episodes with frames for task {task_index}")
        if valid_candidates and len(valid_candidates) < len(episodes_for_task):
            logging.debug(
                "InjectDemoIndexes: filtered out %d prompt episodes without enough frames for task %d",
                len(episodes_for_task) - len(valid_candidates),
                task_index,
            )
        elif not valid_candidates and fallback_candidates:
            logging.warning(
                "InjectDemoIndexes: using fallback prompt episodes with < %d frames for task %d",
                self.sample_frames,
                task_index,
            )
        if split == "train" and self.random_select:
            k = min(self.sample_episodes, len(candidates))
            if self.seed_base is None:
                selected_episodes = random.sample(candidates, k)
            else:
                rng = random.Random(
                    stable_sample_seed(
                        self.seed_base,
                        "InjectDemoIndexes",
                        task_index,
                        data.get("index"),
                        data.get("frame_index"),
                        data.get("episode_index"),
                    )
                )
                selected_episodes = rng.sample(candidates, k)
        else:
            selected_episodes = candidates[: self.sample_episodes]
        if not selected_episodes:
            raise ValueError(f"InjectDemoIndexes: unable to choose prompt episodes for task {task_index}")

        # 2) choose frames per episode
        dem_prompt_indexes: list[list[int]] = []

        for ep in selected_episodes:
            frame_idxs = self.episode_to_indexes.get(int(ep), [])
            n = len(frame_idxs)
            if n >= self.sample_frames:
                pos = np.linspace(0, n - 1, num=self.sample_frames, dtype=int)
                # pos_list.append(pos)
                chosen = [frame_idxs[p] for p in pos]
            elif n > 0:
                logging.warning(
                    "InjectDemoIndexes: padding episode %s for task %d because only %d frames available",
                    ep,
                    task_index,
                    n,
                )
                chosen = list(frame_idxs)
                pad_value = frame_idxs[-1]
                while len(chosen) < self.sample_frames:
                    chosen.append(pad_value)
            else:
                raise ValueError(f"InjectDemoIndexes: episode {ep} for task {task_index} has no frames available")

            dem_prompt_indexes.append(chosen)

        # 3) attach to data
        data["selected_episode"] = np.array(selected_episodes, dtype=np.int32)
        data["dem_prompt_indexes"] = dem_prompt_indexes

        # XJ: === Update inference cache ===
        if split == "test":
            prompt = data.get("prompt", "")
            if hasattr(prompt, "item"):
                prompt = prompt.item()
            logging.info(
                "[InjectDemoIndexes] Test task: '%s' (index=%d), demo episode(s): %s, frame indices: %s",
                prompt,
                task_index,
                selected_episodes,
                dem_prompt_indexes,
            )
            self._cache["task_index"] = task_index
            self._cache["selected_episode"] = np.array(selected_episodes, dtype=np.int32)
            self._cache["dem_prompt_indexes"] = dem_prompt_indexes

        return data


@dataclasses.dataclass(frozen=True)
class InjectDemoFromCustomDataset(DataTransformFn):
    """Populates in-context demonstration fields by calling a CustomLeRobotDataset directly.

    Drop-in replacement for `InjectDemoIndexes` + `AddImagePromptTransform` +
    `AddStatesActionsPromptTransform` at policy-serving time. Avoids the JSON state/action
    cache files that the older eval path depended on.

    On call, reads ``task_index`` from the data dict and delegates to
    ``CustomLeRobotDataset.load_incontext_demonstration`` to populate:
        - ``dem_prompt_images`` (dict {"image", "wrist_image"} of torch tensors)
        - ``dem_prompt_states`` (torch tensor)
        - ``dem_prompt_actions`` (torch tensor)
        - ``selected_episode`` (np.ndarray[int32])

    For ``split == "test"``, demos are cached per ``task_index`` so all frames of a rollout
    see the same demonstration (matches `InjectDemoIndexes` test-split caching).
    """

    dataset: Any  # CustomLeRobotDataset; typed as Any to avoid a hard import cycle.

    def __post_init__(self):
        object.__setattr__(
            self,
            "_cache",
            {
                "task_index": None,
                "dem_prompt_images": None,
                "dem_prompt_states": None,
                "dem_prompt_actions": None,
                "selected_episode": None,
            },
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        task_index = int(data["task_index"])
        split = data.get("split", "train")

        if split == "test" and self._cache["task_index"] == task_index:
            data["dem_prompt_images"] = self._cache["dem_prompt_images"]
            data["dem_prompt_states"] = self._cache["dem_prompt_states"]
            data["dem_prompt_actions"] = self._cache["dem_prompt_actions"]
            data["selected_episode"] = self._cache["selected_episode"]
            return data

        # Mirror InjectDemoIndexes (transforms.py:252-256): on test split, demo
        # selection is deterministic regardless of the dataset's random_select flag,
        # so repeated eval runs see the same demos. The dataset's random_select is
        # honored for train split only.
        saved_random_select = self.dataset.random_select
        if split == "test":
            self.dataset.random_select = False
        try:
            demo = self.dataset.load_incontext_demonstration(current_ep_idx=-1, task_index=task_index)
        finally:
            self.dataset.random_select = saved_random_select
        data["dem_prompt_images"] = demo["dem_prompt_images"]
        data["dem_prompt_states"] = demo["dem_prompt_states"]
        data["dem_prompt_actions"] = demo["dem_prompt_actions"]
        data["selected_episode"] = demo["selected_episode"]

        if split == "test":
            prompt = data.get("prompt", "")
            if hasattr(prompt, "item"):
                prompt = prompt.item()
            logging.info(
                "[InjectDemoFromCustomDataset] Test task: '%s' (index=%d), demo episode(s): %s",
                prompt,
                task_index,
                demo["selected_episode"].tolist(),
            )
            self._cache["task_index"] = task_index
            self._cache["dem_prompt_images"] = demo["dem_prompt_images"]
            self._cache["dem_prompt_states"] = demo["dem_prompt_states"]
            self._cache["dem_prompt_actions"] = demo["dem_prompt_actions"]
            self._cache["selected_episode"] = demo["selected_episode"]

        return data


def save_episode_states_to_json(episode_to_all_states: dict[int, np.ndarray], filename: str):
    """
    Converts each NumPy array to a Python list, then dumps to JSON.
    """
    filename = Path(filename)  # Convert string path to a Path object

    json_dict = {}
    for episode_id, states_array in episode_to_all_states.items():
        json_dict[str(episode_id)] = states_array.tolist()

    with filename.open("w") as f:  # Use Path.open()
        json.dump(json_dict, f)


def load_episode_states_from_json(filename: str) -> dict[int, np.ndarray]:
    """
    Loads the JSON file and reconstructs each list into a NumPy array.
    """
    filename = Path(filename)  # Convert string path to Path
    with filename.open("r") as f:  # Use Path.open() instead of open()
        json_dict = json.load(f)

    episode_to_all_states = {}
    for episode_id_str, state_list in json_dict.items():
        episode_id = int(episode_id_str)
        episode_to_all_states[episode_id] = np.array(state_list, dtype=np.float32)

    return episode_to_all_states


def tree_stack_np(list_of_trees, axis=0):
    """
    Stack a list of similarly structured PyTrees along `axis`,
    ensuring the leaves are NumPy arrays.
    """

    def stack_fn(*leaves):
        return np.stack(leaves, axis=axis)

    return jax.tree_map(stack_fn, *list_of_trees)


@dataclasses.dataclass(frozen=True)
class AddImagePromptTransform(DataTransformFn):
    """Stacks image prompts per episode into dicts of arrays by key."""

    dataset: any

    def __post_init__(self):
        # XJ: inference cache
        object.__setattr__(
            self,
            "_cache",
            {
                "dem_prompt_indexes": None,
                "dem_prompt_images": None,
                "dem_prompt_images_mask": None,
                "split": None,
            },
        )

    # def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
    #     idx_lists: List[List[int]] = data.get("dem_prompt_indexes", [])
    #     # split = data.get("split", "train")

    #     # XJ: inference cache hit
    #     # if (
    #     #     split == "test"
    #     #     and self._cache["dem_prompt_indexes"] == idx_lists
    #     # ):
    #     #     data["dem_prompt_images"] = self._cache["dem_prompt_images"]
    #     #     data["dem_prompt_images_mask"] = self._cache["dem_prompt_images_mask"]
    #     #     return data

    #     # otherwise, normal routine
    #     images_dict: Dict[str, List[np.ndarray]] = {}
    #     masks_dict: Dict[str, List[np.ndarray]] = {}
    #     for idx_list in idx_lists:
    #         items = [self.dataset[int(i)] for i in idx_list]
    #         stacked = tree_stack_np(items)
    #         for name, img_arr in stacked["image"].items():
    #             images_dict.setdefault(name, []).append(img_arr)
    #         for name, mask_arr in stacked["image_mask"].items():
    #             masks_dict.setdefault(name, []).append(mask_arr)

    #     # Stack: assemble outputs
    #     imgs = {name: np.stack(arrs, axis=0) for name, arrs in images_dict.items()}
    #     msks = {name: np.stack(arrs, axis=0) for name, arrs in masks_dict.items()}

    #     # Single-episode squeeze
    #     if len(idx_lists) == 1:
    #         imgs = {name: arr[0] for name, arr in imgs.items()}
    #         msks = {name: arr[0] for name, arr in msks.items()}

    #     # Attach to data
    #     data["dem_prompt_images"] = imgs
    #     data["dem_prompt_images_mask"] = msks

    # if split == "test":
    #     self._cache["dem_prompt_indexes"] = idx_lists
    #     self._cache["dem_prompt_images"] = imgs
    #     self._cache["dem_prompt_images_mask"] = msks

    #     return data

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        idx_lists: list[list[int]] = data.get("dem_prompt_indexes", [])

        # ---- Check cache for test split ----
        split = data.get("split", "train")
        if split == "test" and self._cache["split"] == "test" and self._cache["dem_prompt_indexes"] == idx_lists:
            # Use cached results
            data["dem_prompt_images"] = self._cache["dem_prompt_images"]
            data["dem_prompt_images_mask"] = self._cache["dem_prompt_images_mask"]
            return data

        images_dict: dict[str, list[np.ndarray]] = {}
        masks_dict: dict[str, list[np.ndarray]] = {}
        for idx_list in idx_lists:
            if len(idx_list) == 0:
                raise ValueError(f"Empty idx_list found! dem_prompt_indexes={idx_lists}")
            items = [self.dataset[int(i)] for i in idx_list]
            if len(items) == 0:
                raise ValueError(f"No items fetched! idx_list={idx_list}")
            stacked = tree_stack_np(items)
            for name, img_arr in stacked["image"].items():
                images_dict.setdefault(name, []).append(img_arr)
            for name, mask_arr in stacked["image_mask"].items():
                masks_dict.setdefault(name, []).append(mask_arr)

        # assemble outputs
        imgs = {name: np.stack(arrs, axis=0) for name, arrs in images_dict.items()}
        msks = {name: np.stack(arrs, axis=0) for name, arrs in masks_dict.items()}

        # if only one episode, squeeze the episode dimension
        if len(idx_lists) == 1:
            data["dem_prompt_images"] = {name: arr[0] for name, arr in imgs.items()}
            data["dem_prompt_images_mask"] = {name: arr[0] for name, arr in msks.items()}
        else:
            data["dem_prompt_images"] = imgs
            data["dem_prompt_images_mask"] = msks

        # DEBUG: Print loaded demonstration info
        # print("\n" + "="*80)
        # print("LOADED IN-CONTEXT DEMONSTRATIONS:")
        # # Print the text prompt if available
        # if "prompt" in data:
        #     prompt_str = data["prompt"].item() if hasattr(data["prompt"], 'item') else str(data["prompt"])
        #     print(f"  Text prompt: '{prompt_str}'")
        # print(f"  Demo indexes: {idx_lists}")
        # print(f"  Number of demo episodes: {len(idx_lists)}")
        # for name, img_arr in data["dem_prompt_images"].items():
        #     print(f"  Camera '{name}': shape={img_arr.shape}, dtype={img_arr.dtype}")
        # print("="*80 + "\n")

        # ---- Update cache for test split ----
        if split == "test":
            self._cache["split"] = "test"
            self._cache["dem_prompt_indexes"] = idx_lists
            self._cache["dem_prompt_images"] = data["dem_prompt_images"]
            self._cache["dem_prompt_images_mask"] = data["dem_prompt_images_mask"]

        return data


# @dataclasses.dataclass(frozen=True)
# class AddStatesActionsPromptTransform(DataTransformFn):
#     """Adds state/action sequences for multiple episodes."""
#     dataset: any  # the underlying dataset from which to fetch demo items

#     max_len: int = 32

#     episode_to_all_states: Dict[int, np.ndarray] = dataclasses.field(init=False)
#     episode_to_all_first_actions: Dict[int, np.ndarray] = dataclasses.field(init=False)

#     states_cache_path: str = "metadata/libero/episode_states_cache.json"
#     actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
#     episode_to_indexes_file: str = "metadata/libero/episode_to_indexes.json"


#     def __post_init__(self):
#         try:
#             states = load_episode_states_from_json(self.states_cache_path)
#             actions = load_episode_states_from_json(self.actions_cache_path)
#         except FileNotFoundError:
#             with Path(self.episode_to_indexes_file).open("r") as f:
#                 raw = json.load(f)
#             idx_map = {int(k): v for k, v in raw.items()}
#             states, actions = {}, {}
#             for ep, idxs in tqdm(idx_map.items(), desc="Building caches", total=len(idx_map)):
#                 state_list, action_list = [], []
#                 for idx in idxs:
#                     item = self.dataset[int(idx)]
#                     state_list.append(item["state"])
#                     action_list.append(item["actions"][0])
#                 states[ep] = np.stack(state_list, axis=0)
#                 actions[ep] = np.stack(action_list, axis=0)
#             save_episode_states_to_json(states, self.states_cache_path)
#             save_episode_states_to_json(actions, self.actions_cache_path)
#         object.__setattr__(self, "episode_to_all_states", states)
#         object.__setattr__(self, "episode_to_all_first_actions", actions)

#         # XJ: inference cache
#         object.__setattr__(self, "_cache", {
#             "selected_episode": None,
#             "states": None,
#             "states_mask": None,
#             "actions": None,
#             "actions_mask": None,
#         })

#     def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         eps: List[int] = data.get("selected_episode", [])
#         split = data.get("split", "train")

#         # XJ: cache hit
#         if split == "test" and self._cache["selected_episode"] == list(eps):
#             if len(eps) == 1:
#                 data["dem_prompt_all_states"] = self._cache["states"][0]
#                 data["dem_prompt_all_states_mask"] = self._cache["states_mask"][0]
#                 data["dem_prompt_all_actions"] = self._cache["actions"][0]
#                 data["dem_prompt_all_actions_mask"] = self._cache["actions_mask"][0]
#             else:
#                 data["dem_prompt_all_states"] = self._cache["states"]
#                 data["dem_prompt_all_states_mask"] = self._cache["states_mask"]
#                 data["dem_prompt_all_actions"] = self._cache["actions"]
#                 data["dem_prompt_all_actions_mask"] = self._cache["actions_mask"]
#             return data


#         states_b, states_mask_b, actions_b, actions_mask_b = [], [], [], []
#         for ep in eps:
#             all_states = self.episode_to_all_states[ep]
#             all_actions = self.episode_to_all_first_actions[ep]

#             # Sample/pad states
#             t, d = all_states.shape
#             if t >= self.max_len:
#                 idxs = np.linspace(0, t - 1, num=self.max_len, dtype=int)
#                 sampled_states = all_states[idxs]
#                 state_mask = np.ones((self.max_len,), dtype=bool)
#             else:
#                 sampled_states = np.zeros((self.max_len, d), dtype=all_states.dtype)
#                 sampled_states[:t] = all_states
#                 sampled_states[t:] = all_states[t - 1] if t > 0 else 0
#                 state_mask = np.zeros((self.max_len,), dtype=bool)
#                 state_mask[:t] = True

#             # Sample/pad actions
#             t_a, a_dim = all_actions.shape
#             if t_a >= self.max_len:
#                 idxs_a = np.linspace(0, t_a - 1, num=self.max_len, dtype=int)
#                 sampled_actions = all_actions[idxs_a]
#                 action_mask = np.ones((self.max_len,), dtype=bool)
#             else:
#                 sampled_actions = np.zeros((self.max_len, a_dim), dtype=all_actions.dtype)
#                 sampled_actions[:t_a] = all_actions
#                 sampled_actions[t_a:] = all_actions[t_a - 1] if t_a > 0 else 0
#                 action_mask = np.zeros((self.max_len,), dtype=bool)
#                 action_mask[:t_a] = True

#             states_b.append(sampled_states)
#             states_mask_b.append(state_mask)
#             actions_b.append(sampled_actions)
#             actions_mask_b.append(action_mask)

#         stacked_states = np.stack(states_b, axis=0)
#         stacked_states_mask = np.stack(states_mask_b, axis=0)
#         stacked_actions = np.stack(actions_b, axis=0)
#         stacked_actions_mask = np.stack(actions_mask_b, axis=0)

#         if len(eps) == 1:
#             data["dem_prompt_all_states"] = stacked_states[0]
#             data["dem_prompt_all_states_mask"] = stacked_states_mask[0]
#             data["dem_prompt_all_actions"] = stacked_actions[0]
#             data["dem_prompt_all_actions_mask"] = stacked_actions_mask[0]
#         else:
#             data["dem_prompt_all_states"] = stacked_states
#             data["dem_prompt_all_states_mask"] = stacked_states_mask
#             data["dem_prompt_all_actions"] = stacked_actions
#             data["dem_prompt_all_actions_mask"] = stacked_actions_mask

#         # XJ: update cache
#         if split == "test":
#             self._cache["selected_episode"] = list(eps)
#             self._cache["states"] = stacked_states
#             self._cache["states_mask"] = stacked_states_mask
#             self._cache["actions"] = stacked_actions
#             self._cache["actions_mask"] = stacked_actions_mask

#         return data


@dataclasses.dataclass(frozen=True)
class AddStatesActionsPromptTransform(DataTransformFn):
    """
    Adds state/action sequences for multiple episodes.
    """

    dataset: any  # the underlying dataset from which to fetch demo items

    max_len: int = 32
    demo_state_dim: int | None = None

    episode_to_all_states: dict[int, np.ndarray] = dataclasses.field(init=False)
    episode_to_all_first_actions: dict[int, np.ndarray] = dataclasses.field(init=False)

    states_cache_path: str = "metadata/libero/episode_states_cache.json"
    actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"

    # Padding mode: how to sample/pad when episode length differs from max_len
    # "keep_all": Keep all L frames when L < max_len, then pad with last frame (current behavior)
    # "linspace_repeat": Always use linspace sampling, then repeat last sample if needed (training behavior)
    padding_mode: str = "keep_all"

    # Whether to mask padded frames as valid (True) or invalid (False)
    # True matches training behavior where repeated frames have mask=True
    # False is current inference behavior where padded frames have mask=False
    mask_padding_as_valid: bool = False

    def __post_init__(self):
        # Validate padding_mode
        if self.padding_mode not in {"keep_all", "linspace_repeat"}:
            raise ValueError(
                f"Invalid padding_mode: '{self.padding_mode}'. " f"Must be 'keep_all' or 'linspace_repeat'."
            )

        expected_idx_map: dict[int, list[int]] | None = None
        # ---- Load / Build caches for states & actions ----
        try:
            states = load_episode_states_from_json(self.states_cache_path)
            actions = load_episode_states_from_json(self.actions_cache_path)
        except FileNotFoundError:
            idx_map = lookup_tables.episode_to_indexes_for_dataset(self.dataset)
            expected_idx_map = expected_idx_map or idx_map
            states, actions = {}, {}
            for ep, idxs in tqdm(idx_map.items(), desc="Building caches", total=len(idx_map)):
                state_list, action_list = [], []
                for idx in idxs:
                    item = self.dataset[int(idx)]
                    state_list.append(item["state"])  # shape: [D_s]
                    action_list.append(item["actions"][0])  # shape: [D_a]  (keep your original choice)
                states[ep] = np.stack(state_list, axis=0)  # [T, D_s]
                actions[ep] = np.stack(action_list, axis=0)  # [T, D_a]
            save_episode_states_to_json(states, self.states_cache_path)
            save_episode_states_to_json(actions, self.actions_cache_path)

        states = self._maybe_slice_states(states)
        object.__setattr__(self, "episode_to_all_states", states)
        object.__setattr__(self, "episode_to_all_first_actions", actions)

        # ---- Inference cache ----
        object.__setattr__(
            self,
            "_cache",
            {
                "selected_episode": None,
                "states": None,
                "states_mask": None,
                "actions": None,
                "actions_mask": None,
            },
        )

    def _maybe_slice_states(self, states: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        if self.demo_state_dim is None:
            return states
        sliced: dict[int, np.ndarray] = {}
        for ep, arr in states.items():
            arr_np = np.asarray(arr, dtype=np.float32)
            sliced[ep] = arr_np[..., : self.demo_state_dim]
        return sliced

    # ---------- helpers ----------
    @staticmethod
    def _even_sample_len(T: int, max_len: int) -> np.ndarray:
        """linspace indices in [0, T-1], length=max_len; assume T>=1."""
        if max_len <= 0:
            return np.empty((0,), dtype=int)
        if T <= 1:
            return np.zeros((max_len,), dtype=int)
        return np.linspace(0, T - 1, num=max_len, dtype=int)

    def _linspace_sample_and_pad(self, arr: np.ndarray, s: int, e: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample using linspace, then repeat last sample if needed (training behavior).
        Mimics CustomLeRobotDataset padding strategy.

        Returns: (sampled, mask) where mask indicates valid frames.
        """
        T_total = arr.shape[0]
        s = max(0, min(s, T_total))
        e = max(s + 1, min(e, T_total))  # ensure at least 1 frame
        window = arr[s:e]  # [L, D]
        L = window.shape[0]
        D = window.shape[1] if window.ndim > 1 else 1

        # Always use linspace sampling for num_samples = min(L, max_len)
        num_samples = min(L, self.max_len)
        idxs = self._even_sample_len(L, num_samples)
        sampled_frames = window[idxs]  # [num_samples, D]

        # If we need more frames, repeat the last sampled frame
        if num_samples < self.max_len:
            pad_count = self.max_len - num_samples
            # Repeat last sampled frame
            repeat_shape = (pad_count,) + (1,) * (sampled_frames.ndim - 1)
            last_frame = sampled_frames[-1:] if num_samples > 0 else np.zeros((1, D), dtype=arr.dtype)
            repeated_frames = np.repeat(last_frame, pad_count, axis=0)

            # Concatenate original samples with repeated frames
            sampled = np.concatenate([sampled_frames, repeated_frames], axis=0)

            # Mask handling based on mask_padding_as_valid
            if self.mask_padding_as_valid:
                # Training behavior: all frames (including repeated) are marked as valid
                mask = np.ones((self.max_len,), dtype=bool)
            else:
                # Current inference behavior: only original samples are valid
                mask = np.zeros((self.max_len,), dtype=bool)
                mask[:num_samples] = True
        else:
            # No padding needed, all frames are from linspace sampling
            sampled = sampled_frames
            mask = np.ones((self.max_len,), dtype=bool)

        return sampled, mask

    def _window_sample_and_pad(self, arr: np.ndarray, s: int, e: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample / pad from window [s, e) to max_len.
        Returns: (sampled, mask) with mask True for valid frames.

        Dispatches to appropriate implementation based on padding_mode.
        """
        if self.padding_mode == "linspace_repeat":
            # Training-style: always linspace sample, repeat last if needed
            return self._linspace_sample_and_pad(arr, s, e)

        # Default "keep_all" mode: original implementation
        T_total = arr.shape[0]
        s = max(0, min(s, T_total))
        e = max(s + 1, min(e, T_total))  # ensure at least 1 frame
        window = arr[s:e]  # [L, D]
        L = window.shape[0]
        D = window.shape[1] if window.ndim > 1 else 1

        if self.max_len <= L:
            idxs = self._even_sample_len(L, self.max_len)
            sampled = window[idxs]
            mask = np.ones((self.max_len,), dtype=bool)
        else:
            sampled = np.zeros((self.max_len, D), dtype=arr.dtype)
            sampled[:L] = window
            sampled[L:] = window[-1] if L > 0 else 0
            # Apply mask_padding_as_valid for backward compatibility control
            if self.mask_padding_as_valid:
                mask = np.ones((self.max_len,), dtype=bool)
            else:
                mask = np.zeros((self.max_len,), dtype=bool)
                mask[:L] = True
        return sampled, mask

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        eps: list[int] = data.get("selected_episode", [])
        split = data.get("split", "train")

        # ---- Cache hit for test split ----
        if split == "test" and self._cache["selected_episode"] == list(eps):
            if len(eps) == 1:
                data["dem_prompt_all_states"] = self._cache["states"][0]
                data["dem_prompt_all_states_mask"] = self._cache["states_mask"][0]
                data["dem_prompt_all_actions"] = self._cache["actions"][0]
                data["dem_prompt_all_actions_mask"] = self._cache["actions_mask"][0]
            else:
                data["dem_prompt_all_states"] = self._cache["states"]
                data["dem_prompt_all_states_mask"] = self._cache["states_mask"]
                data["dem_prompt_all_actions"] = self._cache["actions"]
                data["dem_prompt_all_actions_mask"] = self._cache["actions_mask"]
            return data

        states_b, states_mask_b, actions_b, actions_mask_b = [], [], [], []

        for ep in eps:
            if ep not in self.episode_to_all_states or ep not in self.episode_to_all_first_actions:
                raise KeyError(
                    f"[AddStatesActionsPromptTransform] episode={ep} missing from cache; check whether states/actions cache covers the task subset."
                )
            all_states = self.episode_to_all_states[ep]  # [T, D_s]
            all_actions = self.episode_to_all_first_actions[ep]  # [T, D_a]
            T = all_states.shape[0]

            # Sample whole episode
            s_rel, e_rel = 0, T

            # Sample / pad within window
            sampled_states, state_mask = self._window_sample_and_pad(all_states, s_rel, e_rel)
            sampled_actions, action_mask = self._window_sample_and_pad(all_actions, s_rel, e_rel)

            states_b.append(sampled_states)
            states_mask_b.append(state_mask)
            actions_b.append(sampled_actions)
            actions_mask_b.append(action_mask)

        # Stack batch dimension
        stacked_states = np.stack(states_b, axis=0)
        stacked_states_mask = np.stack(states_mask_b, axis=0)
        stacked_actions = np.stack(actions_b, axis=0)
        stacked_actions_mask = np.stack(actions_mask_b, axis=0)

        # DEBUG: Print loaded state/action prompt info
        # print("\n" + "="*80)
        # print("LOADED STATE/ACTION PROMPTS:")
        # Single-episode squeeze
        if len(eps) == 1:
            data["dem_prompt_all_states"] = stacked_states[0]
            data["dem_prompt_all_states_mask"] = stacked_states_mask[0]
            data["dem_prompt_all_actions"] = stacked_actions[0]
            data["dem_prompt_all_actions_mask"] = stacked_actions_mask[0]
        else:
            data["dem_prompt_all_states"] = stacked_states
            data["dem_prompt_all_states_mask"] = stacked_states_mask
            data["dem_prompt_all_actions"] = stacked_actions
            data["dem_prompt_all_actions_mask"] = stacked_actions_mask

        # ---- Update cache for test split ----
        if split == "test":
            self._cache["selected_episode"] = list(eps)
            self._cache["states"] = stacked_states
            self._cache["states_mask"] = stacked_states_mask
            self._cache["actions"] = stacked_actions
            self._cache["actions_mask"] = stacked_actions_mask

        return data


@dataclasses.dataclass(frozen=True)
class AddDemoPromptTransform(DataTransformFn):
    # TODO: divid it into two parts: 1) add demo prompt, 2) add states and actions
    dataset: any  # the underlying dataset from which to fetch demo items
    max_len: int = 32

    # These fields are not provided at initialization by the user.
    episode_to_all_states: dict[int, np.ndarray] = dataclasses.field(init=False)
    episode_to_all_first_actions: dict[int, np.ndarray] = dataclasses.field(init=False)

    states_cache_path: str = "metadata/libero/episode_states_cache.json"
    actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"

    def __post_init__(self):
        # TODO: consider delta actions here
        # TODO: refactor the code. consider the case only using seen tasks
        # --- Option A: Try to load precomputed states/actions from JSON cache ---
        try:
            states = load_episode_states_from_json(self.states_cache_path)
            actions = load_episode_states_from_json(self.actions_cache_path)
            print("Loaded states/actions from JSON cache.")
        except FileNotFoundError:
            # --- Option B: Build from scratch, then save ---
            episode_to_indexes = lookup_tables.episode_to_indexes_for_dataset(self.dataset)

            states = {}
            actions = {}

            for episode_id, idx_list in tqdm(
                episode_to_indexes.items(),
                desc="Building lookup tables for episodes",
                total=len(episode_to_indexes),
            ):
                states_list = []
                first_actions_list = []
                for idx in idx_list:
                    item = self.dataset[int(idx)]
                    states_list.append(item["state"])
                    first_actions_list.append(item["actions"][0])
                assert len(states_list) > 0
                assert len(first_actions_list) > 0
                states[episode_id] = np.stack(states_list, axis=0)
                actions[episode_id] = np.stack(first_actions_list, axis=0)
            save_episode_states_to_json(states, self.states_cache_path)
            save_episode_states_to_json(actions, self.actions_cache_path)
            print("Built and saved states/actions JSON cache.")

        object.__setattr__(self, "episode_to_all_states", states)
        object.__setattr__(self, "episode_to_all_first_actions", actions)

    def __call__(self, data: dict) -> dict:
        """
        Transforms a single data item by:
          1. Fetching demonstration prompt items from the underlying dataset based on 'dem_prompt_indexes'.
          2. Retrieving and padding all states and the first actions for the selected episode.
          3. Storing the demonstration prompt items along with padded states/actions and their masks.
        """
        # 1) Retrieve demonstration prompt items using the provided indexes.
        dem_prompt_indexes = data.get("dem_prompt_indexes", [])
        # TODO: check, there may be a bug to include self.dataset in AddDemoPromptTransform
        dem_prompt_items = [self.dataset[int(idx)] for idx in dem_prompt_indexes]
        dem_prompt_items = tree_stack_np(dem_prompt_items)
        data["dem_prompt_items"] = dem_prompt_items
        # 2) Retrieve precomputed states and actions for the selected episode.
        # Here we assume that the key "selected_episode" exists in the data.
        episode_id = data["selected_episode"]
        all_states = self.episode_to_all_states[episode_id]  # shape: (T, D)
        all_actions_first = self.episode_to_all_first_actions[episode_id]  # shape: (T, A)

        # --- Process States Separately ---
        t, d = all_states.shape
        if t >= self.max_len:
            # Uniformly sample self.max_len states if enough are available.
            state_indices = np.linspace(0, t - 1, num=self.max_len, dtype=int)
            sampled_states = all_states[state_indices]
            states_mask = np.ones((self.max_len,), dtype=bool)
        else:
            # Otherwise, pad with the last valid state.
            sampled_states = np.zeros((self.max_len, d), dtype=all_states.dtype)
            sampled_states[:t] = all_states
            if t > 0:
                sampled_states[t:] = all_states[t - 1]
            states_mask = np.zeros((self.max_len,), dtype=bool)
            states_mask[:t] = np.True_

        # --- Process Actions Separately ---
        t_a, a_dim = all_actions_first.shape
        if t_a >= self.max_len:
            # Uniformly sample self.max_len actions if enough are available.
            action_indices = np.linspace(0, t_a - 1, num=self.max_len, dtype=int)
            sampled_actions = all_actions_first[action_indices]
            actions_mask = np.ones((self.max_len,), dtype=bool)
        else:
            # Otherwise, pad with the last valid action.
            sampled_actions = np.zeros((self.max_len, a_dim), dtype=all_actions_first.dtype)
            sampled_actions[:t_a] = all_actions_first
            if t_a > 0:
                sampled_actions[t_a:] = all_actions_first[t_a - 1]
            actions_mask = np.zeros((self.max_len,), dtype=bool)
            actions_mask[:t_a] = np.True_

        # 3) Store the padded states, actions, and their masks in the data dict.
        data["dem_prompt_all_states"] = sampled_states
        data["dem_prompt_all_states_mask"] = states_mask
        data["dem_prompt_all_actions"] = sampled_actions
        data["dem_prompt_all_actions_mask"] = actions_mask

        return data


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
            # DEBUG: Print injected prompt
            # print(f"[InjectDefaultPrompt] Injected prompt: '{self.prompt}'")
        elif "prompt" in data:
            # DEBUG: Print existing prompt
            prompt_str = data["prompt"].item() if hasattr(data["prompt"], "item") else str(data["prompt"])
            # print(f"[InjectDefaultPrompt] Using existing prompt: '{prompt_str}'")
        return data


def _pad_norm_stats_identity(stats: NormStats, target_dim: int) -> NormStats:
    """Extend stats to target_dim with identity values so extra dims pass through Normalize unchanged."""
    cur_dim = stats.mean.shape[-1]
    if cur_dim >= target_dim:
        return stats
    pad = target_dim - cur_dim

    def _extend(arr, fill):
        if arr is None:
            return None
        return np.concatenate([arr, np.full((*arr.shape[:-1], pad), fill, dtype=arr.dtype)], axis=-1)

    return NormStats(
        mean=_extend(stats.mean, 0.0),
        std=_extend(stats.std, 1.0),
        q01=_extend(stats.q01, -1.0),
        q99=_extend(stats.q99, 1.0),
    )


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False
    # Optional mapping from data keys to norm_stats keys for aliasing
    # Example: {"dem_prompt_states": "state", "dem_prompt_actions": "actions"}
    norm_stats_aliases: dict[str, str] | None = None
    # Optional mapping from alias key to target trailing dim. Used when the aliased
    # data is zero-padded beyond its native dim before Normalize runs (e.g. demo
    # actions padded from 7 to demo_action_dim=32): the aliased stats are extended
    # with identity values (mean 0, std 1, q01 -1, q99 1) so padding dims pass
    # through unchanged, matching a pad-after-normalize layout.
    norm_stats_alias_pad_dims: dict[str, int] | None = None

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)
        # Validate that aliases point to existing norm_stats keys.
        if self.norm_stats_aliases is not None and self.norm_stats:
            flat_stats = flatten_dict(self.norm_stats)
            for alias_key, target_key in self.norm_stats_aliases.items():
                if target_key not in flat_stats:
                    raise ValueError(
                        f"Alias '{alias_key}' points to non-existent norm_stats key '{target_key}'. "
                        f"Available keys: {list(flat_stats.keys())}"
                    )
        # Cache the alias-expanded, flat norm_stats once. All fields are
        # frozen-dataclass attrs so the cache is valid for the lifetime of
        # this transform; this avoids re-flattening (3×) and re-unflattening
        # (2×) on every __call__. Built via _expand_norm_stats_with_aliases so
        # alias padding (norm_stats_alias_pad_dims) has a single source of truth.
        if self.norm_stats:
            object.__setattr__(self, "_flat_expanded_norm_stats", flatten_dict(self._expand_norm_stats_with_aliases()))
        else:
            object.__setattr__(self, "_flat_expanded_norm_stats", {})

    def __call__(self, data: DataDict) -> DataDict:
        if not self.norm_stats:
            return data
        fn = self._normalize_quantile if self.use_quantiles else self._normalize
        selector = self._flat_expanded_norm_stats
        flat_data = flatten_dict(data)
        if self.strict:
            for k in selector:
                if k not in flat_data:
                    raise ValueError(f"Selector key {k} not found in tree")
        out = {k: (fn(v, selector[k]) if k in selector else v) for k, v in flat_data.items()}
        return unflatten_dict(out)

    def _expand_norm_stats_with_aliases(self) -> at.PyTree[NormStats]:
        """Create an expanded norm_stats dict that includes alias mappings.

        For each alias, add an entry in norm_stats that points to the same
        NormStats object as the target key. This allows demo data keys to
        use the same normalization statistics as current observation keys.

        Returns:
            Expanded norm_stats dict with aliases resolved.
        """
        if self.norm_stats_aliases is None:
            return self.norm_stats

        # Flatten to work with simple string keys
        flat_stats = flatten_dict(self.norm_stats)

        # Add alias entries (shallow copy - same NormStats objects)
        for alias_key, target_key in self.norm_stats_aliases.items():
            # Only add if alias doesn't already exist (original takes precedence)
            if alias_key not in flat_stats:
                stats = flat_stats[target_key]
                pad_dim = (self.norm_stats_alias_pad_dims or {}).get(alias_key)
                if pad_dim is not None:
                    stats = _pad_norm_stats_identity(stats, pad_dim)
                flat_stats[alias_key] = stats

        # Unflatten back to nested structure
        return unflatten_dict(flat_stats)

    def _normalize(self, x, stats: NormStats):
        return (x - stats.mean) / (stats.std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # Optional mapping from data keys to norm_stats keys for aliasing
    # Example: {"dem_prompt_states": "state", "dem_prompt_actions": "actions"}
    norm_stats_aliases: dict[str, str] | None = None

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

        # Validate that aliases point to existing norm_stats keys.
        if self.norm_stats_aliases is not None and self.norm_stats:
            flat_stats = flatten_dict(self.norm_stats)
            for alias_key, target_key in self.norm_stats_aliases.items():
                if target_key not in flat_stats:
                    raise ValueError(
                        f"Alias '{alias_key}' points to non-existent norm_stats key '{target_key}'. "
                        f"Available keys: {list(flat_stats.keys())}"
                    )
        # See Normalize.__post_init__ for the cache rationale.
        if self.norm_stats:
            flat = flatten_dict(self.norm_stats)
            if self.norm_stats_aliases is not None:
                for alias_key, target_key in self.norm_stats_aliases.items():
                    if alias_key not in flat:
                        flat[alias_key] = flat[target_key]
            object.__setattr__(self, "_flat_expanded_norm_stats", flat)
        else:
            object.__setattr__(self, "_flat_expanded_norm_stats", {})

    def __call__(self, data: DataDict) -> DataDict:
        if not self.norm_stats:
            return data
        fn = self._unnormalize_quantile if self.use_quantiles else self._unnormalize
        selector = self._flat_expanded_norm_stats
        flat_data = flatten_dict(data)
        # Unnormalize originally passed strict=True; preserve that contract.
        for k in selector:
            if k not in flat_data:
                raise ValueError(f"Selector key {k} not found in tree")
        out = {k: (fn(v, selector[k]) if k in selector else v) for k, v in flat_data.items()}
        return unflatten_dict(out)

    def _expand_norm_stats_with_aliases(self) -> at.PyTree[NormStats]:
        """Create an expanded norm_stats dict that includes alias mappings.

        For each alias, add an entry in norm_stats that points to the same
        NormStats object as the target key. This allows demo data keys to
        use the same normalization statistics as current observation keys.

        Returns:
            Expanded norm_stats dict with aliases resolved.
        """
        if self.norm_stats_aliases is None:
            return self.norm_stats

        # Flatten to work with simple string keys
        flat_stats = flatten_dict(self.norm_stats)

        # Add alias entries (shallow copy - same NormStats objects)
        for alias_key, target_key in self.norm_stats_aliases.items():
            # Only add if alias doesn't already exist (original takes precedence)
            if alias_key not in flat_stats:
                flat_stats[alias_key] = flat_stats[target_key]

        # Unflatten back to nested structure
        return unflatten_dict(flat_stats)

    def _unnormalize(self, x, stats: NormStats):
        return x * (stats.std + 1e-6) + stats.mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        # Resize demonstration prompt images if present (for CustomLeRobotDataset)
        if "dem_prompt_images" in data:
            data["dem_prompt_images"] = {
                k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["dem_prompt_images"].items()
            }
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()
        tokens, token_masks = self.tokenizer.tokenize(prompt)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class TokenizeFASTIncontextInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    max_incontext_steps: int = 4

    def _encode_state_sequence(self, states: np.ndarray) -> np.ndarray:
        discretized = np.digitize(np.asarray(states), bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        base = self.tokenizer._paligemma_tokenizer.vocab_size() - 1 - self.tokenizer._fast_skip_tokens
        return (base - discretized.astype(np.int32)).reshape(-1)

    def _encode_action_sequence(self, actions: np.ndarray) -> np.ndarray:
        fast_tokens = self.tokenizer._fast_tokenizer(np.asarray(actions)[None])[0]
        return self.tokenizer._act_tokens_to_paligemma_tokens(fast_tokens).astype(np.int32)

    def _pad_tokens(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_len = self.tokenizer._max_len
        tokens = np.asarray(tokens, dtype=np.int32)
        if tokens.size > max_len:
            logging.warning(
                "Incontext token length (%d) exceeds max length (%d); truncating.",
                tokens.size,
                max_len,
            )
            tokens = tokens[:max_len]
        padded = np.zeros(max_len, dtype=np.int32)
        mask = np.zeros(max_len, dtype=bool)
        padded[: tokens.size] = tokens
        mask[: tokens.size] = True
        return padded, mask

    def _concat_encoded_states(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states)
        if states.ndim == 2:
            chunks = [self._encode_state_sequence(states[: self.max_incontext_steps])]
        elif states.ndim >= 3:
            chunks = [self._encode_state_sequence(ep[: self.max_incontext_steps]) for ep in states]
        else:
            raise ValueError("demonstration states must be at least 2-D")
        if not chunks:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate(chunks, axis=0)

    def _concat_encoded_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions)
        if actions.ndim == 2:
            chunks = [self._encode_action_sequence(actions[: self.max_incontext_steps])]
        elif actions.ndim >= 3:
            chunks = [self._encode_action_sequence(ep[: self.max_incontext_steps]) for ep in actions]
        else:
            raise ValueError("demonstration actions must be at least 2-D")
        if not chunks:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate(chunks, axis=0)

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("TokenizeFASTIncontextInputs requires a 'prompt' field")

        demo_states = data.get("dem_prompt_all_states")
        demo_actions = data.get("dem_prompt_all_actions")
        if demo_states is None or demo_actions is None:
            raise ValueError("Incontext FAST tokenization requires demo states/actions")

        state_tokens = self._concat_encoded_states(demo_states)
        action_tokens = self._concat_encoded_actions(demo_actions)

        padded_state_tokens, state_mask = self._pad_tokens(state_tokens)
        padded_action_tokens, action_mask = self._pad_tokens(action_tokens)

        data["tokenized_incontext_states"] = padded_state_tokens
        data["tokenized_incontext_states_mask"] = state_mask
        data["incontext_states_ar_mask"] = np.zeros_like(padded_state_tokens, dtype=np.int32)
        data["incontext_states_loss_mask"] = np.zeros_like(padded_state_tokens, dtype=bool)

        data["tokenized_incontext_actions"] = padded_action_tokens
        data["tokenized_incontext_actions_mask"] = action_mask
        data["incontext_actions_ar_mask"] = np.zeros_like(padded_action_tokens, dtype=np.int32)
        data["incontext_actions_loss_mask"] = np.zeros_like(padded_action_tokens, dtype=bool)

        state = data["state"]
        actions = data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        data["tokenized_prompt"] = tokens
        data["tokenized_prompt_mask"] = token_mask
        data["token_ar_mask"] = ar_mask
        data["token_loss_mask"] = loss_mask

        return data


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
