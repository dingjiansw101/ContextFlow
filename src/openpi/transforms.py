from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
from pathlib import Path
import random
import re
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable, Optional, List, Dict

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize
from tqdm import tqdm

# from openpi.training.data_loader import Dataset

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")

def reindex_filtered_dict(data: Dict[str, Any]) -> Dict[str, Any]:
# TODO: check this function
    new_data = {}
    current_frame_index = 0

    for ep_idx in data:
        num_frames = len(data[ep_idx])
        new_data[ep_idx] = list(range(current_frame_index, current_frame_index + num_frames))
        current_frame_index += num_frames
    return new_data


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
    A data transform that reads two JSON files containing:
      1) task_to_episode: {task_index (str): list of episode_index (int)}
      2) episode_to_indexes: {episode_index (str): list of index_idx (int)}
    and converts their keys to integers.
    """

    task_to_episode_path: Path = Path("metadata/libero/task_to_episode.json")
    episode_to_indexes_path: Path = Path("metadata/libero/episode_to_indexes.json")
    sample_frames: int = 16
    random_select: bool = True
    train_episode_index_list: Optional[List[int]] = None

    def __post_init__(self):

        # Load JSON files using Path.open()
        with self.task_to_episode_path.open("r") as f:
            task_to_episode_str = json.load(f)
        with self.episode_to_indexes_path.open("r") as f:
            episode_to_indexes_str = json.load(f)

        # Convert dictionary keys from strings to integers
        task_to_episode = {int(k): v for k, v in task_to_episode_str.items()}
        # import ipdb; ipdb.set_trace()
        if self.train_episode_index_list is None:
            episode_to_indexes = {int(k): v for k, v in episode_to_indexes_str.items()}
        else:
            episode_to_indexes = {int(k): v for k, v in episode_to_indexes_str.items() if int(k) in self.train_episode_index_list}

            # XIANJIE: if train-test split, test episodes are removed and the corresponding frames are removed
            # which leads to non-continuous frame index
            # the frame index could exceed the length of LeRobot dataset (number of frames of all the parquet files)
            # therefore, the frame index must be reindexed, in the continuous manner.
            # LeRobot dataset follows the order of "train_episode_index_list"
            # so we can simple reindex the frame index in the following way:
            episode_to_indexes = reindex_filtered_dict(episode_to_indexes)

        # Store these dictionaries on the frozen dataclass
        object.__setattr__(self, "task_to_episode", task_to_episode)
        object.__setattr__(self, "episode_to_indexes", episode_to_indexes)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Example transform: randomly select a demonstration as prompt for the data,
        storing both the chosen indexes and the retrieved items.
        """
        # 1) Randomly pick an episode for this task
        task_index = int(data["task_index"])
        episodes_for_task = self.task_to_episode.get(task_index, [])

        split = data.get("split", "train")
        
        if split == "train" and self.random_select:
            selected_episode = random.choice(episodes_for_task)
        else:
            selected_episode = episodes_for_task[0] if episodes_for_task else None


        # 2) Get all indexes for that episode
        all_indexes = self.episode_to_indexes.get(selected_episode, [])

        # 3) Uniformly sample up to self.sample_frames frames
        total_frames = len(all_indexes)
        if total_frames > self.sample_frames:
            # Generate self.sample_frames evenly spaced positions
            positions = np.linspace(0, total_frames - 1, num=self.sample_frames)
            positions = np.round(positions).astype(int).tolist()
            chosen_indexes = [all_indexes[pos] for pos in positions]
        else:
            chosen_indexes = all_indexes

        # 4) Store both the chosen indexes and the items in `data`
        data["dem_prompt_indexes"] = np.array(chosen_indexes)
        data["selected_episode"] = selected_episode

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
class AddDemoPromptTransform(DataTransformFn):
    # TODO: divid it into two parts: 1) add demo prompt, 2) add states and actions
    dataset: any  # the underlying dataset from which to fetch demo items
    max_len: int = 32 

    # These fields are not provided at initialization by the user.
    episode_to_all_states: dict[int, np.ndarray] = dataclasses.field(init=False)
    episode_to_all_first_actions: dict[int, np.ndarray] = dataclasses.field(init=False)

    states_cache_path: str = "metadata/libero/episode_states_cache.json"
    actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
    episode_to_indexes_file: str = "metadata/libero/episode_to_indexes.json"
    
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
            episode_to_indexes_path = Path(self.episode_to_indexes_file)
            with episode_to_indexes_path.open("r") as f:
                episode_to_indexes_str = json.load(f)
            episode_to_indexes = {int(k): v for k, v in episode_to_indexes_str.items()}

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
                import ipdb; ipdb.set_trace()
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
        # jax.debug.print("self.max_len: {}", self.max_len)
        # 2) Retrieve precomputed states and actions for the selected episode.
        # Here we assume that the key "selected_episode" exists in the data.
        episode_id = data["selected_episode"]
        # jax.debug.print("episode_id: {}", episode_id)   
        all_states = self.episode_to_all_states[episode_id]       # shape: (T, D)
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
class AddPointTrackPromptTransform(DataTransformFn):
    """
    Data transform that adds a demonstration prompt of precomputed track states
    for a selected episode. Pads or samples to a fixed sequence length.
    """
    max_len: int = 32                     # target sequence length
    tracks_path: str = "metadata/libero/episode_tracks_combined.json"

    # populated in __post_init__, mapping episode_id -> np.ndarray of shape (T, F)
    episode_to_tracks: Dict[int, np.ndarray] = dataclasses.field(init=False)

    def __post_init__(self):
        # Load precomputed track sequences from JSON cache
        ep_tracks = load_episode_states_from_json(self.tracks_path)
        object.__setattr__(self, "episode_to_tracks", ep_tracks)
        print(f"Loaded {len(ep_tracks)} episodes of track data from {self.tracks_path}.")

    def __call__(self, data: Dict) -> Dict:
        # Retrieve the ID of the selected episode
        episode_id = data["selected_episode"]
        # Look up the full track sequence for this episode
        tracks = self.episode_to_tracks[episode_id]    # shape (T, F)
        T, F = tracks.shape

        # If we have at least max_len frames, sample uniformly
        if T >= self.max_len:
            indices = np.linspace(0, T - 1, num=self.max_len, dtype=int)
            sampled_tracks = tracks[indices]
            mask = np.ones((self.max_len,), dtype=bool)
        else:
            # Otherwise pad to max_len by repeating the last frame
            sampled_tracks = np.zeros((self.max_len, F), dtype=tracks.dtype)
            sampled_tracks[:T] = tracks
            if T > 0:
                sampled_tracks[T:] = tracks[T - 1]
            mask = np.zeros((self.max_len,), dtype=bool)
            mask[:T] = np.True_

        # Store into the data dict for downstream use
        data["dem_prompt_tracks"] = sampled_tracks         # np.ndarray (max_len, F)
        data["dem_prompt_tracks_mask"] = mask              # np.ndarray (max_len,)

        return data

@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

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

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

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
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
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


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


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
