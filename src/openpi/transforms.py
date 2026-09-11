from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import logging
import re
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

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
class InjectDemoFromCustomDataset(DataTransformFn):
    """Populates in-context demonstration fields by calling a CustomLeRobotDataset directly.

    The demonstrations are read from the same dataset implementation used by training.

    On call, reads ``task_index`` from the data dict and delegates to
    ``CustomLeRobotDataset.load_incontext_demonstration`` to populate:
        - ``dem_prompt_images`` (dict {"image", "wrist_image"} of torch tensors)
        - ``dem_prompt_states`` (torch tensor)
        - ``dem_prompt_actions`` (torch tensor)
        - ``selected_episode`` (np.ndarray[int32])

    For ``split == "test"``, demos are cached per ``task_index`` so all frames of a rollout
    see the same demonstration.
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

        # On the test split, demo selection is deterministic regardless of the dataset's random_select flag,
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


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
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
