from collections.abc import Iterator, Sequence
import json
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
from tqdm import tqdm

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


def tree_stack_np(list_of_trees, axis=0):
    """
    Stack a list of similarly structured PyTrees along `axis`,
    ensuring the leaves are NumPy arrays.
    """

    def stack_fn(*leaves):
        return np.stack(leaves, axis=axis)

    return jax.tree_map(stack_fn, *list_of_trees)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


def save_episode_states_to_json(episode_to_all_states: dict[int, np.ndarray], filename: str):
    """
    Converts each NumPy array to a Python list, then dumps to JSON.
    """
    # Convert episode_id to string (JSON keys must be strings)
    json_dict = {}
    for episode_id, states_array in episode_to_all_states.items():
        json_dict[str(episode_id)] = states_array.tolist()

    with open(filename, "w") as f:
        json.dump(json_dict, f)


def load_episode_states_from_json(filename: str) -> dict[int, np.ndarray]:
    """
    Loads the JSON file and reconstructs each list into a NumPy array.
    """
    with open(filename) as f:
        json_dict = json.load(f)

    episode_to_all_states = {}
    for episode_id_str, state_list in json_dict.items():
        episode_id = int(episode_id_str)
        # Recreate the NumPy array (optionally specify dtype if needed)
        episode_to_all_states[episode_id] = np.array(state_list, dtype=np.float32)

    return episode_to_all_states


class AddDemoPromptDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._max_len = 512
        # --- Option A: If cache exists, load from JSON ---
        try:
            self.episode_to_all_states = load_episode_states_from_json("metadata/libero/episode_states_cache.json")
            self.episode_to_all_first_actions = load_episode_states_from_json(
                "metadata/libero/episode_actions_first_cache.json"
            )
            print("Loaded states/actions from JSON cache.")
        except FileNotFoundError:
            # --- Option B: Build from scratch, then save ---
            # Load or build the episode_to_indexes
            episode_to_indexes_path = "metadata/libero/episode_to_indexes.json"
            with open(episode_to_indexes_path) as f:
                episode_to_indexes_str = json.load(f)
            self.episode_to_indexes = {int(k): v for k, v in episode_to_indexes_str.items()}

            self.episode_to_all_states = {}
            self.episode_to_all_first_actions = {}

            # for episode_id, idx_list in self.episode_to_indexes.items():
            for episode_id, idx_list in tqdm(
                self.episode_to_indexes.items(),
                desc="Building lookup tables for episodes",
                total=len(self.episode_to_indexes),
            ):
                states_list = []
                first_actions_list = []
                for idx in idx_list:
                    item = self._dataset[idx]
                    states_list.append(item["state"])
                    first_actions_list.append(item["actions"][0])

                assert len(states_list) > 0
                assert len(first_actions_list) > 0
                self.episode_to_all_states[episode_id] = np.stack(states_list, axis=0)
                self.episode_to_all_first_actions[episode_id] = np.stack(first_actions_list, axis=0)
            # Save to JSON so next time we can load it
            save_episode_states_to_json(self.episode_to_all_states, "metadata/libero/episode_states_cache.json")
            save_episode_states_to_json(
                self.episode_to_all_first_actions, "metadata/libero/episode_actions_first_cache.json"
            )
            print("Built and saved states/actions JSON cache.")

    def __getitem__(self, index: SupportsIndex) -> T_co:
        item = self._dataset[index]
        dem_prompt_indexes = item.get("dem_prompt_indexes", [])
        dem_prompt_items = [self._dataset[int(idx)] for idx in dem_prompt_indexes]
        dem_prompt_items = tree_stack_np(dem_prompt_items)
        item["dem_prompt_items"] = dem_prompt_items

        # Suppose item["episode_id"] tells us which episode
        episode_id = item["selected_episode"]

        # Retrieve precomputed states and first actions
        all_states = self.episode_to_all_states[episode_id]  # shape: (T, D)
        all_actions_first = self.episode_to_all_first_actions[episode_id]  # shape: (T, A)

        # --- Pad States ---
        t, d = all_states.shape
        length_to_copy = min(t, self._max_len)

        padded_states = np.zeros((self._max_len, d), dtype=all_states.dtype)
        # Copy the real portion
        padded_states[:length_to_copy] = all_states[:length_to_copy]
        # Pad with the last valid state (if any)
        if length_to_copy < self._max_len and t > 0:
            padded_states[length_to_copy:] = all_states[length_to_copy - 1]

        # Create a boolean mask for states
        states_mask = np.full((self._max_len,), fill_value=np.False_, dtype=bool)
        states_mask[:length_to_copy] = np.True_

        # --- Pad Actions ---
        t_actions, a_dim = all_actions_first.shape
        length_to_copy_actions = min(t_actions, self._max_len)

        padded_actions = np.zeros((self._max_len, a_dim), dtype=all_actions_first.dtype)
        # Copy the real portion
        padded_actions[:length_to_copy_actions] = all_actions_first[:length_to_copy_actions]
        # Pad with the last valid action (if any)
        if length_to_copy_actions < self._max_len and t_actions > 0:
            padded_actions[length_to_copy_actions:] = all_actions_first[length_to_copy_actions - 1]

        # Create a boolean mask for actions
        actions_mask = np.full((self._max_len,), fill_value=np.False_, dtype=bool)
        actions_mask[:length_to_copy_actions] = np.True_

        # Store them in the item
        item["dem_prompt_all_states"] = padded_states
        item["dem_prompt_all_states_mask"] = states_mask
        item["dem_prompt_all_actions"] = padded_actions
        item["dem_prompt_all_actions_mask"] = actions_mask

        return item

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, local_files_only=data_config.local_files_only)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
        local_files_only=data_config.local_files_only,
    )
    if data_config.prompt_from_task:
        # TODO: language instrucitons of libero are stored here
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats
    # import ipdb; ipdb.set_trace()
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader)


def create_incontext_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_dataset(data_config, config.model)
    # import ipdb; ipdb.set_trace()
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    dataset = AddDemoPromptDataset(dataset)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    # import ipdb; ipdb.set_trace()
    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader, dataset: Dataset):
            self._data_config = data_config
            self._data_loader = data_loader
            self._dataset = dataset

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                # import ipdb; ipdb.set_trace()
                # yield _model.Observation.from_dict(batch), batch["actions"]
                yield _model.ObservationIncontext.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader, dataset)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
