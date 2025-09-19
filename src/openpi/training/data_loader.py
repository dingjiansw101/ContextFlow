from collections.abc import Iterator, Sequence
import json
import multiprocessing
import os
from pathlib import Path
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

import dataclasses
from typing import Any, Dict, Sequence as _Seq  # 不覆盖上面的 Sequence

@dataclasses.dataclass
class DebugProbe:
    tag: str = "probe"
    keys: _Seq[str] = ("episode_index", "frame_index", "index", "task_index")
    max_print: int = 5       # 仅打印前 N 次
    every_n: int = 0         # 或每 N 次打印一次；0 表示不开

    def __post_init__(self):
        self._count = 0

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self._count += 1
        do_print = (self._count <= self.max_print) or (self.every_n and self._count % self.every_n == 0)
        if do_print:
            fields = " ".join(f"{k}={data.get(k, None)}" for k in self.keys)
            print(f"[{self.tag}] {fields}")
        return data

def _instrument_transforms(transforms: Sequence[_transforms.DataTransformFn],
                           *,
                           prefix: str = "T",
                           keys: tuple[str, ...] = ("episode_index", "frame_index", "index", "task_index"),
                           max_print: int = 5) -> list[_transforms.DataTransformFn]:
    """在每个 transform 前插入 DebugProbe(tag='prefix#idx:ClassName')。"""
    out: list[_transforms.DataTransformFn] = []
    for i, t in enumerate(transforms):
        tag = f"{prefix}#{i}:{t.__class__.__name__}"
        out.append(DebugProbe(tag=tag, keys=keys, max_print=max_print))
        out.append(t)
    return out


# TODO: refactor: checking passed train_episode is None or not
def is_effective_none(x):
    if x is None:
        return True
    if isinstance(x, tuple) and len(x) == 1 and x[0] is None:
        return True
    return False


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

        # Xianjie: for data transform debug
        # self._transform_list = transforms


    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


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
        # TODO: Xianjie: reduendant check of None type
        # Xianjie: either None or training episode index list
        episodes=data_config.train_episode if not is_effective_none(data_config.train_episode) else None,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
        local_files_only=data_config.local_files_only,
    )
    # import ipdb; ipdb.set_trace()
    if data_config.prompt_from_task:
        # TODO: language instrucitons of libero are stored here
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


# def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
#     """Transform the dataset by applying the data transforms."""
#     norm_stats = {}
#     if data_config.repo_id != "fake" and not skip_norm_stats:
#         if data_config.norm_stats is None:
#             raise ValueError(
#                 "Normalization stats not found. "
#                 "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
#             )
#         norm_stats = data_config.norm_stats
        
#     return TransformedDataset(
#         dataset,
#         [
#             *data_config.repack_transforms.inputs,
#             *data_config.data_transforms.inputs,
#             _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
#             *data_config.model_transforms.inputs,
#         ],
#     )
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

    # 先把原先的 transforms 串起来（不改变语义）
    seq: list[_transforms.DataTransformFn] = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    # 开关：DL_TRACE=1 时才注入探针（默认不打印）
    if os.environ.get("DL_TRACE", "0") == "1":
        seq = _instrument_transforms(seq, prefix="TF", keys=("episode_index", "frame_index", "index", "task_index"),
                                     max_print=int(os.environ.get("DL_TRACE_MAX", "20")))

    return TransformedDataset(dataset, seq)


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
            self._dataset = dataset


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
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    # dataset_old = AddDemoPromptDataset(dataset)

    if config.model.use_image_prompts:
        print("Using image prompt")
        add_image_transform = _transforms.AddImagePromptTransform(dataset=dataset)
        dataset = TransformedDataset(dataset, [add_image_transform])

    if config.model.use_action_state_prompts:
        print("Using action-state prompt")
        if config.data.episode_to_indexes_file is not None:
            add_demo_transform = _transforms.AddStatesActionsPromptTransform(dataset=dataset, max_len=config.model.sample_actions,
                                                                states_cache_path=config.data.states_cache_path,
                                                                actions_cache_path=config.data.actions_cache_path,
                                                                episode_to_indexes_file=config.data.episode_to_indexes_file,
                                                                all_episode_stage = config.data.all_episode_stage,
                                                                )                                        
        else:
            add_demo_transform = _transforms.AddStatesActionsPromptTransform(dataset=dataset, max_len=config.model.sample_actions,
                                                                states_cache_path=config.data.states_cache_path,
                                                                actions_cache_path=config.data.actions_cache_path,
                                                                all_episode_stage = config.data.all_episode_stage)
        dataset = TransformedDataset(dataset, [add_demo_transform])

    if config.model.use_point_track_prompts:
        print("Using point track prompt")
        add_point_track_transform = _transforms.AddPointTrackPromptTransform(
            max_len=config.model.sample_actions,
            tracks_path=config.data.tracks_path)
        dataset = TransformedDataset(dataset, [add_point_track_transform])
    # jax.tree_util.tree_all(jax.tree_map(np.allclose, dataset[0], dataset_old[0]))
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
