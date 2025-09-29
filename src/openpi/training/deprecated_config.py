"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias, Optional, List, Union, Iterable, List
import os, re, json, jsonlines, sys, importlib, pathlib

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.pi0_incontext as pi0_incontext
import openpi.models.pi0_incontextv2 as pi0_incontextv2
import openpi.models.pi0_incontextv3 as pi0_incontextv3
import openpi.models.pi0_incontextv4 as pi0_incontextv4
import openpi.models.pi0_incontextv6 as pi0_incontextv6
import openpi.models.pi0_incontextv7 as pi0_incontextv7
import openpi.models.pi0_incontextv8 as pi0_incontextv8
import openpi.models.pi0_incontextv9 as pi0_incontextv9
import openpi.models.pi0_incontextv10 as pi0_incontextv10
import openpi.models.pi0_incontextv11 as pi0_incontextv11
import openpi.models.pi0_incontextv12 as pi0_incontextv12
import openpi.models.pi0_light as pi0Light
import openpi.models.deprecated_pi0light_incontextv12 as deprecated_pi0light_incontextv12
import openpi.models.pi0_light_incontextv12 as pi0_light_incontextv12
import openpi.models.pi0_incontextv12_dummy as pi0_incontextv12_dummy



import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_mobile_policy as aloha_mobile_policy
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_incontext_policy as libero_incontext_policy
import openpi.policies.aloha_mobile_incontext_policy as aloha_incontext_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.rlbench_gripper_policy as rlbench_gripper_policy
import openpi.policies.rlbench_joint_policy as rlbench_joint_policy
import openpi.policies.robocasa_insertion_policy as robocasa_insertion_policy
import openpi.policies.robocasa_human_policy as robocasa_human_policy
import openpi.policies.robocasa_human_three_image_policy as robocasa_human_three_image_policy
# import openpi.policies.robocasa_human_three_image_base_obs_policy as robocasa_human_three_image_base_obs_policy
import openpi.policies.robocasa_single_task_policy as robocasa_single_task_policy
import openpi.policies.robocasa_human_three_image_incontext_policy as robocasa_human_three_image_incontext_policy
import openpi.policies.robocasa_mg_three_image_policy as robocasa_mg_three_image_policy
import openpi.policies.robocasa_mg_three_image_incontext_policy as robocasa_mg_three_image_incontext_policy


import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

from pathlib import Path

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter
_NAME_RE = re.compile(r"(\d+)") 

from pathlib import Path

def get_project_root() -> Path:
    if "OPENPI_PROJECT_ROOT" in os.environ:
        return Path(os.environ["OPENPI_PROJECT_ROOT"]).expanduser().resolve()
    return Path(__file__).resolve().parents[3]

PROJECT_ROOT = get_project_root()

DEFAULT_LIBERO_EPISODE_JSON = str(Path("~/.cache/huggingface/lerobot/physical-intelligence/libero/meta/episodes.jsonl").expanduser())
#"/home/dingj0b/.cache/huggingface/lerobot/physical-intelligence/libero/meta/episodes.jsonl"

DEFAULT_LIBERO_TEST_TASK = [
        # 10
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        "put both the alphabet soup and the tomato sauce in the basket",
        # goal
        "put the bowl on the plate",
        "put the bowl on the stove",
        # object
        "pick up the milk and place it in the basket",
        "pick up the tomato sauce and place it in the basket",
        # spatial
        "pick up the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl next to the plate and place it on the plate",
]
DEFAULT_LIBERO_TEST_TASK_V2 = [
        # 10
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        "pick up the book and place it in the back compartment of the caddy",
        # goal
        # "turn on the stove",
        "open the middle drawer of the cabinet",
        # object
        "pick up the milk and place it in the basket",
        "pick up the tomato sauce and place it in the basket",
        # spatial
        "pick up the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl next to the plate and place it on the plate",
]

DEFAULT_LIBERO_TEST_TASK_V3 = [
        # 10
        "turn on the stove and put the moka pot on it",
        "put both the cream cheese box and the butter in the basket",
        # goal
        "put the wine bottle on the rack",
        "put the cream cheese in the bowl",
        # object
        "pick up the ketchup and place it in the basket",
        "pick up the bbq sauce and place it in the basket",
        # spatial
        "pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
]
DEFAULT_LIBERO_TEST_TASK_V4 = [
        # 10
        "put both the alphabet soup and the cream cheese box in the basket",
        "put both moka pots on the stove",
        # goal
        "open the top drawer and put the bowl inside",
        "put the wine bottle on top of the cabinet",
        # object
        "pick up the butter and place it in the basket",
        "pick up the salad dressing and place it in the basket",
        # spatial
        "pick up the black bowl on the wooden cabinet and place it on the plate",
        "pick up the black bowl from table center and place it on the plate",
]

DEFAULT_ROBOCASA_EPISODE_JSON = str(Path("~/.cache/huggingface/lerobot/daixianjie/robocasa_human_lerobot/meta/episodes.jsonl").expanduser())
#"/home/dingj0b/.cache/huggingface/lerobot/daixianjie/robocasa_human_lerobot/meta/episodes.jsonl"


DEFAULT_ROBOCASA_TEST_TASK = [
    str(PROJECT_ROOT / "examples" / "robocasa" / "robocasa_human_tasks.json")
]

DEFAULT_ROBOCASA_MG_EPISODE_JSON = str(Path("~/.cache/huggingface/lerobot/daixianjie/robocasa_mg_lerobot/meta/episodes.jsonl").expanduser())
#"/home/dingj0b/.cache/huggingface/lerobot/daixianjie/robocasa_mg_lerobot/meta/episodes.jsonl"

DEFAULT_ROBOCASA_MG_TEST_TASK = [
    str(PROJECT_ROOT / "examples" / "robocasa" / "robocasa_mg_tasks.json")
]

DEFAULT_ROBOCASA_MG_TEST_TASK_WITHOUT_OPENDOUBLEDOOR = [
    str(PROJECT_ROOT / "examples" / "robocasa" / "robocasa_mg_tasks_without_open_double_door.json")
]


# --- helper, keep tiny & local ---
def _basename(x: str) -> str:
    return os.path.basename(str(x)).strip()

def _stem(x: str) -> str:
    return os.path.splitext(_basename(x))[0]

def _normalize_episode_name(x: str) -> str:
    """
    Normalize to basename like 'episode_000937.parquet'.
    Accepts full path or name. Lowercases and strips spaces.
    """
    base = os.path.basename(x).strip()
    return base

def _name_to_index(name: str) -> Optional[int]:
    """
    get int from file name:
      'episode_000012.parquet' -> 12
      'episode_12' -> 12
    fail get None
    """
    m = _NAME_RE.search(_stem(name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _load_name_whitelist(src: Union[str, Path, List[str]]) -> List[str]:
    if isinstance(src, list):
        names = src
    else:
        p = Path(src)
        if not p.exists():
            raise FileNotFoundError(f"keep list not found: {p}")
        if p.suffix.lower() in [".txt", ".list"]:
            with p.open("r") as f:
                names = [line.strip() for line in f if line.strip()]
        elif p.suffix.lower() == ".json":
            with p.open("r") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                names = obj
            elif isinstance(obj, dict):
                names = obj.get("episodes", [])
            else:
                raise ValueError(f"Unsupported JSON content in {p}")
        else:
            raise ValueError(f"Unsupported keep list suffix: {p.suffix}")
    return [str(n).strip() for n in names if str(n).strip()]

def get_kept_episode_indices(
    episodes_jsonl_path: Union[str, Path],
    exclude_task_language: Optional[List[str]],
    include_episode_filenames: Optional[Union[str, Path, List[str]]] = None,
    *,
    verbose: bool = True,
) -> Optional[List[int]]:
    if episodes_jsonl_path is None:
        return None

    ep_path = Path(episodes_jsonl_path)
    if not ep_path.exists():
        raise FileNotFoundError(f"episodes.jsonl file not found at: {ep_path}")

    kept: List[int] = []

    if include_episode_filenames is not None:
        raw_names = _load_name_whitelist(include_episode_filenames)
        idx_whitelist = set()
        bad_names = []
        for n in raw_names:
            idx = _name_to_index(n)
            if idx is None:
                bad_names.append(n)
            else:
                idx_whitelist.add(idx)
        if verbose:
            print(f"[whitelist] loaded {len(raw_names)} names -> {len(idx_whitelist)} indices.")
            if bad_names:
                print(f"[whitelist][warn] failed to parse indices from {len(bad_names)} names (show up to 5): {bad_names[:5]}")

        found_indices = set()
        with jsonlines.open(ep_path, mode="r") as reader:
            for entry in reader:
                ep_idx = entry.get("episode_index")
                if ep_idx is None:
                    continue
                try:
                    ep_idx = int(ep_idx)
                except Exception:
                    continue
                if ep_idx in idx_whitelist:
                    kept.append(ep_idx)
                    found_indices.add(ep_idx)

        if verbose:
            print(f"[whitelist] matched {len(kept)} episodes by index.")
            if len(found_indices) < len(idx_whitelist):
                missing = sorted(idx_whitelist - found_indices)
                print(f"[whitelist][diagnose] {len(idx_whitelist)-len(found_indices)} indices from keep list not found in jsonl (up to 10): {missing[:10]}")

        return kept

    if exclude_task_language is None:
        return None
    if not isinstance(exclude_task_language, list) or not all(isinstance(t, str) for t in exclude_task_language):
        raise TypeError("exclude_task_language must be a list of strings.")

    with jsonlines.open(ep_path, mode='r') as reader:
        for entry in reader:
            if "episode_index" not in entry or "tasks" not in entry:
                raise ValueError(f"Invalid entry (missing 'episode_index' or 'tasks'): {entry}")
            tasks = entry["tasks"]
            if not isinstance(tasks, list):
                raise ValueError(f"'tasks' must be a list of strings, but got: {type(tasks)}")
            if not any(task in exclude_task_language for task in tasks):
                kept.append(int(entry["episode_index"]))

    if verbose:
        print(f"[exclude-by-task] kept {len(kept)} episodes.")
    return kept

def deprecated_get_kept_episode_indices(
    episodes_jsonl_path: Union[str, Path],
    exclude_task_language: List[str]
) -> Optional[List[int]]:
    """
    Filters episode indices from a episodes.jsonl file by excluding those
    whose task descriptions match any entry in the given exclude list.

    Args:
        episodes_jsonl_path (Union[str, Path]): Path to the `episodes.jsonl` file.
        exclude_task_language (List[str]): List of task descriptions to exclude.

    Returns:
        List[int]: List of episode indices to keep (i.e., not excluded).

    Raises:
        TypeError: If argument types are incorrect.
        FileNotFoundError: If the episodes.jsonl file does not exist.
        ValueError: If the file content is malformed or missing required fields.
    """
    if episodes_jsonl_path is None or exclude_task_language is None:
        return None
    # Type checks
    if not isinstance(exclude_task_language, list) or not all(isinstance(t, str) for t in exclude_task_language):
        raise TypeError("exclude_task_language must be a list of strings.")
    
    if not isinstance(episodes_jsonl_path, (str, Path)):
        raise TypeError("episodes_jsonl_path must be a string or Path.")

    episodes_jsonl_path = Path(episodes_jsonl_path)
    if not episodes_jsonl_path.exists():
        raise FileNotFoundError(f"episodes.jsonl file not found at: {episodes_jsonl_path}")

    kept_indices: List[int] = []

    # Read and filter
    with jsonlines.open(episodes_jsonl_path, mode='r') as reader:
        for entry in reader:
            if "episode_index" not in entry or "tasks" not in entry:
                raise ValueError(f"Invalid entry (missing 'episode_index' or 'tasks'): {entry}")

            tasks = entry["tasks"]
            if not isinstance(tasks, list):
                raise ValueError(f"'tasks' must be a list of strings, but got: {type(tasks)}")

            if not any(task in exclude_task_language for task in tasks):
                kept_indices.append(entry["episode_index"])

    return kept_indices


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False

    # Xianjie: add additioanl episode field to enable train-test split
    # the episode arg will be passed to LeRobotDataset.episodes
    train_episode: list[int] | None = None
    

class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI0_INCONTEXT:
                # TODO: do we need to modify the model transform for incontext?
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    # Xianjie: train-test spli config parameters
    # remove_task_list: a list of tasks that need to be removed from training (for test)
    remove_task_list: tyro.conf.Suppress[Optional[List[str]]] = None
    # episode_json_path: a json that contains the episode index and task name
    episode_json_path: tyro.conf.Suppress[Optional[str]] = None
    task_to_episode: tyro.conf.Suppress[Optional[str]] = None
    episode_to_indexes_file: tyro.conf.Suppress[Optional[str]] = None
    # white list: a josn path that contains all training episodes
    keep_episode_filename_list: tyro.conf.Suppress[Optional[Union[str, Path, List[str]]]] = None


    # TODO: Xianjie: maybe use task index? Or take training task description/index as input?
    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        print("asset_id: ", asset_id)
        print("assets.asset_id: ", self.assets.asset_id)
        print("repo_id: ", repo_id)
        # import ipdb; ipdb.set_trace() # TODO: fix this bug, The trossen remote norm stats is used for training and testing
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
            train_episode=get_kept_episode_indices(self.episode_json_path, self.remove_task_list),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            train_episode=get_kept_episode_indices(self.episode_json_path, self.remove_task_list),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        # Use delta actions (not for gripper)
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            train_episode=get_kept_episode_indices(self.episode_json_path, self.remove_task_list),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoIncontextDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True
    states_cache_path: str = "metadata/libero/episode_states_cache.json"
    actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
    task_to_episode: str='metadata/libero/task_to_episode.json'
    episode_to_indexes_file: str='metadata/libero/episode_to_indexes.json'
    tracks_path: str = "metadata/libero/episode_tracks_combined.json"
    libero_input_refactor: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index", 
                        "index": "index",
                        "task_index": "task_index",
                    }
                )
            ]
        )

        # Xianjie: calculate training episode indexi first
        train_epi = get_kept_episode_indices(self.episode_json_path, self.remove_task_list)


        # Prepare data for policy training
        # inject the indexes of demo prompt, TODO: provide json file_paths here
        data_transforms = _transforms.Group(
            inputs=[_transforms.InjectDemoIndexes(sample_frames=model_config.sample_frames, 
                                                  random_select=model_config.random_select,
                                                  sample_episodes=model_config.sample_episodes,
                                                  task_to_episode=self.task_to_episode,
                                                  episode_to_indexes=self.episode_to_indexes_file,
                                                  train_episode_index_list=train_epi)],
            outputs=[],
        )

        # Convert images to uint8 numpy arrays, add masks
        if self.libero_input_refactor:
            data_transforms = data_transforms.push(
                inputs=[
                    libero_incontext_policy.LiberoIncontextInputs_refactor(
                        action_dim=model_config.action_dim, model_type=model_config.model_type
                    )
                ],
                outputs=[libero_incontext_policy.LiberoIncontextOutputs()],
            )
        else:
            data_transforms = data_transforms.push(
                inputs=[
                    libero_incontext_policy.LiberoIncontextInputs(
                        action_dim=model_config.action_dim, model_type=model_config.model_type
                    )
                ],
                outputs=[libero_incontext_policy.LiberoIncontextOutputs()],
            )
        
        # TODO: fix the bug of libero actions.
        # fix it and re-train on libero
        # Use delta actions (not for gripper)
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        # else:
            # import ipdb; ipdb.set_trace()
        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            train_episode=train_epi,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotLiberoStageIncontextDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True
    states_cache_path: str = "metadata/libero/episode_states_cache.json"
    actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
    task_to_episode: str='metadata/libero/task_to_episode.json'
    episode_to_indexes_file: str='metadata/libero/episode_to_indexes.json'
    tracks_path: str = "metadata/libero/episode_tracks_combined.json"
    libero_input_refactor: bool = False
    # white list: a list of training episodes
    all_episode_stage: Optional[Union[str, Path, List[str]]] = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index", 
                        "index": "index",
                        "task_index": "task_index",
                    }
                )
            ]
        )

        # Xianjie: calculate training episode indexi first
        # --- choose train episodes ---
        if self.keep_episode_filename_list is not None:
            # white list
            train_epi = get_kept_episode_indices(
                self.episode_json_path,
                exclude_task_language=None,
                include_episode_filenames=self.keep_episode_filename_list,  
            )
        else:
            train_epi = get_kept_episode_indices(self.episode_json_path, self.remove_task_list)

        # Prepare data for policy training
        # inject the indexes of demo prompt, TODO: provide json file_paths here
        data_transforms = _transforms.Group(
            inputs=[_transforms.InjectDemoIndexes(sample_frames=model_config.sample_frames, 
                                                  random_select=model_config.random_select,
                                                  sample_episodes=model_config.sample_episodes,
                                                  task_to_episode=self.task_to_episode,
                                                  episode_to_indexes=self.episode_to_indexes_file,
                                                  train_episode_index_list=train_epi,
                                                  all_episode_stage = self.all_episode_stage)],
            outputs=[],
        )

        # Convert images to uint8 numpy arrays, add masks
        if self.libero_input_refactor:
            data_transforms = data_transforms.push(
                inputs=[
                    libero_incontext_policy.LiberoIncontextInputs_refactor(
                        action_dim=model_config.action_dim, model_type=model_config.model_type
                    )
                ],
                outputs=[libero_incontext_policy.LiberoIncontextOutputs()],
            )
        else:
            data_transforms = data_transforms.push(
                inputs=[
                    libero_incontext_policy.LiberoIncontextInputs(
                        action_dim=model_config.action_dim, model_type=model_config.model_type
                    )
                ],
                outputs=[libero_incontext_policy.LiberoIncontextOutputs()],
            )
        
        # TODO: fix the bug of libero actions.
        # fix it and re-train on libero
        # Use delta actions (not for gripper)
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        # else:
            # import ipdb; ipdb.set_trace()
        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            train_episode=train_epi,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotAlohaMobileDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    # adapt_to_pi: bool = True
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # import ipdb; ipdb.set_trace()
        # assert model_config.action_dim == 16
        data_transforms = _transforms.Group(
            inputs=[
                aloha_mobile_policy.AlohaMobileInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)
            ],
            outputs=[aloha_mobile_policy.AlohaMobileOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            # TODO: for base action, is it delta?
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1, -1, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            train_episode=get_kept_episode_indices(self.episode_json_path, self.remove_task_list),
        )
    
@dataclasses.dataclass(frozen=True)
class LeRobotAlohaMobileIncontextDataConfig(DataConfigFactory):
    states_cache_path: str = "metadata/aloha_pen_uncap/episode_states_cache.json"
    actions_cache_path: str = "metadata/aloha_pen_uncap/episode_actions_first_cache.json"
    tracks_path: str = "metadata/aloha_pen_uncap/episode_tracks_combined.json"
    libero_input_refactor: bool = False
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    # TODO: check the issue of default prompt
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    # adapt_to_pi: bool = True
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # import ipdb; ipdb.set_trace()
        # assert model_config.action_dim == 16

        # TODO: generate the indexes for aloha mobile data
        train_epi = get_kept_episode_indices(self.episode_json_path, self.remove_task_list)

        data_transforms = _transforms.Group(
            inputs=[_transforms.InjectDemoIndexes(
                                                  task_to_episode="metadata/aloha_pen_uncap/task_to_episode.json",
                                                  episode_to_indexes="metadata/aloha_pen_uncap/episode_to_indexes.json",
                                                  sample_frames=model_config.sample_frames, 
                                                  random_select=model_config.random_select,
                                                  sample_episodes=model_config.sample_episodes,
                                                  train_episode_index_list=train_epi)],
            outputs=[],
        )

        data_transforms = data_transforms.push(
            inputs=[
                aloha_incontext_policy.AlohaMobileIncontextInputs(
                    action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi
                )
            ],
            outputs=[aloha_mobile_policy.AlohaMobileOutputs(adapt_to_pi=self.adapt_to_pi)],

        )
        # data_transforms = _transforms.Group(
        #     inputs=[
        #         aloha_mobile_policy.AlohaMobileInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)
        #     ],
        #     outputs=[aloha_mobile_policy.AlohaMobileOutputs(adapt_to_pi=self.adapt_to_pi)],
        # )
        if self.use_delta_joint_actions:
            # TODO: for base action, is it delta?
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1, -1, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        # TODO: change it to support multi-task?
        # model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            train_episode=get_kept_episode_indices(self.episode_json_path, self.remove_task_list),
        )

@dataclasses.dataclass(frozen=True)
class LeRobotRLBenchJointDataConfig(DataConfigFactory):
    # deprecated
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        # "action_joint_velocity": "action_joint_velocity",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[rlbench_joint_policy.RLBenchJointInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[rlbench_joint_policy.RLBenchJointOutputs()],
        )
        
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotRLBenchGripperDataConfig(DataConfigFactory):
    # deprecated
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[rlbench_gripper_policy.RLBenchGripperInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[rlbench_gripper_policy.RLBenchGripperOutputs()],
        )
        
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
        
@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaInsertionDataConfig(DataConfigFactory):
    # deprecated
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image_left",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[robocasa_insertion_policy.RobocasaInsertionInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[robocasa_insertion_policy.RobocasaInsertionOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
        
@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaHumanDataConfig(DataConfigFactory):
    # deprecated
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image_left",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[robocasa_human_policy.RobocasaHumanInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[robocasa_human_policy.RobocasaHumanOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
    
@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaHumanThreeImageDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image_left": "image_left",
                        "observation/image_right": "image_right",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # XJ: debug (libero use same json for train-test split instead of a list of strings)
        if self.remove_task_list:
            with open(self.remove_task_list[0], "r") as f:
                remove_test_tasks = json.load(f)["test_tasks"]

            # XJ: calculate training episode indexi first
            train_epi = get_kept_episode_indices(self.episode_json_path, remove_test_tasks)
        else:
            train_epi = get_kept_episode_indices(self.episode_json_path, self.remove_task_list)

        data_transforms = _transforms.Group(
            inputs=[robocasa_human_three_image_policy.RobocasaHumanThreeImageInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[robocasa_human_three_image_policy.RobocasaHumanThreeImageOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            train_episode=train_epi,
        )
        
@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaMgThreeImageDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image_left": "image_left",
                        "observation/image_right": "image_right",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # XJ: debug (libero use same json for train-test split instead of a list of strings)
        if self.remove_task_list:
            with open(self.remove_task_list[0], "r") as f:
                remove_test_tasks = json.load(f)["test_tasks"]

            # XJ: calculate training episode indexi first
            train_epi = get_kept_episode_indices(self.episode_json_path, remove_test_tasks)
        else:
            train_epi = get_kept_episode_indices(self.episode_json_path, self.remove_task_list)

        data_transforms = _transforms.Group(
            inputs=[robocasa_mg_three_image_policy.RobocasaMgThreeImageInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[robocasa_mg_three_image_policy.RobocasaMgThreeImageOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            train_episode=train_epi,
        )
 
@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaHumanThreeImageIncontextDataConfig(DataConfigFactory):
    states_cache_path: str = "metadata/robocasa/episode_states_cache.json"
    actions_cache_path: str = "metadata/robocasa/episode_actions_cache.json"
    task_to_episode: str='metadata/robocasa/task_to_episode.json'
    episode_to_indexes_file: str='metadata/robocasa/episode_to_indexes.json'
    tracks_path: str = "metadata/robocasa/episode_tracks_combined.json"
    # # XJ: deprecated flags
    # use_delta_joint_actions: bool = False
    # robocasa_input_refactor: bool = False
    
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image_left": "image_left",
                        "observation/image_right": "image_right",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index", 
                        "index": "index",
                        "task_index": "task_index",
                    }
                )
            ]
        )
        
        # XJ: debug (libero use same json for train-test split instead of a list of strings)
        if self.remove_task_list:
            with open(self.remove_task_list[0], "r") as f:
                remove_test_tasks = json.load(f)["test_tasks"]

            # XJ: calculate training episode indexi first
            train_epi = get_kept_episode_indices(self.episode_json_path, remove_test_tasks)
        else:
            train_epi = get_kept_episode_indices(self.episode_json_path, self.remove_task_list)
        
        # Prepare data for policy training
        # inject the indexes of demo prompt, TODO: provide json file_paths here
        data_transforms = _transforms.Group(
            inputs=[_transforms.InjectDemoIndexes(sample_frames=model_config.sample_frames, 
                                                  random_select=model_config.random_select,
                                                  sample_episodes=model_config.sample_episodes,
                                                  task_to_episode=self.task_to_episode,
                                                  episode_to_indexes=self.episode_to_indexes_file,
                                                  train_episode_index_list=train_epi)],
            outputs=[],
        )
        
        data_transforms = data_transforms.push(
                inputs=[
                    robocasa_human_three_image_incontext_policy.RobocasaHumanThreeImageIncontextInputs(
                        action_dim=model_config.action_dim, model_type=model_config.model_type
                    )
                ],
                outputs=[robocasa_human_three_image_incontext_policy.RobocasaHumanThreeImageIncontextOutputs()],
        )
        
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            train_episode=train_epi,
        )    
          

@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaMgThreeImageIncontextDataConfig(DataConfigFactory):
    states_cache_path: str = "metadata/robocasa_mg/episode_states_cache.json"
    actions_cache_path: str = "metadata/robocasa_mg/episode_actions_cache.json"
    task_to_episode: str='metadata/robocasa_mg/task_to_episode.json'
    episode_to_indexes_file: str='metadata/robocasa_mg/episode_to_indexes.json'
    tracks_path: str = "metadata/robocasa_mg/episode_tracks_combined.json"
    # # XJ: deprecated flags
    # use_delta_joint_actions: bool = False
    # robocasa_input_refactor: bool = False
    
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image_left": "image_left",
                        "observation/image_right": "image_right",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index", 
                        "index": "index",
                        "task_index": "task_index",
                    }
                )
            ]
        )
        
        # XJ: debug
        if self.remove_task_list:
            with open(self.remove_task_list[0], "r") as f:
                remove_test_tasks = json.load(f)["test_tasks"]

            # XJ: calculate training episode indexi first
            train_epi = get_kept_episode_indices(self.episode_json_path, remove_test_tasks)
        else:
            train_epi = get_kept_episode_indices(self.episode_json_path, self.remove_task_list)
        
        # Prepare data for policy training
        # inject the indexes of demo prompt, TODO: provide json file_paths here
        data_transforms = _transforms.Group(
            inputs=[_transforms.InjectDemoIndexes(sample_frames=model_config.sample_frames, 
                                                  random_select=model_config.random_select,
                                                  sample_episodes=model_config.sample_episodes,
                                                  task_to_episode=self.task_to_episode,
                                                  episode_to_indexes=self.episode_to_indexes_file,
                                                  train_episode_index_list=train_epi)],
            outputs=[],
        )
        
        data_transforms = data_transforms.push(
                inputs=[
                    robocasa_mg_three_image_incontext_policy.RobocasaMgThreeImageIncontextInputs(
                        action_dim=model_config.action_dim, model_type=model_config.model_type
                    )
                ],
                outputs=[robocasa_mg_three_image_incontext_policy.RobocasaMgThreeImageIncontextOutputs()],
        )
        
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            train_episode=train_epi,
        ) 

@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaSingleTaskThreeImageDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image_left": "image_left",
                        "observation/image_right": "image_right",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[robocasa_single_task_policy.RobocasaSingleTaskThreeImageInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[robocasa_single_task_policy.RobocasaSingleTaskThreeImageOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )       


# @dataclasses.dataclass(frozen=True)
# class LeRobotRobocasaHumanThreeImageBaseObsDataConfig(DataConfigFactory):
#     @override
#     def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
#         repack_transform = _transforms.Group(
#             inputs=[
#                 _transforms.RepackTransform(
#                     {
#                         "observation/image_left": "image_left",
#                         "observation/image_right": "image_right",
#                         "observation/wrist_image": "wrist_image",
#                         "observation/state": "state",
#                         "actions": "actions",
#                         "prompt": "prompt",
#                     }
#                 )
#             ]
#         )

#         data_transforms = _transforms.Group(
#             inputs=[robocasa_human_three_image_base_obs_policy.RobocasaHumanThreeImageBaseObsInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
#             outputs=[robocasa_human_three_image_base_obs_policy.RobocasaHumanThreeImageBaseObsOutputs()],
#         )
#         model_transforms = ModelTransformFactory()(model_config)
#         return dataclasses.replace(
#             self.create_base_config(assets_dirs),
#             repack_transforms=repack_transform,
#             data_transforms=data_transforms,
#             model_transforms=model_transforms,
#         )     
 
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    # XJ: for cunstomizer image encoder
    vision_weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints" #"/ibex/tmp/c2090/openpi_explore_storage/checkpoints" 

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 5_000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # In`fe`rence Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
    ),
    TrainConfig(
        name="pi0_aloha_mobile",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="mobile_trossen",
            ),
        ),
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
    ),
    TrainConfig(
        name="pi0_aloha_pen_uncap_b5_low_mem_finetune",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=10_000,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_aloha_pen_uncap_b5_low_mem_finetune_trossen_norm",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_fast_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=10_000,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    TrainConfig(
        name="pi0_aloha_pen_uncap_b5_low_mem_finetune_trossen_normv2",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    TrainConfig(
        name="pi0_aloha_pen_uncap_b5_trossen_norm",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_workers=16,
        num_train_steps=20_000,
    ),

    # TODO: check the prompt
    TrainConfig(
        name="pi0_aloha_pen_uncap_incontextv12_low_mem_finetune_sample2_actionssample32_random_select",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            # default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_fast_aloha_pen_uncap_b5",
        model=pi0_fast.Pi0FASTConfig(action_dim=16, action_horizon=50, max_token_len=576),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=10_000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_fast_aloha_pen_uncap_b5_trossen_norm",
        model=pi0_fast.Pi0FASTConfig(action_horizon=50, max_token_len=576),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_fast_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=10_000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_fast_aloha_pen_uncap_low_mem_finetune_trossen_norm",
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora", action_horizon=50, max_token_len=576),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_fast_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=10_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=16, action_horizon=50, max_token_len=448, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_aloha_pen_uncap_low_mem_finetune",
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora", action_horizon=50, max_token_len=576),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=10_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=16, action_horizon=50, max_token_len=448, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_aloha_pen_uncap_low_mem_finetune_bs30",
        batch_size=30,
        wandb_enabled=False,
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora", max_token_len=300),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_fast_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=16, action_horizon=40, max_token_len=400, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    TrainConfig(
        name="pi0_libero_incontext_low_mem_finetune",
        model=pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        batch_size=36,
        # wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_libero_incontext_low_mem_finetune_sample2",
        model=pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=8, random_select=False,
            # paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=False, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=8, random_select=False,
            # paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_libero_incontext_low_mem_finetune_sample2_random_select",
        model=pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=8,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=8,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=36,
        # wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_libero_incontext_low_mem_finetune_sample2_actionssample32",
        model=pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),
    
    TrainConfig(
        name="pi0_libero_incontext_low_mem_finetune_sample2_actionssample32_without_delta",
        model=pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json"
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64_train_split",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64_train_split_inference",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample128",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=128, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=128, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64_long",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=400_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_without_delta",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json"
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv4_low_mem_finetune_sample2_actionssample32_random_select_without_delta",
        model=pi0_incontextv4.Pi0IncontextConfigv4(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json"
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv4.Pi0IncontextConfigv4(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv4.Pi0IncontextConfigv4(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv4.Pi0IncontextConfigv4(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv4.Pi0IncontextConfigv4(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv4.Pi0IncontextConfigv4(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv6.Pi0IncontextConfigv6(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv6.Pi0IncontextConfigv6(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv6.Pi0IncontextConfigv6(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv6.Pi0IncontextConfigv6(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        # batch_size=36,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_without_text_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, use_text_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, use_text_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        # batch_size=36,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_without_text_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m",
             action_expert_variant="gemma_300m_lora", sample_frames=2, 
             sample_actions=32, random_select=True, use_text_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, use_text_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7a_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, block_attention=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        # batch_size=36,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7a_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv11_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv11.Pi0IncontextConfigv11(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, block_attention=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        # num_train_steps=22_500,
        freeze_filter=pi0_incontextv11.Pi0IncontextConfigv11(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        # batch_size=36,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv11_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv11.Pi0IncontextConfigv11(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        # num_train_steps=22_500,
        freeze_filter=pi0_incontextv11.Pi0IncontextConfigv11(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    # in 7b, we full finetune the llm    
    TrainConfig(
        name="pi0_libero_incontextv7b_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, block_attention=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=2, # 
        num_workers=8,
        # batch_size=36,
        batch_size=32,# 
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7b_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m",
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_10_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=32, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=32, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_10_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=32, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=32, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),


    TrainConfig(
        name="pi0_libero_incontextv10_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv10_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv10_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=4, 
            sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", 
            sample_frames=4, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv10_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=4, 
            sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=4, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora",
              sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=64, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora",
              sample_frames=2, sample_actions=64, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, sample_episodes=4,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, sample_episodes=4,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_8_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_8_low_mem_finetune_sample2_actionssample32_random_select_without_delta",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_8_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),


    TrainConfig(
        name="pi0_libero_incontextv7_7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_debug",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora",
              sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=True,
              use_action_state_prompts=True, sample_episodes=2,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_5_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_5_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv7_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv8_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/episode_tracks_combined_all.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32,
              random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv8_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32,
              random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/episode_tracks_combined_all.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    # in the 8_1 version, we used dense point tracks
    TrainConfig(
        name="pi0_libero_incontextv8_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, point_track_dim=444,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/episode_tracks_combined_grid32_all.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32,
              random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv8_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32,
              random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, point_track_dim=444,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/episode_tracks_combined_grid32_all.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    # in the 8_1_1 version, we used all 
    TrainConfig(
        name="pi0_libero_incontextv8_1_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True, point_track_dim=444,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/episode_tracks_combined_grid32_all.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32,
              random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv8_1_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32,
              random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True, point_track_dim=444,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/episode_tracks_combined_grid32_all.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    # in the 8_2 version, we used video tokens
    TrainConfig(
        name="pi0_libero_incontextv8_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, point_track_dim=784,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/video_tokens_merged.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20,
              random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, point_track_dim=784,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv8_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20,
              random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, point_track_dim=784,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/video_tokens_merged.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, point_track_dim=784,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),


    # in the 8_2 version, we used all the prompts
    TrainConfig(
        name="pi0_libero_incontextv8_2_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20, 
            random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True, point_track_dim=784,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/video_tokens_merged.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20,
              random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True, point_track_dim=784,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv8_2_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20,
              random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True, point_track_dim=784,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/video_tokens_merged.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20, 
            random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True, point_track_dim=784,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv6_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv6.Pi0IncontextConfigv6(
             sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv6_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv6.Pi0IncontextConfigv6(
             sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),


    TrainConfig(
        name="pi0_libero_incontextv3_low_mem_finetune_sample2_actionssample32_random_select_without_delta",
        model=pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json"
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv3_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv3_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv3_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv3_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),


    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_states_cache_debug",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache_debug.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache_debug.json"
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample8",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=8, random_select=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=8, random_select=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        wandb_enabled=False,
    ),

    # Xianjie: pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32 with train_test_split
    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            # Xianjie: newly added para for train-test split
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_splitv2",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            # Xianjie: newly added para for train-test split
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_embedding_train_split",
        model=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_embedding_train_split_inference",
        model=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v3",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK_V3,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v4",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK_V4,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=16, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=16, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=100,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=16, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=16, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_8_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=64,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_8_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_10_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=8,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=8,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=64,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_10_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=8,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=8,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_11_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=4, sample_actions=32, random_select=True, sample_episodes=4,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=4, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=64,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_11_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=4, sample_actions=32, random_select=True, sample_episodes=4,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=4, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),


    # TODO: libero_refactor is not tested. train and test it
    TrainConfig(
        name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
            libero_input_refactor=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
            libero_input_refactor=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    # TODO: 12_7 is not finished yet, finish it
    TrainConfig(
        name="pi0_libero_incontextv12_7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, causal_attention=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
            # libero_input_refactor=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, causal_attention=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    # ablation of causal attention
    TrainConfig(
        name="pi0_libero_incontextv12_7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, causal_attention=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
            # libero_input_refactor=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, causal_attention=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),


    TrainConfig(
        name="pi0_libero_incontextv12_6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, avg_current_img=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, avg_current_img=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_5_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False, use_image_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False, use_image_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_5_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False, use_image_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False, use_image_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_3_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_3_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),


    TrainConfig(
        name="pi0_libero_incontextv12_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_text_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_text_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_text_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_text_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),


    TrainConfig(
        name="pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=8, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=8, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),


    TrainConfig(
        name="pi0_libero_incontextv9_low_mem_finetune_sample2_actionssample32_random_select_without_delta",
        model=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_v9_low_mem_finetune_sample2_actionssample32_random_select_without_delta",
        model=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False, use_action_state_prompts=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False, use_action_state_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),


    TrainConfig(
        name="pi0_libero_incontextv9_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=40_000,
        freeze_filter=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=16,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv9_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        # weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=40_000,
        freeze_filter=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv9_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=2,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_incontextv9_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_debug",
        model=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        # weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=1,
        # num_workers=2,
        batch_size=32,
        # wandb_enabled=False,
    ),


    TrainConfig(
        name="pi0_libero_incontextv9_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        # weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_libero_low_mem_finetune_split_train",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=36,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune_split_train_v3",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            remove_task_list=DEFAULT_LIBERO_TEST_TASK_V3,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=36,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune_split_train_v4",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            remove_task_list=DEFAULT_LIBERO_TEST_TASK_V4,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=36,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune_split_inference",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=4,
        batch_size=32,
    ),
    # Xianjie:
    TrainConfig(
        # XIANJIE: no lora; no split; with delta
        name="pi0_libero_incontextv2_sample2_actionssample64",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),
    # Xianjie:
    TrainConfig(
        # XIANJIE: no lora; with split; with delta
        name="pi0_libero_incontextv2_sample2_actionssample64_train_split",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),   
    # Xianjie:
    TrainConfig(
        # XIANJIE: no lora; with split; with delta
        name="pi0_libero_incontextv2_sample2_actionssample64_test_split",
        model=pi0_incontextv2.Pi0IncontextConfigv2(
            sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ), 
    #
    # Pi0 Light
    #
    # Xianjie
    # TrainConfig(
    #     # no delta with split
    #     name="pi0light_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
    #     model=deprecated_pi0light_incontextv12.Pi0LightIncontextConfigv12(
    #         prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
    #         sample_frames=2, sample_actions=32, random_select=True, siglip_variant="Ti/16"
    #     ),  
    #     data=LeRobotLiberoIncontextDataConfig(
    #         repo_id="physical-intelligence/libero",
    #         base_config=DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #         use_delta_joint_actions=False,
    #         states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
    #         actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
    #         remove_task_list=DEFAULT_LIBERO_TEST_TASK,
    #         episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
    #         npz_path="gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz", # Ti/16
    #     ),
    #     num_train_steps=20_000,
    #     freeze_filter=deprecated_pi0light_incontextv12.Pi0LightIncontextConfigv12(
    #         prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
    #         sample_frames=2, sample_actions=32, random_select=True, 
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    #     num_workers=16,
    #     # num_workers=1,
    #     batch_size=32,
    #     # wandb_enabled=False,
    # ),
    # TrainConfig(
    #     # no delta with split
    #     name="pi0light_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
    #     model=deprecated_pi0light_incontextv12.Pi0LightIncontextConfigv12(
    #         prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
    #         sample_frames=2, sample_actions=32, random_select=True, siglip_variant="Ti/16"
    #     ),  
    #     data=LeRobotLiberoIncontextDataConfig(
    #         repo_id="physical-intelligence/libero",
    #         base_config=DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #         use_delta_joint_actions=False,
    #         states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
    #         actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
    #         # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
    #         # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
    #         npz_path="gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz", # Ti/16
    #     ),
    #     num_train_steps=20_000,
    #     freeze_filter=deprecated_pi0light_incontextv12.Pi0LightIncontextConfigv12(
    #         prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
    #         sample_frames=2, sample_actions=32, random_select=True, 
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    #     num_workers=16,
    #     # num_workers=1,
    #     batch_size=32,
    #     # wandb_enabled=False,
    # ),
    
    #
    # Fine-tuning Libero configs.
    #
    TrainConfig(
        name="pi0_libero",
        model=pi0.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        num_train_steps=10_000,
    ),

    TrainConfig(
        name="pi0_libero_without_delta",
        model=pi0.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # num_train_steps=10_000,
    ),


    TrainConfig(
        name="pi0_libero_zero",
        model=pi0.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        num_train_steps=10_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        num_train_steps=10_000,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        wandb_enabled=False,
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    
    # #
    # # XJ: Fine-tuning RLBench configs
    # #
    # TrainConfig(
    #     name="pi0_rlbench_gripper_low_mem_finetune_train",
    #     model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    #     data=LeRobotRLBenchGripperDataConfig(
    #         repo_id="daixianjie/rlbench_lerobot_train",
    #         base_config=DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     num_train_steps=20_000,
    #     freeze_filter=pi0.Pi0Config(
    #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    #     ).get_freeze_filter(),
    #     # Turn off EMA for LoRA finetuning.
    #     ema_decay=None,
    #     num_workers=4,
    #     batch_size=36,
    # ),
    
    # TrainConfig(
    #     name="pi0_rlbench_joint_low_mem_finetune_train",
    #     # Here is an example of loading a pi0 model for LoRA fine-tuning.
    #     model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    #     data=LeRobotRLBenchJointDataConfig(
    #         repo_id="daixianjie/rlbench_joint_vel_action_lerobot_train",
    #         base_config=DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     num_train_steps=20_000,
    #     freeze_filter=pi0.Pi0Config(
    #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    #     ).get_freeze_filter(),
    #     # Turn off EMA for LoRA finetuning.
    #     ema_decay=None,
    #     num_workers=4,
    #     batch_size=36,
    # ),
    
    #
    # XJ: Fine-tuning robocasa configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    # TrainConfig(
    #     # no delta with split
    #     name="pi0_robocasa_insertion_low_mem_finetune_train",
    #     # Here is an example of loading a pi0 model for LoRA fine-tuning.
    #     model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    #     data=LeRobotRobocasaInsertionDataConfig(
    #         repo_id="daixianjie/robocasa_insertion_lerobot",
    #         base_config=DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     num_train_steps=140_000,
    #     # The freeze filter defines which parameters should be frozen during training.
    #     # We have a convenience function in the model config that returns the default freeze filter
    #     # for the given model config for LoRA finetuning. Just make sure it matches the model config
    #     # you chose above.
    #     freeze_filter=pi0.Pi0Config(
    #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    #     ).get_freeze_filter(),
    #     # Turn off EMA for LoRA finetuning.
    #     ema_decay=None,
    #     num_workers=16,
    #     batch_size=32,
    # ),
    TrainConfig(
        # no delta with split
        name="pi0_robocasa_human_low_mem_finetune_train",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotRobocasaHumanDataConfig(
            repo_id="daixianjie/robocasa_human_lerobot",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    
    TrainConfig(
        # no delta with split
        name="pi0_robocasa_human_three_image_low_mem_finetune_train",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotRobocasaHumanThreeImageDataConfig(
            repo_id="daixianjie/robocasa_human_lerobot",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=200_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    
    # TrainConfig(
    #     # no delta with split
    #     name="pi0_robocasa_human_three_image_base_obs_low_mem_finetune_train",
    #     # Here is an example of loading a pi0 model for LoRA fine-tuning.
    #     model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    #     data=LeRobotRobocasaHumanThreeImageBaseObsDataConfig(
    #         repo_id="daixianjie/robocasa_human_lerobot",
    #         base_config=DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     num_train_steps=20_000,
    #     # The freeze filter defines which parameters should be frozen during training.
    #     # We have a convenience function in the model config that returns the default freeze filter
    #     # for the given model config for LoRA finetuning. Just make sure it matches the model config
    #     # you chose above.
    #     freeze_filter=pi0.Pi0Config(
    #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    #     ).get_freeze_filter(),
    #     # Turn off EMA for LoRA finetuning.
    #     ema_decay=None,
    #     num_workers=16,
    #     batch_size=32,
    # ),  
    
    TrainConfig(
        # no delta with split
        name="pi0light_robocasa_human_three_image_low_mem_finetune_train",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0Light.Pi0LightConfig(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", siglip_variant="Ti/16"),  # So400m/14, Ti/16, S/32
        data=LeRobotRobocasaHumanThreeImageDataConfig(
            repo_id="daixianjie/robocasa_human_lerobot",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            # npz_path="gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz", # S/32
            npz_path="gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz", # Ti/16
        ),
        num_train_steps=100_000,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 100_000,
            decay_lr= 2.5e-6),
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0Light.Pi0LightConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ), 
    TrainConfig(
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini.py pi0mini_robocasa_human_three_image_low_mem_finetune_train --exp-name=pi0mini_robocasa_human_three_image_low_mem_finetune_train_1000k --overwrite
        name="pi0mini_robocasa_human_three_image_low_mem_finetune_train",
        model=pi0Light.Pi0LightConfig(paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaHumanThreeImageDataConfig(
            repo_id="daixianjie/robocasa_human_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 1000_000,
            decay_lr= 2.5e-6),
        num_train_steps=1000_000,
        freeze_filter=pi0Light.Pi0LightConfig(
            paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=36,
    ),    
    TrainConfig(
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext.py pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train --exp-name=pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train --project-name=pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train --overwrite
        name="pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaHumanThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_human_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa/episode_to_indexes.json',
            states_cache_path="metadata/robocasa/episode_states_cache.json",
            actions_cache_path="metadata/robocasa/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 500_000,
            decay_lr= 2.5e-6),
        num_train_steps=500_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=36,
    ), 
    TrainConfig(
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext.py pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train --exp-name=pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train --project-name=pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train --overwrite
        name="pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaHumanThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_human_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa/episode_to_indexes.json',
            states_cache_path="metadata/robocasa/episode_states_cache.json",
            actions_cache_path="metadata/robocasa/episode_actions_cache.json",
            # remove_task_list=DEFAULT_ROBOCASA_TEST_TASK,
            # episode_json_path=DEFAULT_ROBOCASA_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 500_000,
            decay_lr= 2.5e-6),
        num_train_steps=500_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=36,
    ), 
    
    
    # TrainConfig(
    #     # no delta with split
    #     name="pi0_robocasa_turnonmicrowave_three_image_low_mem_finetune_train",
    #     # Here is an example of loading a pi0 model for LoRA fine-tuning.
    #     model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    #     data=LeRobotRobocasaSingleTaskThreeImageDataConfig(
    #         repo_id="daixianjie/robocasa_turnonmicrowave_lerobot",
    #         base_config=DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     num_train_steps=20_000,
    #     # The freeze filter defines which parameters should be frozen during training.
    #     # We have a convenience function in the model config that returns the default freeze filter
    #     # for the given model config for LoRA finetuning. Just make sure it matches the model config
    #     # you chose above.
    #     freeze_filter=pi0.Pi0Config(
    #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    #     ).get_freeze_filter(),
    #     # Turn off EMA for LoRA finetuning.
    #     ema_decay=None,
    #     num_workers=16,
    #     batch_size=32,
    # ),
    # TrainConfig(
    #     # no delta with split
    #     name="pi0mini_robocasa_turnonmicrowave_three_image_low_mem_finetune_train",
    #     model=pi0Light.Pi0LightConfig(paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
    #     data=LeRobotRobocasaSingleTaskThreeImageDataConfig(
    #         repo_id="daixianjie/robocasa_turnonmicrowave_lerobot",
    #         base_config=DataConfig(
    #             local_files_only=False,  
    #             prompt_from_task=True,
    #         ),
    #     ),
    #     vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
    #         npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
    #     ),
    #     weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     lr_schedule = _optimizer.CosineDecaySchedule(
    #         warmup_steps = 1_000,
    #         peak_lr= 2.5e-5,
    #         decay_steps= 500_000,
    #         decay_lr= 2.5e-6),
    #     num_train_steps=500_000,
    #     save_interval = 50_000,
    #     freeze_filter=pi0Light.Pi0LightConfig(
    #         paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    #     num_workers=16,
    #     batch_size=36,
    # ),
    
    # TrainConfig(
    #     # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini.py pi0mini_robocasa_human_three_image_base_eef_low_mem_finetune_train --exp-name=pi0mini_robocasa_human_three_image_base_eef_low_mem_finetune_train --project-name=pi0mini_robocasa_human_three_image_base_eef_low_mem_finetune_train --overwrite
    #     # this exp use customized paligemma and different pre-trained img encoder 
    #     # which has train_test_split; without delta; language prompt; lora; 20k
    #     name="pi0mini_robocasa_human_three_image_base_eef_low_mem_finetune_train",
    #     model=pi0Light.Pi0LightConfig(paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
    #     data=LeRobotRobocasaSingleTaskThreeImageDataConfig(
    #         repo_id="daixianjie/robocasa_human_lerobot",
    #         base_config=DataConfig(
    #             local_files_only=False,  
    #             prompt_from_task=True,
    #         ),
    #     ),
    #     vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
    #         npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
    #     ),
    #     weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     lr_schedule = _optimizer.CosineDecaySchedule(
    #         warmup_steps = 1_000,
    #         peak_lr= 2.5e-5,
    #         decay_steps= 500_000,
    #         decay_lr= 2.5e-6),
    #     num_train_steps=500_000,
    #     freeze_filter=pi0Light.Pi0LightConfig(
    #         paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    #     num_workers=8,
    #     batch_size=36,
    # ),  
    TrainConfig(
        name="pi0tiny_robocasa_mg_three_image_train_split_large_lr",
        model=pi0Light.Pi0LightConfig(
            vocab_size=50_000, 
            paligemma_variant="gemma_A", action_expert_variant="gemma_B", 
            freeze_llm_embedder=False, freeze_img_encoder=False, 
            siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        # freeze_filter=pi0Light.Pi0LightConfig(
        #     vocab_size=50_000, paligemma_variant="gemma_A", action_expert_variant="gemma_B", freeze_llm_embedder=False, freeze_img_encoder = False, siglip_variant="S/16",
        # ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0tiny_robocasa_mg_three_image_train_split",
        model=pi0Light.Pi0LightConfig(
            vocab_size=50_000, 
            paligemma_variant="gemma_A", action_expert_variant="gemma_B", 
            freeze_llm_embedder=False, freeze_img_encoder=False, 
            siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 1_000_000,
            decay_lr= 2.5e-6),
        num_train_steps=1_000_000,
        # freeze_filter=pi0Light.Pi0LightConfig(
        #     vocab_size=50_000, paligemma_variant="gemma_A", action_expert_variant="gemma_B", freeze_llm_embedder=False, freeze_img_encoder = False, siglip_variant="S/16",
        # ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0tiny_robocasa_mg_three_image_inference",
        model=pi0Light.Pi0LightConfig(
            vocab_size=50_000, 
            paligemma_variant="gemma_A", action_expert_variant="gemma_B", 
            freeze_llm_embedder=False, freeze_img_encoder=False, 
            siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 500_000,
            decay_lr= 2.5e-6),
        num_train_steps=500_000,
        # freeze_filter=pi0Light.Pi0LightConfig(
        #     vocab_size=50_000, paligemma_variant="gemma_A", action_expert_variant="gemma_B", freeze_llm_embedder=False, freeze_img_encoder = False, siglip_variant="S/16",
        # ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0tiny_incontext_robocasa_mg_three_image_train_split_large_lr",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        # freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
        #     vocab_size=50_000, 
        #     prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
        #     sample_frames=2, sample_actions=32, random_select=True,  
        #     freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16").get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    
    TrainConfig(
        name="pi0tiny_incontext_robocasa_mg_three_image_train_split",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 500_000,
            decay_lr= 2.5e-6),
        num_train_steps=500_000,
        # freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
        #     vocab_size=50_000, 
        #     prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
        #     sample_frames=2, sample_actions=32, random_select=True,  
        #     freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16").get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    TrainConfig(
        name="pi0tiny_incontext_robocasa_mg_three_image_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 500_000,
            decay_lr= 2.5e-6),
        num_train_steps=500_000,
        # freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
        #     vocab_size=50_000, 
        #     prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
        #     sample_frames=2, sample_actions=32, random_select=True,  
        #     freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16").get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    
    TrainConfig(
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini.py pi0mini_robocasa_mg_three_image_low_mem_finetune_train --exp-name=pi0mini_robocasa_mg_three_image_low_mem_finetune_train --overwrite
        name="pi0mini_robocasa_mg_three_image_low_mem_finetune_train",
        model=pi0Light.Pi0LightConfig(paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 2_000_000,
            decay_lr= 2.5e-6),
        num_train_steps=2_000_000,
        freeze_filter=pi0Light.Pi0LightConfig(
            paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),  
    TrainConfig(
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini.py pi0mini_robocasa_mg_three_image_low_mem_finetune_train_split --exp-name=pi0mini_robocasa_mg_three_image_low_mem_finetune_train_split --overwrite
        name="pi0mini_robocasa_mg_three_image_low_mem_finetune_train_split",
        model=pi0Light.Pi0LightConfig(paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 2_000_000,
            decay_lr= 2.5e-6),
        num_train_steps=2_000_000,
        freeze_filter=pi0Light.Pi0LightConfig(
            paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
        
    TrainConfig(
        # no delta with split
        name="pi0_robocasa_mg_three_image_low_mem_finetune_train",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotRobocasaMgThreeImageDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 500_000,
            decay_lr= 2.5e-6),
        num_train_steps=500_000,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    
    TrainConfig(
        # no delta with split
        name="pi0_robocasa_mg_three_image_low_mem_finetune_train_split",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotRobocasaMgThreeImageDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 500_000,
            decay_lr= 2.5e-6),
        num_train_steps=500_000,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    
    TrainConfig(
        # no delta with split
        name="pi0_incontext_robocasa_mg_three_image_low_mem_finetune_train",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 500_000,
            decay_lr= 2.5e-6),
        num_train_steps=500_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    
    TrainConfig(
        # no delta with split
        name="pi0_incontext_robocasa_mg_three_image_low_mem_finetune_inference",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 500_000,
            decay_lr= 2.5e-6),
        num_train_steps=500_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),

    TrainConfig(
        name="pi0mini_incontext_robocasa_mg_three_image_low_mem_finetune",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 500_000,
            decay_lr= 2.5e-6),
        num_train_steps=500_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 

    TrainConfig(
        name="pi0mini_incontext_robocasa_mg_three_image_low_mem_finetune_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 500_000,
            decay_lr= 2.5e-6),
        num_train_steps=500_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    # This config is used to demonstrate how to train on a simple simulated environment.
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        # no delta with split
        name="debug_robocasa_human_three_image_low_mem_finetune_train",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        # model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),  
        model=pi0Light.Pi0LightConfig(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", siglip_variant="Ti/16"),  # So400m/14, Ti/16, S/32
        data=LeRobotRobocasaHumanThreeImageDataConfig(
            repo_id="daixianjie/robocasa_human_lerobot",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            # npz_path="gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz", # S/32
            npz_path="gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz", # Ti/16
        ),
        num_train_steps=5_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0Light.Pi0LightConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        num_workers=4,
        batch_size=4,
    ),
    TrainConfig(
        # no delta with split
        name="debug_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=deprecated_pi0light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, siglip_variant="Ti/16"
        ),  
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz", # Ti/16
        ),
        num_train_steps=5_000,
        freeze_filter=deprecated_pi0light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    

    TrainConfig(
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini.py debug_pi0mini_libero_low_mem_finetune_split_train --exp-name=debug_pi0mini_libero_low_mem_finetune_split_train_ibex --overwrite
        # this exp use customized paligemma and different pre-trained img encoder 
        # which has train_test_split; without delta; language prompt; lora; 20k
        name="debug_pi0mini_libero_low_mem_finetune_split_train",
        model=pi0Light.Pi0LightConfig(paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0Light.Pi0LightConfig(
            paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=36,
    ),
    
    # XJ_REBUTAL
    ## XJ: debug pi0mini incontext code on libero
    TrainConfig(
        name="pi0mini_incontext_libero_low_mem_finetune_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    
    TrainConfig(
        name="pi0mini_incontext_libero_low_mem_finetune_train",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    
    ## XJ: try incontext_v12_1 with more prompt images 
    TrainConfig(
        name="pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_train_large_lr",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=8, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=8, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    TrainConfig(
        name="pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_train",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=8, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 1_000_000,
            decay_lr= 2.5e-6),
        num_train_steps=1_000_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=8, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 

    TrainConfig(
        name="pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=8, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            # remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            # episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 1_000_000,
            decay_lr= 2.5e-6),
        num_train_steps=1_000_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=8, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 

    #
    # XJ libero_with_depth: just to pull newly generated libero dataset with depth image but with more episodes
    #
    # TrainConfig(
    #     name="pi0_depth_libero_low_mem_finetune",
    #     model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    #     data=LeRobotLiberoDataConfig(
    #         repo_id="daixianjie/libero_with_depth",
    #         base_config=DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     # num_train_steps=30_000,
    #     num_train_steps=40_000,
    #     freeze_filter=pi0.Pi0Config(
    #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    #     num_workers=4,
    #     batch_size=36,
    # ),
    TrainConfig(
        name="debug_pi0tiny_incontext_robocasa_mg_three_image_train_split",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=False, use_action_state_prompts=False),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        # freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
        #     vocab_size=50_000, 
        #     prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
        #     sample_frames=2, sample_actions=32, random_select=True,  
        #     freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16").get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    TrainConfig(
        name="debug_pi0tiny_incontext_robocasa_mg_three_image_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=False, use_action_state_prompts=False),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        # freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
        #     vocab_size=50_000, 
        #     prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
        #     sample_frames=2, sample_actions=32, random_select=True,  
        #     freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16").get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    TrainConfig(
        name="debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_train_split",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    TrainConfig(
        name="debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    TrainConfig(
        name="debug_prompt_no_random_select_pi0tiny_incontext_robocasa_mg_three_image_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=False,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    TrainConfig(
        name="debug_proprio_prompt_pi0tiny_incontext_robocasa_mg_three_image_train_split",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=False, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    TrainConfig(
        name="debug_proprio_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=False, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    
    # XJ: final
    TrainConfig(
        name="final_boost_img_prompt_pi0tiny_incontext_robocasa_mg_three_image_large_lr_1M_train_split",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=8, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 5e-4,
            decay_steps= 500_000,
            decay_lr= 5e-5),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    TrainConfig(
        name="final_boost_img_prompt_pi0tiny_incontext_robocasa_mg_three_image_large_lr_1M_all",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=8, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 5e-4,
            decay_steps= 500_000,
            decay_lr= 5e-5),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    # XJ: final_2
    # debug much larger lr
    TrainConfig(
        name="final_pi0tiny_incontext_robocasa_mg_three_image_largest_lr_train_split",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-3,
            decay_steps= 500_000,
            decay_lr= 2.5e-4),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    TrainConfig(
        name="final_pi0tiny_incontext_robocasa_mg_three_image_largest_lr_train_all",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-3,
            decay_steps= 500_000,
            decay_lr= 2.5e-4),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    # debug large lr + 1M
    TrainConfig(
        name="pi0tiny_incontext_robocasa_mg_three_image_large_lr_train_split_1M",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 1_000_000,
            decay_lr= 2.5e-5),
        num_train_steps=1_000_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),

    # debug low action horizon
    TrainConfig(
        name="debug_low_action_horizon",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True,
            action_horizon = 20),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    TrainConfig(
        name="debug_low_action_horizon_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True,
            action_horizon = 20),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    # debug img encoder: freeze img encoder to check if the pretrained weight is loaded or not
    TrainConfig(
        name="debug_img_encoder",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=True, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    # debug training without open double doors
    TrainConfig(
        name="debug_train_without_open_double_door",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK_WITHOUT_OPENDOUBLEDOOR,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-4,
            decay_steps= 500_000,
            decay_lr= 2.5e-5),
        num_train_steps=500_000,
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ), 
    
    ## rebuttal exp
    # InSpire Training Setting
    # pi0 incontext v12 none lora
    TrainConfig(
        name="pi0_libero90_incontextv12_finetune",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="daixianjie/libero_90_lerobot",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode='metadata/libero_90/task_to_episode.json',
            episode_to_indexes_file='metadata/libero_90/episode_to_indexes.json',
            states_cache_path="metadata/libero_90/episode_states_cache.json",
            actions_cache_path="metadata/libero_90/episode_actions_cache.json",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 30_000,
            decay_lr= 2.5e-6),
        num_train_steps=30_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=128,
        # wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_libero90_incontextv12_finetune_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode='metadata/libero/task_to_episode.json',
            episode_to_indexes_file='metadata/libero/episode_to_indexes.json',
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 30_000,
            decay_lr= 2.5e-6),
        num_train_steps=30_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=128,
        # wandb_enabled=False,
    ),
    # pi0 incontextv12 lora
    TrainConfig(
        name="pi0_libero90_incontextv12_low_mem_finetune",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="daixianjie/libero_90_lerobot",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode='metadata/libero_90/task_to_episode.json',
            episode_to_indexes_file='metadata/libero_90/episode_to_indexes.json',
            states_cache_path="metadata/libero_90/episode_states_cache.json",
            actions_cache_path="metadata/libero_90/episode_actions_cache.json",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 30_000,
            decay_lr= 2.5e-6),
        num_train_steps=30_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_libero90_incontextv12_low_mem_finetune_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode='metadata/libero/task_to_episode.json',
            episode_to_indexes_file='metadata/libero/episode_to_indexes.json',
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 30_000,
            decay_lr= 2.5e-6),
        num_train_steps=30_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),
    # pi0 
    TrainConfig(
        name="pi0_libero90_low_mem_finetune",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="daixianjie/libero_90_lerobot",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 30_000,
            decay_lr= 2.5e-6),
        num_train_steps=30_000,        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0_libero90_low_mem_finetune_inference",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 30_000,
            decay_lr= 2.5e-6),
        num_train_steps=30_000,        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ),
    
    # more incontext images
    TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_train_split_v1",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=8, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK_V2,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=8, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_inference",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=8, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps=20_000,        
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=8, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),
    # video tokens
    TrainConfig(
        name="pi0_libero_incontextv12_video_prompt_low_mem_finetune_train_split",
        model=pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=784,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/video_tokens_merged.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK_V2,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=784,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_libero_incontextv12_video_prompt_low_mem_finetune_inference",
        model=pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=784,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/video_tokens_merged.json",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=784,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    # point track tokens
    TrainConfig(
        name="pi0_libero_incontextv12_point_track_low_mem_finetune_train_split",
        model=pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=256,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/episode_tracks_combined_all.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK_V2,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=256,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_libero_incontextv12_point_track_low_mem_finetune_inference",
        model=pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=256,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            tracks_path="metadata/libero/episode_tracks_combined_all.json",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=256,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    # totally initialized ICFM (ViT-B-16) deprecated: shouldn't be using lora since it's random
    TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_random_init_train_split",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
            ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK_V2,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps=20_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_random_init_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
            ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps=20_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    # total init without lora:
    TrainConfig(
        name="pi0_libero_incontextv12_random_init_train_split",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
            ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=DEFAULT_LIBERO_TEST_TASK_V2,
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,

        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps=20_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0_libero_incontextv12_random_init_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
            ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps=20_000,
        freeze_filter=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    
    # stage-wise incontext 
    TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_clean_stage_wise_prompt_train_all",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoStageIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            keep_episode_filename_list="/home/dingj0b/dingjian/openpi_explore/project/openpi/examples/libero/all_stage_clean.json",
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
            all_episode_stage = "/home/dingj0b/dingjian/openpi_explore/project/openpi/examples/libero/all_segments_summary.json",

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_noisy_stage_wise_prompt_train_all",
        model=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoStageIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            keep_episode_filename_list="/home/dingj0b/dingjian/openpi_explore/project/openpi/examples/libero/all_stage_noisy.json",
            episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
            all_episode_stage = "/home/dingj0b/dingjian/openpi_explore/project/openpi/examples/libero/all_segments_summary.json",

        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),
    
    # scale-up: large lr + 
    TrainConfig(
        name="pi0tiny_incontext_robocasa_mg_three_image_scaleup_train_split",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            remove_task_list=DEFAULT_ROBOCASA_MG_TEST_TASK,
            episode_json_path=DEFAULT_ROBOCASA_MG_EPISODE_JSON,
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 10_000,
            peak_lr= 5e-4,
            decay_steps= 1_500_000,
            decay_lr= 5e-5),
        num_train_steps=1_500_000,
        ema_decay=0.999,
        # num_worker per GPU
        num_workers=8,
        # batch size in total (bs_per_gpu = batch_size / #_GPUs)
        batch_size=128,
    ),
    TrainConfig(
        name="pi0tiny_incontext_robocasa_mg_three_image_scaleup_inference",
        model=pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16",
            use_image_prompts=True, use_action_state_prompts=True),
        data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
            repo_id="daixianjie/robocasa_mg_lerobot",
            base_config=DataConfig(
                local_files_only=False,  
                prompt_from_task=True,
            ),
            task_to_episode='metadata/robocasa_mg/task_to_episode.json',
            episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
            states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
            actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
        ),
        vision_weight_loader=weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=weight_loaders.EmptyLoader(),
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 10_000,
            peak_lr= 5e-4,
            decay_steps= 1_500_000,
            decay_lr= 5e-5),
        num_train_steps=1_500_000,
        ema_decay=0.999,
        # num_worker per GPU
        num_workers=8,
        # batch size in total (bs_per_gpu = batch_size / #_GPUs)
        batch_size=128,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
