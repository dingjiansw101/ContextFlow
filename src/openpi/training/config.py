"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias, Optional, List, Union, Iterable
import os, re, json, jsonlines, sys, importlib

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.pi0_incontextv12 as pi0_incontextv12
import openpi.models.pi0_incontextv18 as pi0_incontextv18
import openpi.models.pi0_light as pi0Light
import openpi.models.deprecated_pi0light_incontextv12 as deprecated_pi0light_incontextv12
import openpi.models.pi0_light_incontextv12 as pi0_light_incontextv12
import openpi.models.pi0_light_incontextv14 as pi0_light_incontextv14

import openpi.models.pi0_light_incontextv12_separate_img_proj as pi0_light_incontextv12_separate_img_proj

import openpi.models.tokenizer as _tokenizer

import openpi.policies.droid_policy as droid_policy




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
        "turn on the stove",
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


# correspond to v4 in google sheet
DEFAULT_LIBERO_TEST_TASK_V5 = [
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "pick up the black bowl on the ramekin and place it on the plate",
        "pick up the chocolate pudding and place it in the basket",
        "pick up the orange juice and place it in the basket",
        "put the wine bottle on the rack",
        "put the bowl on top of the cabinet",
        "put the yellow and white mug in the microwave and close it",
        "put the black bowl in the bottom drawer of the cabinet and close it",
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
    # If true, will use create_custom_incontext_data_loader instead of create_incontext_data_loader.
    use_custom_dataloader: bool = False
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
    
    model_summary_json: str | None = None
    # XJ: add override assets dir to avoid creating redundant assets folders/files
    assets_repo_override: str | None = None
    
    @property
    def assets_dirs(self) -> pathlib.Path:
        if self.assets_repo_override is not None:
            return (pathlib.Path(self.assets_base_dir) / self.assets_repo_override).resolve()
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

def _discover_child_modules() -> list[str]:
    """自动发现同目录下 config_*.py（排除 config.py 本身）"""
    pkg_dir = pathlib.Path(__file__).parent
    base_pkg = __name__.rsplit(".", 1)[0]  # 例如 openpi.training
    out = []
    for p in pkg_dir.glob("config_*.py"):
        if p.name == "config.py":
            continue
        out.append(f"{base_pkg}.{p.stem}")  # e.g. openpi.training.config_aloha
    out.sort()
    return out

def _load_fragments(module_names: list[str]) -> list[TrainConfig]:
    """调用各子文件的 build(api) 汇总 TrainConfig 列表"""
    out: list[TrainConfig] = []
    api = sys.modules[__name__]  # 把当前模块对象传入
    for m in module_names:
        mod = importlib.import_module(m)
        build = getattr(mod, "build", None)
        if callable(build):
            out.extend(build(api))
    return out

# Use `get_config` if you need to get a config by name in your code.
_MODULES = _discover_child_modules()

_CONFIGS: list[TrainConfig] = _load_fragments(_MODULES)

# 重名保护
_names = [c.name for c in _CONFIGS]
_dups = {n for n in _names if _names.count(n) > 1}
if _dups:
    raise ValueError(f"Duplicate TrainConfig names: {_dups}")

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
