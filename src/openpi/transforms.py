from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
from pathlib import Path
import random
import re
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable, Optional, List, Dict, Tuple

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

# def reindex_filtered_dict(data: Dict[str, Any], episode_order: Optional[Sequence[int]] = None) -> Dict[str, Any]:
#     """Reindex per-episode frame lists to be contiguous within a filtered subset.

#     Args:
#         data: Mapping from episode index to a list of global frame indices. Assumes episodes are
#             unique keys and lists preserve the relative ordering within each episode.
#         episode_order: Optional sequence specifying the iteration order for episodes. When provided,
#             only episodes present in both the order list and `data` are considered, preserving the
#             dataset's filtered ordering.

#     Returns:
#         Dict mapping the same episode indices to contiguous frame indices starting from zero.
#     """
#     new_data: Dict[int, list[int]] = {}
#     current_frame_index = 0

#     if episode_order is None:
#         episode_iter = data.keys()
#     else:
#         episode_iter = [ep for ep in episode_order if ep in data]

#     for ep_idx in episode_iter:
#         frames = data[ep_idx]
#         num_frames = len(frames)
#         new_data[ep_idx] = list(range(current_frame_index, current_frame_index + num_frames))
#         current_frame_index += num_frames

#     return new_data

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

    task_to_episode: str = "metadata/libero/task_to_episode.json"
    episode_to_indexes: str = "metadata/libero/episode_to_indexes.json"

    sample_frames: int = 2
    random_select: bool = True
    sample_episodes: int = 1
    train_episode_index_list: Optional[List[int]] = None
    
    # XJ(stage-wise prompt)
    all_episode_stage: Optional[str] = None # json path or None
    override_index: Optional[bool] = False # only used when sample_frames == 2
    provided_stage_key: str = "stage_rank"   # <- 新增：从 data 里读取的键名


    def __post_init__(self):
        task_to_episode_path = Path(self.task_to_episode)
        episode_to_indexes_path = Path(self.episode_to_indexes)
        # Load JSON files using Path.open()
        with task_to_episode_path.open("r") as f:
            task_to_episode_str = json.load(f)
        with episode_to_indexes_path.open("r") as f:
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
            episode_to_indexes = reindex_filtered_dict(episode_to_indexes)#, self.train_episode_index_list)

        # XJ(stage-wise prompt)
        # ---- XJ: filter task_to_episode only when all_episode_stage is not None ----
        if self.all_episode_stage is not None and self.train_episode_index_list is not None:
            # 1) 允许集合：白名单 ∩ episode_to_indexes 中“存在且帧数>0”的 episode
            allowed_from_list = {int(x) for x in self.train_episode_index_list}
            available_eps = {
                int(ep)
                for ep, idxs in episode_to_indexes.items()
                if isinstance(idxs, list) and len(idxs) > 0
            }
            allowed = allowed_from_list & available_eps  # 关键：与可用键做交集

            # 2) 过滤 task_to_episode（运行时类型统一到 int）
            filtered_task_to_episode = {
                int(task): [
                    int(ep) for ep in (eps or [])
                    if int(ep) in allowed
                ]
                for task, eps in task_to_episode.items()
            }
            task_to_episode = filtered_task_to_episode

    
        # Store these dictionaries on the frozen dataclass
        object.__setattr__(self, "task_to_episode", task_to_episode)
        object.__setattr__(self, "episode_to_indexes", episode_to_indexes)
        
        # XJ(stage-wise prompt)
        # ---- XJ: preprocess all_episode_stage（list -> dict index）----
        stage_map: Optional[Dict[str, List[Dict[str, Any]]]] = None
        if self.all_episode_stage is not None:
            with Path(self.all_episode_stage).open("r") as f:
                stage_list = json.load(f)  # a list, each element of which is dict（including episode_name, segments）

            # build a lookup table of {episode_name -> segments(list of dict)} 
            tmp = {}
            for entry in stage_list:
                name = entry.get("episode_name")
                segments = entry.get("segments")
                if isinstance(name, str) and isinstance(segments, list):
                    # segment
                    norm = []
                    for i, seg in enumerate(segments):
                        if not isinstance(seg, dict):
                            continue
                        s = int(seg.get("start", 0))
                        e = int(seg.get("end", 0))
                        if e > s:  # left close right open interval: [s, e)
                            norm.append({
                                "id": int(seg.get("id", i)),
                                "start": s,
                                "end": e,
                                "label": str(seg.get("label", "")),
                            })
                    if norm:
                        tmp[name] = norm
            stage_map = tmp if tmp else None
        object.__setattr__(self, "_episode_stage_map", stage_map)

        # XJ: Initialize inference cache
        object.__setattr__(self, "_cache", {
            "task_index": None,  # type: Optional[int]
            "selected_episode": None,  # type: Optional[np.ndarray]
            "dem_prompt_indexes": None,  # type: Optional[List[List[int]]]
            "provided_stage_rank": None,
        })
        
    # XJ(stage-wise prompt): casr episode_index（e.g: 47426）to standard file name
    @staticmethod
    def _index_to_episode_name(ep_idx: int) -> str:
        return f"episode_{ep_idx:06d}.parquet"

    # XJ(stage-wise prompt): based on frame_idx_in_episode, find the stage in segments
    @staticmethod
    def _find_stage(segments: List[Dict[str, Any]], frame_idx_in_episode: int) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        for rank, seg in enumerate(segments):
            s, e = seg["start"], seg["end"]
            if s <= frame_idx_in_episode < e:
                return rank, seg
        return None, None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Example transform: randomly select a demonstration as prompt for the data,
        storing both the chosen indexes and the retrieved items.
        """
        # 1) Randomly pick an episode for this task
        task_index = int(data["task_index"])
        episodes_for_task = self.task_to_episode.get(task_index, [])

        split = data.get("split", "train")
        
        # XJ(stage-wise prompt)
        # provided_rank = data.get(self.provided_stage_key, None) 

        
        # === XJ: Inference cache hit ===
        if self._cache["task_index"] == task_index and split == "test":
            data["selected_episode"] = self._cache["selected_episode"]
            data["dem_prompt_indexes"] = self._cache["dem_prompt_indexes"]
            return data
        
        # === Otherwise: generate prompt ===
        # 1) choose episodes
        if split == "train" and self.random_select:
            k = min(self.sample_episodes, len(episodes_for_task))
            selected_episodes = random.sample(episodes_for_task, k)
        else:
            selected_episodes = episodes_for_task[: self.sample_episodes]


        # 2) choose frames per episode
        dem_prompt_indexes: List[List[int]] = []
        # pos_list: List[List[int]] = []
        
        # XJ(stage-wise prompt)
        cur_ep_idx = data.get("episode_index", None)
        cur_frame_in_ep = data.get("frame_index", None)  
        # if override，with missing key，raise error
        if self.all_episode_stage is not None and self.override_index and self.sample_frames == 2:
            if cur_ep_idx is None or cur_frame_in_ep is None:
                raise ValueError(
                    "InjectDemoIndexes override needs both `episode_index` (int) and `frame_index` (int) in `data`."
                )
        # XJ(stage-wise prompt): find the stage number for current training sample
        cur_rank = None
        cur_stage = None
        if self.all_episode_stage is not None and self.override_index and self.sample_frames == 2 and self._episode_stage_map is not None:
            try:
                # cast the episode index to actual episode name
                cur_ep_name = self._index_to_episode_name(int(cur_ep_idx))
                segments = self._episode_stage_map.get(cur_ep_name, None)
                if segments:
                    cur_rank, cur_stage = self._find_stage(segments, int(cur_frame_in_ep))
            except Exception:
                # error
                raise ValueError("InjectDemoIndexes (XJ): cannot find the accurate stage number for current training sample")
                # cur_rank, cur_stage = None, None
                
                
        for ep in selected_episodes:
            frame_idxs = self.episode_to_indexes.get(ep, [])
            n = len(frame_idxs)
            if n > self.sample_frames:
                pos = np.linspace(0, n - 1, num=self.sample_frames, dtype=int)
                # pos_list.append(pos)
                chosen = [frame_idxs[p] for p in pos]
            else:
                chosen = frame_idxs
                raise ValueError(f"InjectDemoIndexes (XJ): too few frames! len(frame_idxs)={n} in episode {selected_episodes} of task {task_index}, n<=self.sample_frames")

            # XJ(stage-wise prompt): override dem_prompt_indexes only when sample 2 frames
            if self.all_episode_stage is not None and self.override_index and self.sample_frames == 2 and self._episode_stage_map is not None:
                if cur_stage is not None:
                    # override "chosen" by the first and last frames of the current stage (map to the frame doamin of selected demo-episode) 
                    s_rel = int(cur_stage["start"])
                    e_rel = int(cur_stage["end"]) - 1  # last frame = end-1
                    if e_rel < s_rel:
                        e_rel = s_rel

                    # clip frame index to the frame domain of the selected demo-episode: [0, n-1]
                    def _clip(i: int) -> int:
                        return min(max(i, 0), n - 1)

                    s_rel = _clip(s_rel)
                    e_rel = _clip(e_rel)

                    # IMPORTANT!：episode_to_indexes[ep] cast lcoal frame index to global frame index 
                    chosen = [int(frame_idxs[s_rel]), int(frame_idxs[e_rel])]
                
                
            dem_prompt_indexes.append(chosen)

        # 3) attach to data
        data["selected_episode"] = np.array(selected_episodes, dtype=np.int32)
        data["dem_prompt_indexes"] = dem_prompt_indexes
        
        # XJ: === Update inference cache ===
        if split == "test":
            self._cache["task_index"] = task_index
            self._cache["selected_episode"] = np.array(selected_episodes, dtype=np.int32)
            self._cache["dem_prompt_indexes"] = dem_prompt_indexes

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
class AddCurrentFramesSequenceTransform(DataTransformFn):
    """
    给“当前样本”补上同 episode 的 N 帧序列（图像/状态/动作），
    并把 actions 从 [H,A] 改成 [N,H,A]，供 fused loss 使用。
    """
    dataset: any
    episode_to_indexes_file: Optional[str] = "metadata/libero/episode_to_indexes.json"
    n_frames: int = 4
    sampling: str = "random"       # "uniform" | "around" | "random" | "random_stratified"
    train_episode_index_list: Optional[List[int]] = None

    # 采样细化参数（可选）
    keep_anchor_when_random: bool = True          # random/random_stratified 下尽量包含 anchor
    enforce_unique: bool = True                   # 是否强制采样索引唯一
    debug_checks: bool = True                     # 是否启用强断言

    # 随机性控制：若不传，则用 numpy 全局 RNG（多进程下每 worker 会不同）
    seed_base: Optional[int] = None               # 可给每个实例一个基种子，保证可复现（可选）

    def __post_init__(self):
        # 读 episode->全局帧列表
        epi2idx = None
        if self.episode_to_indexes_file is not None:
            p = Path(self.episode_to_indexes_file)
            if p.exists():
                with p.open("r") as f:
                    raw = json.load(f)  # { "123": [global_idx, ...], ... }
                if self.train_episode_index_list is None:
                    epi2idx = {int(k): v for k, v in raw.items()}
                else:
                    allowed = set(int(ep) for ep in self.train_episode_index_list)
                    filt = {int(k): v for k, v in raw.items() if int(k) in allowed}
                    epi2idx = reindex_filtered_dict(filt)  # 你的已有逻辑
        object.__setattr__(self, "_epi2idx", epi2idx)

        # 内部 RNG（注意 dataclass frozen，需要用 object.__setattr__）
        if self.seed_base is not None:
            rng = np.random.default_rng(int(self.seed_base))
        else:
            rng = np.random.default_rng()
        object.__setattr__(self, "_rng", rng)

        # 基础参数校验
        if self.debug_checks:
            assert self.n_frames >= 2, "n_frames 必须 >= 2 才有意义（否则请禁用本 transform 或走单帧）"
            assert self.sampling in {"uniform", "around", "random", "random_stratified"}, \
                f"sampling 不支持：{self.sampling}"

    # -----------------------------
    # 采样子程序
    # -----------------------------
    def _ensure_unique(self, idxs: List[int]) -> List[int]:
        if not self.enforce_unique:
            return idxs
        uniq = list(dict.fromkeys(int(x) for x in idxs))  # 稳定去重
        if len(uniq) != len(idxs):
            raise AssertionError(f"[AddCurrentFramesSequenceTransform] 采样索引存在重复：{idxs} -> {uniq}")
        return uniq

    def _rng_for_sample(self, ep_idx: int, anchor_local_idx: Optional[int]) -> np.random.Generator:
        """
        为了 worker/epoch 稳定性，你也可以把 dataloader 的 global_step / epoch
        混入 seed，这里先保持简单：实例级 RNG。
        """
        return self._rng

    def _pick_indices_uniform(self, n_total: int) -> List[int]:
        # 均匀采样：linspace + 四舍五入（向下取整）
        loc = np.linspace(0, n_total - 1, num=self.n_frames, dtype=int).tolist()
        return [int(i) for i in loc]

    def _pick_indices_around(self, n_total: int, anchor_local_idx: Optional[int]) -> List[int]:
        if anchor_local_idx is None:
            return self._pick_indices_uniform(n_total)
        half = max(1, self.n_frames // 2)
        s = max(0, min(anchor_local_idx - half + 1, n_total - self.n_frames))
        e = min(n_total, s + self.n_frames)
        return list(range(int(s), int(e)))

    def _pick_indices_random(self, n_total: int, anchor_local_idx: Optional[int], rng: np.random.Generator) -> List[int]:
        """
        完全随机：不放回采样。若 keep_anchor_when_random=True 且给定 anchor，则包含 anchor。
        """
        if self.n_frames > n_total:
            raise ValueError(f"随机采样所需帧数 n_frames={self.n_frames} > 该 episode 总帧数 n_total={n_total}")
        if self.keep_anchor_when_random and anchor_local_idx is not None and 0 <= anchor_local_idx < n_total:
            # 先固定 anchor，再从余集合随机补齐
            rest = np.delete(np.arange(n_total), anchor_local_idx)
            k = self.n_frames - 1
            choose = rng.choice(rest, size=k, replace=False)
            loc = np.concatenate([[anchor_local_idx], choose])
        else:
            loc = rng.choice(n_total, size=self.n_frames, replace=False)
        loc = np.sort(loc)  # 保持时间顺序（可选）
        return [int(i) for i in loc]

    def _pick_indices_random_stratified(self, n_total: int, anchor_local_idx: Optional[int],
                                        rng: np.random.Generator) -> List[int]:
        """
        分段随机：把序列划分成 n_frames 个区段，每段均匀长度内随机取 1 帧，
        同时尽量包含 anchor（若在某段内则把该段的随机点替换为 anchor）。
        这样既随机，又避免长期偏向前/后端。
        """
        if self.n_frames > n_total:
            raise ValueError(f"分段随机所需帧数 n_frames={self.n_frames} > n_total={n_total}")
        bounds = np.linspace(0, n_total, num=self.n_frames + 1, dtype=int)  # 段边界，右开
        loc = []
        anchor_used = False
        for s, e in zip(bounds[:-1], bounds[1:]):
            e = max(e, s + 1)  # 防止空段
            if (self.keep_anchor_when_random and
                (anchor_local_idx is not None) and
                (s <= anchor_local_idx < e) and
                not anchor_used):
                loc.append(int(anchor_local_idx))
                anchor_used = True
            else:
                loc.append(int(rng.integers(s, e)))  # [s, e)
        loc.sort()
        return loc

    def _pick_indices(self, frame_list: List[int], anchor_local_idx: Optional[int]) -> List[int]:
        """
        从“episode 内的局部索引”空间采样（返回局部索引），稍后会映射为全局帧索引。
        """
        n = len(frame_list)
        if n == 0 or self.n_frames <= 1:
            raise ValueError("[AddCurrentFramesSequenceTransform] episode 为空或 n_frames <= 1。")

        rng = self._rng_for_sample(ep_idx=-1, anchor_local_idx=anchor_local_idx)  # ep_idx 非必需
        if self.sampling == "uniform":
            loc = self._pick_indices_uniform(n)
        elif self.sampling == "around":
            loc = self._pick_indices_around(n, anchor_local_idx)
        elif self.sampling == "random":
            loc = self._pick_indices_random(n, anchor_local_idx, rng)
        elif self.sampling == "random_stratified":
            loc = self._pick_indices_random_stratified(n, anchor_local_idx, rng)
        else:
            raise ValueError(f"未知 sampling: {self.sampling}")

        # 唯一性与范围断言
        loc = [int(i) for i in loc]
        if self.debug_checks:
            assert len(loc) == self.n_frames, f"采样数量应等于 n_frames={self.n_frames}，got {len(loc)}"
            if self.enforce_unique:
                assert len(set(loc)) == len(loc), f"采样局部索引不唯一：{loc}"
            assert all(0 <= i < n for i in loc), f"采样局部索引越界：{loc} with n={n}"

        # 映射为全局帧索引
        chosen_global = [int(frame_list[i]) for i in loc]
        if self.debug_checks:
            # 全局索引要能在 frame_list 中找到
            back = [frame_list.index(g) for g in chosen_global]
            assert all(0 <= b < n for b in back), f"全局索引无法回查局部：{chosen_global}"
        return chosen_global

    # -----------------------------
    # 主入口
    # -----------------------------
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 仅训练启用；推理/验证保持单帧
        split = data.get("split", "train")
        if split != "train" or self.n_frames <= 1:
            return data

        # 基本字段断言
        if self.debug_checks:
            assert "episode_index" in data, "data 缺少 episode_index"
        ep_idx = int(data["episode_index"])

        # anchor（局部帧号），仅在 around/random* 下用于“尽量包含”
        anchor_local = data.get("frame_index", None)
        anchor_local = int(anchor_local) if anchor_local is not None else None

        # 取得 episode 的全局帧列表
        if self._epi2idx is None or ep_idx not in self._epi2idx:
            raise ValueError("[AddCurrentFramesSequenceTransform] 无法找到该 episode 的帧索引列表 "
                             f"(ep={ep_idx})，请确认 episode_to_indexes_file 是否正确。")
        frame_list = self._epi2idx[ep_idx]
        if self.debug_checks:
            assert isinstance(frame_list, list) and len(frame_list) >= self.n_frames, \
                f"episode={ep_idx} 的帧数不足（{len(frame_list)} < n_frames={self.n_frames}）"

        # 采样全局帧索引
        chosen_global = self._pick_indices(frame_list, anchor_local)

        # 选取并堆叠
        items = [self.dataset[int(i)] for i in chosen_global]
        if self.debug_checks:
            assert len(items) == self.n_frames, f"items 数量异常：{len(items)} vs n_frames={self.n_frames}"
        if len(items) == 0:
            raise ValueError(f"[AddCurrentFramesSequenceTransform] empty chosen_global: {chosen_global}")

        stacked = tree_stack_np(items)  # 要求返回每个叶子在前面新增 N 维

        # 从 stacked 组织输出
        # 约定：stacked["image"][cam] -> [N,H,W,3], stacked["image_mask"][cam] -> [N]
        #      stacked["state"] -> [N,A], stacked["actions"] -> [N,H,A]
        images_seq = {name: arr for name, arr in stacked["image"].items()}
        masks_seq  = {name: arr for name, arr in stacked["image_mask"].items()}
        state_seq  = stacked["state"]
        act_seq    = stacked["actions"]

        # 强断言（形状/类型）
        if self.debug_checks:
            N = self.n_frames
            # 图像/掩码键一致
            assert set(images_seq.keys()) == set(masks_seq.keys()), \
                f"image 与 image_mask 视角键不一致：{images_seq.keys()} vs {masks_seq.keys()}"
            # 形状检查
            for cam, arr in images_seq.items():
                assert arr.ndim == 4 and arr.shape[0] == N, \
                    f"images_seq[{cam}] 期望 [N,H,W,3]，got {arr.shape}"
            for cam, arr in masks_seq.items():
                assert arr.ndim == 1 and arr.shape[0] == N and arr.dtype == np.bool_, \
                    f"masks_seq[{cam}] 期望 [N] 且 bool，got {arr.shape}, {arr.dtype}"
            assert state_seq.ndim == 2 and state_seq.shape[0] == N, \
                f"state_seq 期望 [N,A]，got {state_seq.shape}"
            assert act_seq.ndim == 3 and act_seq.shape[0] == N, \
                f"actions_seq 期望 [N,H,A]，got {act_seq.shape}"
            # 单调性（按 chosen_global 排序应与时间一致）
            assert chosen_global == sorted(chosen_global), \
                f"chosen_global 未按时间升序：{chosen_global}"
            # 唯一性
            if self.enforce_unique:
                assert len(set(chosen_global)) == len(chosen_global), \
                    f"chosen_global 存在重复：{chosen_global}"

        # 写回 data（下游 fused 会期望这些键）
        data["current_images_seq"]       = images_seq
        data["current_image_masks_seq"]  = masks_seq
        data["current_state_seq"]        = state_seq.astype(np.float32, copy=False)
        data["actions_seq"]              = act_seq.astype(np.float32, copy=False)

        return data
    
    
@dataclasses.dataclass(frozen=True)
class AddImagePromptTransform(DataTransformFn):
    """Stacks image prompts per episode into dicts of arrays by key."""
    dataset: any
    
    def __post_init__(self):
        # XJ: inference cache
        object.__setattr__(self, "_cache", {
            "dem_prompt_indexes": None,
            "dem_prompt_images": None,
            "dem_prompt_images_mask": None,
            "split": None,
        })

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
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        idx_lists: List[List[int]] = data.get("dem_prompt_indexes", [])
        
        # ---- Check cache for test split ----
        split = data.get("split", "train")
        if (
            split == "test"
            and self._cache["split"] == "test"
            and self._cache["dem_prompt_indexes"] == idx_lists
        ):
            # Use cached results
            data["dem_prompt_images"] = self._cache["dem_prompt_images"]
            data["dem_prompt_images_mask"] = self._cache["dem_prompt_images_mask"]
            return data
        
        
        images_dict: Dict[str, List[np.ndarray]] = {}
        masks_dict: Dict[str, List[np.ndarray]] = {}
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
    Stage-aware (windowed) sampling strictly by stage rank if stage JSON is provided.
    """
    dataset: any  # the underlying dataset from which to fetch demo items

    max_len: int = 32

    episode_to_all_states: Dict[int, np.ndarray] = dataclasses.field(init=False)
    episode_to_all_first_actions: Dict[int, np.ndarray] = dataclasses.field(init=False)

    states_cache_path: str = "metadata/libero/episode_states_cache.json"
    actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
    episode_to_indexes_file: str = "metadata/libero/episode_to_indexes.json"

    # Stage JSON (list of dicts; each entry has "episode_name" and "segments":[{start,end,label,id}, ...])
    all_episode_stage: Optional[str] = None

    def __post_init__(self):
        # ---- Load / Build caches for states & actions ----
        try:
            states = load_episode_states_from_json(self.states_cache_path)
            actions = load_episode_states_from_json(self.actions_cache_path)
        except FileNotFoundError:
            with Path(self.episode_to_indexes_file).open("r") as f:
                raw = json.load(f)
            idx_map = {int(k): v for k, v in raw.items()}
            states, actions = {}, {}
            for ep, idxs in tqdm(idx_map.items(), desc="Building caches", total=len(idx_map)):
                state_list, action_list = [], []
                for idx in idxs:
                    item = self.dataset[int(idx)]
                    state_list.append(item["state"])        # shape: [D_s]
                    action_list.append(item["actions"][0])  # shape: [D_a]  (keep your original choice)
                states[ep] = np.stack(state_list, axis=0)      # [T, D_s]
                actions[ep] = np.stack(action_list, axis=0)    # [T, D_a]
            save_episode_states_to_json(states, self.states_cache_path)
            save_episode_states_to_json(actions, self.actions_cache_path)

        object.__setattr__(self, "episode_to_all_states", states)
        object.__setattr__(self, "episode_to_all_first_actions", actions)

        # ---- Build stage map: {episode_name -> segments} ----
        stage_map: Optional[Dict[str, List[Dict[str, Any]]]] = None
        if self.all_episode_stage is not None:
            with Path(self.all_episode_stage).open("r") as f:
                stage_list = json.load(f)  # list of dicts

            tmp: Dict[str, List[Dict[str, Any]]] = {}
            for entry in stage_list:
                name = entry.get("episode_name")
                segments = entry.get("segments")
                if isinstance(name, str) and isinstance(segments, list):
                    norm = []
                    for i, seg in enumerate(segments):
                        if not isinstance(seg, dict):
                            continue
                        s = int(seg.get("start", 0))
                        e = int(seg.get("end", 0))
                        if e > s:  # half-open [s, e)
                            norm.append({
                                "id": int(seg.get("id", i)),
                                "start": s,
                                "end": e,
                                "label": str(seg.get("label", "")),
                            })
                    if norm:
                        tmp[name] = norm
            stage_map = tmp if tmp else None

        object.__setattr__(self, "_episode_stage_map", stage_map)

        # ---- Inference cache ----
        object.__setattr__(self, "_cache", {
            "selected_episode": None,
            "cur_ep_idx": None,
            "cur_frame_in_ep": None,
            "states": None,
            "states_mask": None,
            "actions": None,
            "actions_mask": None,
        })

    # ---------- helpers ----------
    @staticmethod
    def _index_to_episode_name(ep_idx: int) -> str:
        # LIBERO-style naming
        return f"episode_{ep_idx:06d}.parquet"

    @staticmethod
    def _find_stage_by_frame(segments: List[Dict[str, Any]], frame_idx_in_episode: int) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        """Return (rank, segment) where segment satisfies [start, end) covering frame_idx."""
        for rank, seg in enumerate(segments):
            s, e = seg["start"], seg["end"]
            if s <= frame_idx_in_episode < e:
                return rank, seg
        return None, None

    @staticmethod
    def _even_sample_len(T: int, max_len: int) -> np.ndarray:
        """linspace indices in [0, T-1], length=max_len; assume T>=1."""
        if max_len <= 0:
            return np.empty((0,), dtype=int)
        if T <= 1:
            return np.zeros((max_len,), dtype=int)
        return np.linspace(0, T - 1, num=max_len, dtype=int)

    def _window_sample_and_pad(self, arr: np.ndarray, s: int, e: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample / pad from window [s, e) to max_len.
        Returns: (sampled, mask) with mask True for valid frames.
        """
        T_total = arr.shape[0]
        s = max(0, min(s, T_total))
        e = max(s + 1, min(e, T_total))  # ensure at least 1 frame
        window = arr[s:e]                 # [L, D]
        L = window.shape[0]
        D = window.shape[1] if window.ndim > 1 else 1

        if L >= self.max_len:
            idxs = self._even_sample_len(L, self.max_len)
            sampled = window[idxs]
            mask = np.ones((self.max_len,), dtype=bool)
        else:
            sampled = np.zeros((self.max_len, D), dtype=arr.dtype)
            sampled[:L] = window
            sampled[L:] = window[-1] if L > 0 else 0
            mask = np.zeros((self.max_len,), dtype=bool)
            mask[:L] = True
        return sampled, mask

    @staticmethod
    def _pick_segment_by_rank(segments: List[Dict[str, Any]], rank: int) -> Dict[str, Any]:
        """Rank-only alignment: pick segment by rank; fallback to last if out-of-range."""
        if not segments:
            raise ValueError("segments is empty")
        if 0 <= rank < len(segments):
            return segments[rank]
        return segments[-1]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        eps: List[int] = data.get("selected_episode", [])
        split = data.get("split", "train")

        # Current sample's (episode, frame) to locate current stage rank
        cur_ep_idx = data.get("episode_index", None)   # int episode index
        cur_frame_in_ep = data.get("frame_index", None)  # relative frame-in-episode
        
        # if self._episode_stage_map is not None:  # 或用: if self.all_episode_stage is not None:
        #     if cur_ep_idx is None or cur_frame_in_ep is None:
        #         raise ValueError(
        #             "[Stage] Stage map provided but current sample lacks required keys: "
        #             f"episode_index={cur_ep_idx}, frame=(frame_index/index)={cur_frame_in_ep}. "
        #             "Fix upstream repack (map 'index' to 'frame_index') or disable stage mode."
        #         )
        
        # ---- Cache hit for test split ----
        if (
            split == "test"
            and self._cache["selected_episode"] == list(eps)
            and self._cache["cur_ep_idx"] == cur_ep_idx
            and self._cache["cur_frame_in_ep"] == cur_frame_in_ep
        ):
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

        # XJ(stage-wise prompt): ---- Determine current stage rank (if available) ----
        cur_rank: Optional[int] = None
        cur_stage: Optional[Dict[str, Any]] = None
        if self._episode_stage_map is not None and cur_ep_idx is not None and cur_frame_in_ep is not None:
            # assert (s_rel != 0 or e_rel != T), "[Stage] stage map present but sampling full episode; expected stage window."

            try:
                cur_ep_name = self._index_to_episode_name(int(cur_ep_idx))
                segments = self._episode_stage_map.get(cur_ep_name, None)
                if segments:
                    cur_rank, cur_stage = self._find_stage_by_frame(segments, int(cur_frame_in_ep))
            except Exception:
                raise NotImplementedError
                # cur_rank, cur_stage = None, None

        states_b, states_mask_b, actions_b, actions_mask_b = [], [], [], []

        for ep in eps:
            all_states = self.episode_to_all_states[ep]             # [T, D_s]
            all_actions = self.episode_to_all_first_actions[ep]     # [T, D_a]
            T = all_states.shape[0]

            # Default window: whole episode
            s_rel, e_rel = 0, T

            # XJ(stage-wise prompt): If we have current stage rank and demo segments, align strictly by rank
            if self._episode_stage_map is not None and cur_stage is not None and cur_rank is not None:
                demo_name = self._index_to_episode_name(int(ep))
                demo_segments = self._episode_stage_map.get(demo_name, None)

                if demo_segments:
                    seg = self._pick_segment_by_rank(demo_segments, int(cur_rank))
                    s_rel, e_rel = int(seg["start"]), int(seg["end"])
                else:
                    # No segments for this demo episode: map current stage window as-is
                    s_rel, e_rel = int(cur_stage["start"]), int(cur_stage["end"])

                # Clip to [0, T] with at least one frame
                s_rel = max(0, min(s_rel, T))
                e_rel = max(s_rel + 1, min(e_rel, T))
                
            # ---- XJ: sanity checks for stage usage ----
            # if self._episode_stage_map is not None:
            #     # 必须找到 stage
            #     assert cur_stage is not None, \
            #         f"[Stage] Stage map provided but current sample (ep={cur_ep_idx}, frame={cur_frame_in_ep}) not mapped to any stage"
            #     # 必须不是整段
            #     assert not (s_rel == 0 and e_rel == T), \
            #         f"[Stage] Stage map provided but sampling full episode for ep={ep}"

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
            self._cache["cur_ep_idx"] = cur_ep_idx
            self._cache["cur_frame_in_ep"] = cur_frame_in_ep
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
    """Adds track sequences for multiple episodes."""
    max_len: int = 32
    tracks_path: str = "metadata/libero/episode_tracks_combined.json"
    episode_to_tracks: Dict[int, np.ndarray] = dataclasses.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "episode_to_tracks", load_episode_states_from_json(self.tracks_path))

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        eps: List[int] = data.get("selected_episode", [])
        tracks_b, tracks_mask_b = [], []
        for ep in eps:
            tr = self.episode_to_tracks[ep]
            T, F = tr.shape
            if T >= self.max_len:
                idxs = np.linspace(0, T - 1, num=self.max_len, dtype=int)
                sampled_tr = tr[idxs]
                tracks_mask = np.ones((self.max_len,), dtype=bool)
            else:
                sampled_tr = np.zeros((self.max_len, F), dtype=tr.dtype)
                sampled_tr[:T] = tr
                sampled_tr[T:] = tr[T - 1] if T > 0 else 0
                tracks_mask = np.zeros((self.max_len,), dtype=bool)
                tracks_mask[:T] = True
            tracks_b.append(sampled_tr)
            tracks_mask_b.append(tracks_mask)

        stacked_tracks = np.stack(tracks_b, axis=0)
        stacked_tracks_mask = np.stack(tracks_mask_b, axis=0)

        if len(eps) == 1:
            data["dem_prompt_tracks"] = stacked_tracks[0]
            data["dem_prompt_tracks_mask"] = stacked_tracks_mask[0]
        else:
            data["dem_prompt_tracks"] = stacked_tracks
            data["dem_prompt_tracks_mask"] = stacked_tracks_mask
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
