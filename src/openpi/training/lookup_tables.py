"""In-code generation of the task→episode and episode→frame lookup tables.

These tables used to be precomputed into ``metadata/<name>/task_to_episode.json``
and ``metadata/<name>/episode_to_indexes.json`` and then read back at train /
eval time. Both are fully determined by the LeRobot dataset metadata
(``meta/tasks.jsonl`` + ``meta/episodes.jsonl``), which LeRobotDataset already
loads, so deriving them here removes the precompute step and the risk of a stale
JSON silently disagreeing with the dataset it describes.

Cost is ~10 ms for a dataset the size of LIBERO, and no frame data is touched —
these read metadata only, so they work on machines where the parquet shards are
not present.
"""

from __future__ import annotations

import functools

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import get_episode_data_index


def build_task_to_episode(
    meta: LeRobotDatasetMetadata, episodes: list[int] | None = None
) -> dict[int, list[int]]:
    """Map each task index to the episode indices that carry that task.

    Args:
        meta: Metadata of the dataset to describe.
        episodes: When given, restrict the result to these episode indices — the
            same list handed to ``LeRobotDataset(episodes=...)``. Episodes absent
            from the dataset must not be offered as demo candidates, since callers
            resolve them against the dataset's own episode ordering.

    Returns:
        ``{task_index: [episode_index, ...]}``, episodes ascending.
    """
    task_to_index = meta.task_to_task_index
    keep = None if episodes is None else {int(ep) for ep in episodes}

    task_to_episode: dict[int, list[int]] = {}
    for episode in meta.episodes:
        episode_index = int(episode["episode_index"])
        if keep is not None and episode_index not in keep:
            continue
        for task in episode["tasks"]:
            task_to_episode.setdefault(int(task_to_index[task]), []).append(episode_index)
    return {task: sorted(eps) for task, eps in sorted(task_to_episode.items())}


def build_episode_to_indexes(
    meta: LeRobotDatasetMetadata, episodes: list[int] | None = None
) -> dict[int, list[int]]:
    """Map each episode index to its frame indices in the dataset's index space.

    Delegates the cumulative-offset arithmetic to LeRobot's own
    ``get_episode_data_index``, so a filtered dataset is reindexed exactly the way
    LeRobotDataset lays its frames out. That replaces the previous approach of
    reindexing a full-dataset JSON by hand, which was only correct when the kept
    episode list happened to be sorted.

    Args:
        meta: Metadata of the dataset to describe.
        episodes: When given, the episode indices kept by the dataset, in the
            order the dataset uses.

    Returns:
        ``{episode_index: [frame_index, ...]}``.
    """
    data_index = get_episode_data_index(meta.episodes, episodes)
    episode_ids = [int(ep) for ep in episodes] if episodes is not None else range(len(meta.episodes))
    starts = data_index["from"].tolist()
    ends = data_index["to"].tolist()
    return {
        int(episode): list(range(int(start), int(end)))
        for episode, start, end in zip(episode_ids, starts, ends, strict=True)
    }


def episode_to_indexes_for_dataset(dataset) -> dict[int, list[int]]:
    """Derive the episode→frame table from an already-constructed dataset.

    Accepts a LeRobotDataset or anything wrapping one via ``_dataset`` (e.g.
    ``TransformedDataset``), so callers that hold a dataset do not need to
    re-resolve the repo id or re-read metadata off disk.
    """
    inner = dataset
    for _ in range(8):  # bounded: wrappers nest a couple of levels at most
        if hasattr(inner, "meta"):
            return build_episode_to_indexes(inner.meta, getattr(inner, "episodes", None))
        if not hasattr(inner, "_dataset"):
            break
        inner = inner._dataset  # noqa: SLF001 — unwrapping TransformedDataset is the point
    raise TypeError(
        f"Cannot derive episode_to_indexes from {type(dataset).__name__}: no LeRobot dataset found. "
        "Expected a LeRobotDataset or a wrapper exposing it as ._dataset."
    )


@functools.cache
def _load_metadata(repo_id: str, root: str | None, local_files_only: bool) -> LeRobotDatasetMetadata:  # noqa: FBT001
    return LeRobotDatasetMetadata(repo_id, root=root, local_files_only=local_files_only)


def lookup_tables_for_repo(
    repo_id: str,
    *,
    root: str | None = None,
    episodes: list[int] | None = None,
    local_files_only: bool = False,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Build both lookup tables for a dataset repo, reading metadata only.

    The metadata object is cached per (repo_id, root, local_files_only), so
    repeated calls across transforms in one process cost a dict rebuild rather
    than re-reading ``meta/*.jsonl``.

    Returns:
        ``(task_to_episode, episode_to_indexes)``
    """
    meta = _load_metadata(repo_id, root, local_files_only)
    return build_task_to_episode(meta, episodes), build_episode_to_indexes(meta, episodes)
