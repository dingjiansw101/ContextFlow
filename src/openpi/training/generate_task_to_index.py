"""Refactored dataset sanity‑check script.

This script builds lookup tables (task‑to‑episode and episode‑to‑indexes) for any
OpenPI dataset configuration and optionally prints a few example samples before
and after transformation.

Usage (CLI):
    uv run src/openpi/training/generate_task_to_index.py --config pi0_libero90_low_mem_finetune --output_dir metadata/libero_90

You can also import and call `test_dataset` programmatically:

    from test_dataset_refactor import test_dataset
    cfg = _config.get_config("pi0_libero")
    test_dataset(cfg, output_dir="metadata/libero")
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple, Union

import numpy as np

from openpi.training import config as _config
from openpi.training.data_loader import create_dataset, transform_dataset

_LOOKUP_COLUMNS = ["task_index", "episode_index", "index"]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def build_lookup_tables(hf_dataset) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Build task‑to‑episode and episode‑to‑index lookup tables.

    Reads only the three integer columns as arrow → numpy and groups them by
    sorting once. Indexing ``hf_dataset[i]`` per frame instead would decode the
    image columns on every row (they live inside the parquet, ``total_videos: 0``),
    which costs ~2.5 ms/frame — 11+ minutes on LIBERO versus ~0.2 s here.

    Args:
        hf_dataset: A HuggingFace dataset containing the columns
            ``task_index``, ``episode_index``, and ``index`` (per‑frame index).

    Returns:
        (task_to_episode, episode_to_indexes)
            task_to_episode: {task_index: [episode_index, ...]}
            episode_to_indexes: {episode_index: [index, ...]}
    """
    # select_columns honours any indices mapping (.select()/.shuffle()) while
    # leaving the image columns untouched, so nothing gets decoded.
    table = hf_dataset.select_columns(_LOOKUP_COLUMNS).with_format("arrow")[:]
    task_idx = table.column("task_index").to_numpy(zero_copy_only=False)
    episode_idx = table.column("episode_index").to_numpy(zero_copy_only=False)
    frame_idx = table.column("index").to_numpy(zero_copy_only=False)

    # episode_to_indexes: sort by (episode, index), then split at episode boundaries.
    order = np.lexsort((frame_idx, episode_idx))
    episodes_sorted, frames_sorted = episode_idx[order], frame_idx[order]
    cuts = np.flatnonzero(episodes_sorted[1:] != episodes_sorted[:-1]) + 1
    episode_keys = episodes_sorted[np.concatenate(([0], cuts))]
    episode_to_indexes: Dict[int, List[int]] = dict(
        zip(episode_keys.tolist(), (group.tolist() for group in np.split(frames_sorted, cuts)), strict=True)
    )

    # task_to_episode: unique (task, episode) pairs come back lexsorted and deduped,
    # so the same boundary trick groups them by task.
    pairs = np.unique(np.stack([task_idx, episode_idx], axis=1), axis=0)
    task_cuts = np.flatnonzero(pairs[1:, 0] != pairs[:-1, 0]) + 1
    task_keys = pairs[np.concatenate(([0], task_cuts)), 0]
    task_to_episode: Dict[int, List[int]] = dict(
        zip(task_keys.tolist(), (group[:, 1].tolist() for group in np.split(pairs, task_cuts)), strict=True)
    )
    return task_to_episode, episode_to_indexes


def save_lookup_tables(
    task_to_episode: Dict[int, List[int]],
    episode_to_indexes: Dict[int, List[int]],
    output_dir: str = ".",
) -> None:
    """Save lookup dictionaries as JSON files.

    Files written:
        - ``task_to_episode.json``
        - ``episode_to_indexes.json``

    Args:
        task_to_episode: Mapping from task index to episode indices.
        episode_to_indexes: Mapping from episode index to frame indices.
        output_dir: Directory where the JSON files will be created.
    """
    os.makedirs(output_dir, exist_ok=True)

    task_path = os.path.join(output_dir, "task_to_episode.json")
    epi_path = os.path.join(output_dir, "episode_to_indexes.json")

    with open(task_path, "w", encoding="utf‑8") as f:
        json.dump(task_to_episode, f, indent=2)
    with open(epi_path, "w", encoding="utf‑8") as f:
        json.dump(episode_to_indexes, f, indent=2)

    print(f"✅ Wrote task_to_episode → {task_path}")
    print(f"✅ Wrote episode_to_indexes → {epi_path}")


# -----------------------------------------------------------------------------
# Main test function
# -----------------------------------------------------------------------------

def test_dataset(
    cfg: Union[str, "_config.TrainConfig"],
    output_dir: str = "metadata/",
    *,
    skip_norm_stats: bool = True,
    preview_samples: int = 0,
) -> None:
    """Sanity‑check a dataset configuration and generate lookup metadata.

    Args:
        cfg: Either the *name* of a registered train config (``"pi0_libero"``) or
            an already instantiated ``TrainConfig`` object.
        output_dir: Directory where ``task_to_episode.json`` and
            ``episode_to_indexes.json`` will be saved.
        skip_norm_stats: If ``True`` (default) skip normalisation statistics when
            running :pyfunc:`openpi.training.data_loader.transform_dataset`.
        preview_samples: Print the first *N* sample keys before and after
            transforms for quick inspection (default: ``0`` = none).
    """
    # Resolve config name → object if necessary
    config_obj = _config.get_config(cfg) if isinstance(cfg, str) else cfg

    # Construct data config and dataset
    data_cfg = config_obj.data.create(config_obj.assets_dirs, config_obj.model)
    raw_dataset = create_dataset(data_cfg, config_obj.model)

    # Build lookup tables from the underlying HF dataset
    try:
        task_to_episode, episode_to_index = build_lookup_tables(raw_dataset._dataset.hf_dataset)
    except:
        task_to_episode, episode_to_index = build_lookup_tables(raw_dataset.hf_dataset)
    save_lookup_tables(task_to_episode, episode_to_index, output_dir)

    # Optionally preview raw samples
    if preview_samples > 0:
        print("\n--- Raw dataset sample keys ---")
        for i in range(min(preview_samples, len(raw_dataset))):
            print(raw_dataset[i].keys())

    # Apply OpenPI transforms
    processed_dataset = transform_dataset(raw_dataset, data_cfg, skip_norm_stats=skip_norm_stats)

    # Optionally preview processed samples
    if preview_samples > 0:
        print("\n--- Processed dataset sample keys ---")
        for i in range(min(preview_samples, len(processed_dataset))):
            print(processed_dataset[i].keys())


# -----------------------------------------------------------------------------
# CLI entry‑point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata and sanity‑check an OpenPI dataset.")

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Name of a registered training config (e.g. 'pi0_libero') or a Python path to a config object.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="metadata/",
        help="Directory where metadata JSONs will be saved (default: %(default)s)",
    )
    parser.add_argument(
        "--skip_norm_stats",
        action="store_true",
        help="Skip normalisation stats when transforming the dataset (default: False)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="Print keys of the first N samples before and after transforms (default: 0)",
    )

    args = parser.parse_args()

    test_dataset(
        cfg=args.config,
        output_dir=args.output_dir,
        skip_norm_stats=args.skip_norm_stats,
        preview_samples=args.preview,
    )
