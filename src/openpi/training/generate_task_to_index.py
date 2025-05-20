"""Refactored dataset sanity‑check script.

This script builds lookup tables (task‑to‑episode and episode‑to‑indexes) for any
OpenPI dataset configuration and optionally prints a few example samples before
and after transformation.

Usage (CLI):
    uv run src/openpi/training/generate_task_to_index.py --config pi0mini_robocasa_human_three_image_low_mem_finetune_train --output_dir metadata/robocasa

You can also import and call `test_dataset` programmatically:

    from test_dataset_refactor import test_dataset
    cfg = _config.get_config("pi0_libero")
    test_dataset(cfg, output_dir="metadata/libero")
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from tqdm import tqdm

from openpi.training import config as _config
from openpi.training.data_loader import create_dataset, transform_dataset

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def build_lookup_tables(hf_dataset) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Build task‑to‑episode and episode‑to‑index lookup tables.

    Args:
        hf_dataset: A HuggingFace dataset containing the columns
            ``task_index``, ``episode_index``, and ``index`` (per‑frame index).
            The values may be Python integers or 0‑dim tensors.

    Returns:
        (task_to_episode, episode_to_indexes)
            task_to_episode: {task_index: [episode_index, ...]}
            episode_to_indexes: {episode_index: [index, ...]}
    """
    task_to_episode: Dict[int, set[int]] = defaultdict(set)
    episode_to_indexes: Dict[int, List[int]] = defaultdict(list)

    for i in tqdm(range(len(hf_dataset)), desc="Building lookup tables"):
        row = hf_dataset[i]
        task_idx = int(row["task_index"])      # e.g. tensor(3) -> 3
        episode_idx = int(row["episode_index"])
        index_idx = int(row["index"])

        task_to_episode[task_idx].add(episode_idx)
        episode_to_indexes[episode_idx].append(index_idx)

    # Convert sets to sorted lists to make JSON deterministic
    task_to_episode_sorted: Dict[int, List[int]] = {
        t: sorted(list(eps)) for t, eps in task_to_episode.items()
    }
    episode_to_indexes_sorted: Dict[int, List[int]] = {
        e: sorted(idx_list) for e, idx_list in episode_to_indexes.items()
    }
    return task_to_episode_sorted, episode_to_indexes_sorted


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
    # import ipdb; ipdb.set_trace()
    config_obj = _config.get_config(cfg) if isinstance(cfg, str) else cfg

    # Construct data config and dataset
    data_cfg = config_obj.data.create(config_obj.assets_dirs, config_obj.model)
    raw_dataset = create_dataset(data_cfg, config_obj.model)

    # import ipdb; ipdb.set_trace()
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
