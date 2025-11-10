#!/usr/bin/env python3
"""Fast cache generation script for state-action demonstration data.

This script generates cache files for in-context learning by batch-loading
episode data using efficient .select() operations and multiprocessing.

Usage:
    uv run scripts/generate_cache.py <config_name> [--num-workers N]

Example:
    uv run scripts/generate_cache.py pi0_aloha_objects_task_suite_incontextv18_low_mem_finetune_sample_frames8
"""

import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import openpi.training.config as _config
import openpi.training.data_loader as data_loader
import openpi.transforms as _transforms


def save_episode_cache_to_json(episode_data: dict[int, np.ndarray], filename: str) -> None:
    """Save episode cache data to JSON file.

    Args:
        episode_data: Dict mapping episode_id -> numpy array of states or actions
        filename: Output JSON file path
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    json_dict = {str(ep): arr.tolist() for ep, arr in episode_data.items()}

    with filename.open("w") as f:
        json.dump(json_dict, f)

    print(f"[INFO] Saved cache to {filename} ({len(episode_data)} episodes)")

_WORKER_DATASET = None
_WORKER_TRANSFORMS = None


def _init_worker(dataset, transforms):
    global _WORKER_DATASET, _WORKER_TRANSFORMS
    _WORKER_DATASET = dataset
    _WORKER_TRANSFORMS = transforms


def process_episode_batch(args: tuple[Any, ...]) -> tuple[int, np.ndarray, np.ndarray]:
    """Process a single episode by batch-loading all its frames.

    Args:
        args: Tuple that includes episode_id, frame_indices, and optionally dataset/transforms.

    Returns:
        Tuple of (episode_id, states_array, actions_array)
    """
    if len(args) == 4:
        episode_id, frame_indices, raw_dataset, transforms = args
    else:
        episode_id, frame_indices = args
        raw_dataset = _WORKER_DATASET
        transforms = _WORKER_TRANSFORMS

    # Batch-load all frames for this episode using .select()
    # This is much faster than loading frames one-by-one
    frames = raw_dataset.select(frame_indices)

    state_list = []
    action_list = []

    # Apply transforms to each frame
    for i in range(len(frame_indices)):
        # Extract single frame from the dataset
        # HuggingFace Dataset supports integer indexing directly
        item = frames[i]

        # Apply transform pipeline
        if transforms is not None:
            item = transforms(item)

        # Extract state and first action from action horizon
        state_list.append(item["state"].numpy() if torch.is_tensor(item["state"]) else item["state"])

        # Get first action from action horizon (typically actions is [horizon, action_dim])
        actions = item["actions"]
        if torch.is_tensor(actions):
            actions = actions.numpy()

        # Handle both [horizon, action_dim] and [action_dim] shapes
        first_action = actions[0] if len(actions.shape) > 1 else actions
        action_list.append(first_action)

    # Stack into arrays
    states_array = np.stack(state_list, axis=0)
    actions_array = np.stack(action_list, axis=0)

    return episode_id, states_array, actions_array


def build_cache_parallel(config: _config.TrainConfig, num_workers: int = 8) -> None:
    """Build state-action caches using parallel processing.

    Args:
        config: Training configuration containing data config and cache paths
        num_workers: Number of parallel worker processes
    """
    print(f"[INFO] Building caches with {num_workers} workers...")

    # Load dataset and assemble only the transforms needed for states/actions
    # Note: We skip normalization/model transforms so cache stores raw values
    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = data_loader.create_dataset(data_config, config.model)

    default_prompt = getattr(config.data, "default_prompt", None) or ""
    state_action_transforms = [
        _transforms.InjectDefaultPrompt(default_prompt),
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ]
    transforms = _transforms.compose(state_action_transforms) if state_action_transforms else None

    # Underlying HuggingFace dataset (unwrap if needed)
    hf_dataset = raw_dataset
    while hasattr(hf_dataset, "_dataset"):
        hf_dataset = hf_dataset._dataset

    if hasattr(hf_dataset, "hf_dataset"):
        hf_dataset = hf_dataset.hf_dataset

    print(f"[INFO] Dataset loaded: {len(hf_dataset)} total frames")

    # Load episode-to-indexes mapping
    episode_to_indexes_file = Path(config.data.episode_to_indexes_file)
    if not episode_to_indexes_file.exists():
        raise FileNotFoundError(f"Episode indexes file not found: {episode_to_indexes_file}")

    with episode_to_indexes_file.open("r") as f:
        raw_mapping = json.load(f)

    episode_to_indexes = {int(k): v for k, v in raw_mapping.items()}
    print(f"[INFO] Loaded episode mapping: {len(episode_to_indexes)} episodes")

    # Prepare tasks for multiprocessing
    tasks = list(episode_to_indexes.items())

    # Process episodes in parallel with progress bar
    states_cache = {}
    actions_cache = {}

    print(f"[INFO] Processing {len(tasks)} episodes...")

    if num_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(hf_dataset, transforms),
        ) as pool:
            with tqdm(total=len(tasks), desc="Processing episodes", unit="ep") as pbar:
                for episode_id, states_array, actions_array in pool.imap_unordered(
                    process_episode_batch, tasks, chunksize=1
                ):
                    states_cache[episode_id] = states_array
                    actions_cache[episode_id] = actions_array
                    pbar.update(1)
                    pbar.set_postfix({
                        "states_shape": states_array.shape,
                        "actions_shape": actions_array.shape
                    })
    else:
        # Single-process mode (useful for debugging)
        for task in tqdm(tasks, desc="Processing episodes", unit="ep"):
            episode_id, states_array, actions_array = process_episode_batch(task + (hf_dataset, transforms))
            states_cache[episode_id] = states_array
            actions_cache[episode_id] = actions_array

    # Save caches to disk
    print("[INFO] Saving caches to disk...")
    save_episode_cache_to_json(states_cache, config.data.states_cache_path)
    save_episode_cache_to_json(actions_cache, config.data.actions_cache_path)

    # Print summary statistics
    total_states_mb = sum(arr.nbytes for arr in states_cache.values()) / 1024 / 1024
    total_actions_mb = sum(arr.nbytes for arr in actions_cache.values()) / 1024 / 1024
    total_frames = sum(len(arr) for arr in states_cache.values())

    print("\n[INFO] Cache generation completed!")
    print(f"  - Episodes: {len(states_cache)}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - States cache: {total_states_mb:.2f} MB")
    print(f"  - Actions cache: {total_actions_mb:.2f} MB")
    print(f"  - Total cache size: {(total_states_mb + total_actions_mb):.2f} MB")


if __name__ == "__main__":
    import argparse

    # Parse config name from CLI (compatible with training script CLI)
    config = _config.cli()

    # Add optional num_workers argument
    parser = argparse.ArgumentParser(description="Generate episode caches for in-context learning")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8, use 1 for debugging)"
    )

    # Parse only known args to avoid conflicts with config CLI args
    args, _ = parser.parse_known_args()

    build_cache_parallel(config, num_workers=args.num_workers)
