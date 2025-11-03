#!/usr/bin/env python3
"""Compare data outputs from two different dataloader implementations.

Tests if CustomLeRobotDataset produces the same data as the regular LeRobotDataset
with transforms when using the same random seed.

Usage:
    # Compare with default configs and seed:
    python src/openpi/training/dataloader_comparison_test.py

    # Compare with specific seed:
    python src/openpi/training/dataloader_comparison_test.py --seed 42

    # Compare more batches:
    python src/openpi/training/dataloader_comparison_test.py --num-batches 5

    # Compare custom configs:
    python src/openpi/training/dataloader_comparison_test.py \
        --config1 pi0mini_incontext_libero_custom_dataset_debug \
        --config2 pi0mini_incontext_libero_low_mem_finetune_train_debug_baseline
"""

from __future__ import annotations

import argparse
import dataclasses
import random
from typing import Any

import jax
import numpy as np
import torch

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def set_all_seeds(seed: int) -> None:
    """Set random seeds for all libraries to ensure deterministic behavior."""
    print(f"Setting all random seeds to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # JAX uses explicit PRNG keys, but we can set numpy seed which JAX may use internally
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compare_arrays(
    arr1: np.ndarray | jax.Array,
    arr2: np.ndarray | jax.Array,
    name: str,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> tuple[bool, str]:
    """Compare two arrays for equality.

    Args:
        arr1: First array
        arr2: Second array
        name: Name of the field being compared
        rtol: Relative tolerance for float comparison
        atol: Absolute tolerance for float comparison

    Returns:
        Tuple of (is_match, message)
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    # Check shapes
    if arr1.shape != arr2.shape:
        return False, f"  ✗ {name}: Shape mismatch - {arr1.shape} vs {arr2.shape}"

    # Check dtypes
    if arr1.dtype != arr2.dtype:
        return False, f"  ✗ {name}: Dtype mismatch - {arr1.dtype} vs {arr2.dtype}"

    # Check values
    if np.issubdtype(arr1.dtype, np.floating):
        # Float comparison with tolerance
        if np.allclose(arr1, arr2, rtol=rtol, atol=atol):
            return True, f"  ✓ {name}: {arr1.shape} {arr1.dtype} - values match (within tolerance)"
        else:
            max_diff = np.abs(arr1 - arr2).max()
            return False, f"  ✗ {name}: {arr1.shape} {arr1.dtype} - values differ (max diff: {max_diff:.6e})"
    else:
        # Exact comparison for int/bool
        if np.array_equal(arr1, arr2):
            return True, f"  ✓ {name}: {arr1.shape} {arr1.dtype} - values match (exact)"
        else:
            num_diff = np.sum(arr1 != arr2)
            return False, f"  ✗ {name}: {arr1.shape} {arr1.dtype} - {num_diff} values differ"


def compare_dicts(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    name: str = "root",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> tuple[bool, list[str]]:
    """Recursively compare two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary
        name: Current path name for nested dicts
        rtol: Relative tolerance for float comparison
        atol: Absolute tolerance for float comparison

    Returns:
        Tuple of (all_match, messages)
    """
    messages = []
    all_match = True

    # Check for missing/extra keys
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    if keys1 != keys2:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        if missing_in_2:
            messages.append(f"  ✗ {name}: Keys only in dict1: {missing_in_2}")
            all_match = False
        if missing_in_1:
            messages.append(f"  ✗ {name}: Keys only in dict2: {missing_in_1}")
            all_match = False

    # Compare common keys
    for key in keys1 & keys2:
        val1 = dict1[key]
        val2 = dict2[key]
        full_name = f"{name}/{key}" if name != "root" else key
        # if "states" in key:
        #     import ipdb; ipdb.set_trace()
        #     print(f"val1: {val1}")
        #     print(f"val2: {val2}")
        #     print(f"full_name: {full_name}")
        #     print(f"rtol: {rtol}")
        #     print(f"atol: {atol}")
        #     print(f"messages: {messages}")
        #     print(f"all_match: {all_match}")
        if isinstance(val1, dict) and isinstance(val2, dict):
            # Recursive comparison for nested dicts
            match, sub_messages = compare_dicts(val1, val2, full_name, rtol, atol)
            messages.extend(sub_messages)
            all_match = all_match and match
        elif isinstance(val1, (np.ndarray, jax.Array)) or isinstance(val2, (np.ndarray, jax.Array)):
            # Array comparison
            match, msg = compare_arrays(val1, val2, full_name, rtol, atol)
            messages.append(msg)
            all_match = all_match and match
        elif isinstance(val1, (int, float, str, bool)) and isinstance(val2, (int, float, str, bool)):
            # Scalar comparison
            if val1 == val2:
                messages.append(f"  ✓ {full_name}: {val1} == {val2}")
            else:
                messages.append(f"  ✗ {full_name}: {val1} != {val2}")
                all_match = False
        else:
            # Type mismatch or unsupported type
            messages.append(f"  ? {full_name}: Type mismatch or unsupported - {type(val1)} vs {type(val2)}")
            all_match = False

    return all_match, messages


def compare_observations(
    obs1: _model.ObservationIncontext,
    obs2: _model.ObservationIncontext,
    batch_idx: int,
) -> tuple[bool, list[str]]:
    """Compare two ObservationIncontext objects.

    Args:
        obs1: First observation
        obs2: Second observation
        batch_idx: Batch index for logging

    Returns:
        Tuple of (all_match, messages)
    """
    messages = [f"\nBatch {batch_idx} - ObservationIncontext comparison:"]
    all_match = True

    # Convert to dicts for comparison
    obs1_dict = dataclasses.asdict(obs1)
    obs2_dict = dataclasses.asdict(obs2)

    match, sub_messages = compare_dicts(obs1_dict, obs2_dict, "observation")
    messages.extend(sub_messages)
    all_match = all_match and match

    return all_match, messages


def filter_messages(messages: list[str]) -> list[str]:
    """Filter messages to show summary for matches and details for mismatches."""
    matched = []
    mismatched = []

    for msg in messages:
        if msg.strip().startswith('✓'):
            matched.append(msg)
        elif msg.strip().startswith('✗') or msg.strip().startswith('?'):
            mismatched.append(msg)
        else:
            # Header messages, keep them
            mismatched.append(msg)

    result = []
    if matched:
        result.append(f"  ✓ {len(matched)} keys matched (types and values)")
    result.extend(mismatched)

    return result


def compare_dataloaders(
    config1_name: str,
    config2_name: str,
    num_batches: int = 3,
    seed: int = 42,
    batch_size: int = 2,
) -> bool:
    """Compare outputs from two dataloaders.

    Args:
        config1_name: Name of first config
        config2_name: Name of second config
        num_batches: Number of batches to compare
        seed: Random seed
        batch_size: Batch size for both loaders

    Returns:
        True if all batches match, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"DATALOADER COMPARISON TEST")
    print(f"{'='*80}")
    print(f"Config 1: {config1_name}")
    print(f"Config 2: {config2_name}")
    print(f"Batch size: {batch_size}")
    print(f"Num batches: {num_batches}")
    print(f"Random seed: {seed}")
    print(f"{'='*80}\n")

    # Set all seeds
    set_all_seeds(seed)

    # Create first dataloader
    print(f"Creating dataloader 1: {config1_name}")
    config1 = _config.get_config(config1_name)
    config1 = dataclasses.replace(
        config1,
        batch_size=batch_size,
        num_workers=0,  # Deterministic: no multiprocessing
    )

    # Override random_select=False for deterministic demo selection (if applicable)
    if hasattr(config1.data, 'random_select'):
        config1 = dataclasses.replace(
            config1,
            data=dataclasses.replace(config1.data, random_select=False)
        )

    if config1.use_custom_dataloader:
        # Check custom dataloader version
        version = getattr(config1.data, 'custom_dataloader_version', 'v1')
        if version == "v2":
            loader1 = _data_loader.create_custom_incontext_data_loaderv2(
                config1,
                skip_norm_stats=False,
                num_batches=num_batches,
            )
        else:  # v1 or default
            loader1 = _data_loader.create_custom_incontext_data_loader(
                config1,
                skip_norm_stats=False,
                num_batches=num_batches,
            )
    else:
        loader1 = _data_loader.create_incontext_data_loader(
            config1,
            skip_norm_stats=False,
            num_batches=num_batches,
        )

    # Reset seeds before creating second loader
    set_all_seeds(seed)

    # Create second dataloader
    print(f"Creating dataloader 2: {config2_name}")
    config2 = _config.get_config(config2_name)
    config2 = dataclasses.replace(
        config2,
        batch_size=batch_size,
        num_workers=0,  # Deterministic: no multiprocessing
    )

    # Override random_select=False for deterministic demo selection (if applicable)
    if hasattr(config2.model, 'random_select'):
        config2 = dataclasses.replace(
            config2,
            model=dataclasses.replace(config2.model, random_select=False)
        )

    if config2.use_custom_dataloader:
        # Check custom dataloader version
        version = getattr(config2.data, 'custom_dataloader_version', 'v1')
        if version == "v2":
            loader2 = _data_loader.create_custom_incontext_data_loaderv2(
                config2,
                skip_norm_stats=False,
                num_batches=num_batches,
            )
        else:  # v1 or default
            loader2 = _data_loader.create_custom_incontext_data_loader(
                config2,
                skip_norm_stats=False,
                num_batches=num_batches,
            )
    else:
        loader2 = _data_loader.create_incontext_data_loader(
            config2,
            skip_norm_stats=False,
            num_batches=num_batches,
        )

    print(f"\nComparing {num_batches} batches...\n")

    # Compare batches
    iter1 = iter(loader1)
    iter2 = iter(loader2)

    all_batches_match = True

    for i in range(num_batches):
        obs1, actions1 = next(iter1)
        obs2, actions2 = next(iter2)

        # Compare observations
        match, messages = compare_observations(obs1, obs2, i)
        filtered_messages = filter_messages(messages)
        for msg in filtered_messages:
            print(msg)
        all_batches_match = all_batches_match and match

        # Compare actions
        print(f"\nBatch {i} - Actions comparison:")
        match, msg = compare_arrays(actions1, actions2, "actions")
        print(msg)
        all_batches_match = all_batches_match and match

        print(f"\n{'-'*80}")

    # Summary
    print(f"\n{'='*80}")
    if all_batches_match:
        print("✓ SUCCESS: All batches match!")
    else:
        print("✗ FAILURE: Some batches differ")
    print(f"{'='*80}\n")

    return all_batches_match


def main():
    """Run dataloader comparison test."""
    parser = argparse.ArgumentParser(description="Compare dataloader outputs")
    parser.add_argument(
        "--config1",
        default="pi0mini_incontext_libero_custom_dataset_debug",
        help="First config name",
    )
    parser.add_argument(
        "--config2",
        default="pi0mini_incontext_libero_low_mem_finetune_train_debug_baseline",
        help="Second config name",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=3,
        help="Number of batches to compare",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size",
    )

    args = parser.parse_args()

    success = compare_dataloaders(
        args.config1,
        args.config2,
        args.num_batches,
        args.seed,
        args.batch_size,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
