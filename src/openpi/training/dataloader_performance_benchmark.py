#!/usr/bin/env python3
"""Benchmark data loader performance for CustomLeRobotDataset vs LeRobotDataset.

Compares:
1. pi0mini_incontext_libero_custom_dataset_debug (CustomLeRobotDataset)
2. pi0mini_incontext_libero_low_mem_finetune_train_debug_baseline (LeRobotDataset + transforms)

Usage:
    # Basic run with default parameters (batch_sizes=[2, 16], workers=[0, 2], batches=30):
    python src/openpi/training/dataloader_performance_benchmark.py

    # Custom batch sizes and workers:
    python src/openpi/training/dataloader_performance_benchmark.py \
        --batch-sizes 2 16 32 \
        --num-workers 0 2 4

    # More batches for better statistics:
    python src/openpi/training/dataloader_performance_benchmark.py \
        --num-batches 50

    # Compare different configs:
    python src/openpi/training/dataloader_performance_benchmark.py \
        --config1 pi0mini_incontext_libero_custom_dataset_debug \
        --config2 pi0mini_incontext_libero_low_mem_finetune_train_debug_baseline

Output:
    - Loader creation time
    - Warmup batch time (includes JIT compilation)
    - Mean/std/min/max batch times
    - Throughput (batches/second)
    - Memory usage (peak increase)
    - Side-by-side comparison table with % differences
    - Summary showing which implementation is faster
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import statistics
import time
from typing import Any

import psutil

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass
class TimingResult:
    """Results from benchmarking a single data loader configuration."""

    config_name: str
    batch_size: int
    num_workers: int
    num_batches: int

    # Timing metrics (seconds)
    loader_creation_time: float
    warmup_batch_time: float
    mean_batch_time: float
    std_batch_time: float
    min_batch_time: float
    max_batch_time: float
    total_iteration_time: float

    # Throughput
    batches_per_sec: float

    # Memory (bytes)
    memory_before: int
    memory_peak: int
    memory_after: int

    def memory_increase_mb(self) -> float:
        """Memory increase during iteration in MB."""
        return (self.memory_peak - self.memory_before) / 1024 / 1024


def get_memory_usage() -> int:
    """Get current RSS memory usage in bytes."""
    process = psutil.Process()
    return process.memory_info().rss


def benchmark_loader(
    config_name: str,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    skip_norm_stats: bool = True,
    sample_frames: int | None = None,
) -> TimingResult:
    """Benchmark a single data loader configuration.

    Args:
        config_name: Name of the config to benchmark
        batch_size: Batch size to use
        num_workers: Number of workers for data loading
        num_batches: Number of batches to iterate through
        skip_norm_stats: Whether to skip normalization stats
        sample_frames: Override sample_frames in config (if applicable)

    Returns:
        TimingResult containing all benchmark metrics
    """
    print(f"\nBenchmarking: {config_name}")
    print(f"  batch_size={batch_size}, num_workers={num_workers}, num_batches={num_batches}, sample_frames={sample_frames}")

    # Force garbage collection before measuring
    gc.collect()
    memory_before = get_memory_usage()

    # Measure loader creation time
    t_start = time.perf_counter()
    config = _config.get_config(config_name)
    config = dataclasses.replace(config, batch_size=batch_size, num_workers=num_workers)

    # Override sample_frames if specified (for custom dataset configs)
    if sample_frames is not None:
        # Override in model config if it has sample_frames
        if hasattr(config.model, 'sample_frames'):
            config = dataclasses.replace(
                config,
                model=dataclasses.replace(config.model, sample_frames=sample_frames)
            )
        # Override in data config if it has sample_frames
        if hasattr(config.data, 'sample_frames'):
            config = dataclasses.replace(
                config,
                data=dataclasses.replace(config.data, sample_frames=sample_frames)
            )

    # Determine which create function to use
    if "custom_dataset" in config_name:
        data_loader = _data_loader.create_custom_incontext_data_loader(
            config,
            skip_norm_stats=skip_norm_stats,
            num_batches=num_batches,
        )
    else:
        data_loader = _data_loader.create_incontext_data_loader(
            config,
            skip_norm_stats=skip_norm_stats,
            num_batches=num_batches,
        )
    loader_creation_time = time.perf_counter() - t_start
    print(f"  Loader created in {loader_creation_time:.3f}s")

    # Warmup: first batch (includes JIT compilation, caching)
    data_iter = iter(data_loader)
    t_warmup_start = time.perf_counter()
    _ = next(data_iter)
    warmup_batch_time = time.perf_counter() - t_warmup_start
    print(f"  Warmup batch: {warmup_batch_time:.3f}s")

    # Benchmark: iterate through num_batches
    batch_times = []
    memory_peak = memory_before

    t_total_start = time.perf_counter()
    for i in range(num_batches - 1):  # -1 because warmup consumed one
        t_batch_start = time.perf_counter()
        _ = next(data_iter)
        batch_time = time.perf_counter() - t_batch_start
        batch_times.append(batch_time)

        # Track peak memory
        current_memory = get_memory_usage()
        memory_peak = max(memory_peak, current_memory)

        if (i + 1) % 10 == 0:
            print(f"  Batch {i + 1}/{num_batches - 1}: {batch_time:.3f}s")

    total_iteration_time = time.perf_counter() - t_total_start

    # Measure memory after iteration
    gc.collect()
    memory_after = get_memory_usage()

    # Calculate statistics
    mean_batch_time = statistics.mean(batch_times)
    std_batch_time = statistics.stdev(batch_times) if len(batch_times) > 1 else 0.0
    min_batch_time = min(batch_times)
    max_batch_time = max(batch_times)
    batches_per_sec = (num_batches - 1) / total_iteration_time

    result = TimingResult(
        config_name=config_name,
        batch_size=batch_size,
        num_workers=num_workers,
        num_batches=num_batches,
        loader_creation_time=loader_creation_time,
        warmup_batch_time=warmup_batch_time,
        mean_batch_time=mean_batch_time,
        std_batch_time=std_batch_time,
        min_batch_time=min_batch_time,
        max_batch_time=max_batch_time,
        total_iteration_time=total_iteration_time,
        batches_per_sec=batches_per_sec,
        memory_before=memory_before,
        memory_peak=memory_peak,
        memory_after=memory_after,
    )

    print(f"  Mean batch time: {mean_batch_time:.3f}s ± {std_batch_time:.3f}s")
    print(f"  Throughput: {batches_per_sec:.2f} batches/sec")
    print(f"  Memory increase: {result.memory_increase_mb():.1f} MB")

    return result


def format_comparison_table(results: list[tuple[TimingResult, TimingResult]]) -> str:
    """Format comparison table for two configs.

    Args:
        results: List of (custom_result, regular_result) tuples

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("DATA LOADER PERFORMANCE COMPARISON")
    lines.append("=" * 100)

    for custom, regular in results:
        lines.append(f"\nConfiguration: batch_size={custom.batch_size}, num_workers={custom.num_workers}")
        lines.append("-" * 100)

        # Header
        lines.append(f"{'Metric':<40} {'CustomDataset':>20} {'RegularDataset':>20} {'Diff %':>15}")
        lines.append("-" * 100)

        # Loader creation
        diff_pct = ((custom.loader_creation_time - regular.loader_creation_time) / regular.loader_creation_time) * 100
        lines.append(f"{'Loader creation time (s)':<40} {custom.loader_creation_time:>20.3f} {regular.loader_creation_time:>20.3f} {diff_pct:>14.1f}%")

        # Warmup batch
        diff_pct = ((custom.warmup_batch_time - regular.warmup_batch_time) / regular.warmup_batch_time) * 100
        lines.append(f"{'Warmup batch time (s)':<40} {custom.warmup_batch_time:>20.3f} {regular.warmup_batch_time:>20.3f} {diff_pct:>14.1f}%")

        # Mean batch time
        diff_pct = ((custom.mean_batch_time - regular.mean_batch_time) / regular.mean_batch_time) * 100
        lines.append(f"{'Mean batch time (s)':<40} {custom.mean_batch_time:>20.3f} {regular.mean_batch_time:>20.3f} {diff_pct:>14.1f}%")

        # Std batch time
        lines.append(f"{'Std batch time (s)':<40} {custom.std_batch_time:>20.3f} {regular.std_batch_time:>20.3f} {'':<15}")

        # Min/Max batch time
        lines.append(f"{'Min batch time (s)':<40} {custom.min_batch_time:>20.3f} {regular.min_batch_time:>20.3f} {'':<15}")
        lines.append(f"{'Max batch time (s)':<40} {custom.max_batch_time:>20.3f} {regular.max_batch_time:>20.3f} {'':<15}")

        # Throughput
        diff_pct = ((custom.batches_per_sec - regular.batches_per_sec) / regular.batches_per_sec) * 100
        lines.append(f"{'Throughput (batches/sec)':<40} {custom.batches_per_sec:>20.2f} {regular.batches_per_sec:>20.2f} {diff_pct:>14.1f}%")

        # Total iteration time
        diff_pct = ((custom.total_iteration_time - regular.total_iteration_time) / regular.total_iteration_time) * 100
        lines.append(f"{'Total iteration time (s)':<40} {custom.total_iteration_time:>20.3f} {regular.total_iteration_time:>20.3f} {diff_pct:>14.1f}%")

        # Memory
        custom_mem_mb = custom.memory_increase_mb()
        regular_mem_mb = regular.memory_increase_mb()
        diff_pct = ((custom_mem_mb - regular_mem_mb) / regular_mem_mb) * 100 if regular_mem_mb > 0 else 0
        lines.append(f"{'Memory increase (MB)':<40} {custom_mem_mb:>20.1f} {regular_mem_mb:>20.1f} {diff_pct:>14.1f}%")

        lines.append("-" * 100)

    lines.append("=" * 100)

    # Summary
    lines.append("\nSUMMARY:")
    for custom, regular in results:
        if custom.mean_batch_time < regular.mean_batch_time:
            winner = "CustomDataset"
            speedup = (regular.mean_batch_time / custom.mean_batch_time - 1) * 100
        else:
            winner = "RegularDataset"
            speedup = (custom.mean_batch_time / regular.mean_batch_time - 1) * 100

        lines.append(f"  batch_size={custom.batch_size}, workers={custom.num_workers}: "
                    f"{winner} is {speedup:.1f}% faster")

    lines.append("=" * 100 + "\n")

    return "\n".join(lines)


def main():
    """Run benchmark comparison."""
    parser = argparse.ArgumentParser(description="Benchmark data loader performance")
    parser.add_argument(
        "--config1",
        default="pi0mini_incontext_libero_custom_dataset_debug",
        help="First config (CustomDataset)",
    )
    parser.add_argument(
        "--config2",
        default="pi0mini_incontext_libero_low_mem_finetune_train_debug_baseline",
        help="Second config (Regular)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[2, 16],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--num-workers",
        nargs="+",
        type=int,
        default=[0, 2],
        help="Worker counts to test",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=30,
        help="Number of batches to iterate",
    )
    parser.add_argument(
        "--skip-norm-stats",
        action="store_true",
        default=True,
        help="Skip normalization stats",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=None,
        help="Override sample_frames for custom dataset config (if applicable)",
    )

    args = parser.parse_args()

    print(f"Starting benchmark comparison:")
    print(f"  Config 1 (Custom): {args.config1}")
    print(f"  Config 2 (Regular): {args.config2}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Num batches: {args.num_batches}")
    if args.sample_frames is not None:
        print(f"  Sample frames override: {args.sample_frames}")

    results = []

    for batch_size in args.batch_sizes:
        for num_workers in args.num_workers:
            # Benchmark custom dataset
            custom_result = benchmark_loader(
                args.config1,
                batch_size,
                num_workers,
                args.num_batches,
                args.skip_norm_stats,
                args.sample_frames,
            )

            # Benchmark regular dataset
            regular_result = benchmark_loader(
                args.config2,
                batch_size,
                num_workers,
                args.num_batches,
                args.skip_norm_stats,
                args.sample_frames,
            )

            results.append((custom_result, regular_result))

    # Print comparison table
    print(format_comparison_table(results))


if __name__ == "__main__":
    main()
