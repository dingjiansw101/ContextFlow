#!/usr/bin/env python3
"""
Convert MimicDroid RoboCasa HDF5 trajectories into Lerobot-style Parquet shards
suitable for OpenPI in-context training.

Each HDF5 under the input directory can contain multiple demos (groups
`/data/demo_*`). The script treats every group as a standalone episode and writes
it to `data/chunk-XXX/episode_XXXXXX.parquet`, accompanied by metadata files in
`meta/` (info.json, stats.json, episodes.jsonl, tasks.jsonl).

Example:
    python convert_mimicdroid_to_parquet.py \
        --input-root MimicDroidDataset/training \
        --output-root converted/mimicdroid_train \
        --state-key observations/robot0_proprio-state \
        --fps 20

Requirements:
    pip install h5py numpy pyarrow tqdm
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


@dataclass
class StatsAccumulator:
    """Track running statistics for vector features."""

    dim: int

    def __post_init__(self) -> None:
        self.count = 0
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sumsq = np.zeros(self.dim, dtype=np.float64)
        self.min = np.full(self.dim, np.inf, dtype=np.float64)
        self.max = np.full(self.dim, -np.inf, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        if batch.size == 0:
            return
        assert batch.ndim == 2 and batch.shape[1] == self.dim, (
            f"Expected (N,{self.dim}) but got {batch.shape}"
        )
        b = batch.astype(np.float64)
        self.count += b.shape[0]
        self.sum += b.sum(axis=0)
        self.sumsq += np.square(b).sum(axis=0)
        self.min = np.minimum(self.min, b.min(axis=0))
        self.max = np.maximum(self.max, b.max(axis=0))

    def finalize(self) -> Dict[str, List[float]]:
        if self.count == 0:
            raise ValueError("No samples seen; cannot finalize statistics.")
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - np.square(mean), 1e-12)
        std = np.sqrt(var)
        return {
            "count": int(self.count),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing MimicDroid HDF5 files (e.g., MimicDroidDataset/training)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination directory for Parquet files and metadata.",
    )
    parser.add_argument(
        "--state-key",
        default="observations/robot0_proprio-state",
        help="HDF5 dataset path to use as the low-dimensional observation.",
    )
    parser.add_argument(
        "--action-key",
        default="actions",
        help="HDF5 dataset path to use as the action vector.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Sampling frequency used to generate timestamps.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of episodes per output data chunk directory.",
    )
    parser.add_argument(
        "--pattern",
        default="demo.hdf5",
        help="Only process files whose name matches this pattern (use demo_im128_notp.hdf5 for vision).",
    )
    parser.add_argument(
        "--task-prefix",
        default="training",
        help="Label prefix used when deriving task names from paths.",
    )
    return parser.parse_args()


def find_hdf5_files(root: Path, pattern: str) -> List[Path]:
    return sorted(path for path in root.rglob(pattern) if path.is_file())


def task_name_from_path(h5_path: Path, task_prefix: str) -> str:
    parts = list(h5_path.parts)
    if "TaskDemos" in parts:
        idx = parts.index("TaskDemos")
        if idx + 2 < len(parts):
            return f"{parts[idx + 1]}_{parts[idx + 2]}"
        return parts[idx + 1] if idx + 1 < len(parts) else h5_path.stem
    parent = h5_path.parent.name
    return f"{task_prefix}_{parent}"


def fixed_size_list(array: np.ndarray, dtype: pa.DataType) -> pa.FixedSizeListArray:
    list_size = array.shape[1]
    flattened = array.reshape(-1)
    pa_array = pa.array(flattened, type=dtype)
    return pa.FixedSizeListArray.from_arrays(pa_array, list_size)


def ensure_dirs(output_root: Path) -> Tuple[Path, Path]:
    data_dir = output_root / "data"
    meta_dir = output_root / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, meta_dir


def write_parquet_episode(
    out_path: Path,
    episode_index: int,
    task_index: int,
    state: np.ndarray,
    action: np.ndarray,
    fps: float,
    global_index_start: int,
) -> int:
    num_frames = state.shape[0]
    frame_index = np.arange(num_frames, dtype=np.int32)
    timestamp = frame_index.astype(np.float32) / float(fps)

    data = {
        "index": pa.array(
            np.arange(global_index_start, global_index_start + num_frames, dtype=np.int64)
        ),
        "episode_index": pa.array(np.full(num_frames, episode_index, dtype=np.int32)),
        "frame_index": pa.array(frame_index),
        "task_index": pa.array(np.full(num_frames, task_index, dtype=np.int32)),
        "timestamp": pa.array(timestamp.astype(np.float32)),
        "observation/state": fixed_size_list(state.astype(np.float32), pa.float32()),
        "actions": fixed_size_list(action.astype(np.float32), pa.float32()),
    }

    table = pa.table(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd", compression_level=3)
    return global_index_start + num_frames


def main() -> None:
    args = parse_args()
    data_dir, meta_dir = ensure_dirs(args.output_root)

    h5_files = find_hdf5_files(args.input_root, args.pattern)
    if not h5_files:
        raise FileNotFoundError(
            f"No HDF5 files matching '{args.pattern}' found under {args.input_root}"
        )

    task_to_index: Dict[str, int] = {}
    tasks_meta: List[Dict[str, object]] = []
    episodes_meta: List[Dict[str, object]] = []

    episode_stats: Dict[str, StatsAccumulator] | None = None
    global_row_index = 0
    episode_counter = 0

    for h5_path in tqdm(h5_files, desc="Processing HDF5 files"):
        task_name = task_name_from_path(h5_path, args.task_prefix)
        if task_name not in task_to_index:
            task_idx = len(task_to_index)
            task_to_index[task_name] = task_idx
            tasks_meta.append(
                {"task_index": task_idx, "task_name": task_name, "source_path": str(h5_path.parent)}
            )
        else:
            task_idx = task_to_index[task_name]

        with h5py.File(h5_path, "r") as f:
            demo_root = f["data"] if "data" in f else f
            for demo_name in demo_root:
                grp = demo_root[demo_name]
                if args.state_key not in grp:
                    raise KeyError(
                        f"{args.state_key} not found in {h5_path}::{demo_name}. "
                        "Specify --state-key to match the dataset structure."
                    )
                if args.action_key not in grp:
                    raise KeyError(
                        f"{args.action_key} not found in {h5_path}::{demo_name}. "
                        "Specify --action-key to match the dataset structure."
                    )

                state = np.array(grp[args.state_key], dtype=np.float32)
                action = np.array(grp[args.action_key], dtype=np.float32)

                if state.shape[0] != action.shape[0]:
                    raise ValueError(
                        f"Mismatched lengths in {h5_path}::{demo_name}: "
                        f"{state.shape[0]} vs {action.shape[0]}"
                    )

                if episode_stats is None:
                    episode_stats = {
                        "observation/state": StatsAccumulator(dim=state.shape[1]),
                        "actions": StatsAccumulator(dim=action.shape[1]),
                    }

                episode_stats["observation/state"].update(state)
                episode_stats["actions"].update(action)

                chunk_id = episode_counter // args.chunk_size
                out_chunk_dir = data_dir / f"chunk-{chunk_id:03d}"
                out_file = out_chunk_dir / f"episode_{episode_counter:06d}.parquet"
                global_row_index = write_parquet_episode(
                    out_file,
                    episode_counter,
                    task_idx,
                    state,
                    action,
                    args.fps,
                    global_row_index,
                )

                episodes_meta.append(
                    {
                        "episode_index": episode_counter,
                        "task_index": task_idx,
                        "num_frames": int(state.shape[0]),
                        "hdf5_path": str(h5_path),
                        "demo_key": demo_name,
                    }
                )
                episode_counter += 1

    if episode_stats is None:
        raise RuntimeError("No episodes processed; aborting.")

    info = {
        "dataset_name": "MimicDroidConverted",
        "source_root": str(args.input_root),
        "num_episodes": episode_counter,
        "fps": args.fps,
        "state_key": args.state_key,
        "action_key": args.action_key,
        "version": 1,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    stats = {name: acc.finalize() for name, acc in episode_stats.items()}
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    with (meta_dir / "episodes.jsonl").open("w") as f:
        for item in episodes_meta:
            f.write(json.dumps(item) + "\n")

    with (meta_dir / "tasks.jsonl").open("w") as f:
        for item in tasks_meta:
            f.write(json.dumps(item) + "\n")

    print(
        f"Conversion finished: {episode_counter} episodes written to {args.output_root}. "
        f"Metadata stored under {meta_dir}."
    )


if __name__ == "__main__":
    main()
