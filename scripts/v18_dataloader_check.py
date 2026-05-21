"""Deterministic batch artifact dump for the v18 in-context dataloader.

Used to verify exact behavior preservation across dataloader refactors. Loads
``pi0_libero_incontextv18_low_mem_finetune_sample_frames8`` with fixed seeds
and a synchronous (num_workers=0) loader, fetches a fixed number of items
directly from the underlying dataset, hashes their contents, and writes a JSON
report plus per-item .npz dumps.

Run BEFORE the refactor to capture the baseline, AFTER the refactor to verify
identical artifacts.

Usage:
  uv run python scripts/v18_dataloader_check.py \
      --output-dir /tmp/openpi_v18_consistency/<run_id>/<label>

Compare two runs with `diff` on the JSON summary (and optional .npz inspection).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (int, float, bool)):
        return np.asarray(value)
    return None


def _flatten_record(record, prefix=""):
    """Flatten nested dict/list-of-tensors into (key, ndarray) pairs."""
    out = {}
    if isinstance(record, dict):
        for k, v in record.items():
            out.update(_flatten_record(v, f"{prefix}{k}/"))
    else:
        arr = _to_numpy(record)
        if arr is not None:
            out[prefix.rstrip("/")] = arr
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi0_libero_incontextv18_low_mem_finetune_sample_frames8")
    parser.add_argument("--num-items", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-arrays", action="store_true", help="Dump per-item .npz")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    # Defer directory creation until we have results to write, so an aborted
    # run does not leave an empty dir behind.
    config = _config.get_config(args.config)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Build the raw custom dataset + transform stack, just like
    # create_custom_incontext_data_loader does, but without the TorchDataLoader.
    dataset = _data_loader.create_custom_dataset(data_config, config.model, config.data)
    dataset = _data_loader.transform_dataset(
        dataset,
        data_config,
        skip_norm_stats=False,
        norm_stats_aliases=getattr(config.data, "norm_stats_aliases", None),
    )

    # Pin both stdlib random and numpy random for deterministic demo selection.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_len = len(dataset)
    indices = []
    idx = args.start_index
    while len(indices) < args.num_items:
        indices.append(idx % dataset_len)
        idx += args.stride

    records = []
    timings = []
    for sample_i, dataset_idx in enumerate(indices):
        t0 = time.perf_counter()
        item = dataset[dataset_idx]
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        flat = _flatten_record(item)
        digest_items = {}
        for key in sorted(flat.keys()):
            arr = flat[key]
            digest_items[key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "hash": _hash_array(arr),
                "mean": float(arr.mean()) if arr.dtype.kind in "fui" and arr.size else None,
                "min": float(arr.min()) if arr.dtype.kind in "fui" and arr.size else None,
                "max": float(arr.max()) if arr.dtype.kind in "fui" and arr.size else None,
            }
        records.append({
            "sample_i": sample_i,
            "dataset_idx": int(dataset_idx),
            "elapsed_seconds": elapsed,
            "fields": digest_items,
        })
        if args.save_arrays:
            np.savez(output_dir / f"sample_{sample_i:02d}.npz", **flat)

    summary = {
        "config": args.config,
        "dataset_size": dataset_len,
        "num_items": args.num_items,
        "start_index": args.start_index,
        "stride": args.stride,
        "seed": args.seed,
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "git_branch": os.popen("git rev-parse --abbrev-ref HEAD").read().strip(),
        "per_sample": records,
        "timings_seconds": {
            "min": float(np.min(timings)),
            "mean": float(np.mean(timings)),
            "max": float(np.max(timings)),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Wrote {summary_path}")
    print(f"Per-sample timing min/mean/max (s): "
          f"{summary['timings_seconds']['min']:.4f} / "
          f"{summary['timings_seconds']['mean']:.4f} / "
          f"{summary['timings_seconds']['max']:.4f}")


if __name__ == "__main__":
    main()
