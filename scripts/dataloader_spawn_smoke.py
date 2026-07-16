"""Smoke-test the production dataloader path with spawned workers.

This is intentionally different from v18_dataloader_check.py: it exercises the
TorchDataLoader path used by training, including multiprocessing spawn and
persistent workers. Run it from a real .py file, not from stdin, because Python
spawn needs to be able to import __main__.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import random
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import numpy as np
import torch

import openpi.models.model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _to_numpy(value) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _create_loader(config: _config.TrainConfig, *, num_batches: int, num_workers: int, shuffle: bool):
    match config.model.model_type:
        case _model.ModelType.PI0 | _model.ModelType.PI0_FAST:
            return _data_loader.create_data_loader(
                config,
                skip_norm_stats=False,
                num_batches=num_batches,
                num_workers=num_workers,
                shuffle=shuffle,
            )
        case _model.ModelType.PI0_INCONTEXT | _model.ModelType.PI0_FAST_INCONTEXT:
            if config.use_custom_dataloader:
                return _data_loader.create_custom_incontext_data_loader(
                    config,
                    skip_norm_stats=False,
                    num_batches=num_batches,
                    num_workers=num_workers,
                    shuffle=shuffle,
                )
            return _data_loader.create_incontext_data_loader(
                config,
                skip_norm_stats=False,
                num_batches=num_batches,
                num_workers=num_workers,
                shuffle=shuffle,
            )
        case _:
            raise ValueError(f"Unsupported model type: {config.model.model_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ContextFlow")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--assets-base-dir")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = _config.get_config(args.config)
    if args.assets_base_dir is not None:
        config = dataclasses.replace(config, assets_base_dir=args.assets_base_dir)
    config = dataclasses.replace(config, batch_size=args.batch_size)

    print(
        "building loader",
        {
            "config": args.config,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_batches": args.num_batches,
            "shuffle": not args.no_shuffle,
            "jax_devices": [str(device) for device in jax.devices()],
        },
        flush=True,
    )
    loader = _create_loader(
        config,
        num_batches=args.num_batches,
        num_workers=args.num_workers,
        shuffle=not args.no_shuffle,
    )

    t0 = time.perf_counter()
    for batch_i, (obs, actions) in enumerate(loader):
        actions_np = _to_numpy(actions)
        message = {
            "batch": batch_i,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
            "actions_shape": list(actions_np.shape),
            "actions_dtype": str(actions_np.dtype),
        }
        if getattr(obs, "incontext_selected_episode", None) is not None:
            selected_episode = _to_numpy(obs.incontext_selected_episode)
            message["selected_episode_min"] = int(selected_episode.min())
            message["selected_episode_max"] = int(selected_episode.max())
        print(message, flush=True)

    print(f"ok: consumed {args.num_batches} batches", flush=True)


if __name__ == "__main__":
    main()
