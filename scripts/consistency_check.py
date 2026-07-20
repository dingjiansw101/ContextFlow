from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tree_fingerprint(tree: Any) -> tuple[str, list[str]]:
    import jax

    h = hashlib.sha256()
    shapes: list[str] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        key = jax.tree_util.keystr(path)
        arr = np.asarray(leaf)
        h.update(key.encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes(order="C"))
        shapes.append(f"{key}:{arr.shape}@{arr.dtype}")
    return h.hexdigest(), shapes


def _set_local_files_only(config_mod, cfg):
    base_data = cfg.data.base_config or config_mod.DataConfig()
    return dataclasses.replace(
        cfg.data,
        base_config=dataclasses.replace(base_data, local_files_only=True),
    )


def _make_config(config_mod, args: argparse.Namespace):
    import jax

    cfg = config_mod.get_config(args.config)
    data = _set_local_files_only(config_mod, cfg)
    deterministic_data_seed = args.deterministic_data_seed
    if args.deterministic_data and deterministic_data_seed is None:
        deterministic_data_seed = args.seed
    if args.deterministic_data and hasattr(data, "seed_base"):
        data = dataclasses.replace(data, seed_base=deterministic_data_seed)

    # Demo selection uses the real random_select=True path; determinism comes from
    # seed_base (set above via --deterministic-data). Cross-branch comparison against a
    # reference like aloha-dev requires that reference to carry the same dormant seed_base
    # plumbing (behavior-preserving when unset); see ALOHA_CACHE_FREE_CONSISTENCY.md.
    return dataclasses.replace(
        cfg,
        seed=args.seed,
        batch_size=max(1, jax.device_count()),
        num_workers=0,
        fsdp_devices=max(1, jax.device_count()),
        wandb_enabled=False,
        num_train_steps=1,
        save_interval=1,
        log_interval=1,
        overwrite=True,
        resume=False,
        exp_name=f"{args.config}_consistency",
        checkpoint_base_dir=args.checkpoint_base_dir,
        data=data,
    )


def _make_loader(kind: str, trainer, data_loader_mod, cfg, data_sharding):
    if kind == "refactor":
        return trainer._create_train_data_loader(cfg, sharding=data_sharding, num_workers=0, shuffle=True)  # noqa: SLF001
    if kind == "standard":
        return data_loader_mod.create_data_loader(cfg, sharding=data_sharding, num_workers=0, shuffle=True)
    if kind == "incontext":
        if getattr(cfg, "use_custom_dataloader", False):
            return data_loader_mod.create_custom_incontext_data_loader(
                cfg,
                sharding=data_sharding,
                num_workers=0,
                shuffle=True,
            )
        return data_loader_mod.create_incontext_data_loader(cfg, sharding=data_sharding, num_workers=0, shuffle=True)
    if kind == "fast_mini_v14":
        if cfg.use_custom_dataloader:
            version = getattr(cfg.data, "custom_dataloader_version", "v1")
            if version == "v2":
                return data_loader_mod.create_custom_incontext_data_loaderv2(
                    cfg,
                    sharding=data_sharding,
                    num_workers=0,
                    shuffle=True,
                )
            return data_loader_mod.create_custom_incontext_data_loader(
                cfg,
                sharding=data_sharding,
                num_workers=0,
                shuffle=True,
            )
        return data_loader_mod.create_incontext_data_loader(cfg, sharding=data_sharding, num_workers=0, shuffle=True)
    raise ValueError(f"Unknown loader kind: {kind}")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    logging.basicConfig(level=logging.WARNING)

    from flax.training import common_utils
    import jax
    import jax.numpy as jnp

    import openpi.training.config as config_mod
    import openpi.training.data_loader as data_loader_mod
    import openpi.training.sharding as sharding

    trainer = _load_module("consistency_trainer", args.trainer)

    jax.config.update("jax_threefry_partitionable", val=True)
    jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))

    cfg = _make_config(config_mod, args)
    rng = jax.random.key(cfg.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    loader = _make_loader(args.loader_kind, trainer, data_loader_mod, cfg, data_sharding)
    batch = next(iter(loader))
    batch_hash, batch_shapes = _tree_fingerprint(batch)

    result = {
        "status": "ok",
        "worktree": str(Path.cwd()),
        "config": args.config,
        "trainer": args.trainer,
        "loader_kind": args.loader_kind,
        "devices": [str(d) for d in jax.devices()],
        "seed": cfg.seed,
        "deterministic_data": bool(args.deterministic_data),
        "deterministic_data_seed": getattr(cfg.data, "seed_base", None),
        "batch_hash": batch_hash,
        "batch_shapes": batch_shapes,
        "metrics": None,
    }
    if args.batch_only:
        return result

    train_state, train_state_sharding = trainer.init_train_state(cfg, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)

    ptrain_step = jax.jit(
        functools.partial(trainer.train_step, cfg),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    with sharding.set_mesh(mesh):
        new_state, info = ptrain_step(train_rng, train_state, batch)
    jax.block_until_ready(new_state)
    reduced_info = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest([info])))
    result["metrics"] = {k: float(v) for k, v in reduced_info.items()}
    return result


def _write_output_json(path: str, result: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic one-batch or one-step training check.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--loader-kind", required=True, choices=["refactor", "standard", "incontext", "fast_mini_v14"])
    parser.add_argument("--batch-only", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--checkpoint-base-dir", default="/tmp/openpi_consistency_checkpoints")
    parser.add_argument("--deterministic-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic-data-seed", type=int)
    args = parser.parse_args()

    try:
        result = _run(args)
    except Exception as exc:
        result = {
            "status": "error",
            "worktree": str(Path.cwd()),
            "config": args.config,
            "trainer": args.trainer,
            "loader_kind": args.loader_kind,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
        }

    if args.output_json:
        _write_output_json(args.output_json, result)
    print("CONSISTENCY_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
