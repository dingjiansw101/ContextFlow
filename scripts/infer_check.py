"""Deterministic inference consistency fixture for the aloha cache-free comparison.

Drives create_trained_policy_incontext directly (no server) with a fixed synthetic
observation. Demo selection is deterministic on split="test" (first candidate episode of
the task) on both the legacy cache-based serving path (aloha-dev) and the cache-free
InjectDemoFromCustomDataset path, so the whole transform + model chain must produce the
same actions for the same checkpoint.

Run with JAX_DEFAULT_MATMUL_PRECISION=float32 in the environment for stable matmuls.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task-index", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy_incontext(
        train_config,
        args.ckpt,
        inference_dtype="float32",
    )

    rng = np.random.default_rng(0)
    obs = {
        "state": rng.uniform(-1.0, 1.0, size=(14,)).astype(np.float32),
        "images": {
            "cam_high": rng.integers(0, 256, size=(3, 480, 640), dtype=np.uint8),
            "cam_left_wrist": rng.integers(0, 256, size=(3, 480, 640), dtype=np.uint8),
            "cam_right_wrist": rng.integers(0, 256, size=(3, 480, 640), dtype=np.uint8),
        },
        "prompt": "perform the task",
        "task_index": args.task_index,
        "split": "test",
    }

    result = policy.infer(obs)
    actions = np.asarray(result["actions"], dtype=np.float64)

    fixture_hash = hashlib.sha256()
    fixture_hash.update(obs["state"].tobytes())
    for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        fixture_hash.update(obs["images"][cam].tobytes())

    out = {
        "status": "ok",
        "config": args.config,
        "ckpt": args.ckpt,
        "task_index": args.task_index,
        "fixture_hash": fixture_hash.hexdigest(),
        "actions_shape": list(actions.shape),
        "actions_hash_f32": hashlib.sha256(actions.astype(np.float32).tobytes()).hexdigest(),
        "actions_stats": {
            "min": float(actions.min()),
            "max": float(actions.max()),
            "mean": float(actions.mean()),
            "norm": float(np.linalg.norm(actions)),
        },
        "actions_first_row": [float(x) for x in actions.reshape(actions.shape[0], -1)[0][:16]],
        "extra_keys": sorted(k for k in result if k != "actions"),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    np.save(str(args.output_json) + ".npy", actions)
    print("INFER_RESULT " + json.dumps(out, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
