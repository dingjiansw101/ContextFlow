"""Run a fresh one-GPU experiment using an already prepared reproduction dataset."""

import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(sys.argv[1]).resolve()
    experiment = sys.argv[2]
    if not experiment.replace("_", "").isalnum():
        raise ValueError("Experiment must contain only letters, numbers and underscores")
    worker_path = root / "launcher/login-pipeline-9965b02da61c-1acc95a2a1e4/contextflow_repro_worker.py"
    spec = importlib.util.spec_from_file_location("prepared_worker", worker_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.Worker(root, experiment)
    receipt = root / "receipts" / (experiment + ".json")
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists():
            raise FileExistsError(receipt)
        prepared = worker.require_complete("prepare")
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worker.source, text=True).strip()
        if actual != prepared["source_sha"] or module.sha256(worker.source / "scripts/train.py") != prepared["runtime_source_sha256"]:
            raise RuntimeError("Prepared training source changed")
        worker.validate()
        worker.record.update(experiment=experiment, gpu_count=1, num_workers=8, batch_size=32,
                             num_train_steps=20000, seed=42, source_sha=actual)
        module.save_json(receipt, worker.record)
        try:
            # Keep batch, optimizer, data split, model and seed unchanged. Only worker
            # count and offline I/O differ from the released training configuration.
            entry = '''import dataclasses, runpy, sys
import jax
from openpi.training.config import get_config, DataConfig
assert len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu"
cfg = get_config("ContextFlow")
assert cfg.batch_size == 32 and cfg.num_train_steps == 20000 and cfg.seed == 42
cfg = dataclasses.replace(cfg, exp_name=sys.argv[1], num_workers=8, wandb_enabled=False,
    data=dataclasses.replace(cfg.data, base_config=dataclasses.replace(cfg.data.base_config or DataConfig(), local_files_only=True)))
if sys.argv[2] == "smoke":
    cfg = dataclasses.replace(cfg, num_train_steps=2, save_interval=1, log_interval=1)
assert not cfg.checkpoint_dir.exists(), str(cfg.checkpoint_dir)
runpy.run_path("scripts/train.py", run_name="contextflow_pi1_entry")["main"](cfg)
'''
            env = dict(worker.env, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", UV_OFFLINE="1")
            command = [worker.uv, "run", "--no-project", "--python", worker.python, "python", "-c", entry]
            worker.run(command + [experiment + "_smoke", "smoke"], experiment + "-smoke", env=env)
            worker.run(command + [experiment, "train"], experiment + "-train", env=env)
            checkpoint = worker.source / "checkpoints/ContextFlow" / experiment / "19999"
            metadata = json.loads((checkpoint / "_CHECKPOINT_METADATA").read_text())
            if not metadata.get("commit_timestamp_nsecs"):
                raise RuntimeError("Final checkpoint has not committed")
            norms = list((checkpoint / "assets").rglob("norm_stats.json"))
            original_norm = worker.source / "assets/ContextFlow/physical-intelligence/libero/norm_stats.json"
            if len(norms) != 1 or json.loads(norms[0].read_text()) != json.loads(original_norm.read_text()):
                raise RuntimeError("Checkpoint normalization changed")
            files = sorted(p for component in ("params", "assets") for p in (checkpoint / component).rglob("*") if p.is_file())
            if not (checkpoint / "params/manifest.ocdbt").is_file():
                raise RuntimeError("Parameter manifest missing")
            files.append(checkpoint / "_CHECKPOINT_METADATA")
            with (checkpoint / "EVAL_SHA256SUMS").open("x") as manifest:
                for path in files:
                    manifest.write(f"{module.sha256(path)}  {path.relative_to(checkpoint)}\n")
            worker.record.update(status="completed", checkpoint=str(checkpoint),
                                 checkpoint_norm_stats_sha256=module.sha256(norms[0]),
                                 inference_bytes=sum(p.stat().st_size for p in files))
        except BaseException as error:
            worker.record.update(status="failed", error=str(error))
            raise
        finally:
            worker.record["finished_at"] = module.utc()
            module.save_json(receipt, worker.record)


if __name__ == "__main__":
    main()
