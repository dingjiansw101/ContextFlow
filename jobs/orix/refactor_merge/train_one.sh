#!/usr/bin/env bash
#SBATCH --job-name=openpi_refactor_merge_train
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --partition=batch-h100
#SBATCH --qos=batch
#SBATCH --chdir=/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge

set -euo pipefail

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge}"
CONFIG="${CONFIG:?set CONFIG}"
EXP_NAME="${EXP_NAME:-${CONFIG}_refactor_merge}"
NUM_WORKERS="${NUM_WORKERS:-32}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-}"
DISABLE_CUDNN_FMHA="${DISABLE_CUDNN_FMHA:-0}"

cd "$REPO"
mkdir -p logs errs

export PATH="$HOME/.local/bin:$PATH"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
case "$DISABLE_CUDNN_FMHA" in
    1|true|TRUE|yes|YES)
        export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_cudnn_fmha=false"
        ;;
esac

if [ "$(ulimit -Sn)" -lt 65536 ]; then
    ulimit -n 65536
fi

PYTHONPATH=src uv run python - "$CONFIG" "$ASSETS_BASE_DIR" <<'PY'
import dataclasses
from pathlib import Path
import sys

from openpi.training import config as _config

cfg = _config.get_config(sys.argv[1])
if sys.argv[2]:
    cfg = dataclasses.replace(cfg, assets_base_dir=sys.argv[2])
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
if data_cfg.repo_id != "fake" and data_cfg.norm_stats is None:
    raise SystemExit(f"Missing norm stats for asset_id={data_cfg.asset_id} under {cfg.assets_dirs}")

paths = []
for attr in ("episode_to_indexes_file", "states_cache_path", "actions_cache_path", "task_to_episode", "task_to_episode_path"):
    value = getattr(cfg.data, attr, None)
    if value:
        paths.append(Path(value))

missing = [str(path) for path in paths if not path.exists()]
if missing:
    raise SystemExit("Missing metadata/cache paths:\n" + "\n".join(missing))

print(f"Preflight OK for {cfg.name}: assets={cfg.assets_dirs}, asset_id={data_cfg.asset_id}")
PY

trap 'kill -TERM "$pid" 2>/dev/null; wait "$pid"' SIGTERM
train_args=(
    "$CONFIG"
    --project-name=openpi
    --exp-name="$EXP_NAME"
    --resume
    --num-workers="$NUM_WORKERS"
    --fsdp-devices="$FSDP_DEVICES"
)
if [ -n "$ASSETS_BASE_DIR" ]; then
    train_args+=(--assets-base-dir="$ASSETS_BASE_DIR")
fi

uv run scripts/train.py "${train_args[@]}" &
pid=$!
wait "$pid"
