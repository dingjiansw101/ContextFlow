#!/usr/bin/env bash
#SBATCH --job-name=train_v18_sf8_refactor_merge
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --partition=freecycle-h100,freecycle-h200
#SBATCH --qos=freecycle
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@60
#SBATCH --chdir=/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge

set -euo pipefail

REPO=/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge
CONFIG="pi0_libero_incontextv18_low_mem_finetune_sample_frames8"
EXP_NAME="${CONFIG}_refactor_merge"
ASSETS_BASE_DIR="/mnt/data/u/dingj0b/code/openpi_libero/openpi/assets"

cd "$REPO"
mkdir -p logs errs

export PATH="$HOME/.local/bin:$PATH"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"

ulimit -n 65536 || true

PYTHONPATH=src uv run python - "$CONFIG" "$ASSETS_BASE_DIR" <<'PY'
import dataclasses
from pathlib import Path
import sys

from openpi.training import config as _config

cfg = _config.get_config(sys.argv[1])
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
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py "$CONFIG" \
        --project-name=openpi \
        --exp-name="$EXP_NAME" \
        --assets-base-dir="$ASSETS_BASE_DIR" \
        --resume &
pid=$!
wait "$pid"
