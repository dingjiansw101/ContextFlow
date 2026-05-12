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

cd "$REPO"
mkdir -p logs errs

export PATH="$HOME/.local/bin:$PATH"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

if [ "$(ulimit -Sn)" -lt 65536 ]; then
    ulimit -n 65536
fi

PYTHONPATH=src uv run python - "$CONFIG" <<'PY'
from pathlib import Path
import sys

from openpi.training import config as _config

cfg = _config.get_config(sys.argv[1])
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
uv run scripts/train.py "$CONFIG" \
    --project-name=openpi \
    --exp-name="$EXP_NAME" \
    --resume \
    --num-workers="$NUM_WORKERS" \
    --fsdp-devices="$FSDP_DEVICES" &
pid=$!
wait "$pid"
