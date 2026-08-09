#!/usr/bin/env bash
#SBATCH --job-name=train_ContextFlow_regen
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --partition=freecycle-h100,freecycle-h200
#SBATCH --qos=freecycle
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@60
#SBATCH --chdir=/mnt/data/u/dingj0b/code/openpi-rename
#
# ContextFlow (renamed pi0_libero_incontextv18_..._sample_frames8) trained with the
# REGENERATED metadata (generate_task_to_index.py --config pi0_libero), 2 GPUs.
# Mirrors jobs/orix/refactor_merge/train_v18_sample_frames8.sh with:
#   - new config/assets names (ContextFlow / ContextFlow_Plain)
#   - gpu:2 + mem 200G (was gpu:4 / 400G)
#   - SEED env -> --seed + per-seed exp name
#   - --keep-period=100000 (avoid keeping every 5k-step checkpoint)
#   - pip cuDNN preferred over the broken system cuDNN 9.10.1
# usage: sbatch --export=ALL,SEED=1 jobs/orix/rename/train_contextflow_regen.sh

set -euo pipefail

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi-rename}"
CONFIG="ContextFlow"
SEED="${SEED:?set SEED via sbatch --export=ALL,SEED=N}"
EXP_NAME="${EXP_NAME:-ContextFlow_regen_meta_seed${SEED}}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-$REPO/assets}"
ASSETS_NAME="ContextFlow_Plain"

cd "$REPO"
mkdir -p logs errs

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"
# orix system cuDNN 9.10.1 breaks JAX (CUDNN_STATUS_NOT_SUPPORTED); prefer pip cuDNN.
CUDNN_LIB="$REPO/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib"
[ -d "$CUDNN_LIB" ] && export LD_LIBRARY_PATH="$CUDNN_LIB:${LD_LIBRARY_PATH:-}"
ulimit -n 65536 || true

if ! find "${ASSETS_BASE_DIR}/${ASSETS_NAME}" -maxdepth 4 -type f -name 'norm_stats*' 2>/dev/null | grep -q .; then
    echo "Missing ${ASSETS_BASE_DIR}/${ASSETS_NAME} norm stats." >&2
    exit 66
fi
# The task->episode / episode->indexes tables are derived at runtime from the
# LeRobot dataset metadata (src/openpi/training/lookup_tables.py), so there is
# no precomputed JSON left to check for here.

pid=""
forward_term() {
    if [[ -n "${pid}" ]]; then
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" || true
    fi
}
trap forward_term SIGTERM

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py "$CONFIG" \
        --project-name=openpi \
        --exp-name="$EXP_NAME" \
        --assets-base-dir="$ASSETS_BASE_DIR" \
        --num-workers=16 \
        --seed="$SEED" \
        --keep-period=100000 \
        --resume &
pid=$!
wait "$pid"
