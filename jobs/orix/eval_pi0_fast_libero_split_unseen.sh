#!/usr/bin/env bash
#SBATCH --partition=batch-h100
#SBATCH --qos=batch
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err

set -euo pipefail

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi_libero/openpi}"
SPLIT="${SPLIT:?set SPLIT to 1 or 2}"
RUN_ID="${RUN_ID:-orix_20260501}"

case "$SPLIT" in
    1|2) ;;
    *) echo "unsupported split for this eval wrapper: $SPLIT" >&2; exit 64 ;;
esac

cd "$REPO"
mkdir -p logs errs

export PATH="$HOME/.local/bin:$PATH"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-$HOME/nvidia-egl}"
export LD_LIBRARY_PATH="$HOME/nvidia-egl/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

NAME="pi0_fast_libero_split${SPLIT}"
CHECKPOINT_DIR="checkpoints/${NAME}/${NAME}/19999"

bash jobs/local/eval_pi0_fast_libero_split_unseen.sh "$SPLIT" "$CHECKPOINT_DIR" "$RUN_ID"
