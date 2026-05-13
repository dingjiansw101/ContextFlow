#!/usr/bin/env bash
#SBATCH --job-name=pi0_libero_incontextv18_low_mem_finetune_sample_frames8_orix
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --partition=batch-h100
#SBATCH --qos=batch
#SBATCH --chdir=/home/dingj0b/code/openpi_libero/openpi

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs errs

export PATH="$HOME/.local/bin:$PATH"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"

CONFIG="pi0_libero_incontextv18_low_mem_finetune_sample_frames8"
EXP_NAME="${CONFIG}_orix"
ASSETS_NAME="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor"

if ! find "assets/${ASSETS_NAME}" -maxdepth 4 -type f -name 'norm_stats*' 2>/dev/null | grep -q .; then
    echo "Missing assets/${ASSETS_NAME} norm stats; run scripts/compute_norm_stats.py --config-name ${CONFIG} first." >&2
    exit 66
fi

trap 'kill -TERM "$pid" 2>/dev/null; wait "$pid"' SIGTERM
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py "$CONFIG" \
        --project-name=openpi \
        --exp-name="$EXP_NAME" \
        --resume &
pid=$!
wait "$pid"
