#!/usr/bin/env bash
#SBATCH --job-name=eval_v18_plus_libero90_refactor_merge
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --partition=batch-h100
#SBATCH --qos=batch
#SBATCH --chdir=/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge

set -euo pipefail

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge}"
CONFIG="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_plus_libero90"
POLICY_CONFIG="ContextFlow_plus_libero90"  # served config (renamed); CONFIG stays old for the on-disk ckpt path
EXP_NAME="${CONFIG}_refactor_merge"
ITER="${ITER:-69999}"
CHECKPOINT_DIR="checkpoints/${CONFIG}/${EXP_NAME}/${ITER}"
RUN_ID="${RUN_ID:-orix_refactor_merge_$(date +%Y%m%d)}"
TASK_SPLIT="${TASK_SPLIT:-split0}"
SUITE_LIST="${SUITE_LIST:-spatial object goal 10}"
ASSETS_BASE_DIR="/mnt/data/u/dingj0b/code/openpi_libero/openpi/assets"
DISABLE_CUDNN_FMHA="${DISABLE_CUDNN_FMHA:-0}"

cd "$REPO"
mkdir -p logs errs

export REPO
export TASK_SPLIT
export SUITE_LIST
export ASSETS_BASE_DIR
export DISABLE_CUDNN_FMHA

bash jobs/local/eval_pi0_libero_incontext_unseen.sh \
    "$EXP_NAME" "$POLICY_CONFIG" "$CHECKPOINT_DIR" "$RUN_ID"
