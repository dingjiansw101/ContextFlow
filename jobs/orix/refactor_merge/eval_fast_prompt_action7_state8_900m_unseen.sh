#!/usr/bin/env bash
#SBATCH --job-name=eval_fast_seq_900m_refactor_merge
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

REPO=/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge
CONFIG="pi0_fast_incontext_prompt_action_7_state_8_train_split_900m"
POLICY_CONFIG="pi0_fast_incontext_prompt_action_7_state_8_inference_900m"
EXP_NAME="${CONFIG}_refactor_merge"
ITER="${ITER:-19999}"
CHECKPOINT_DIR="checkpoints/${CONFIG}/${EXP_NAME}/${ITER}"
RUN_ID="${RUN_ID:-orix_refactor_merge_$(date +%Y%m%d)}"
TASK_SPLIT="${TASK_SPLIT:-split0}"
SUITE_LIST="${SUITE_LIST:-spatial object goal 10}"
ASSETS_BASE_DIR="/mnt/data/u/dingj0b/code/openpi_fastincontext/openpi/assets"
DISABLE_CUDNN_FMHA="${DISABLE_CUDNN_FMHA:-0}"

cd "$REPO"
mkdir -p logs errs

bash jobs/orix/refactor_merge/eval_libero_unseen_one.sh \
    "$REPO" "$CONFIG" "$POLICY_CONFIG" "$EXP_NAME" "$CHECKPOINT_DIR" "$RUN_ID" "$TASK_SPLIT" "$SUITE_LIST" "$ASSETS_BASE_DIR" "$DISABLE_CUDNN_FMHA"
