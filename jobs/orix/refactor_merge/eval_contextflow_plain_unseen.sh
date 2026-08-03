#!/usr/bin/env bash
#SBATCH --job-name=eval_ContextFlow_Plain_unseen
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --partition=batch-h100
#SBATCH --qos=batch
#SBATCH --chdir=/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge
#
# New-name orix unseen eval for ContextFlow_Plain.
# Renamed from eval_v12_refactor_incontext_unseen.sh (config
# pi0_libero_refactor_incontextv12_..._dataset_refactor -> ContextFlow_Plain, exp
# _refactor_merge -> ContextFlow_Plain_refactor_merge). Requires the orix checkout on the
# rename branch (get_config('ContextFlow_Plain') + ./assets/ContextFlow_Plain/).
set -euo pipefail

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge}"
CONFIG=ContextFlow_Plain
POLICY_CONFIG=ContextFlow_Plain
EXP_NAME="${EXP_NAME:-ContextFlow_Plain_refactor_merge}"
ITER="${ITER:-19999}"
CHECKPOINT_DIR="checkpoints/${CONFIG}/${EXP_NAME}/${ITER}"
RUN_ID="${RUN_ID:-orix_rename_$(date +%Y%m%d)}"
TASK_SPLIT="${TASK_SPLIT:-split0}"
SUITE_LIST="${SUITE_LIST:-spatial object goal 10}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-$REPO/assets}"
DISABLE_CUDNN_FMHA="${DISABLE_CUDNN_FMHA:-0}"

cd "$REPO"; mkdir -p logs errs
export REPO TASK_SPLIT SUITE_LIST ASSETS_BASE_DIR DISABLE_CUDNN_FMHA

bash jobs/local/eval_pi0_libero_incontext_unseen.sh \
    "$EXP_NAME" "$POLICY_CONFIG" "$CHECKPOINT_DIR" "$RUN_ID"
