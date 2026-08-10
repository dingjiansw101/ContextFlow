#!/usr/bin/env bash
#SBATCH --job-name=eval_ContextFlow_unseen
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
# New-name orix unseen eval for ContextFlow.
# Renamed from eval_v18_sample_frames8_unseen.sh (config
# pi0_libero_incontextv18_low_mem_finetune_sample_frames8 -> ContextFlow, exp _4gpu ->
# ContextFlow_4gpu). Requires the orix checkout to be on the rename branch so that
# get_config('ContextFlow') resolves and ./assets/ContextFlow_Plain/ holds the norm stats.
set -euo pipefail

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge}"
CONFIG=ContextFlow
POLICY_CONFIG=ContextFlow
EXP_NAME="${EXP_NAME:-ContextFlow_4gpu}"
ITER="${ITER:-19999}"
CHECKPOINT_DIR="checkpoints/${CONFIG}/${EXP_NAME}/${ITER}"
RUN_ID="${RUN_ID:-orix_rename_$(date +%Y%m%d)}"
SUITE_LIST="${SUITE_LIST:-spatial object goal 10}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-$REPO/assets}"
DISABLE_CUDNN_FMHA="${DISABLE_CUDNN_FMHA:-0}"

cd "$REPO"; mkdir -p logs errs
export REPO SUITE_LIST ASSETS_BASE_DIR DISABLE_CUDNN_FMHA

bash jobs/local/eval_pi0_libero_incontext_unseen.sh \
    "$EXP_NAME" "$POLICY_CONFIG" "$CHECKPOINT_DIR" "$RUN_ID"
