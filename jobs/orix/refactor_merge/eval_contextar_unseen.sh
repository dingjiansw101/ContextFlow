#!/usr/bin/env bash
#SBATCH --job-name=eval_ContextAR_unseen
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
# New-name orix unseen eval for ContextAR (LIBERO pi0-FAST in-context).
# Renamed from eval_fast_prompt_action7_state8_unseen.sh. Historically served the untouched
# sibling `pi0_fast_incontext_prompt_action_7_state_8_inference`; here we serve the NEW name
# `ContextAR` (serve-identical: same dims, same assets key debug_pi0_fast_libero_incontext_inference,
# policy dataset spans all episodes so train-time remove_task_list is irrelevant). Requires the
# orix checkout on the rename branch so get_config('ContextAR') resolves.
set -euo pipefail

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge}"
CONFIG=ContextAR
POLICY_CONFIG=ContextAR
EXP_NAME="${EXP_NAME:-ContextAR_base}"
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
