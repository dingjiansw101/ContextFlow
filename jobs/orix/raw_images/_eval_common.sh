#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:?CONFIG must be set by the wrapper}"
: "${POLICY_CONFIG:?POLICY_CONFIG must be set by the wrapper}"
: "${ASSETS_BASE_DIR:?ASSETS_BASE_DIR must be set by the wrapper}"
: "${JOB_TAG:?JOB_TAG must be set by the wrapper}"

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi-refactor_v18_raw_images}"
EXP_NAME="${EXP_NAME:-${CONFIG}_raw_images}"
ITER="${ITER:-19999}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${CONFIG}/${EXP_NAME}/${ITER}}"
RUN_ID="${RUN_ID:-orix_raw_images_${JOB_TAG}_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
TASK_SPLIT="${TASK_SPLIT:-split0}"
SUITE_LIST="${SUITE_LIST:-spatial object goal 10}"
DISABLE_CUDNN_FMHA="${DISABLE_CUDNN_FMHA:-0}"
SKIP_LOG_TO_SHEET="${SKIP_LOG_TO_SHEET:-1}"
LiberoVenv="${LiberoVenv:-/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge/examples/libero/.venv}"

cd "$REPO"
mkdir -p logs errs

export REPO
export LiberoVenv
export TASK_SPLIT
export SUITE_LIST
export ASSETS_BASE_DIR
export DISABLE_CUDNN_FMHA
export SKIP_LOG_TO_SHEET

bash jobs/local/eval_pi0_libero_incontext_unseen.sh \
    "$EXP_NAME" "$POLICY_CONFIG" "$CHECKPOINT_DIR" "$RUN_ID"
