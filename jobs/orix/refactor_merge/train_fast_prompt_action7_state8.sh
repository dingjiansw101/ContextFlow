#!/usr/bin/env bash
#SBATCH --job-name=train_fast_seq_refactor_merge
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --partition=freecycle-h100,freecycle-h200
#SBATCH --qos=freecycle
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@60
#SBATCH --chdir=/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge

set -euo pipefail

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi-refactor_refactor_merge}"
CONFIG="pi0_fast_incontext_prompt_action_7_state_8_train_split"  # on-disk exp/ckpt name (pre-rename)
POLICY_CONFIG="ContextAR"  # get_config lookup (renamed)
EXP_NAME="${CONFIG}_refactor_merge"
ASSETS_BASE_DIR="/mnt/data/u/dingj0b/code/openpi_fastincontext/openpi/assets"
ASSETS_NAME="debug_pi0_fast_libero_incontext_inference"

cd "$REPO"
mkdir -p logs errs

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"
ulimit -n 65536 || true

if ! find "${ASSETS_BASE_DIR}/${ASSETS_NAME}" -maxdepth 4 -type f -name 'norm_stats*' 2>/dev/null | grep -q .; then
    echo "Missing ${ASSETS_BASE_DIR}/${ASSETS_NAME} norm stats; run scripts/compute_norm_stats.py --config-name ${POLICY_CONFIG} first." >&2
    exit 66
fi

pid=""
forward_term() {
    if [[ -n "${pid}" ]]; then
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" || true
    fi
}
trap forward_term SIGTERM

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py "$POLICY_CONFIG" \
        --project-name=openpi \
        --exp-name="$EXP_NAME" \
        --assets-base-dir="$ASSETS_BASE_DIR" \
        --resume &
pid=$!
wait "$pid"
