#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:?CONFIG must be set by the wrapper}"
: "${ASSETS_BASE_DIR:?ASSETS_BASE_DIR must be set by the wrapper}"
: "${ASSETS_NAME:?ASSETS_NAME must be set by the wrapper}"

REPO="${REPO:-/mnt/data/u/dingj0b/code/openpi-refactor_v18_raw_images}"
EXP_NAME="${EXP_NAME:-${CONFIG}_raw_images}"
NUM_WORKERS="${NUM_WORKERS:-32}"

cd "$REPO"
mkdir -p logs errs

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"
ulimit -n 65536 || true

if ! find "${ASSETS_BASE_DIR}/${ASSETS_NAME}" -maxdepth 4 -type f -name 'norm_stats*' 2>/dev/null | grep -q .; then
    echo "Missing ${ASSETS_BASE_DIR}/${ASSETS_NAME} norm stats; run scripts/compute_norm_stats.py --config-name ${CONFIG} first." >&2
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
    uv run scripts/train.py "$CONFIG" \
        --project-name=openpi \
        --exp-name="$EXP_NAME" \
        --assets-base-dir="$ASSETS_BASE_DIR" \
        --num-workers="$NUM_WORKERS" \
        --resume &
pid=$!
wait "$pid"
