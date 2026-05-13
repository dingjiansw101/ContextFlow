#!/usr/bin/env bash
#SBATCH --job-name=openpi_refactor_merge_eval
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
CONFIG="${CONFIG:?set CONFIG}"
POLICY_CONFIG="${POLICY_CONFIG:-$CONFIG}"
EXP_NAME="${EXP_NAME:-${CONFIG}_refactor_merge}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${CONFIG}/${EXP_NAME}/19999}"
RUN_ID="${RUN_ID:-orix_refactor_merge_$(date +%Y%m%d)}"
TASK_SPLIT="${TASK_SPLIT:-split0}"
SUITE_LIST="${SUITE_LIST:-spatial object goal 10}"
LiberoVenv="${LiberoVenv:-examples/libero/.venv}"
ProjectPython="${ProjectPython:-}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-}"
DISABLE_CUDNN_FMHA="${DISABLE_CUDNN_FMHA:-0}"

cd "$REPO"
mkdir -p logs errs

LOG_DIR="logs/${EXP_NAME}/${RUN_ID}"
VIDEO_DIR="data/libero/${EXP_NAME}/${RUN_ID}"
SERVER_LOG="${LOG_DIR}/${SERVER_LOG_STEM:-server_weight_float32}.log"

mkdir -p "$LOG_DIR" "$VIDEO_DIR"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "checkpoint dir not found: $CHECKPOINT_DIR" >&2
    exit 66
fi
if [ ! -x "${LiberoVenv}/bin/python" ]; then
    echo "LIBERO python not found: ${LiberoVenv}/bin/python" >&2
    exit 66
fi

PYTHONPATH=src uv run python - "$POLICY_CONFIG" "$ASSETS_BASE_DIR" <<'PY'
import dataclasses
import sys

from openpi.training import config as _config

cfg = _config.get_config(sys.argv[1])
if sys.argv[2]:
    cfg = dataclasses.replace(cfg, assets_base_dir=sys.argv[2])
data_cfg = cfg.data.create_policy(cfg.assets_dirs, cfg.model)
if data_cfg.repo_id != "fake" and data_cfg.norm_stats is None:
    raise SystemExit(f"Missing policy norm stats for asset_id={data_cfg.asset_id} under {cfg.assets_dirs}")

print(f"Policy preflight OK for {cfg.name}: assets={cfg.assets_dirs}, asset_id={data_cfg.asset_id}")
PY

if [ -n "$ProjectPython" ]; then
    PORT_PYTHON="$ProjectPython"
else
    PORT_PYTHON="$(command -v python3 || command -v python)"
fi

PORT="${PORT:-$("$PORT_PYTHON" - <<'PY'
import socket

s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)}"

export PATH="$HOME/.local/bin:$PATH"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-$HOME/nvidia-egl}"
export LD_LIBRARY_PATH="$HOME/nvidia-egl/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
case "$DISABLE_CUDNN_FMHA" in
    1|true|TRUE|yes|YES)
        export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_cudnn_fmha=false"
        ;;
esac

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting policy server for ${EXP_NAME} with policy config ${POLICY_CONFIG} on port ${PORT}"
policy_args=(
    --loader=INCONTEXT
    --port "$PORT"
    policy:checkpoint
    --policy.inference-dtype=float32
    --policy.config="$POLICY_CONFIG"
    --policy.dir="$CHECKPOINT_DIR"
)

if [ -n "$ProjectPython" ]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        "$ProjectPython" scripts/serve_policy.py "${policy_args[@]}" \
            >"$SERVER_LOG" 2>&1 &
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        uv run scripts/serve_policy.py "${policy_args[@]}" \
            >"$SERVER_LOG" 2>&1 &
fi
SERVER_PID=$!

for _ in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "policy server exited early; see ${SERVER_LOG}" >&2
        exit 67
    fi
    if grep -q "Creating server" "$SERVER_LOG" 2>/dev/null; then
        break
    fi
    sleep 2
done
if ! grep -q "Creating server" "$SERVER_LOG" 2>/dev/null; then
    echo "policy server did not become ready; see ${SERVER_LOG}" >&2
    exit 68
fi

source "${LiberoVenv}/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:$PWD:$PWD/packages/openpi-client/src:$PWD/third_party/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

run_suite() {
    local suite="$1"
    local stem="$2"
    local results_path="${LOG_DIR}/${stem}_unseen_results.json"
    if [ -s "$results_path" ]; then
        echo "Skipping ${suite}; existing result found at ${results_path}"
        return
    fi
    echo "Starting ${suite} held-out eval for ${EXP_NAME} (${TASK_SPLIT})"
    env -u CUDA_VISIBLE_DEVICES python examples/libero/main_incontext_unseen.py \
        --args.host 127.0.0.1 \
        --args.port "$PORT" \
        --args.task_suite_name "$suite" \
        --args.task_split "$TASK_SPLIT" \
        --args.video_out_path "${VIDEO_DIR}/${stem}" \
        --args.results_out_path "$results_path" \
        >"${LOG_DIR}/${stem}_unseen_weight_float32.log" 2>&1
}

run_requested_suite() {
    case "$1" in
        spatial|libero_spatial) run_suite libero_spatial spatial ;;
        object|libero_object) run_suite libero_object object ;;
        goal|libero_goal) run_suite libero_goal goal ;;
        10|libero_10) run_suite libero_10 10 ;;
        *) echo "unsupported suite: $1" >&2; exit 64 ;;
    esac
}

for suite in $SUITE_LIST; do
    run_requested_suite "$suite"
done

echo "Completed held-out eval for ${EXP_NAME}; results in ${LOG_DIR}"
