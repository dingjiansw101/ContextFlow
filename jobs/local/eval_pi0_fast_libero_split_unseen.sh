#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <split-id 0..2> <checkpoint-dir> [run-id]" >&2
    exit 64
fi

SPLIT_ID="$1"
CHECKPOINT_DIR="$2"
RUN_ID="${3:-kw60803_20260501}"

case "$SPLIT_ID" in
    0|1|2) ;;
    *) echo "unsupported split id: $SPLIT_ID" >&2; exit 64 ;;
esac

REPO="${REPO:-$PWD}"
cd "$REPO"

NAME="pi0_fast_libero_split${SPLIT_ID}"
TASK_SPLIT="split${SPLIT_ID}"
LOG_DIR="logs/${NAME}/${RUN_ID}"
VIDEO_DIR="data/libero/${NAME}/${RUN_ID}"

mkdir -p "$LOG_DIR" "$VIDEO_DIR"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "checkpoint dir not found: $CHECKPOINT_DIR" >&2
    exit 66
fi
if [ ! -f assets/pi0_fast_libero_split0/physical-intelligence/libero/norm_stats.json ]; then
    echo "Missing assets/pi0_fast_libero_split0 norm stats on eval machine." >&2
    exit 66
fi

PORT="${PORT:-$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)}"

export PATH="$HOME/.local/bin:$PATH"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting policy server for ${NAME} on port ${PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/serve_policy.py --port "$PORT" \
        policy:checkpoint --policy.config="$NAME" --policy.dir="$CHECKPOINT_DIR" \
        >"${LOG_DIR}/server_weight_float32.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "policy server exited early; see ${LOG_DIR}/server_weight_float32.log" >&2
        exit 67
    fi
    if grep -q "Creating server" "${LOG_DIR}/server_weight_float32.log" 2>/dev/null; then
        break
    fi
    sleep 2
done
if ! grep -q "Creating server" "${LOG_DIR}/server_weight_float32.log" 2>/dev/null; then
    echo "policy server did not become ready; see ${LOG_DIR}/server_weight_float32.log" >&2
    exit 68
fi

source examples/libero/.venv/bin/activate
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
    echo "Starting ${suite} held-out eval for ${NAME} (${TASK_SPLIT})"
    env -u CUDA_VISIBLE_DEVICES python examples/libero/main_incontext_unseen.py \
        --args.host 127.0.0.1 \
        --args.port "$PORT" \
        --args.task_suite_name "$suite" \
        --args.task_split "$TASK_SPLIT" \
        --args.video_out_path "${VIDEO_DIR}/${stem}" \
        --args.results_out_path "$results_path" \
        >"${LOG_DIR}/${stem}_unseen_weight_float32.log" 2>&1
}

run_suite libero_spatial spatial
run_suite libero_object object
run_suite libero_goal goal
run_suite libero_10 10

echo "Completed held-out eval for ${NAME}; results in ${LOG_DIR}"
