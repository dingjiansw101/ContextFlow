#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <run-name> <inference-config> <checkpoint-dir> [run-id]" >&2
    echo "  <run-name>           directory stem under logs/ and data/libero/ (e.g. pi0_libero_incontextv18..._orix)" >&2
    echo "  <inference-config>   policy.config arg for serve_policy_incontext.py (typically <CONFIG>_inference)" >&2
    echo "  <checkpoint-dir>     orbax checkpoint directory" >&2
    echo "  [run-id]             tag for outputs; defaults to orix_<date>" >&2
    echo "env overrides: TASK_SPLIT (default split0), ASSETS_NAME, SUITE_LIST, PORT, CUDA_VISIBLE_DEVICES, LiberoVenv, ProjectPython" >&2
    exit 64
fi

NAME="$1"
CONFIG_NAME="$2"
CHECKPOINT_DIR="$3"
RUN_ID="${4:-orix_$(date +%Y%m%d)}"

REPO="${REPO:-$PWD}"
cd "$REPO"

TASK_SPLIT="${TASK_SPLIT:-split0}"
ASSETS_NAME="${ASSETS_NAME:-pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor}"
LOG_DIR="logs/${NAME}/${RUN_ID}"
VIDEO_DIR="data/libero/${NAME}/${RUN_ID}"
ProjectPython="${ProjectPython:-}"
LiberoVenv="${LiberoVenv:-examples/libero/.venv}"
SERVER_LOG="${LOG_DIR}/${SERVER_LOG_STEM:-server_weight_float32}.log"

mkdir -p "$LOG_DIR" "$VIDEO_DIR"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "checkpoint dir not found: $CHECKPOINT_DIR" >&2
    exit 66
fi
if ! find "assets/${ASSETS_NAME}" -maxdepth 4 -type f -name 'norm_stats*' 2>/dev/null | grep -q .; then
    echo "Missing assets/${ASSETS_NAME} norm stats on eval machine." >&2
    exit 66
fi

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

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting in-context policy server for ${NAME} (config=${CONFIG_NAME}) on port ${PORT}"
if [ -n "$ProjectPython" ]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
        "$ProjectPython" scripts/serve_policy_incontext.py --port "$PORT" \
            policy:checkpoint --policy.inference_dtype=float32 \
            --policy.config="$CONFIG_NAME" --policy.dir="$CHECKPOINT_DIR" \
            >"$SERVER_LOG" 2>&1 &
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
        uv run scripts/serve_policy_incontext.py --port "$PORT" \
            policy:checkpoint --policy.inference_dtype=float32 \
            --policy.config="$CONFIG_NAME" --policy.dir="$CHECKPOINT_DIR" \
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

if [ ! -x "${LiberoVenv}/bin/python" ]; then
    echo "LIBERO python not found: ${LiberoVenv}/bin/python" >&2
    exit 66
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

run_requested_suite() {
    case "$1" in
        spatial|libero_spatial) run_suite libero_spatial spatial ;;
        object|libero_object) run_suite libero_object object ;;
        goal|libero_goal) run_suite libero_goal goal ;;
        10|libero_10) run_suite libero_10 10 ;;
        *) echo "unsupported suite: $1" >&2; exit 64 ;;
    esac
}

for suite in ${SUITE_LIST:-spatial object goal 10}; do
    run_requested_suite "$suite"
done

echo "Completed held-out eval for ${NAME}; results in ${LOG_DIR}"

# Auto-sync results to the openpi LIBERO eval tracker (skipped when SKIP_LOG_TO_SHEET=1).
if [ "${SKIP_LOG_TO_SHEET:-0}" != "1" ] && command -v claude >/dev/null 2>&1; then
    TRAINING_MACHINE="${TRAINING_MACHINE:-orix}"
    EVAL_MACHINE="${EVAL_MACHINE:-orix}"
    SHEET_URL="https://docs.google.com/spreadsheets/d/16It_o0GO_eYTpek65dSKr3sB0TOc_4FXZ5Uqwp9gKjU/edit?gid=715723535#gid=715723535"
    claude -p "/log-to-sheet Read eval results from ${LOG_DIR}/*_results.json and sync to ${SHEET_URL} tab 'pi0 libero'. Do not overwrite existing rows. Include config name=${NAME}, checkpoint path=${CHECKPOINT_DIR}, eval log path=${LOG_DIR}, Training Machine=${TRAINING_MACHINE}, Eval Machine=${EVAL_MACHINE} metadata." \
        || echo "log-to-sheet sync failed (non-fatal); results still at ${LOG_DIR}" >&2
fi
