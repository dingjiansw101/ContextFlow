#!/usr/bin/env bash
# Unseen in-context LIBERO eval for the RENAMED configs (ContextFlow / ContextFlow_Plain /
# ContextAR) on visioncair, using BORROWED venvs so the locked rename worktree needs no
# `uv sync`.
#
# Mirrors jobs/local/eval_unseen_A_v18_local_normdemo_fix.sh, adapted to:
#   - serve the NEW config name via a borrowed JAX venv + PYTHONPATH=src (renamed code)
#   - use a borrowed LIBERO client venv + the main checkout's third_party/libero
#   - system nvidia EGL (visioncair has libEGL_nvidia + 10_nvidia.json; no ~/nvidia-egl)
#   - policy server does NOT preallocate GPU mem (serve_policy.py forces
#     XLA_PYTHON_CLIENT_PREALLOCATE=false), so multiple suites can share one GPU.
#
# usage: eval_incontext_unseen_local.sh <run-name> <policy-config> <checkpoint-dir> [run-id]
# env overrides: REPO TASK_SPLIT SUITE_LIST GPUS SERVER_PY CLIENT_PY LIBERO_TP SKIP_SHEET_SYNC
set -uo pipefail

NAME="${1:?run-name}"; POLICY_CONFIG="${2:?policy-config}"; CKPT_DIR="${3:?checkpoint-dir}"
RUN_ID="${4:-visioncair_$(date +%Y%m%d)}"

REPO="${REPO:-$PWD}"
TASK_SPLIT="${TASK_SPLIT:-split0}"
SUITE_LIST="${SUITE_LIST:-spatial object goal 10}"
GPUS=(${GPUS:-0 1 2 4})
SERVER_PY="${SERVER_PY:-/home/dingj0b/code/openpi/.venv/bin/python}"
CLIENT_PY="${CLIENT_PY:-/home/dingj0b/code/openpi/examples/libero/.venv/bin/python}"
LIBERO_TP="${LIBERO_TP:-/home/dingj0b/code/openpi/third_party/libero}"
BUNDLED_CUDA_NVCC="${BUNDLED_CUDA_NVCC:-$(dirname "$(dirname "$SERVER_PY")")/lib/python3.11/site-packages/nvidia/cuda_nvcc}"

cd "$REPO"
LOG_DIR="logs/${NAME}/${RUN_ID}"
VIDEO_DIR="data/libero_incontext/${NAME}/${RUN_ID}"
mkdir -p "$LOG_DIR" "$VIDEO_DIR"

[ -d "$CKPT_DIR" ]   || { echo "ERROR: checkpoint dir missing: $CKPT_DIR" >&2; exit 66; }
[ -x "$SERVER_PY" ]  || { echo "ERROR: server python missing: $SERVER_PY" >&2; exit 66; }
[ -x "$CLIENT_PY" ]  || { echo "ERROR: client python missing: $CLIENT_PY" >&2; exit 66; }

# --- server (JAX) env ---
export JAX_DEFAULT_MATMUL_PRECISION=float32
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=$BUNDLED_CUDA_NVCC"
export PATH="$BUNDLED_CUDA_NVCC/bin:$PATH"
# --- simulator (EGL) env: system nvidia vendor ---
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"

suite_full() { case "$1" in
    spatial) echo libero_spatial;; object) echo libero_object;;
    goal) echo libero_goal;; 10) echo libero_10;; *) echo "$1";; esac; }

SUITES=($SUITE_LIST)
PIDS=(); trap 'kill $(jobs -p) 2>/dev/null || true' EXIT
i=0
for s in "${SUITES[@]}"; do
    SUITE=$(suite_full "$s"); STEM="$s"
    GPU=${GPUS[$(( i % ${#GPUS[@]} ))]}
    RES="$LOG_DIR/${STEM}_unseen_results.json"
    if [ -s "$RES" ]; then echo "[$SUITE] existing result -> skip"; i=$((i+1)); continue; fi
    PORT=$("$SERVER_PY" -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
    echo "[$SUITE] GPU $GPU port $PORT -> $RES"
    (
        CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="src:packages/openpi-client/src" \
            "$SERVER_PY" scripts/serve_policy.py --loader=INCONTEXT --port "$PORT" policy:checkpoint \
            --policy.inference-dtype=float32 --policy.config="$POLICY_CONFIG" --policy.dir="$CKPT_DIR" \
            > "$LOG_DIR/${STEM}_server.log" 2>&1 &
        SPID=$!
        trap 'kill $SPID 2>/dev/null || true' EXIT
        for _ in $(seq 1 300); do
            kill -0 $SPID 2>/dev/null || { echo "[$SUITE] server exited early; see ${STEM}_server.log" >&2; exit 67; }
            grep -q "Creating server" "$LOG_DIR/${STEM}_server.log" 2>/dev/null && break
            sleep 2
        done
        grep -q "Creating server" "$LOG_DIR/${STEM}_server.log" 2>/dev/null || { echo "[$SUITE] server not ready" >&2; exit 68; }
        # robosuite substring-parses CUDA_VISIBLE_DEVICES -> must be unset for the sim
        env -u CUDA_VISIBLE_DEVICES PYTHONPATH="$PWD:$PWD/packages/openpi-client/src:$LIBERO_TP" \
            "$CLIENT_PY" examples/libero/main_incontext_unseen.py \
            --args.host 127.0.0.1 --args.port "$PORT" \
            --args.task_suite_name "$SUITE" --args.task_split "$TASK_SPLIT" \
            --args.video_out_path "$VIDEO_DIR/$STEM" \
            --args.results_out_path "$RES" \
            > "$LOG_DIR/${STEM}_client.log" 2>&1
        echo "[$SUITE] client done"
    ) &
    PIDS+=($!); i=$((i+1))
done

FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done

echo "=== results for ${NAME} (${RUN_ID}) ==="
for s in "${SUITES[@]}"; do
    RES="$LOG_DIR/${s}_unseen_results.json"
    if [ -f "$RES" ]; then
        "$SERVER_PY" -c "import json;d=json.load(open('$RES'));s=d['summary'];print('$s unseen_SR=%s (%s/%s) tasks=%s'%(s.get('unseen_success_rate'),s.get('total_successes'),s.get('total_episodes'),s.get('num_unseen_tasks')))"
    else
        echo "$s MISSING"; FAIL=1
    fi
done

# Sync eval results to the rename-validation sheet. Default-skipped here: this background
# job performs the sheet sync + refactor/ECCV comparison itself (SKIP_SHEET_SYNC=1). Set
# SKIP_SHEET_SYNC=0 to let the script auto-sync via the log-to-sheet skill instead.
if [ "${SKIP_SHEET_SYNC:-1}" != "1" ] && command -v claude >/dev/null 2>&1; then
    claude -p "/log-to-sheet Read eval results from ${LOG_DIR}/*_results.json and sync to https://docs.google.com/spreadsheets/d/1OPys1ohYcw0lUdGW3W1c02dp0lMm-AP-NnYWhG8IecI/edit?gid=1035082397#gid=1035082397 tab 'rename'. Do not overwrite existing rows. Include config name (${NAME}), policy config (${POLICY_CONFIG}), checkpoint path (${CKPT_DIR}), eval log path (${LOG_DIR}), Training Machine, Eval Machine: visioncair." \
        || echo "(log-to-sheet failed; results JSONs are in ${LOG_DIR})"
fi

echo "=== done (FAIL=$FAIL); logs in $LOG_DIR ==="
exit $FAIL
