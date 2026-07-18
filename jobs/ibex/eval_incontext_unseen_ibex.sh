#!/bin/bash
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=eval_incontext_unseen
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=errs/%j-%x.err
#
# Unseen in-context LIBERO eval for a RENAMED config (ContextFlow / ContextFlow_Plain /
# ContextAR) on ibex. Mirrors the proven ibex recipe
# (openpi_repro_nw16/jobs/ibex/eval_v18_sf8_repro_nw16_unseen.sh): single server, suites run
# sequentially (memory-safe on one GPU). Adapted to serve the NEW config name from the renamed
# code via PYTHONPATH=src + borrowed ibex venvs (no uv sync in this work dir).
#
# Required env: NAME POLICY_CONFIG CKPT_DIR    optional: RUN_ID SUITES REPO SKIP_SHEET_SYNC
#   sbatch --export=ALL,NAME=ContextFlow,POLICY_CONFIG=ContextFlow,CKPT_DIR=/…/19999 jobs/ibex/eval_incontext_unseen_ibex.sh
set -uo pipefail

REPO="${REPO:-/ibex/user/dingj0b/code/openpi_rename_contextflow}"
cd "$REPO"; mkdir -p logs errs

: "${NAME:?set NAME}"; : "${POLICY_CONFIG:?set POLICY_CONFIG}"; CKPT="${CKPT_DIR:?set CKPT_DIR}"
RUN_ID="${RUN_ID:-ibex_$(date +%Y%m%d)}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"; SUITES="${SUITES//,/ }"
[ -d "$CKPT" ] || { echo "ERROR: missing checkpoint $CKPT" >&2; exit 2; }

SERVER_PY=/ibex/user/dingj0b/code/openpi/.venv/bin/python
CLIENT_PY=/ibex/user/dingj0b/code/openpi/examples/libero/.venv/bin/python
LIBERO_TP=/ibex/user/dingj0b/code/openpi/third_party/libero
NVCC=/ibex/user/dingj0b/code/openpi/.venv/lib/python3.11/site-packages/nvidia/cuda_nvcc

# proven ibex EGL + CUDA recipe
export CUDA_HOME=/home/dingj0b/dingjian/cuda-12.1
export PATH="$PATH:/home/dingj0b/dingjian/cuda-12.1/bin:$NVCC/bin:$HOME/.local/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/dingj0b/dingjian/cuda-12.1/lib64:$HOME/.mujoco/mujoco210/bin"
export MUJOCO_GL=egl
export JAX_DEFAULT_MATMUL_PRECISION=float32
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=$NVCC"

LOG_DIR="logs/${NAME}/${RUN_ID}"; mkdir -p "$LOG_DIR"
PORT=$("$SERVER_PY" -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
echo "port=$PORT  config=$POLICY_CONFIG  ckpt=$CKPT"

PYTHONPATH="src:packages/openpi-client/src" "$SERVER_PY" scripts/serve_policy.py \
    --loader=INCONTEXT --port "$PORT" policy:checkpoint \
    --policy.inference-dtype=float32 --policy.config="$POLICY_CONFIG" --policy.dir="$CKPT" \
    > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do
    kill -0 $SERVER_PID 2>/dev/null || { echo "server died early"; tail -25 "$LOG_DIR/server.log"; exit 67; }
    grep -q "Creating server" "$LOG_DIR/server.log" 2>/dev/null && { echo "server ready"; break; }
    sleep 3
done
grep -q "Creating server" "$LOG_DIR/server.log" 2>/dev/null || { echo "server not ready"; tail -25 "$LOG_DIR/server.log"; exit 68; }

FAIL=0
for suite in $SUITES; do
    stem=${suite#libero_}
    RES="$LOG_DIR/${stem}_unseen_results.json"
    if [ -s "$RES" ]; then echo "skip $suite (exists)"; continue; fi
    echo "eval $suite (unseen, split0) ..."
    PYTHONPATH="$PWD:$PWD/packages/openpi-client/src:$LIBERO_TP" "$CLIENT_PY" \
        examples/libero/main_incontext_unseen.py \
        --args.host 127.0.0.1 --args.port "$PORT" \
        --args.task_suite_name "$suite" --args.task_split split0 \
        --args.results_out_path "$RES" \
        > "$LOG_DIR/${stem}_client.log" 2>&1 || FAIL=1
done

echo "=== results ($NAME) ==="
for suite in $SUITES; do
    stem=${suite#libero_}; F="$LOG_DIR/${stem}_unseen_results.json"
    [ -f "$F" ] && "$SERVER_PY" -c "import json;d=json.load(open('$F'));s=d['summary'];print('$suite unseen_SR=%s (%s/%s)'%(s.get('unseen_success_rate'),s.get('total_successes'),s.get('total_episodes')))" || { echo "$suite MISSING"; FAIL=1; }
done

# Sheet sync default-off (the driving session syncs the rename tab + comparison itself).
if [ "${SKIP_SHEET_SYNC:-1}" != "1" ] && command -v claude >/dev/null 2>&1; then
    claude -p "/log-to-sheet Read eval results from $LOG_DIR/*_results.json and sync to https://docs.google.com/spreadsheets/d/1OPys1ohYcw0lUdGW3W1c02dp0lMm-AP-NnYWhG8IecI/edit?gid=1035082397#gid=1035082397 tab 'rename'. Do not overwrite existing rows. Config=$NAME, ckpt=$CKPT, Eval Machine: ibex." || true
fi
exit $FAIL
