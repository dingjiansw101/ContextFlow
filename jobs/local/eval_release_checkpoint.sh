#!/usr/bin/env bash
# Run one released checkpoint on the four LIBERO suites with the current checkout.
# Usage: bash jobs/local/eval_release_checkpoint.sh CHECKPOINT NAME GPU RUN_ID [TRIALS] [SCOPE]
set -euo pipefail
CHECKPOINT="${1:?checkpoint directory}"
NAME="${2:?checkpoint name}"
GPU="${3:?GPU index}"
RUN_ID="${4:?unique run identifier}"
TRIALS="${5:-50}"
SCOPE="${6:-unseen}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
OUT="logs/release_eval/$RUN_ID/$NAME"
[[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT" >&2; exit 1; }
[[ -d "$CHECKPOINT/params" && -d "$CHECKPOINT/assets" ]] || exit 1
mkdir -p "$OUT"
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src:$REPO/third_party/libero"
export JAX_DEFAULT_MATMUL_PRECISION=float32
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
CUDA_NVCC="$REPO/.venv/lib/python3.11/site-packages/nvidia/cuda_nvcc"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=$CUDA_NVCC"
export PATH="$CUDA_NVCC/bin:$PATH"
export LIBERO_CONFIG_PATH="$REPO/$OUT/libero_config"
uv run --no-sync python - "$OUT" "$CHECKPOINT" "$TRIALS" "$SCOPE" "$GPU" <<'PY'
import json, pathlib, subprocess, sys
out, checkpoint, trials, scope, gpu = sys.argv[1:]
root = pathlib.Path.cwd()
config = root / out / 'libero_config'
config.mkdir()
libero = root / 'third_party/libero/libero/libero'
paths = dict(benchmark_root=str(libero), bddl_files=str(libero/'bddl_files'),
             init_states=str(libero/'init_files'), assets=str(libero/'assets'), datasets=str(libero.parent/'datasets'))
# JSON is valid YAML and avoids changing the user's global LIBERO configuration.
(config/'config.yaml').write_text(json.dumps(paths))
(root/out/'run.json').write_text(json.dumps(dict(checkpoint=checkpoint, trials=int(trials), scope=scope,
    gpu=int(gpu), seed=7, inference_dtype='float32', git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
PY
PORT=$(uv run --no-sync python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
CUDA_VISIBLE_DEVICES="$GPU" uv run --no-sync python scripts/serve_policy.py \
    --loader INCONTEXT --port "$PORT" policy:checkpoint --policy.config ContextFlow \
    --policy.dir "$CHECKPOINT" --policy.inference-dtype float32 > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
READY=0
for ((attempt=0; attempt<450; attempt++)); do
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -50 "$OUT/server.log"; exit 1; }
    if rg -q 'server listening on' "$OUT/server.log"; then READY=1; break; fi
    sleep 2
done
[[ "$READY" == 1 ]] || { echo 'Server startup timed out'; exit 1; }
FILTER=()
case "$SCOPE" in unseen) FILTER+=(--unseen-only);; all) ;; *) echo "Unknown scope: $SCOPE"; exit 1;; esac
for SUITE in ${SUITES:-libero_spatial libero_object libero_goal libero_10}; do
    echo "[$NAME] Evaluating $SUITE ($TRIALS trials per task)"
    timeout --signal=TERM 4h env -u CUDA_VISIBLE_DEVICES \
        uv run --no-project --python "$REPO/examples/libero/.venv/bin/python" python \
        examples/libero/main_incontext.py --host 127.0.0.1 --port "$PORT" \
        --task-suite-name "$SUITE" --num-trials-per-task "$TRIALS" --seed 7 "${FILTER[@]}" \
        --video-out-path "$OUT/videos/$SUITE" --results-out-path "$OUT/${SUITE}.json" \
        > "$OUT/${SUITE}.log" 2>&1
    # A nonzero exit already fails above. Catch swallowed rollout errors too;
    # robosuite can emit ignored destructor tracebacks after a successful exit.
    if rg -q 'Caught exception:' "$OUT/${SUITE}.log"; then
        echo "Evaluation error: $OUT/${SUITE}.log"; tail -50 "$OUT/${SUITE}.log"; exit 1
    fi
    [[ -s "$OUT/${SUITE}.json" ]] || exit 1
    uv run --no-sync python - "$OUT/${SUITE}.json" "$TRIALS" "$SCOPE" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
trials = int(sys.argv[2])
tasks = 2 if sys.argv[3] == 'unseen' else 10
assert len(result['per_task_results']) == tasks
assert all(row['episodes'] == trials for row in result['per_task_results'])
assert result['summary']['total_episodes'] == tasks * trials
print(result['summary'])
PY
    echo "[$NAME] Completed $SUITE"
done
echo "[$NAME] Evaluation complete: $OUT"
