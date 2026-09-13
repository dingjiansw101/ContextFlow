#!/usr/bin/env bash
# Usage: bash jobs/local/eval_reproduction.sh CHECKPOINT EXPERIMENT GPU
set -euo pipefail
CHECKPOINT="${1:?checkpoint}"
EXPERIMENT="${2:?experiment}"
GPU="${3:?free local GPU}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
OUT="$REPO/logs/reproduction_eval/$EXPERIMENT"
[[ ! -e "$OUT" && -s "$CHECKPOINT/EVAL_SHA256SUMS" ]] || exit 1
mkdir -p "$OUT/libero_config"
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src:$REPO/third_party/libero"
export JAX_DEFAULT_MATMUL_PRECISION=float32 XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU"
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LIBERO_CONFIG_PATH="$OUT/libero_config"
CUDA_NVCC="$REPO/.venv/lib/python3.11/site-packages/nvidia/cuda_nvcc"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=$CUDA_NVCC"
export PATH="$CUDA_NVCC/bin:$PATH"
uv run --no-sync python - "$OUT" "$CHECKPOINT" "$GPU" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, checkpoint, gpu = sys.argv[1:]
out, checkpoint = pathlib.Path(out), pathlib.Path(checkpoint)
libero = pathlib.Path.cwd() / 'third_party/libero/libero/libero'
(out/'libero_config/config.yaml').write_text(json.dumps(dict(
    benchmark_root=str(libero), bddl_files=str(libero/'bddl_files'),
    init_states=str(libero/'init_files'), assets=str(libero/'assets'),
    datasets=str(pathlib.Path.home()/'.cache/huggingface/lerobot'))))
(out/'run.json').write_text(json.dumps(dict(checkpoint=str(checkpoint), gpu=int(gpu),
    seed=7, trials_per_task=50, scope='unseen', inference_dtype='float32',
    suites=['libero_spatial','libero_object','libero_goal','libero_10'],
    manifest_sha256=hashlib.sha256((checkpoint/'EVAL_SHA256SUMS').read_bytes()).hexdigest(),
    eval_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
PY
PORT=$(uv run --no-sync python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
CUDA_VISIBLE_DEVICES="$GPU" uv run --no-sync python scripts/serve_policy.py \
    --loader INCONTEXT --port "$PORT" policy:checkpoint --policy.config ContextFlow \
    --policy.dir "$CHECKPOINT" --policy.inference-dtype float32 > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
READY=0
for ((attempt=0; attempt<900; attempt++)); do
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -c 5000 "$OUT/server.log"; exit 1; }
    if rg -q 'server listening on' "$OUT/server.log"; then READY=1; break; fi
    sleep 2
done
[[ "$READY" == 1 ]] || exit 1
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    timeout --signal=TERM --kill-after=60s 8h env -u CUDA_VISIBLE_DEVICES \
        uv run --no-project --python "$REPO/examples/libero/.venv/bin/python" python \
        examples/libero/main_incontext.py --host 127.0.0.1 --port "$PORT" \
        --task-suite-name "$SUITE" --num-trials-per-task 50 --seed 7 --unseen-only \
        --video-out-path "$OUT/videos/$SUITE" --results-out-path "$OUT/$SUITE.json" \
        > "$OUT/$SUITE.log" 2>&1
    if rg -q 'Caught exception:' "$OUT/$SUITE.log"; then exit 1; fi
    uv run --no-sync python - "$OUT/$SUITE.json" <<'PY'
import json, sys
from openpi.training.config_libero import LIBERO_UNSEEN_TASKS
result = json.load(open(sys.argv[1]))
rows = result['per_task_results']
assert len(rows) == 2 and len({r['task_description'] for r in rows}) == 2
assert all(r['episodes'] == 50 and r['category'] == 'unseen' and r['task_description'] in LIBERO_UNSEEN_TASKS for r in rows)
assert result['summary']['total_episodes'] == 100
assert result['summary']['total_successes'] == sum(r['successes'] for r in rows)
print(sys.argv[1], result['summary'], flush=True)
PY
done
date -Iseconds > "$OUT/eval.complete"
