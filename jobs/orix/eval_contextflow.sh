#!/usr/bin/env bash
# Submit with --dependency=afterok:<training-job> after checking the final checkpoint path.
# Usage: sbatch-pi jobs/orix/eval_contextflow.sh CHECKPOINT OUTPUT_DIRECTORY
# Uses the existing LIBERO Python 3.8 environment without modifying it.
#SBATCH --job-name=cf-readme-eval
#SBATCH --account=pi-elhosemh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/u/dingj0b/contextflow/logs/eval-%j.log
set -euo pipefail
CHECKPOINT="${1:?final checkpoint directory}"
OUT="${2:?unused absolute output directory}"
REPO=/home/dingj0b/code/contextflow
CLIENT_ENV=/home/dingj0b/code/openpi-refactor_refactor_merge/examples/libero/.venv
cd "$REPO"
[[ "$OUT" == /* && ! -e "$OUT" ]] || { echo "Output must be an unused absolute path: $OUT"; exit 1; }
[[ -d "$CHECKPOINT/params" && -d "$CHECKPOINT/assets" ]] || { echo "Checkpoint incomplete: $CHECKPOINT"; exit 1; }
[[ -f .venv/pyvenv.cfg && -f "$CLIENT_ENV/pyvenv.cfg" ]] || exit 1
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mnt/data/u/dingj0b/contextflow/uv-cache
export HF_HOME=/mnt/data/u/dingj0b/contextflow/huggingface
export LEROBOT_HOME=/mnt/data/u/dingj0b/contextflow/lerobot
export OPENPI_DATA_HOME=/mnt/data/u/dingj0b/contextflow/openpi-cache
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src:$REPO/third_party/libero"
export JAX_DEFAULT_MATMUL_PRECISION=float32
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_DIRS="$HOME/nvidia-egl"
export LD_LIBRARY_PATH="$HOME/nvidia-egl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The client clears CUDA_VISIBLE_DEVICES; select the physical GPU assigned by SLURM.
export MUJOCO_EGL_DEVICE_ID="${SLURM_JOB_GPUS:?SLURM must assign exactly one GPU}"
[[ "$MUJOCO_EGL_DEVICE_ID" =~ ^[0-9]+$ ]] || { echo 'Expected one numeric physical GPU ID'; exit 1; }
CUDA_NVCC="$REPO/.venv/lib/python3.11/site-packages/nvidia/cuda_nvcc"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=$CUDA_NVCC"
export PATH="$CUDA_NVCC/bin:$PATH"
export LIBERO_CONFIG_PATH="$OUT/libero_config"
mkdir -p "$OUT"
uv run --no-sync python - "$OUT" "$CHECKPOINT" "$CLIENT_ENV" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

out, checkpoint, client_env = map(Path, sys.argv[1:])
libero = Path.cwd() / 'third_party/libero/libero/libero'
config = out / 'libero_config'
config.mkdir()
(config / 'config.yaml').write_text(json.dumps(dict(
    benchmark_root=str(libero), bddl_files=str(libero / 'bddl_files'),
    init_states=str(libero / 'init_files'), assets=str(libero / 'assets'),
    datasets=str(libero.parent / 'datasets'))))
stats = checkpoint / 'assets/physical-intelligence/libero/norm_stats.json'
(out / 'run.json').write_text(json.dumps(dict(
    checkpoint=str(checkpoint), checkpoint_norm_stats_sha256=hashlib.sha256(stats.read_bytes()).hexdigest(),
    eval_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    job_id=os.environ['SLURM_JOB_ID'], node=os.environ['SLURMD_NODENAME'],
    client_env=str(client_env), seed=7, trials_per_task=50, scope='unseen',
    suites=['libero_spatial', 'libero_object'], inference_dtype='float32'), indent=2))
PY
PORT=$(uv run --no-sync python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
uv run --no-sync python scripts/serve_policy.py \
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
for SUITE in libero_spatial libero_object; do
    echo "Evaluating $SUITE: 50 trials per unseen task"
    timeout --signal=TERM --kill-after=60s 8h env -u CUDA_VISIBLE_DEVICES \
        uv run --no-project --python "$CLIENT_ENV/bin/python" python \
        examples/libero/main_incontext.py --host 127.0.0.1 --port "$PORT" \
        --task-suite-name "$SUITE" --num-trials-per-task 50 --seed 7 --unseen-only \
        --video-out-path "$OUT/videos/$SUITE" --results-out-path "$OUT/$SUITE.json" \
        > "$OUT/$SUITE.log" 2>&1
    if rg -q 'Caught exception:' "$OUT/$SUITE.log"; then
        tail -50 "$OUT/$SUITE.log"; exit 1
    fi
    uv run --no-sync python - "$OUT/$SUITE.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
rows = result['per_task_results']
assert len(rows) == 2, rows
assert len({row['task_description'] for row in rows}) == 2
assert all(row['episodes'] == 50 and row['category'] == 'unseen' for row in rows)
assert all(0 <= row['successes'] <= row['episodes'] for row in rows)
assert result['summary']['total_episodes'] == 100
assert result['summary']['total_successes'] == sum(row['successes'] for row in rows)
print(sys.argv[1], result['summary'])
PY
done
date -Iseconds > "$OUT/eval.complete"
echo "Evaluation complete: $OUT"
