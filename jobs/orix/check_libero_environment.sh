#!/usr/bin/env bash
# Diagnostic only: render each held-out Spatial/Object task without running a policy.
# May run as an overlapping step of an existing allocation with spare GPU memory:
# srun --jobid JOB --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=1 bash jobs/orix/check_libero_environment.sh
set -euo pipefail
[[ "${SLURM_NTASKS:-1}" == 1 ]] || { echo 'Specify --ntasks=1 for this diagnostic'; exit 1; }
cd /home/dingj0b/code/contextflow
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mnt/data/u/dingj0b/contextflow/uv-cache
export LEROBOT_HOME=/mnt/data/u/dingj0b/contextflow/lerobot
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PWD/third_party/libero"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_DIRS="$HOME/nvidia-egl"
export LD_LIBRARY_PATH="$HOME/nvidia-egl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The one-GPU SLURM step exposes one EGL device, enumerated as zero.
export MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH
LIBERO_CONFIG_PATH=$(mktemp -d /mnt/data/u/dingj0b/contextflow/logs/libero-check.XXXXXX)
uv run --no-sync python - <<'PY'
import json
import os
from pathlib import Path

libero = Path.cwd() / 'third_party/libero/libero/libero'
(Path(os.environ['LIBERO_CONFIG_PATH']) / 'config.yaml').write_text(json.dumps(dict(
    benchmark_root=str(libero), bddl_files=str(libero / 'bddl_files'),
    init_states=str(libero / 'init_files'), assets=str(libero / 'assets'),
    datasets=str(libero.parent / 'datasets'))))
PY
env -u CUDA_VISIBLE_DEVICES uv run --no-project \
    --python /home/dingj0b/code/openpi-refactor_refactor_merge/examples/libero/.venv/bin/python python - <<'PY'
import pathlib
import numpy as np
from examples.libero import main_incontext as client

mapping = client.get_task_to_index_mapping(client.LIBERO_TASKS_JSONL)
for name in ('libero_spatial', 'libero_object'):
    suite = client.benchmark.get_benchmark_dict()[name]()
    heldout = [i for i in range(suite.n_tasks) if suite.get_task(i).language in client.LIBERO_UNSEEN_TASKS]
    assert len(heldout) == 2, (name, heldout)
    for i in heldout:
        task = suite.get_task(i)
        assert task.language in mapping
        states = suite.get_task_init_states(i)
        assert len(states) >= 50
        env, description = client._get_libero_env(task, 256, 7)
        try:
            env.reset()
            env.set_init_state(states[0])
            obs, _, _, _ = env.step(client.LIBERO_DUMMY_ACTION)
            for key in ('agentview_image', 'robot0_eye_in_hand_image'):
                frame = np.asarray(obs[key])
                assert frame.shape == (256, 256, 3), (key, frame.shape)
                assert frame.dtype == np.uint8
                assert frame.max() > frame.min(), (key, 'flat render')
            print('RENDER_OK', name, i, description, flush=True)
        finally:
            env.close()
print('LIBERO_ENVIRONMENT_OK', flush=True)
PY
