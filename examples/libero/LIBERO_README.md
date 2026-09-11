# LIBERO: Training and Evaluating the In-Context Models

Complete the [repository installation](../../README.md#installation) first. Run the commands below from the repository root to train and evaluate **ContextFlow** on LIBERO.

## 1. Environments

Two Python environments are involved:

- the **repo environment** (Python 3.11, managed by `uv`) — training and the policy server.
- the **LIBERO client environment** (Python 3.8) — runs the simulator and the evaluation clients.

Create the client environment:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
uv pip install --python examples/libero/.venv/bin/python \
    -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python examples/libero/.venv/bin/python \
    -e packages/openpi-client -e third_party/libero
```

Alternatively, with Docker:

```bash
# Grant access to the X11 server:
sudo xhost +local:docker
export SERVER_ARGS="--env LIBERO"
docker compose -f examples/libero/compose.yml up --build
```

## 2. Required Artifacts

| Artifact | Where to get it | Needed for |
| --- | --- | --- |
| LIBERO dataset (`physical-intelligence/libero`) | auto-downloaded from HuggingFace on first use | training and evaluation |
| Checkpoints | Google Drive (see [root README](../../README.md#in-context-model-checkpoints-google-drive)) | evaluation |

Follow the [checkpoint download instructions](../../README.md#in-context-model-checkpoints-google-drive) to evaluate released weights, then skip to evaluation below.

Before starting the evaluation client, ensure the dataset is available under `$LEROBOT_HOME/physical-intelligence/libero` (`LEROBOT_HOME` defaults to `~/.cache/huggingface/lerobot`). If the simulator runs on a separate machine, place the dataset's `meta/` directory at that location on the client machine.

## 3. Training

Train with the config name and an experiment name (checkpoints go to `./checkpoints/<config>/<exp-name>/<step>/`):

```bash
uv run scripts/compute_norm_stats.py --config-name ContextFlow
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ContextFlow \
    --exp-name=my_run
```

- Config name: `ContextFlow`.
- ContextFlow trains for 20k steps with `batch_size=32`, starting from the π₀ base checkpoint (auto-downloaded from S3).
- Training excludes the eight held-out tasks (`remove_task_list=LIBERO_UNSEEN_TASKS` in the config).
- Training defaults to `--seed=42` and `--data.seed-base=1` for deterministic per-sample demonstration selection. Override these for repeated runs and use a fresh `--exp-name` per run.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` lets JAX use 90% of GPU memory (default 75%). Multi-GPU: run under `CUDA_VISIBLE_DEVICES=0,1` (data-parallel sharding is automatic across visible devices).

## 4. Evaluation

Evaluation is server/client: the policy server runs in the repo environment on a GPU, the LIBERO simulator client in the Python 3.8 environment.

**Terminal 1 — policy server** (float32 recommended for in-context models):

```bash
export JAX_DEFAULT_MATMUL_PRECISION=float32
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.inference_dtype=float32 \
    --policy.config=ContextFlow \
    --policy.dir=checkpoints/ContextFlow/ContextFlow_run1/19999
```

For your own trained model, use `checkpoints/ContextFlow/my_run/19999` instead.

**Terminal 2 — evaluation client:**

```bash
export MUJOCO_GL=egl
export PYTHONPATH="$PWD/src:$PWD/third_party/libero:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
# Evaluate unseen tasks
uv run --no-project --python examples/libero/.venv/bin/python python \
    examples/libero/main_incontext.py \
    --task-suite-name libero_spatial \
    --results-out-path logs/eval_results/unseen/libero_spatial.json \
    --video-out-path data/libero_incontext/videos/unseen/libero_spatial
```

Key client arguments:

- `--task-suite-name`: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
- `--num-trials-per-task`: rollouts per task (default 50)
- Only held-out (unseen) tasks are evaluated by default.
- `--unseen-task-index`: evaluate one held-out task by its zero-based index within the suite's unseen tasks; implies unseen-only evaluation.
- `--host` / `--port`: policy server address (default `0.0.0.0:8000`)

To evaluate one unseen task, use `--unseen-task-index 0` or `--unseen-task-index 1` with the desired suite.

To evaluate unseen tasks across all four suites, keep the policy server running and run:

```bash
export MUJOCO_GL=egl
export PYTHONPATH="$PWD/src:$PWD/third_party/libero:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
for suite in libero_spatial libero_object libero_goal libero_10; do
    uv run --no-project --python examples/libero/.venv/bin/python python \
        examples/libero/main_incontext.py \
        --task-suite-name "$suite" --num-trials-per-task 50 \
        --results-out-path "logs/eval_results/unseen_all/${suite}.json" \
        --video-out-path "data/libero_incontext/videos/unseen_all/${suite}"
done
```

Results are written as JSON to `logs/eval_results/` (override with `--results-out-path`). Read `summary.unseen_success_rate` for the unseen success rate, expressed as a fraction between 0 and 1. For repeated evaluations, set `--seed=<n>` and use separate `--results-out-path` and `--video-out-path` locations for each run.

## 5. Troubleshooting

See the [root README troubleshooting table](../../README.md#troubleshooting). The most common LIBERO-specific pitfalls: missing `MUJOCO_GL=egl` (black renders / EGL crashes), and running in-context inference without `float32` precision.
