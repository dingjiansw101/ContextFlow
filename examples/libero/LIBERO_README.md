# LIBERO: Training and Evaluating the In-Context Models

This is the complete guide for running **ContextFlow** on the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO): preparing the dataset and metadata, training, serving a policy, and evaluating on seen/unseen task splits. See the [root README](../../README.md) for an overview of the method and the checkpoint inventory.

This example requires git submodules to be initialized:

```bash
git submodule update --init --recursive
```

## 1. Environments

Two Python environments are involved:

- the **repo environment** (Python 3.11, managed by `uv`) — training and the policy server. Set it up as in the [root README](../../README.md#installation) (`GIT_LFS_SKIP_SMUDGE=1 uv sync`).
- the **LIBERO client environment** (Python 3.8) — runs the simulator and the evaluation clients.

Create the client environment:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip install -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/src:$PWD/third_party/libero
```

Alternatively, with Docker:

```bash
# Grant access to the X11 server:
sudo xhost +local:docker
export SERVER_ARGS="--env LIBERO"
docker compose -f examples/libero/compose.yml up --build
```

Note: when updating `requirements.txt` in this directory, the flag `--extra-index-url https://download.pytorch.org/whl/cu113` must be added to the `uv pip compile` command.

## 2. Required Artifacts

| Artifact | Where to get it | Needed for |
| --- | --- | --- |
| LIBERO dataset (`physical-intelligence/libero`) | auto-downloaded from HuggingFace on first use | training and eval (in-context demos are drawn from it; the eval clients also read its `meta/tasks.jsonl` for task names/order) |
| Checkpoints | Google Drive (see [root README](../../README.md#in-context-model-checkpoints-google-drive)) | evaluation |

Download the Drive artifacts (browser, or rclone with your own Google Drive remote):

```bash
FOLDER=1Bf5j90lifJ9kPy2YSQG1bp5FKWZwzTES
rclone copy --drive-root-folder-id $FOLDER gdrive:ContextFlow/ContextFlow_4gpu/19999 \
    checkpoints/ContextFlow/ContextFlow_4gpu/19999
```

## 3. Dataset Metadata

**Training and evaluation need no precomputed lookup tables.** The task→episode and
episode→frame maps are derived in memory from the LeRobot dataset metadata
(`meta/tasks.jsonl` + `meta/episodes.jsonl`) by
[`src/openpi/training/lookup_tables.py`](../../src/openpi/training/lookup_tables.py), which costs
~10 ms and cannot drift from the dataset it describes. The old
`metadata/<name>/task_to_episode.json` / `episode_to_indexes.json` files have been removed.

The eval clients read the task table directly from the dataset
(`$LEROBOT_HOME/physical-intelligence/libero/meta/tasks.jsonl`, where `LEROBOT_HOME`
defaults to `~/.cache/huggingface/lerobot`), so there is nothing to copy or commit.
A client host therefore needs the dataset present — on a machine that only runs the
simulator, fetch at least its `meta/` directory.

## 4. Training

Train with the config name and an experiment name (checkpoints go to `./checkpoints/<config>/<exp-name>/<step>/`):

```bash
uv run scripts/compute_norm_stats.py --config-name ContextFlow
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ContextFlow \
    --exp-name=my_run --overwrite
```

- Config name: `ContextFlow`.
- ContextFlow trains for 20k steps with `batch_size=32`, starting from the π₀ base checkpoint (auto-downloaded from S3).
- Training excludes the eight held-out tasks (`remove_task_list=LIBERO_UNSEEN_TASKS` in the config).
- Set a seed with `--seed=<n>` for repeated runs; use a fresh `--exp-name` per run.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` lets JAX use 90% of GPU memory (default 75%). Multi-GPU: run under `CUDA_VISIBLE_DEVICES=0,1` (data-parallel sharding is automatic across visible devices).

## 5. Evaluation

Evaluation is server/client: the policy server runs in the repo environment on a GPU, the LIBERO simulator client in the Python 3.8 environment.

**Terminal 1 — policy server** (float32 recommended for in-context models):

```bash
export JAX_DEFAULT_MATMUL_PRECISION=float32
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.inference_dtype=float32 \
    --policy.config=ContextFlow \
    --policy.dir=checkpoints/ContextFlow/ContextFlow_4gpu/19999
```

`serve_policy.py` auto-detects in-context configs and attaches the demo-fetching pipeline (`--loader=INCONTEXT` forces it). The server needs the HuggingFace dataset (auto-downloaded) to fetch demos; the task→episode map is derived from its metadata at startup.

**Terminal 2 — evaluation client** (activate `examples/libero/.venv` first, Section 1):

```bash
# Unseen tasks of a split (the paper's generalization metric)
python examples/libero/main_incontext.py --unseen-only \
    --task-suite-name libero_spatial

# All tasks of a split (reports seen and unseen success rates separately)
python examples/libero/main_incontext.py --task-suite-name libero_spatial

# Plain pi0 / pi0-FAST baseline (no in-context demos)
python examples/libero/main.py --task-suite-name libero_spatial
```

Key client arguments:

- `--task-suite-name`: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
- The seen/unseen assignment is not configurable: the training configs and eval clients share `LIBERO_UNSEEN_TASKS` from `openpi.training.config_libero`. Any suite task not in that tuple counts as seen.
- `--num-trials-per-task`: rollouts per task (default 50)
- `--unseen-only`: evaluate only held-out tasks (default: evaluate all tasks).
- `--unseen-task-index`: evaluate one held-out task by its zero-based index within the suite's unseen tasks; implies unseen-only evaluation.
- `--host` / `--port`: policy server address (default `0.0.0.0:8000`)

`main_incontext_unseen.py` remains a compatibility entry point with unseen-only evaluation enabled by default.

Results are written as JSON to `logs/eval_results/` (override with `--results-out-path`).

**Batch evaluation.** `jobs/local/eval_incontext_unseen_local.sh <run-name> <policy-config> <checkpoint-dir> [run-id]` runs all four suites in parallel across GPUs (one server per suite; env `SUITE_LIST`, `GPUS`, `TASK_SPLIT` to customize). SLURM wrappers: `jobs/orix/refactor_merge/eval_contextflow_unseen.sh` and `jobs/ibex/eval_incontext_unseen_ibex.sh`.

## 6. Troubleshooting

See the [root README troubleshooting table](../../README.md#troubleshooting). The most common LIBERO-specific pitfalls: missing `MUJOCO_GL=egl` (black renders / EGL crashes), and running in-context inference without `float32` precision.
