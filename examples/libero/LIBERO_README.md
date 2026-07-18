# LIBERO: Training and Evaluating the In-Context Models

This is the complete guide for running **ContextFlow**, **ContextFlow-Plain**, and **ContextAR** on the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO): preparing the dataset and metadata, training, serving a policy, and evaluating on seen/unseen task splits. See the [root README](../../README.md) for an overview of the methods and the checkpoint inventory.

This example requires git submodules to be initialized:

```bash
git submodule update --init --recursive
```

## 1. Environments

Two Python environments are involved:

- the **repo environment** (Python 3.11, managed by `uv`) — training, the policy server, metadata generation. Set it up as in the [root README](../../README.md#installation) (`GIT_LFS_SKIP_SMUDGE=1 uv sync`).
- the **LIBERO client environment** (Python 3.8) — runs the simulator and the evaluation clients.

Create the client environment:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
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
| LIBERO dataset (`physical-intelligence/libero`) | auto-downloaded from HuggingFace on first use | training and eval (in-context demos are drawn from it) |
| `metadata/libero/task_to_episode.json` | committed in git; also on [Google Drive](https://drive.google.com/drive/folders/1TJvz-ITv4b99HjiJ27DRk8j0p6b6VaaJ?usp=sharing); or [generate it](#3-generating-the-metadata-from-scratch) | training and eval — maps each task to its demo episodes |
| `metadata/libero/tasks.jsonl` | committed in git (verbatim copy of the dataset's `meta/tasks.jsonl`) | eval clients (task names/order) |
| `libero_task_splits/` (`split0`…`split7`) | committed in git | seen/unseen evaluation |
| `assets/ContextFlow_Plain/` (norm stats) | Google Drive `assets/` | **training only** — inference loads norm stats from the checkpoint |
| Checkpoints | Google Drive (see [root README](../../README.md#in-context-model-checkpoints-google-drive)) | evaluation |

Download the Drive artifacts (browser, or rclone with your own Google Drive remote):

```bash
FOLDER=1TJvz-ITv4b99HjiJ27DRk8j0p6b6VaaJ
rclone copy --drive-root-folder-id $FOLDER gdrive:assets ./assets
rclone copy --drive-root-folder-id $FOLDER gdrive:metadata/libero ./metadata/libero
rclone copy --drive-root-folder-id $FOLDER gdrive:ContextFlow/ContextFlow_4gpu/19999 \
    checkpoints/ContextFlow/ContextFlow_4gpu/19999
```

**What about the big cache files?** The Drive `metadata/libero/` folder also contains `episode_states_without_delta_cache.json` / `episode_actions_without_delta_cache.json` (~150 MB). The paper configs (`ContextFlow`, `ContextFlow_Plain`, `ContextAR`) do **not** use them — they load demonstrations directly from the dataset (`CustomLeRobotDataset`) during both training and inference. The caches are only read by legacy configs that still use the old transform-based pipeline; skip them unless you run those.

> **⚠ Norm stats: download, do not recompute.** The released norm stats in `assets/ContextFlow_Plain` were computed by an earlier generation of these configs that used `delta_action=True`. The current configs set `delta_action=False` but deliberately continue to use the same stats — all released checkpoints were trained with them, and each checkpoint also carries its own copy in `<checkpoint>/assets/`. If you recompute with `scripts/compute_norm_stats.py` under today's configs you will get *different* stats and will not reproduce the released results. Recompute only for a new dataset of your own.

## 3. Generating the Metadata from Scratch

`metadata/libero/task_to_episode.json` (and `episode_to_indexes.json`, produced alongside) can be regenerated from the dataset instead of downloaded. The generator instantiates the full data pipeline, so the norm stats (`assets/ContextFlow_Plain`, Section 2) should be in place first. The config also tries to open the metadata files while it is being constructed, so bootstrap them as empty JSON objects:

```bash
mkdir -p metadata/libero
echo '{}' > metadata/libero/task_to_episode.json
echo '{}' > metadata/libero/episode_to_indexes.json

uv run src/openpi/training/generate_task_to_index.py \
    --config ContextFlow \
    --output_dir metadata/libero
```

This overwrites the placeholders with the real task→episode and episode→frame-index maps. The result is deterministic — regenerating should reproduce the committed `task_to_episode.json` exactly.

`metadata/libero/tasks.jsonl` is simply a copy of the dataset's task table:

```bash
cp ~/.cache/huggingface/lerobot/physical-intelligence/libero/meta/tasks.jsonl metadata/libero/tasks.jsonl
```

## 4. Training

Train with the config name and an experiment name (checkpoints go to `./checkpoints/<config>/<exp-name>/<step>/`):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ContextFlow \
    --exp-name=my_run --overwrite
```

- Config names: `ContextFlow`, `ContextFlow_Plain`, `ContextAR` (see [CONFIG_NAME_MAPPING.md](../../CONFIG_NAME_MAPPING.md) for the mapping from the original training names).
- All three train for 20k steps with `batch_size=32`, starting from the π₀ / π₀-FAST base checkpoints (auto-downloaded from S3).
- The training split excludes the unseen tasks of `split0` (`remove_task_list` in the config); the `*_plus_libero90_split1/2/3` variants exclude the corresponding split's unseen tasks instead.
- Set a seed with `--seed=<n>` for repeated runs; use a fresh `--exp-name` per run.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` lets JAX use 90% of GPU memory (default 75%). Multi-GPU: run under `CUDA_VISIBLE_DEVICES=0,1` (data-parallel sharding is automatic across visible devices).
- If training logs `Norm stats not found ... skipping`, stop — the norm stats are missing (Section 2) and the run would silently train unnormalized.

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

`serve_policy.py` auto-detects in-context configs and attaches the demo-fetching pipeline (`--loader=INCONTEXT` forces it). The server needs `metadata/libero/task_to_episode.json` and the HuggingFace dataset (auto-downloaded) to fetch demos.

**Terminal 2 — evaluation client** (activate `examples/libero/.venv` first, Section 1):

```bash
# Unseen tasks of a split (the paper's generalization metric)
python examples/libero/main_incontext_unseen.py \
    --task-suite-name libero_spatial --task-split split0

# Seen tasks of a split
python examples/libero/main_incontext.py --task-suite-name libero_spatial --task-split split0

# Plain pi0 / pi0-FAST baseline (no in-context demos)
python examples/libero/main.py --task-suite-name libero_spatial
```

Key client arguments:

- `--task-suite-name`: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`
- `--task-split`: `split0` … `split7` (default `split0`)
- `--task-splits-dir`: directory with the split definitions (default: `libero_task_splits` at the repo root, committed in git; each split has `seen_tasks.json` / `unseen_tasks.json`)
- `--num-trials-per-task`: rollouts per task (default 50)
- `--host` / `--port`: policy server address (default `0.0.0.0:8000`)

Results are written as JSON to `logs/eval_results/` (override with `--results-out-path`).

**Batch evaluation.** `jobs/local/eval_incontext_unseen_local.sh <run-name> <policy-config> <checkpoint-dir> [run-id]` runs all four suites in parallel across GPUs (one server per suite; env `SUITE_LIST`, `GPUS`, `TASK_SPLIT` to customize). SLURM wrappers: `jobs/orix/refactor_merge/eval_contextflow{,_plain}_unseen.sh`, `eval_contextar_unseen.sh`, and `jobs/ibex/eval_incontext_unseen_ibex.sh`.

## 6. LIBERO-90 Co-Training

The `*_plus_libero90` configs co-train on LIBERO-90 in addition to the standard suites (`ContextFlow_plus_libero90`, `ContextFlow_Plain_plus_libero90`, `ContextAR_plus_libero90`, each also with `_split1/2/3` variants that hold out the matching split — evaluate those with the matching `--task-split`).

They need two extra artifacts:

1. **The LIBERO-90 LeRobot dataset** [`vo2yager/libero_90`](https://huggingface.co/datasets/vo2yager/libero_90) — auto-downloads from HuggingFace. To rebuild it yourself from the official raw HDF5 demonstrations instead:

   ```bash
   # Download the raw libero_90 HDF5 files (ships inside the libero_100 archive):
   python third_party/libero/benchmark_scripts/download_libero_datasets.py --datasets libero_100

   # Convert to a LeRobot dataset (optionally --push_to_hub):
   uv run examples/libero/convert_libero_raw_hdf5_to_lerobot.py --data_dir /path/to/libero_90
   ```

2. **`metadata/libero_90/task_to_episode.json`** — committed in git; also on Google Drive (`metadata/libero_90/`), or regenerate with `generate_task_to_index.py --output_dir metadata/libero_90` and a libero_90 config (see the script's docstring).

## 7. Troubleshooting

See the [root README troubleshooting table](../../README.md#troubleshooting). The most common LIBERO-specific pitfalls: missing `MUJOCO_GL=egl` (black renders / EGL crashes), missing norm stats (silent "skipping" log line during training), and running in-context inference without `float32` precision.
