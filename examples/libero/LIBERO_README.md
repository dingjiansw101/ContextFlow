# LIBERO: Training and Evaluating the In-Context Models

This is the complete guide for running **ContextFlow** on the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO): preparing the dataset and metadata, training, serving a policy, and evaluating on seen/unseen task splits. See the [root README](../../README.md) for an overview of the method and the checkpoint inventory.

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
| LIBERO dataset (`physical-intelligence/libero`) | auto-downloaded from HuggingFace on first use | training and eval (in-context demos are drawn from it; the eval clients also read its `meta/tasks.jsonl` for task names/order) |
| `assets/ContextFlow/physical-intelligence/libero/` (norm stats) | Google Drive `assets/` (archived under `ContextFlow_Plain/`; rename the top level after download) | **training only** — inference loads norm stats from the checkpoint. The ContextFlow configs resolve this path via `assets_repo_override="ContextFlow"` |
| Checkpoints | Google Drive (see [root README](../../README.md#in-context-model-checkpoints-google-drive)) | evaluation |

Download the Drive artifacts (browser, or rclone with your own Google Drive remote):

```bash
FOLDER=1TJvz-ITv4b99HjiJ27DRk8j0p6b6VaaJ
rclone copy --drive-root-folder-id $FOLDER gdrive:assets ./assets
rclone copy --drive-root-folder-id $FOLDER gdrive:ContextFlow/ContextFlow_4gpu/19999 \
    checkpoints/ContextFlow/ContextFlow_4gpu/19999
```

**What about the big cache files?** The Drive `metadata/libero/` folder also contains `episode_states_without_delta_cache.json` / `episode_actions_without_delta_cache.json` (~150 MB). The `ContextFlow` configs do **not** use them — they load demonstrations directly from the dataset (`CustomLeRobotDataset`) during both training and inference. The caches are only read by legacy configs that still use the old transform-based pipeline; skip them unless you run those.

> **⚠ Norm stats: download, do not recompute.** The released norm stats in `assets/ContextFlow/physical-intelligence/libero` were computed by an earlier generation of these configs that used `use_delta_joint_actions=True`, over the full dataset (no train-split filtering). The current configs set `use_delta_joint_actions=False` but deliberately continue to use the same stats — all released checkpoints were trained with them, and each checkpoint also carries its own copy in `<checkpoint>/assets/`. Recomputing with `scripts/compute_norm_stats.py` under today's configs therefore gives *different* stats (delta actions change the action distribution: e.g. the released action mean for dim 3 is −2.97 ≈ −state mean, where an absolute-action computation gives ≈0) and will not reproduce the released results. Provenance is verified: re-running the computation with the delta setting flipped back on reproduces the released `norm_stats.json` byte-for-byte. Recompute only for a new dataset of your own.

## 3. Generating the Metadata from Scratch

**Training and evaluation need no precomputed lookup tables.** The task→episode and
episode→frame maps are derived in memory from the LeRobot dataset metadata
(`meta/tasks.jsonl` + `meta/episodes.jsonl`) by
[`src/openpi/training/lookup_tables.py`](../../src/openpi/training/lookup_tables.py), which costs
~10 ms and cannot drift from the dataset it describes. The old
`metadata/<name>/task_to_episode.json` / `episode_to_indexes.json` files have been removed.

`src/openpi/training/generate_task_to_index.py` still exists and can write those JSON files, but
only the standalone analysis tools (`scripts/visualize_lerobot.py`,
`scripts/check_libero_prompt_coverage.py`) read them:

```bash
uv run src/openpi/training/generate_task_to_index.py \
    --config pi0_libero \
    --skip_norm_stats \
    --output_dir metadata/libero
```

Two details matter here:

- **Generate with the plain `pi0_libero` config, not an in-context config.** The in-context training configs (`ContextFlow`, …) filter their dataset down to the training episodes (`remove_task_list`), so generating through them would omit the unseen tasks — but evaluation needs demo episodes for unseen tasks too. `pi0_libero` sees the full dataset, and it reads no metadata itself, so there is no bootstrapping problem.
- `--skip_norm_stats` skips the transform sanity check that runs after the files are written; metadata generation itself does not need norm stats.

The eval clients read the task table directly from the dataset
(`$LEROBOT_HOME/physical-intelligence/libero/meta/tasks.jsonl`, where `LEROBOT_HOME`
defaults to `~/.cache/huggingface/lerobot`), so there is nothing to copy or commit.
A client host therefore needs the dataset present — on a machine that only runs the
simulator, fetch at least its `meta/` directory.

## 4. Training

Train with the config name and an experiment name (checkpoints go to `./checkpoints/<config>/<exp-name>/<step>/`):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ContextFlow \
    --exp-name=my_run --overwrite
```

- Config names: `ContextFlow`, `ContextFlow_plus_libero90` (see [CONFIG_NAME_MAPPING.md](../../CONFIG_NAME_MAPPING.md) for the mapping from the original training names).
- All three train for 20k steps with `batch_size=32`, starting from the π₀ / π₀-FAST base checkpoints (auto-downloaded from S3).
- Training excludes the eight held-out tasks (`remove_task_list=LIBERO_UNSEEN_TASKS` in the config).
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

`serve_policy.py` auto-detects in-context configs and attaches the demo-fetching pipeline (`--loader=INCONTEXT` forces it). The server needs the HuggingFace dataset (auto-downloaded) to fetch demos; the task→episode map is derived from its metadata at startup.

**Terminal 2 — evaluation client** (activate `examples/libero/.venv` first, Section 1):

```bash
# Unseen tasks of a split (the paper's generalization metric)
python examples/libero/main_incontext_unseen.py \
    --task-suite-name libero_spatial

# Seen tasks of a split
python examples/libero/main_incontext.py --task-suite-name libero_spatial

# Plain pi0 / pi0-FAST baseline (no in-context demos)
python examples/libero/main.py --task-suite-name libero_spatial
```

Key client arguments:

- `--task-suite-name`: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`. (`libero_90` is training co-data only — the in-context clients reject it, because its task indices are a different space from the demo dataset the server serves from.)
- The seen/unseen assignment is not configurable: the training configs and eval clients share `LIBERO_UNSEEN_TASKS` from `openpi_client.libero_task_split`. Any suite task not in that tuple counts as seen.
- `--num-trials-per-task`: rollouts per task (default 50)
- `--host` / `--port`: policy server address (default `0.0.0.0:8000`)

Results are written as JSON to `logs/eval_results/` (override with `--results-out-path`).

**Batch evaluation.** `jobs/local/eval_incontext_unseen_local.sh <run-name> <policy-config> <checkpoint-dir> [run-id]` runs all four suites in parallel across GPUs (one server per suite; env `SUITE_LIST`, `GPUS`, `TASK_SPLIT` to customize). SLURM wrappers: `jobs/orix/refactor_merge/eval_contextflow_unseen.sh` and `jobs/ibex/eval_incontext_unseen_ibex.sh`.

## 6. LIBERO-90 Co-Training

The `ContextFlow_plus_libero90` config co-trains on LIBERO-90 in addition to the standard suites.

It needs one extra artifact:

1. **The LIBERO-90 LeRobot dataset** [`vo2yager/libero_90`](https://huggingface.co/datasets/vo2yager/libero_90) — auto-downloads from HuggingFace (~63 GB). To rebuild it yourself from the official raw HDF5 demonstrations instead:

   ```bash
   # Download the raw libero_90 HDF5 files (ships inside the libero_100 archive):
   python third_party/libero/benchmark_scripts/download_libero_datasets.py --datasets libero_100

   # Convert to a LeRobot dataset (optionally --push_to_hub):
   uv run examples/libero/convert_libero_raw_hdf5_to_lerobot.py --data_dir /path/to/libero_90
   ```

   The config's `libero_90` dataset spec sets `local_files_only=True` and points `episode_json_path`
   at `~/.cache/huggingface/lerobot/vo2yager/libero_90/meta/episodes.jsonl`, so that path must
   exist before training starts. If you only need the dataset *metadata* — for the task→episode
   tables, or to read `meta/tasks.jsonl` — copying `meta/` alone is enough (~539 KB instead of
   63 GB); the lookup tables never touch the parquet shards. Training on LIBERO-90 frames, of
   course, does need the full dataset.

## 7. Troubleshooting

See the [root README troubleshooting table](../../README.md#troubleshooting). The most common LIBERO-specific pitfalls: missing `MUJOCO_GL=egl` (black renders / EGL crashes), missing norm stats (silent "skipping" log line during training), and running in-context inference without `float32` precision.
