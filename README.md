# ContextFlow: In-Context Vision-Language-Action Models

This repository contains the model and training/evaluation code for **ContextFlow** — a vision-language-action (VLA) model that conditions on **in-context demonstrations** (demo images, states, and actions of a related task) to generalize to unseen tasks without fine-tuning.

It is a fork of [openpi](https://github.com/Physical-Intelligence/openpi) by the [Physical Intelligence team](https://www.physicalintelligence.company/) and builds on their two base models:

- the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based diffusion VLA
- the [π₀-FAST model](https://www.physicalintelligence.company/research/fast), an autoregressive VLA based on the FAST action tokenizer

On top of these we provide the in-context method:

| Method | Base | In-context conditioning | Model config class |
| --- | --- | --- | --- |
| **ContextFlow** | π₀ (flow) | 8 demo frames + 128 demo state/action steps | `ContextFlowConfig` (`src/openpi/models/contextflow.py`) |

It is trained and evaluated on the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO) with a seen/unseen task split, and has ALOHA real-robot variants (`ContextFlow_Aloha`, `Pi0_Aloha`).

## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. The current training script does not support multi-node training.

| Mode                    | Memory Required | Example GPU        |
| ----------------------- | --------------- | ------------------ |
| Inference               | > 16 GB         | RTX 4090           |
| Fine-Tuning (LoRA)      | > 40 GB         | A100 (80GB) / H100 |

The LIBERO ContextFlow runs reported in the paper were trained on 2–4× H100/A100-80GB with `batch_size=32`. The repo has been tested on Ubuntu 22.04.

## Installation

Clone the repo with submodules (the LIBERO simulator and the ALOHA client are submodules; LeRobot itself is pulled by `uv sync`):

```bash
git clone --recurse-submodules <this-repo-url>

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. Once uv is installed, run:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

## Model Checkpoints

### Base models (initialization for training)

Training the in-context models starts from the pre-trained π₀ / π₀-FAST base checkpoints, which are downloaded automatically from Physical Intelligence's S3 bucket on first use (cached in `~/.cache/openpi`; override with `OPENPI_DATA_HOME`):

| Model        | Checkpoint Path                                |
| ------------ | ---------------------------------------------- |
| π₀ base      | `s3://openpi-assets/checkpoints/pi0_base`      |
| π₀-FAST base | `s3://openpi-assets/checkpoints/pi0_fast_base` |

### In-context model checkpoints (Google Drive)

Our trained checkpoints, norm stats, and dataset metadata are hosted in the public Google Drive folder [`ContextFlow_Data`](https://drive.google.com/drive/folders/1TJvz-ITv4b99HjiJ27DRk8j0p6b6VaaJ?usp=sharing). Each checkpoint directory contains `params/` and `assets/` (the norm stats it was trained with), so a downloaded checkpoint is self-sufficient for inference. `MANIFEST.json` in the same folder documents every checkpoint's provenance (original training name, step, source machine).

| Model | Config name | Recommended checkpoint (path inside `ContextFlow_Data`) |
| --- | --- | --- |
| ContextFlow | `ContextFlow` | `ContextFlow/ContextFlow_4gpu/19999` |
| + LIBERO-90 co-training | `ContextFlow_plus_libero90` | see `MANIFEST.json` (one folder per config) |
| ALOHA (real robot) | `ContextFlow_Aloha`, `Pi0_Aloha` | see `MANIFEST.json` |

The Drive folder also holds checkpoints for methods this repo no longer ships configs for (ContextFlow-Plain, ContextAR and their variants); see `MANIFEST.json`. To run those, check out a commit before the configs were pruned.

Download via the browser link above, or with [rclone](https://rclone.org/drive/) (using your own configured Google Drive remote, here called `gdrive:`):

```bash
FOLDER=1TJvz-ITv4b99HjiJ27DRk8j0p6b6VaaJ
# Checkpoint → local layout expected by the eval commands (checkpoints/<config>/<exp>/<step>)
rclone copy --drive-root-folder-id $FOLDER gdrive:ContextFlow/ContextFlow_4gpu/19999 \
    checkpoints/ContextFlow/ContextFlow_4gpu/19999
# Norm stats (needed for training only — inference reads them from the checkpoint)
rclone copy --drive-root-folder-id $FOLDER gdrive:assets ./assets
# LIBERO metadata (task→episode maps; see the LIBERO README)
rclone copy --drive-root-folder-id $FOLDER gdrive:metadata/libero ./metadata/libero
```

> **⚠ Norm stats: download, do not recompute.** The released norm stats live in `assets/ContextFlow/libero`, which both remaining ContextFlow configs point at via `assets_repo_override="ContextFlow"` plus `AssetsConfig(asset_id="libero")`. (Older checkouts and the Google Drive archive use `assets/ContextFlow_Plain/physical-intelligence/libero` — see the migration note in [CONFIG_NAME_MAPPING.md](CONFIG_NAME_MAPPING.md).) They were computed by an earlier generation of the configs that used `use_delta_joint_actions=True`, over the full dataset (provenance verified: re-running the computation with that setting reproduces the released file byte-for-byte). The current configs set `use_delta_joint_actions=False` but intentionally keep reusing those same stats — every released checkpoint was trained with them. Running `scripts/compute_norm_stats.py` with today's configs produces *different* statistics, and models trained or evaluated with mismatched stats will not reproduce the released results. Recompute only when you train on a new dataset of your own.

## Running Inference

Trained in-context policies are created with `create_trained_policy_incontext`, which automatically attaches the demo-fetching pipeline (demos are pulled from the LIBERO dataset at inference time using the `metadata/libero/task_to_episode.json` map — no separate demo files needed):

```python
from openpi.training import config as _config
from openpi.policies import policy_config

config = _config.get_config("ContextFlow")
policy = policy_config.create_trained_policy_incontext(
    config, "checkpoints/ContextFlow/ContextFlow_4gpu/19999"
)
action_chunk = policy.infer(example)["actions"]
```

In practice you will usually run inference through the policy server. For in-context models we recommend float32 matmul precision for stable results:

```bash
export JAX_DEFAULT_MATMUL_PRECISION=float32
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.inference_dtype=float32 \
  --policy.config=ContextFlow \
  --policy.dir=checkpoints/ContextFlow/ContextFlow_4gpu/19999
```

`serve_policy.py` auto-detects in-context configs (`--loader=AUTO` is the default; pass `--loader=INCONTEXT` to force). The server listens on port 8000 by default.

## Remote Inference

The model can run on a different server and stream actions to the robot via a websocket connection (see `src/openpi/serving/websocket_policy_server.py` and `packages/openpi-client/`). This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate. You can test inference without a robot using the [simple client](examples/simple_client/README.md), which sends random observations to the server.

## Training and Evaluating on LIBERO

The full walkthrough — dataset and metadata preparation, training, serving, and seen/unseen evaluation — lives in **[examples/libero/LIBERO_README.md](examples/libero/LIBERO_README.md)**. The short version:

```bash
# 1. Get metadata + norm stats (download from Google Drive, or generate — see the LIBERO README)
rclone copy --drive-root-folder-id 1TJvz-ITv4b99HjiJ27DRk8j0p6b6VaaJ gdrive:metadata/libero ./metadata/libero
rclone copy --drive-root-folder-id 1TJvz-ITv4b99HjiJ27DRk8j0p6b6VaaJ gdrive:assets ./assets

# 2. Train (the LIBERO dataset physical-intelligence/libero auto-downloads from HuggingFace)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ContextFlow --exp-name=my_run --overwrite

# 3. Serve the trained checkpoint (terminal 1)
export JAX_DEFAULT_MATMUL_PRECISION=float32
uv run scripts/serve_policy.py policy:checkpoint --policy.inference_dtype=float32 \
  --policy.config=ContextFlow --policy.dir=checkpoints/ContextFlow/my_run/19999

# 4. Evaluate on unseen tasks (terminal 2, inside the LIBERO client venv)
python examples/libero/main_incontext_unseen.py --task-suite-name libero_spatial --task-split split0
```

Task splits (which LIBERO tasks are seen during training vs held out) are committed in [`libero_task_splits/`](libero_task_splits) (`split0`, with `seen_tasks.json` / `unseen_tasks.json`).

## Repository Structure

- `src/openpi/models/` — model implementations: `contextflow.py` (+ the upstream `pi0.py`, `pi0_fast.py`)
- `src/openpi/training/` — configs (`config_libero.py`, `config_aloha.py`), the in-context dataset (`custom_dataset.py`), metadata generation (`generate_task_to_index.py`)
- `src/openpi/policies/` — policy wrappers, `policy_config.py` (checkpoint → policy, in-context demo pipeline)
- `scripts/` — `train.py`, `serve_policy.py`, `compute_norm_stats.py`
- `examples/libero/` — LIBERO evaluation clients and the [LIBERO guide](examples/libero/LIBERO_README.md)
- `jobs/` — batch eval/training wrappers for local and SLURM machines
- `libero_task_splits/` — seen/unseen task split definitions

## Troubleshooting

| Issue                                     | Resolution                                                                                                                                                                                   |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` fails with dependency conflicts | Try removing the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. Check that you have the latest version of `uv` installed (`uv self update`). |
| Training logs `Norm stats not found ... skipping` | This is a **silent** failure mode: the run continues with no normalization. Make sure `./assets/ContextFlow/libero/` exists (downloaded from Google Drive, then renamed per CONFIG_NAME_MAPPING.md) before starting training. |
| In-context inference results are unstable / degraded | Export `JAX_DEFAULT_MATMUL_PRECISION=float32` and pass `--policy.inference_dtype=float32` to `serve_policy.py`. |
| Training runs out of GPU memory           | Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before training so JAX can use 90% of GPU memory. You can also reduce the batch size, or shard with `fsdp_devices` in the training config. |
| Simulator renders black images / EGL errors during LIBERO eval | Export `MUJOCO_GL=egl` and make sure an NVIDIA EGL vendor library is installed (see `jobs/local/eval_incontext_unseen_local.sh` for a working environment setup). |
| Dataset download fails                    | Check your internet connection. If using `local_files_only=True`, verify the dataset exists locally. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`). |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server. |

## Acknowledgements

This repository is a fork of [openpi](https://github.com/Physical-Intelligence/openpi). We thank the Physical Intelligence team for open-sourcing the π₀ / π₀-FAST models, and the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) team for the benchmark.
