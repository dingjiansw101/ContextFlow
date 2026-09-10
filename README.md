# ContextFlow: In-Context Flow Matching for Robot Manipulation

This repository contains the model and training/evaluation code for **ContextFlow** — a vision-language-action (VLA) model that conditions on **in-context demonstrations** (demo images, states, and actions of a related task) to generalize to unseen tasks without fine-tuning.

It is a fork of [openpi](https://github.com/Physical-Intelligence/openpi) by the [Physical Intelligence team](https://www.physicalintelligence.company/) and builds on their base model, the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based diffusion VLA.

On top of this we provide the in-context method:

| Method | Base | In-context conditioning | Model config class |
| --- | --- | --- | --- |
| **ContextFlow** | π₀ (flow) | 8 demo frames + 128 demo state/action steps | `ContextFlowConfig` (`src/openpi/models/contextflow.py`) |

It is trained and evaluated on the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO) with a seen/unseen task split.

## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. The current training script does not support multi-node training.

| Mode                    | Memory Required | Example GPU        |
| ----------------------- | --------------- | ------------------ |
| Inference               | > 16 GB         | A100 (80GB)        |
| Fine-Tuning (LoRA)       | > 40 GB         | A100 (80GB)        |

The default ContextFlow training configuration uses `batch_size=32`. The repo has been tested on Ubuntu 22.04.

## Installation

Clone the repo with submodules (the LIBERO simulator is a submodule; LeRobot itself is pulled by `uv sync`):

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

### Base model (initialization for training)

Training the in-context models starts from the pre-trained π₀ base checkpoint, which is downloaded automatically from Physical Intelligence's S3 bucket on first use (cached in `~/.cache/openpi`; override with `OPENPI_DATA_HOME`):

| Model   | Checkpoint Path                           |
| ------- | ----------------------------------------- |
| π₀ base | `s3://openpi-assets/checkpoints/pi0_base` |

### In-context model checkpoints (Google Drive)

The Google Drive folder [`ContextFlow_Data`](https://drive.google.com/drive/folders/1Bf5j90lifJ9kPy2YSQG1bp5FKWZwzTES) provides the ContextFlow configuration and trained weights. Download the complete checkpoint directory for inference.

| Model | Config name | Checkpoint path inside `ContextFlow_Data` |
| --- | --- | --- |
| ContextFlow | `ContextFlow` | `ContextFlow/ContextFlow_4gpu/19999` |

Download via the browser link above, or with [rclone](https://rclone.org/drive/) (using your own configured Google Drive remote, here called `gdrive:`):

```bash
FOLDER=1Bf5j90lifJ9kPy2YSQG1bp5FKWZwzTES
# Checkpoint → local layout expected by the eval commands (checkpoints/<config>/<exp>/<step>)
rclone copy --drive-root-folder-id $FOLDER gdrive:ContextFlow/ContextFlow_4gpu/19999 \
    checkpoints/ContextFlow/ContextFlow_4gpu/19999
```

## Running Inference

Trained in-context policies are created with `create_trained_policy_incontext`, which automatically attaches the demo-fetching pipeline (demos are pulled from the LIBERO dataset at inference time, using a task→episode map derived from the dataset's own metadata — no separate demo or index files needed):

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
# 1. Prepare training assets
uv run scripts/compute_norm_stats.py --config-name ContextFlow

# 2. Train (the LIBERO dataset physical-intelligence/libero auto-downloads from HuggingFace)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ContextFlow --exp-name=my_run --overwrite

# 3. Serve the trained checkpoint (terminal 1)
export JAX_DEFAULT_MATMUL_PRECISION=float32
uv run scripts/serve_policy.py policy:checkpoint --policy.inference_dtype=float32 \
  --policy.config=ContextFlow --policy.dir=checkpoints/ContextFlow/my_run/19999

# 4. Evaluate on unseen tasks (terminal 2, inside the LIBERO client venv)
python examples/libero/main_incontext.py --unseen-only --task-suite-name libero_spatial
```

Which LIBERO tasks are held out (unseen) rather than trained on is fixed in code: `LIBERO_UNSEEN_TASKS` in [`src/openpi/training/config_libero.py`](src/openpi/training/config_libero.py) is shared by the training configs and eval clients. Every other task in the four suites is a seen task.

## Repository Structure

- `src/openpi/models/` — model implementations: `contextflow.py` and the upstream `pi0.py`
- `src/openpi/training/` — configs (`config_libero.py`) and the in-context dataset (`custom_dataset.py`)
- `src/openpi/policies/` — policy wrappers, `policy_config.py` (checkpoint → policy, in-context demo pipeline)
- `scripts/` — `train.py`, `serve_policy.py`, `compute_norm_stats.py`
- `examples/libero/` — LIBERO evaluation clients and the [LIBERO guide](examples/libero/LIBERO_README.md)
- `jobs/` — batch eval/training wrappers for local and SLURM machines

## Troubleshooting

| Issue                                     | Resolution                                                                                                                                                                                   |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` fails with dependency conflicts | Try removing the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. Check that you have the latest version of `uv` installed (`uv self update`). |
| In-context inference results are unstable / degraded | Export `JAX_DEFAULT_MATMUL_PRECISION=float32` and pass `--policy.inference_dtype=float32` to `serve_policy.py`. |
| Training runs out of GPU memory           | Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before training so JAX can use 90% of GPU memory. You can also reduce the batch size, or shard with `fsdp_devices` in the training config. |
| Simulator renders black images / EGL errors during LIBERO eval | Export `MUJOCO_GL=egl` and make sure an NVIDIA EGL vendor library is installed (see `jobs/local/eval_incontext_unseen_local.sh` for a working environment setup). |
| Dataset download fails                    | Check your internet connection. If using `local_files_only=True`, verify the dataset exists locally. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`). |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server. |

## Acknowledgements

This repository is a fork of [openpi](https://github.com/Physical-Intelligence/openpi). We thank the Physical Intelligence team for open-sourcing the π₀ model, and the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) team for the benchmark.
