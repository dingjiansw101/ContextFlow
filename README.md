# ContextFlow: In-Context Flow Matching for Robot Manipulation

This repository contains the model and training/evaluation code for **ContextFlow** — a model that conditions on **in-context demonstrations** (demo images, states, and actions of a related task) to generalize to unseen tasks without fine-tuning.

It is a fork of [openpi](https://github.com/Physical-Intelligence/openpi) by the [Physical Intelligence team](https://www.physicalintelligence.company/) and builds on their base model, the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based diffusion VLA.

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
git clone --recurse-submodules https://github.com/dingjiansw101/ContextFlow.git
cd ContextFlow

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
| ContextFlow | `ContextFlow` | `ContextFlow/ContextFlow_run1/19999` |

Download via the browser link above, or with [rclone](https://rclone.org/drive/) (using your own configured Google Drive remote, here called `gdrive:`):

```bash
FOLDER=1Bf5j90lifJ9kPy2YSQG1bp5FKWZwzTES
# Checkpoint → local layout expected by the eval commands (checkpoints/<config>/<exp>/<step>)
rclone copy --drive-root-folder-id $FOLDER gdrive:ContextFlow/ContextFlow_run1/19999 \
    checkpoints/ContextFlow/ContextFlow_run1/19999
```

## Running Experiments on LIBERO

Follow the [LIBERO guide](examples/libero/LIBERO_README.md) to set up the simulator, train ContextFlow, and evaluate a downloaded or newly trained checkpoint. It includes commands for unseen-task evaluation and repeated runs.

`examples/libero/main_incontext.py` evaluates unseen tasks by default and reports their success rate.

The policy server and simulator can run on separate machines; set `--host` on the evaluation client to the server address. To test the server with random observations, see the [simple client](examples/simple_client/README.md).

## Real-World Dataset

The real-world ALOHA dataset is available on Hugging Face: [vo2yager/aloha_incontext](https://huggingface.co/datasets/vo2yager/aloha_incontext). Visit the dataset page to browse and download the data.

## Repository Structure

- `src/openpi/models/` — model implementations: `contextflow.py` and the upstream `pi0.py`
- `src/openpi/training/` — configs (`config_libero.py`) and the in-context dataset (`custom_dataset.py`)
- `src/openpi/policies/` — policy wrappers, `policy_config.py` (checkpoint → policy, in-context demo pipeline)
- `scripts/` — `train.py`, `serve_policy.py`, `compute_norm_stats.py`
- `examples/libero/` — LIBERO evaluation clients and the [LIBERO guide](examples/libero/LIBERO_README.md)

## Troubleshooting

| Issue                                     | Resolution                                                                                                                                                                                   |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` fails with dependency conflicts | Try removing the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. Check that you have the latest version of `uv` installed (`uv self update`). |
| In-context inference results are unstable / degraded | Export `JAX_DEFAULT_MATMUL_PRECISION=float32` and pass `--policy.inference_dtype=float32` to `serve_policy.py`. |
| Training runs out of GPU memory           | Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before training so JAX can use 90% of GPU memory. You can also reduce the batch size, or shard with `fsdp_devices` in the training config. |
| Simulator renders black images / EGL errors during LIBERO eval | Export `MUJOCO_GL=egl` and make sure an NVIDIA EGL vendor library is installed. |
| Dataset download fails                    | Check your internet connection. If using `local_files_only=True`, verify the dataset exists locally. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`). |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server. |

## Acknowledgements

This repository is a fork of [openpi](https://github.com/Physical-Intelligence/openpi). We thank the Physical Intelligence team for open-sourcing the π₀ model, and the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) team for the benchmark.
