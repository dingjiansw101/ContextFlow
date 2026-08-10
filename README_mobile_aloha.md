# Mobile ALOHA Fine-Tuning Guide

This guide walks through the full workflow for taking raw Mobile ALOHA demonstrations,
converting them into the LeRobot format, fine-tuning a Physical Intelligence $\pi_0$
model, and serving the resulting policy.

## Installation

When cloning this repository, include submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Already cloned?
git submodule update --init --recursive
```

We manage Python dependencies with [uv](https://docs.astral.sh/uv/). After
installing uv, set up the environment with:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

`GIT_LFS_SKIP_SMUDGE=1` ensures the LeRobot dependency is fetched correctly.

## Model Checkpoints

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base diffusion [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                | `s3://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `s3://openpi-assets/checkpoints/pi0_fast_base` |

These checkpoints ship with normalization stats and assets that downstream
configs reference.

## Workflow Overview

1. Convert raw Mobile ALOHA data to a LeRobot dataset.
2. Define or reuse a training config and compute normalization statistics.
3. Launch fine-tuning.
4. Serve the fine-tuned policy for evaluation or deployment.

## 1. Convert Your Data to LeRobot

The conversion script reads raw HDF5 episodes and creates a LeRobot dataset in
`~/.cache/huggingface/lerobot/<repo_id>` by default. Run:

```bash
uv run examples/aloha_mobile_real/convert_aloha_mobile_data_to_lerobot.py \
  --raw-dir /path/to/raw/mobile_aloha \
  --repo-id <org>/<dataset-name> \
  --task uncap_pen \
  --is-mobile true
```

Notes:
- Authenticate with Hugging Face before enabling uploads: `source .venv/bin/activate`
  followed by `huggingface-cli login` (the binary lives in `.venv/bin`).
- Use `--no-push-to-hub` if you want to skip uploading. If a login error occurs
  during `push_to_hub`, fix the credentials and re-run the snippet below to upload
  without recomputing the dataset:

  ```bash
  source .venv/bin/activate
  python - <<'PY'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LEROBOT_HOME
repo_id = "<org>/<dataset-name>"
dataset = LeRobotDataset(repo_id=repo_id, root=LEROBOT_HOME)
dataset.push_to_hub()
PY
  ```

- If the raw data is stored on the Hub, pass `--raw-repo-id <org>/<raw-dataset>`
  instead of `--raw-dir` to download episodes automatically.

## 2. Configure Fine-Tuning

Training configs live in `src/openpi/training/config.py`. Copy an existing
Mobile ALOHA entry (for example,
`pi0_aloha_pen_uncap_b5_low_mem_finetune`) and adjust fields such as `repo_id`,
`num_train_steps`, and prompts. Make sure the config references the dataset name
you chose in the previous step.

## 3. Compute Normalization Statistics

```bash
uv run scripts/compute_norm_stats.py --config-name <config_name>
```

This writes statistics into the training artifacts directory the config points to.

## 4. Launch Training

Start fine-tuning with:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name>   --exp-name=<experiment_name>  --overwrite
```

`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` lets JAX use up to 90% of your GPU memory.
Training logs appear in the console, checkpoints go under `checkpoints/`, and
Weights & Biases logging is enabled if your environment is configured.

## 5. Serve the Policy

After training, serve a checkpoint (replace the iteration with the checkpoint you
want):

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=<config_name> \
  --policy.dir=checkpoints/<config_name>/<config_name>/<iteration>
```

The server listens on port 8000 for observation queries. Follow your ALOHA or
Libero evaluation script to stream observations to the server for rollouts.

## Additional Resources

- [ALOHA Pen-Task Hand Naming Convention](ALOHA_DATASET_NAMING.md) — the paper and the
  released dataset use mirror `<left>`/`<right>` conventions for the pen-uncap tasks;
  includes the mapping and a corrected statistics table.
