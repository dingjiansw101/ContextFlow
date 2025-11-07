# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Environment Setup
```bash
# Clone with submodules (required for LeRobot dependency)
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
# Or update submodules if already cloned:
git submodule update --init --recursive

# Install dependencies with uv (https://docs.astral.sh/uv/)
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### Training Workflow
```bash
# 1. Compute normalization statistics (required before first training run)
uv run scripts/compute_norm_stats.py --config-name <config_name>

# 2. Run training (set XLA flag to maximize GPU memory usage)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<experiment_name> [--overwrite]

# Common config names: pi0_fast_libero, pi0_libero, pi0_fast_droid, pi0_aloha_towel
# --overwrite flag overwrites existing checkpoints with the same name
```

### Inference/Serving
```bash
# Spin up a policy server (listens on port 8000 by default)
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=<config_name> \
  --policy.dir=checkpoints/<config_name>/<exp_name>/<iteration>

# Test inference without a robot (generates random observations)
uv run examples/simple_client/simple_client.py --checkpoint-dir <path>
```

### Testing
```bash
# Run all tests
uv run pytest

# Run tests in a specific module
uv run pytest src/openpi/models/pi0_test.py

# Run tests excluding manual ones (manual tests require specific data/models)
uv run pytest -m "not manual"
```

### Code Quality
```bash
# Lint and format code (auto-fix issues)
uv run ruff check . --fix
uv run ruff format .

# Install pre-commit hooks (runs uv-lock and ruff automatically)
uv run pre-commit install
```

## Architecture Overview

### Package Structure

The codebase is organized into six main packages under `src/openpi/`:

- **models/**: Core neural network implementations
  - Base model abstractions (`model.py`)
  - π₀ diffusion model (`pi0.py`)
  - π₀-FAST autoregressive model (`pi0_fast.py`)
  - In-context learning variants (`pi0_incontextv*.py`)
  - Shared components: SigLIP vision encoder, Gemma LLM, LoRA adapters

- **policies/**: Policy wrappers that bridge models to robot environments
  - `policy.py`: Base Policy class with input/output transforms
  - Robot-specific policies: `libero_policy.py`, `aloha_policy.py`, `droid_policy.py`
  - Each defines mapping from robot observations → model inputs and model outputs → robot actions

- **training/**: Training pipeline, data loading, and configuration
  - `config.py`: Central registry of all training configurations
  - `data_loader.py`: LeRobot dataset loading with transform composition
  - `custom_dataset.py`: Extended dataset for in-context learning (loads demo frames)
  - `weight_loaders.py`: Checkpoint loading utilities (supports LoRA, partial loading)

- **serving/**: Production inference infrastructure
  - `websocket_policy_server.py`: Async WebSocket server for remote inference
  - Uses msgpack_numpy serialization for efficient array transfer

- **shared/**: Common utilities
  - `download.py`: S3 checkpoint downloading (caches to `~/.cache/openpi`)
  - `normalize.py`: Data normalization (z-score and quantile-based)
  - `image_tools.py`: Image preprocessing (resize with padding, format conversion)
  - `nnx_utils.py`: JAX/Flax NNX helpers (JIT wrapping, module path filtering)

- **transforms.py**: Composable data transformation pipeline
  - Protocol-based design: `DataTransformFn` for type safety
  - Key transforms: repack, image prompts, state/action prompts, normalization
  - Used in both training and inference

### Model Architecture: π₀ vs π₀-FAST

**π₀ (Diffusion-based)**
- Architecture: PaliGemma (vision-language) + action expert (Gemma)
- Inference: 10 denoising steps to generate action sequences
- Action representation: Continuous vectors with diffusion noise
- Config defaults: `action_horizon=50`, `action_dim=32`
- Use case: Better language grounding, slower inference

**π₀-FAST (Autoregressive)**
- Architecture: Single modified Gemma with KV-cache
- Inference: Single forward pass with autoregressive decoding
- Action representation: Discrete tokens via FAST action tokenizer
- Config defaults: `action_horizon=32`, `action_dim=32`
- Use case: Faster inference, 0-shot generalization (e.g., DROID checkpoint)

**In-Context Models** (`pi0_incontextv*`)
- Extension of base models with demonstration conditioning
- Loads demo frames from similar episodes during training/inference
- Uses `CustomLeRobotDataset` to retrieve demonstration data
- Observation type includes: current frames + demo images/states/actions

### Data Pipeline Flow

```
Raw LeRobot Dataset
    ↓ (repack_transforms: dataset-specific key mapping)
Standardized Format (observation/image, observation/state, actions, prompt)
    ↓ (data_transforms: robot-specific preprocessing)
Normalized Robot Data
    ↓ (model_transforms: model-specific augmentation)
Model-ready Observation
    {
      images: {key: float32[B, H, W, 3]} in [-1, 1],
      image_masks: {key: bool[B]},
      state: float32[B, state_dim],
      tokenized_prompt: int32[B, max_token_len],
      tokenized_prompt_mask: bool[B, max_token_len]
    }
```

Transforms are **composable** and defined in config's `DataConfig`:
- `repack_transforms`: Map dataset keys to standard format
- `data_transforms`: Robot-specific (resize, color space)
- `model_transforms`: Model-specific (augmentation, normalization)

### Configuration System

Training configs follow a nested structure defined in `src/openpi/training/config.py`:

```python
TrainConfig:
  ├── model_config: Pi0Config | Pi0FASTConfig | ...
  │   ├── action_dim, action_horizon, max_token_len
  │   ├── get_freeze_filter() → LoRA pattern
  │   └── inputs_spec() → Expected tensor shapes
  ├── train_config: TrainingHyperparams
  │   ├── learning_rate, batch_size, num_iterations
  │   └── optimizer, checkpoint frequency
  ├── data_config: DataConfig
  │   ├── repo_id: LeRobot dataset identifier
  │   ├── Transform pipeline (repack → data → model)
  │   ├── norm_stats: Precomputed normalization statistics
  │   └── Filter options (task language, episode lists)
  ├── assets_config: AssetsConfig (checkpoints, norm stats paths)
  └── weight_loaders: List[WeightLoader] for pre-training
```

**Important**: All configs are registered in the `_CONFIGS` dict at the bottom of `config.py`. Use `get_config(name)` to retrieve them.

## Key Development Patterns

### Adding a New Robot Platform

To support a new robot, create three components:

1. **Input Transform** (observation → model format):
```python
@dataclasses.dataclass(frozen=True)
class MyRobotInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: ModelType = ModelType.PI0

    def __call__(self, data: dict) -> dict:
        return {
            "state": transforms.pad_to_dim(data["observation/state"], self.action_dim),
            "image": {
                "camera1_rgb": parse_image(data["observation/camera1"]),
                # ... more cameras
            },
            "image_mask": {"camera1_rgb": np.True_, ...},
            "prompt": data.get("prompt", ""),
        }
```

2. **Output Transform** (model actions → robot format):
```python
@dataclasses.dataclass(frozen=True)
class MyRobotOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Extract relevant action dimensions
        return {"actions": np.asarray(data["actions"][:, :robot_action_dim])}
```

3. **Training Config** (in `config.py`):
```python
def get_myrobot_config() -> TrainConfig:
    return TrainConfig(
        model_config=pi0_fast.Pi0FASTConfig(action_dim=robot_action_dim),
        data_config=DataConfig(
            repo_id="huggingface/dataset-name",
            repack_transforms=[...],  # Map dataset keys
            data_transforms=[...],    # Resize, normalize images
            model_transforms=[...],   # Augmentation
            norm_stats=...,           # Computed via compute_norm_stats.py
        ),
        train_config=...,
        assets_config=...,
        weight_loaders=[...],  # Load base model checkpoint
    )

# Register in _CONFIGS dict
_CONFIGS["myrobot"] = lambda: get_myrobot_config()
```

See `src/openpi/policies/libero_policy.py` and `src/openpi/training/config.py` (search for `LiberoInputs`, `LiberoOutputs`, `get_pi0_fast_libero_config`) for complete examples.

### Normalization Statistics Workflow

Models require precomputed normalization stats for actions and states:

1. **Compute stats** (must be done before training):
   ```bash
   uv run scripts/compute_norm_stats.py --config-name myrobot
   ```

2. **Stats are cached** in `~/.cache/openpi/norm_stats/<config_hash>.pkl`

3. **Stats are loaded** during training from cache or config's `norm_stats` field

4. **Stats are saved** with checkpoints in `assets.norm_stats`

5. **Stats are applied** via `NormalizeTransform` in the model transform pipeline

**Important**: If you change the data config (repo_id, episode filters, etc.), you must recompute norm stats.

### Weight Loading and LoRA

To fine-tune from a pre-trained checkpoint:

```python
# In your training config:
weight_loaders = [
    weight_loaders.LocalCheckpointLoader(
        checkpoint_path=download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_base"),
        # Optional: specify LoRA freeze pattern
        freeze_filter=model_config.get_freeze_filter(),
    )
]
```

**LoRA freezing**:
- π₀ base: Freezes vision encoder + LLM, trains only action expert
- π₀-FAST base: Freezes vision encoder, trains LoRA adapters in LLM
- Full fine-tuning: Set `freeze_filter=None`

See `src/openpi/training/weight_loaders.py` for loader implementations.

### Transform Composition

Transforms are **protocol-based** (`DataTransformFn`) and composable:

```python
from openpi import transforms as T

# Compose multiple transforms
pipeline = T.Chain([
    T.RepackTransform(mapping={...}),
    T.NormalizeActionsTransform(norm_stats=...),
    T.NormalizeStateTransform(norm_stats=...),
    T.NormalizeImagesTransform(output_range=(-1, 1)),
])

# Apply to data
processed = pipeline(raw_data)
```

**Key transforms**:
- `RepackTransform`: Remap keys (dataset → standard format)
- `NormalizeActionsTransform` / `NormalizeStateTransform`: Apply norm stats
- `NormalizeImagesTransform`: Scale images to [-1, 1] or [0, 1]
- `AddImagePromptTransform`: Add demo images (for in-context models)
- `AddStatesActionsPromptTransform`: Add demo states/actions
- `ResizeImageTransform`: Resize with aspect ratio preservation

All transforms are **stateless** and **dataclass-frozen** for JAX compatibility.

## Important Technical Details

### JAX and Flax NNX

This codebase uses **JAX** for numerical computation and **Flax NNX** for neural networks:

- Models inherit from `nnx.Module`
- Training uses `nnx.Optimizer` with functional state management
- JIT compilation via `nnx.jit` (wrapped in `nnx_utils.module_jit`)
- **Important**: JAX arrays are immutable; use `.at[].set()` for updates

**Common gotchas**:
- Must use `jax.random.PRNGKey` for randomness (never `np.random`)
- Arrays must be on same device before operations
- JIT requires static shapes; use `jax.lax.cond` instead of Python `if`

### GPU Memory Management

JAX defaults to 75% GPU memory allocation. For training, increase this:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ...
```

Also configure `fsdp_devices` in training config for multi-GPU data parallelism:
```python
train_config = TrainingHyperparams(
    fsdp_devices=4,  # Shard model across 4 GPUs
    ...
)
```

**Note**: Multi-node training is not yet supported.

### Checkpoint Management

Checkpoints use **Orbax** format (JAX PyTree serialization):

- Saved to `checkpoints/<config_name>/<exp_name>/<iteration>/`
- Structure:
  ```
  checkpoint_dir/
    ├── checkpoint         # Orbax checkpoint metadata
    ├── model/             # Model parameters
    └── assets/            # Norm stats, tokenizer, etc.
  ```

**Loading checkpoints**:
```python
from openpi.training import policy_config

policy = policy_config.create_trained_policy(
    config=config,
    checkpoint_dir="path/to/checkpoint/20000"
)
```

### S3 Asset Downloading

Base models and datasets are hosted on S3:

```python
from openpi.shared import download

# Auto-downloads and caches to ~/.cache/openpi
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")

# Override cache location via environment variable
# export OPENPI_DATA_HOME=/custom/cache/path
```

**Supported S3 paths**:
- `s3://openpi-assets/checkpoints/*` - Model checkpoints
- `s3://openpi-assets/datasets/*` - Dataset files

Authentication: Uses AWS credentials from environment or `~/.aws/credentials`.

### Remote Inference Architecture

The WebSocket policy server enables **off-robot inference**:

1. **Server** (GPU machine):
   ```bash
   uv run scripts/serve_policy.py policy:checkpoint --policy.config=... --policy.dir=...
   ```

2. **Client** (robot):
   ```python
   from openpi_client import PolicyClient

   client = PolicyClient(server_url="ws://gpu-server:8000")
   action = client.infer(observation)
   ```

**Protocol**:
- Uses msgpack_numpy for efficient array serialization
- Supports streaming action chunks
- Includes metadata endpoint for capability discovery

See `docs/remote_inference.md` and `packages/openpi-client/` for details.

## Testing Notes

Tests are located throughout the codebase (e.g., `*_test.py` files):

- Unit tests: Fast, no external dependencies
- Integration tests: Marked with `@pytest.mark.manual`, require models/data
- Test fixtures: Defined in `conftest.py`

**Running specific test categories**:
```bash
# All tests except manual
uv run pytest -m "not manual"

# Only model tests
uv run pytest src/openpi/models/

# Specific test file
uv run pytest src/openpi/models/pi0_test.py::test_forward_pass
```

## Common Issues

See the [Troubleshooting section in README.md](README.md#troubleshooting) for common issues and solutions, including:
- Dependency conflicts during `uv sync`
- GPU memory errors during training
- Missing norm stats errors
- Dataset download failures

For bugs or questions not covered in docs, see [CONTRIBUTING.md](CONTRIBUTING.md) for how to file issues or submit PRs.


## TODO list
- [ ] test examples/libero/main_incontext_unseen.py
- [ ] row-gating issue of v17