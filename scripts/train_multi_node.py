# -*- coding: utf-8 -*-
"""
Multi-node / multi-GPU training script (SLURM-aware, JAX/Flax NNX).
This version adds:
  - jax.distributed.initialize() based on SLURM envs (or manual JAX_* envs)
  - rank0-only W&B / checkpoint / progress bar
  - deterministic per-process PRNG via fold_in(seed, rank)
  - robust resume: rank0 restores from disk, then broadcasts arrays to all ranks
  - sanity checks for global device count and fsdp_devices
  - global barriers (synchronization) around critical phases

All comments are written in English for clarity.
"""

import dataclasses
import functools
import logging
import platform
from typing import Any

import os
import socket
import subprocess
import numpy as np

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

# --- Project modules ---
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


# =============================================================================
# Multi-node helpers (SLURM-aware)
# =============================================================================

def _expand_slurm_nodelist(nodelist: str) -> list[str]:
    """Expand SLURM_NODELIST (like 'gpu[01-02]') into concrete hostnames.

    Returns an empty list if scontrol is unavailable or expansion fails.
    """
    try:
        out = subprocess.check_output(["bash", "-lc", f"scontrol show hostnames {nodelist}"], text=True)
        return [h.strip() for h in out.splitlines() if h.strip()]
    except Exception:
        return []


def _infer_dist_from_env() -> dict:
    """Infer distributed setup from environment variables.

    Priority:
      1) Explicit JAX_* + COORDINATOR_HOST(/PORT)
      2) SLURM: SLURM_PROCID / SLURM_NTASKS / SLURM_NODELIST
      3) Fallback: single-process

    Returns a dict with keys:
      {coordinator_address, num_processes, process_id, source}
    Or {} for single-process mode.
    """
    # Explicit overrides (useful outside SLURM or for custom launchers)
    coord_host = os.environ.get("COORDINATOR_HOST")
    coord_port = os.environ.get("COORDINATOR_PORT", "12355")
    proc_id = os.environ.get("JAX_PROCESS_ID")
    num_proc = os.environ.get("JAX_NUM_PROCESSES")
    if coord_host and proc_id and num_proc:
        return {
            "coordinator_address": f"{coord_host}:{coord_port}",
            "num_processes": int(num_proc),
            "process_id": int(proc_id),
            "source": "ENV(JAX_*)",
        }

    # SLURM path
    slurm_procid = os.environ.get("SLURM_PROCID")
    slurm_ntasks = os.environ.get("SLURM_NTASKS")
    slurm_nodelist = os.environ.get("SLURM_NODELIST")
    if slurm_procid is not None and slurm_ntasks is not None and slurm_nodelist:
        hosts = _expand_slurm_nodelist(slurm_nodelist)
        if hosts:
            coord = f"{hosts[0]}:{coord_port}"  # Use the first node as coordinator
            return {
                "coordinator_address": coord,
                "num_processes": int(slurm_ntasks),
                "process_id": int(slurm_procid),
                "source": "SLURM",
            }

    # No distributed signals found → single-process
    return {}


def init_multi_host_if_needed() -> dict:
    """Initialize jax.distributed if multi-process is detected.

    IMPORTANT: must be called BEFORE any jax.devices(), jax.device_count(),
    or mesh/sharding construction.
    """
    info = _infer_dist_from_env()
    if info:
        try:
            jax.distributed.initialize(
                coordinator_address=info["coordinator_address"],
                num_processes=info["num_processes"],
                process_id=info["process_id"],
            )
        except RuntimeError as e:
            # Safe to continue if it was already initialized by the runtime/launcher.
            logging.info(f"[dist] jax.distributed already initialized: {e}")
        logging.info(f"[dist] initialized via {info['source']}: {info}")
    else:
        logging.info("[dist] single-process (no SLURM/JAX_* env detected)")
    return info


def is_coordinator() -> bool:
    """Return True for the global process 0 (rank 0)."""
    try:
        return jax.process_index() == 0
    except Exception:
        return True


def world_size() -> int:
    """Total number of JAX processes (global)."""
    try:
        return jax.process_count()
    except Exception:
        return 1


def rank() -> int:
    """Global rank of the current process."""
    try:
        return jax.process_index()
    except Exception:
        return 0


def sync_barrier(tag: str = "barrier"):
    """Global synchronization barrier across all processes/devices.

    Uses multihost_utils under the hood; falls back to a trivial collective.
    """
    try:
        from jax.experimental import multihost_utils as mhu
        mhu.sync_global_devices(tag)
    except Exception:
        _ = jax.device_get(jnp.sum(jnp.ones(())))


def _materialize_arrays_from_shapes(tree):
    """Replace jax.ShapeDtypeStruct leaves with empty arrays of matching shape/dtype.

    This is useful on non-coordinator ranks when we want to broadcast from rank0:
    all ranks must provide an array to the collective, even if it's just a placeholder.
    """
    def _to_array(x):
        if isinstance(x, jax.ShapeDtypeStruct):
            return jnp.empty(x.shape, x.dtype)
        return x
    return jax.tree.map(
        _to_array,
        tree,
        is_leaf=lambda x: isinstance(x, (jax.ShapeDtypeStruct, jax.Array, np.ndarray)),
    )


def _broadcast_state_from_rank0(state):
    """Broadcast array leaves of `state` from rank0 to all ranks.

    Non-array leaves (e.g., Python scalars, small dataclass metadata) pass through.
    For safety, we first materialize ShapeDtypeStruct leaves to arrays.
    """
    from jax.experimental import multihost_utils as mhu

    state = _materialize_arrays_from_shapes(state)

    def _bcast(x):
        if isinstance(x, (jax.Array, np.ndarray)):
            return mhu.broadcast_one_to_all(x)
        return x

    return jax.tree.map(
        _bcast,
        state,
        is_leaf=lambda x: isinstance(x, (jax.ShapeDtypeStruct, jax.Array, np.ndarray)),
    )


# =============================================================================
# Logging (rank-aware)
# =============================================================================

def init_logging():
    """Rank-aware logging.

    - INFO level on rank0, WARNING on others to reduce noise.
    - Prefix logs with [R{rank}/{world}@{hostname}] for clarity in multi-node runs.
    """
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            try:
                record.rank = rank()
                record.world = world_size()
            except Exception:
                record.rank = 0
                record.world = 1
            record.host = socket.gethostname().split(".")[0]
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s R%(rank)d/%(world)d@%(host)s] "
            "%(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO if is_coordinator() else logging.WARNING)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.handlers[0].setFormatter(formatter)


# =============================================================================
# W&B (rank0 only)
# =============================================================================

def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    """Initialize Weights & Biases.

    - Only rank0 actually creates/updates the run.
    - Other ranks are put into 'disabled' mode to avoid network/file storms.
    """
    if not enabled or not is_coordinator():
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


# =============================================================================
# Weight loading (simple variant)
# =============================================================================

def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Load weights into the target param structure (with validation).

    Returns the subset with concrete arrays (no ShapeDtypeStruct leaves).
    """
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items()
         if not isinstance(v, jax.ShapeDtypeStruct)}
    )


# =============================================================================
# Train state init
# =============================================================================

@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """Create TrainState or its shape spec (when resume=True) with sharding."""
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        # Optionally insert preloaded weights (subset) into the model state
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)  # Validates subset structure
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Cast frozen params to bfloat16 to save memory/bandwidth
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    # Build shape-only TrainState to derive sharding
    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        # For resume, return only the shapes + sharding (real arrays are restored later)
        return train_state_shape, state_sharding

    # Fresh run: load initial weights (subset) then construct real TrainState
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # Donate partial_params buffer
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# =============================================================================
# Single training step
# =============================================================================

@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """One training step: forward, loss, backward, optimizer update, EMA."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Compute gradients only for trainable (unfrozen) params
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # In-place update of model and recover full state
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Useful diagnostics (only counting >1D kernels)
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


# =============================================================================
# Main
# =============================================================================

def main(config: _config.TrainConfig):
    # 1) Setup rank-aware logging early
    init_logging()

    # 2) Initialize multi-host (MUST be before jax.device_count()/mesh creation)
    _ = init_multi_host_if_needed()
    logging.info(
        f"Running on host={platform.node()} | rank={rank()}/{world_size()} | "
        f"local_device_count={jax.local_device_count()} | global_device_count={jax.device_count()}"
    )

    # 3) Sanity checks on global device count and global batch size
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the GLOBAL device count {jax.device_count()}."
        )

    # If you use parameter sharding (FSDP), its device-axis must divide global devices.
    if jax.device_count() % int(config.fsdp_devices) != 0:
        raise ValueError(
            f"config.fsdp_devices={config.fsdp_devices} must divide the GLOBAL device count {jax.device_count()}."
        )

    # 4) JAX runtime options (nice-to-have)
    jax.config.update("jax_threefry_partitionable", True)  # enable partitionable PRNG
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 5) Make per-process PRNG: same base seed, different fold_in per global rank
    base_rng = jax.random.key(config.seed)
    rng = jax.random.fold_in(base_rng, rank())
    train_rng, init_rng = jax.random.split(rng)

    # 6) Build global mesh and sharding
    #    Tip: If you want FSDP only within a single node (common for 8-GPU nodes),
    #    set config.fsdp_devices = jax.local_device_count().
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 7) Checkpoint manager + W&B (rank0 only)
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 8) Optional barrier so that all ranks start data loading together
    sync_barrier("before_data")

    # 9) Data loader: expected to shard by DATA axis so each process/device reads its own slice
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # 10) Initialize TrainState (shape-only on resume=True), then restore/broadcast
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)

    if resuming:
        # Rank0 reads from disk; others wait, then receive broadcast.
        if is_coordinator():
            train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        sync_barrier("after_restore_rank0")
        train_state = _broadcast_state_from_rank0(train_state)
    else:
        # Fresh run: ensure arrays are materialized and compilations are finished
        jax.block_until_ready(train_state)

    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # 11) JIT-compile the training step with proper sharding
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),  # donate TrainState to reduce memory pressure
    )

    # 12) Main training loop
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=not is_coordinator(),  # progress bar only on rank0
    )

    infos = []
    for step in pbar:
        # set_mesh context ensures collectives follow the same mesh topology
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        # Logging/W&B only on rank0 to avoid duplicated logs
        if is_coordinator() and step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        # Next batch
        batch = next(data_iter)

        # Save checkpoints only on rank0
        if is_coordinator() and (((step % config.save_interval) == 0 and step > start_step)
                                 or step == config.num_train_steps - 1):
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    # 13) Graceful shutdown: wait for checkpoint IO on rank0, then sync and exit
    if is_coordinator():
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()
    sync_barrier("before_exit")


if __name__ == "__main__":
    # Entry point for CLI: your _config.cli() should parse args and return TrainConfig.
    main(_config.cli())
