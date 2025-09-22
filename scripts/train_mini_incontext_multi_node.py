import dataclasses
import functools
import logging
import platform
from typing import Any

import flax
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

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

### XJ: multi-node helper

import os
import socket
import subprocess
from contextlib import contextmanager

# ---------- Multi-host helpers (SLURM-aware) ----------

def _expand_slurm_nodelist(nodelist: str) -> list[str]:
    """Return a concrete host list from SLURM_NODELIST, or [] if not available."""
    try:
        out = subprocess.check_output(["bash", "-lc", f"scontrol show hostnames {nodelist}"], text=True)
        return [h.strip() for h in out.splitlines() if h.strip()]
    except Exception:
        return []

def _infer_dist_from_env() -> dict:
    """
    Infer coordinator_address/num_processes/process_id from SLURM (preferred) or generic envs.
    Returns {} if single-process fallback is intended.
    """
    # Prefer explicit overrides if user provides
    coord_host = os.environ.get("COORDINATOR_HOST")
    coord_port = os.environ.get("COORDINATOR_PORT", "12355")  # default port
    proc_id = os.environ.get("JAX_PROCESS_ID")
    num_proc = os.environ.get("JAX_NUM_PROCESSES")

    if coord_host and proc_id and num_proc:
        return {
            "coordinator_address": f"{coord_host}:{coord_port}",
            "num_processes": int(num_proc),
            "process_id": int(proc_id),
            "source": "ENV(JAX_*)"
        }

    # SLURM path
    slurm_procid = os.environ.get("SLURM_PROCID")
    slurm_ntasks = os.environ.get("SLURM_NTASKS")
    slurm_nodelist = os.environ.get("SLURM_NODELIST")

    if slurm_procid is not None and slurm_ntasks is not None and slurm_nodelist:
        hosts = _expand_slurm_nodelist(slurm_nodelist)
        if hosts:
            # coordinator = first host
            coord = f"{hosts[0]}:{coord_port}"
            return {
                "coordinator_address": coord,
                "num_processes": int(slurm_ntasks),
                "process_id": int(slurm_procid),
                "source": "SLURM"
            }

    # No multi-host signals -> single process
    return {}

def init_multi_host_if_needed() -> dict:
    """
    Initialize jax.distributed if running under SLURM or provided JAX_* envs.
    Must be called before any jax.devices() access.
    """
    info = _infer_dist_from_env()
    if info:
        # Avoid double-initialize
        try:
            # if already initialized, this is a no-op
            jax.distributed.initialize(
                coordinator_address=info["coordinator_address"],
                num_processes=info["num_processes"],
                process_id=info["process_id"],
            )
        except RuntimeError as e:
            # Already initialized or similar — safe to continue
            logging.info(f"[dist] jax.distributed already initialized: {e}")
        logging.info(f"[dist] initialized via {info['source']}: {info}")
    else:
        logging.info("[dist] single-process (no SLURM/JAX_* env detected)")
    return info

def is_coordinator() -> bool:
    """Global rank 0."""
    try:
        return jax.process_index() == 0
    except Exception:
        return True

def world_size() -> int:
    try:
        return jax.process_count()
    except Exception:
        return 1

def rank() -> int:
    try:
        return jax.process_index()
    except Exception:
        return 0

def sync_barrier():
    """Cross-host barrier."""
    try:
        from jax.experimental import multihost_utils as mhu
        mhu.sync_global_devices("barrier")
    except Exception:
        # Fallback: force a trivial collective
        _ = jax.device_get(jnp.sum(jnp.ones(())))

### XJ: multi-node helper end


# def init_logging():
#     """Custom logging format for better readability."""
#     level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

#     class CustomFormatter(logging.Formatter):
#         def format(self, record):
#             record.levelname = level_mapping.get(record.levelname, record.levelname)
#             return super().format(record)

#     formatter = CustomFormatter(
#         fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
#         datefmt="%H:%M:%S",
#     )

#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     logger.handlers[0].setFormatter(formatter)

### XJ: multi-node logger
def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            # Prefix with [rank/size] and hostname for multi-host clarity
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
    # Ensure at least one handler
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    ### XJ: multi node helper 
    # if not enabled:
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


# def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
#     """Loads and validates the weights. Returns a loaded subset of the weights."""
#     loaded_params = loader.load(params_shape)
#     at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

#     # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
#     return traverse_util.unflatten_dict(
#         {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
#     )
def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader,
    params_shape: at.Params,
    *,
    vision_encoder_loader: _weight_loaders.WeightLoader | None = None,
    vision_encoder_prefix: str = "PaliGemma/img",
    verbose: bool = False,
) -> at.Params:
    """
    Load model parameters, optionally replacing vision encoder subtree,
    and validate against the full target model structure.
    """

    # Step 1: Load base weights
    loaded_params = loader.load(params_shape)

    if vision_encoder_loader is None:
        at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
        return flax.traverse_util.unflatten_dict({
            k: v for k, v in flax.traverse_util.flatten_dict(loaded_params, sep="/").items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }, sep="/")

    # Step 2: Replace vision encoder subtree (if needed)
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    flat_vision = flax.traverse_util.flatten_dict(vision_encoder_loader.load(params_shape), sep="/")
    flat_expected = flax.traverse_util.flatten_dict(params_shape, sep="/")

    # Step 3a: Remove existing vision encoder keys
    flat_loaded = {
        k: v for k, v in flat_loaded.items()
        if not (k == vision_encoder_prefix or k.startswith(vision_encoder_prefix + "/"))
    }

    # Step 3b: Add vision encoder keys from the override
    flat_loaded.update({
        k: v for k, v in flat_vision.items()
        if k == vision_encoder_prefix or k.startswith(vision_encoder_prefix + "/")
    })
    
    # Step 3c: Fill in known-missing keys (e.g. image_proj,paligemma\llm, all other projection layers) to make check_pytree_equality pass
    for k, v in flat_expected.items():
        if k not in flat_loaded:
            flat_loaded[k] = v

    merged = flax.traverse_util.unflatten_dict(flat_loaded, sep="/")
    
    # Debug: Print vision encoder key diffs after replacement
    if verbose:
        logging.info("[Debug] Checking vision encoder key differences after replacement:")

        replaced_keys = [k for k in flat_vision if k.startswith(vision_encoder_prefix)]
        for k in replaced_keys:
            expected = flat_expected.get(k)
            actual = flat_loaded.get(k)

            if expected is None:
                logging.warning(f"Key '{k}' not found in expected model params.")
                continue
            if actual is None:
                logging.warning(f"Key '{k}' missing in loaded params after replacement.")
                continue
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                logging.warning(
                    f"Key '{k}' shape/dtype mismatch: "
                    f"expected {expected.shape}@{expected.dtype}, got {actual.shape}@{actual.dtype}"
                )
            else:
                logging.debug(f"Key '{k}' matches expected shape and dtype.")


    # Step 4: Validate structure
    at.check_pytree_equality(expected=params_shape, got=merged, check_shapes=True, check_dtypes=True)

    # Step 5: Remove ShapeDtypeStruct
    flat_cleaned = {
        k: v for k, v in flat_loaded.items()
        if not isinstance(v, jax.ShapeDtypeStruct)
    }
    return flax.traverse_util.unflatten_dict(flat_cleaned, sep="/")


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        # import ipdb; ipdb.set_trace()
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
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

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    # XJ:debug
    if isinstance(config.vision_weight_loader, _weight_loaders.NoOpWeightLoader):
        partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    else:
        partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict(), vision_encoder_loader = config.vision_weight_loader)
        
    
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.ObservationIncontext, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.ObservationIncontext, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    # import ipdb; ipdb.set_trace()
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
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

    # Filter out params that aren't kernels.
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


def main(config: _config.TrainConfig):
    init_logging()
   
    ### XJ: multi-node helper 
    #logging.info(f"Running on: {platform.node()}")
    dist_info = init_multi_host_if_needed()
    logging.info(f"Running on: {platform.node()} | rank={rank()}/{world_size()} | local_device_count={jax.local_device_count()} | global_device_count={jax.device_count()}")


    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the GLOBAL device count {jax.device_count()}"
        )

    jax.config.update("jax_threefry_partitionable", True)  # noqa: FBT003
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    ### XJ: multi-node rng
    # rng = jax.random.key(config.seed)
    base_rng = jax.random.key(config.seed)
    rng = jax.random.fold_in(base_rng, rank())
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    ### XJ: multi-node barrier
    sync_barrier()

    data_loader = _data_loader.create_incontext_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    # jax.debug.print("batch = {} ", batch)
    # import ipdb; ipdb.set_trace()

    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    
    ### XJ: multi-node helper
    # jax.block_until_ready(train_state)
    # logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # if resuming:
    #     train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
    
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
    else:
        jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")


    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        ### XJ: multi node helper
        disable=not is_coordinator(),
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        ### XJ; multi node helper
        # if step % config.log_interval == 0:
        if is_coordinator() and step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        ### XJ: multi node helper
        # if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
        if is_coordinator() and (((step % config.save_interval) == 0 and step > start_step) or step == config.num_train_steps - 1):
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    if is_coordinator():
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
