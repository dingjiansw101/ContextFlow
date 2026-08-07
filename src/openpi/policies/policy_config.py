from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.policies.policy_incontext as _policy_incontext
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training.data_loader import create_dataset
from openpi.training.data_loader import transform_dataset
import openpi.transforms as transforms


@dataclasses.dataclass
class PolicyConfig:
    model: _model.BaseModel
    norm_stats: dict[str, transforms.NormStats]

    input_layers: Sequence[transforms.DataTransformFn]
    output_layers: Sequence[transforms.DataTransformFn]

    model_type: _model.ModelType = _model.ModelType.PI0  # TODO: check where used the model_type
    default_prompt: str | None = None
    sample_kwargs: dict[str, Any] | None = None


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    # TODO: check, use_quantile_norm is false in the pi0_aloha_handover, for training and tesging
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    return _policy.Policy(
        model,
        # TODO: check the transforms here, if it is the same as the one in the training
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),  # prompt here is language instruction for a task
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )


def create_trained_policy_incontext(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    inference_dtype: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        inference_dtype: Optional dtype override for inference (e.g., "float32", "bfloat16"). If not
            provided, defaults to bfloat16.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Resolve inference dtype
    if inference_dtype is None:
        # Default to bfloat16 for backward compatibility
        dtype = jnp.bfloat16
    else:
        # Convert string dtype to JAX dtype
        dtype_map = {
            "bfloat16": jnp.bfloat16,
            "float32": jnp.float32,
            "float16": jnp.float16,
        }
        dtype = dtype_map.get(inference_dtype, jnp.bfloat16)

    logging.info(f"Loading model with dtype: {dtype}...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=dtype))
    data_config = train_config.data.create_policy(train_config.assets_dirs, train_config.model)
    # TODO: check, use_quantile_norm is false in the pi0_aloha_handover, for training and tesging
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    # When the data config already populates dem_prompt_* via its own data_transforms
    # (e.g. CustomLeRobotLiberoIncontextDataConfig), we must NOT also build a dataset and
    # append the cache-based AddImagePromptTransform / AddStatesActionsPromptTransform,
    # which would re-read or rebuild JSON state/action caches at startup.
    provides_incontext_demos = getattr(data_config, "provides_incontext_demos", False)

    if not provides_incontext_demos:
        dataset_data_config = dataclasses.replace(data_config, norm_stats=norm_stats)
        dataset = create_dataset(data_config, train_config.model)
        dataset = transform_dataset(dataset, dataset_data_config)
    else:
        dataset = None

    # Mirror training (data_loader.py: transform_dataset forwards
    # config.data.norm_stats_aliases into Normalize). When a config sets aliases —
    # e.g. CustomLeRobotLiberoIncontextDataConfig maps dem_prompt_all_states → "state"
    # and dem_prompt_all_actions → "actions" — eval must apply the same aliases or the
    # demos arrive un-normalized while training had them normalized.
    norm_stats_aliases = getattr(train_config.data, "norm_stats_aliases", None)
    norm_stats_alias_pad_dims = getattr(train_config.data, "norm_stats_alias_pad_dims", None)

    input_transforms = [
        *repack_transforms.inputs,
        transforms.InjectDefaultPrompt(default_prompt),  # prompt here is language instruction for a task
        *data_config.data_transforms.inputs,
        transforms.Normalize(
            norm_stats,
            use_quantiles=data_config.use_quantile_norm,
            norm_stats_aliases=norm_stats_aliases,
            norm_stats_alias_pad_dims=norm_stats_alias_pad_dims,
        ),
        *data_config.model_transforms.inputs,
    ]

    if not provides_incontext_demos:
        if train_config.model.use_image_prompts:
            print("Inference: Adding image prompts")
            input_transforms.append(transforms.AddImagePromptTransform(dataset))

        if train_config.model.use_action_state_prompts:
            print("Inference: Adding action state prompts")
            demo_state_dim = getattr(train_config.data, "demo_state_dim", None)
            padding_mode = getattr(train_config.data, "padding_mode", "keep_all")
            mask_padding_as_valid = getattr(train_config.data, "mask_padding_as_valid", False)
            input_transforms.append(
                transforms.AddStatesActionsPromptTransform(
                    dataset=dataset,
                    max_len=train_config.model.sample_actions,
                    states_cache_path=train_config.data.states_cache_path,
                    actions_cache_path=train_config.data.actions_cache_path,
                    padding_mode=padding_mode,
                    mask_padding_as_valid=mask_padding_as_valid,
                    demo_state_dim=demo_state_dim,
                )
            )

    return _policy_incontext.PolicyIncontext(
        model,
        # TODO: check the transforms here, if it is the same as the one in the training
        transforms=input_transforms,
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
