from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.policies.policy_incontext as _policy_incontext
from openpi.policies.policy_incontext import PolicyFASTIncontext
from openpi.models.pi0_fast_incontext import Pi0FASTIncontextConfig
from openpi.models import tokenizer as _tokenizer
from openpi.models import pi0_fast_incontext_seq as _pi0_fast_incontext_seq
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms
from openpi.training.data_loader import create_dataset, transform_dataset
from openpi.models.pi0_fast_incontext import Pi0FASTIncontextConfig

@dataclasses.dataclass
class PolicyConfig:
    model: _model.BaseModel
    norm_stats: dict[str, transforms.NormStats]

    input_layers: Sequence[transforms.DataTransformFn]
    output_layers: Sequence[transforms.DataTransformFn]

    model_type: _model.ModelType = _model.ModelType.PI0 # TODO: check where used the model_type
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
            transforms.InjectDefaultPrompt(default_prompt), # prompt here is language instruction for a task
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
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    # TODO: check, use_quantile_norm is false in the pi0_aloha_handover, for training and tesging
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    dataset = create_dataset(data_config, train_config.model)
    dataset = transform_dataset(dataset, data_config)

    input_transforms = [
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt), # prompt here is language instruction for a task
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs        ]

    if train_config.model.use_image_prompts:
        print("Inference: Adding image prompts")
        input_transforms.append(
            transforms.AddImagePromptTransform(dataset)
        )

    if train_config.model.use_action_state_prompts:
        print("Inference: Adding action state prompts")
        demo_state_dim = getattr(train_config.data, "demo_state_dim", None)
        if train_config.data.episode_to_indexes_file is not None:
            input_transforms.append(
                        transforms.AddStatesActionsPromptTransform(
                            dataset=dataset,
                            max_len=train_config.model.sample_actions,
                            states_cache_path=train_config.data.states_cache_path,
                            actions_cache_path=train_config.data.actions_cache_path,
                            episode_to_indexes_file=train_config.data.episode_to_indexes_file,
                            padding_mode=train_config.data.padding_mode,
                            mask_padding_as_valid=train_config.data.mask_padding_as_valid,
                            demo_state_dim=demo_state_dim,
                        )
        )
        else:
            input_transforms.append(
                            transforms.AddStatesActionsPromptTransform(
                                dataset=dataset,
                                max_len=train_config.model.sample_actions,
                                states_cache_path=train_config.data.states_cache_path,
                                actions_cache_path=train_config.data.actions_cache_path,
                                padding_mode=train_config.data.padding_mode,
                                mask_padding_as_valid=train_config.data.mask_padding_as_valid,
                                demo_state_dim=demo_state_dim,
                            )
            )

    if isinstance(
        train_config.model,
        (Pi0FASTIncontextConfig, _pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig),
    ):
        fast_policy = create_trained_policy_fast_incontext(
            train_config,
            checkpoint_dir,
            repack_transforms=repack_transforms,
            sample_kwargs=sample_kwargs,
            default_prompt=default_prompt,
            norm_stats=norm_stats,
        )
        return fast_policy

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


def _maybe_fast_tokenizer_path(model_config: _model.BaseModelConfig) -> str:
    return getattr(model_config, "fast_tokenizer_path", "physical-intelligence/fast")


def _build_fast_incontext_transforms(
    train_config: _config.TrainConfig,
    *,
    repack_transforms: transforms.Group,
    default_prompt: str | None,
    norm_stats: dict[str, transforms.NormStats],
) -> tuple[list[transforms.DataTransformFn], list[transforms.DataTransformFn]]:
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = create_dataset(data_config, train_config.model)
    dataset = transform_dataset(dataset, data_config)

    inputs_layers: list[transforms.DataTransformFn] = [
        *repack_transforms.inputs,
        transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    model_config = train_config.model

    if getattr(model_config, "use_image_prompts", False):
        logging.info("Inference: Adding image prompts")
        inputs_layers.append(transforms.AddImagePromptTransform(dataset))

    if getattr(model_config, "use_action_state_prompts", False):
        logging.info("Inference: Adding action state prompts")
        transform_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "max_len": getattr(model_config, "sample_actions", 0),
            "states_cache_path": getattr(train_config.data, "states_cache_path", None),
            "actions_cache_path": getattr(train_config.data, "actions_cache_path", None),
            "demo_state_dim": getattr(train_config.data, "demo_state_dim", None),
        }
        episode_map = getattr(train_config.data, "episode_to_indexes_file", None)
        if episode_map is not None:
            transform_kwargs["episode_to_indexes_file"] = episode_map
        inputs_layers.append(transforms.AddStatesActionsPromptTransform(**transform_kwargs))


    # For Pi0FASTIncontextSeqConfig, TokenizeFASTInputs is already in
    # data_config.model_transforms.inputs (see ModelTransformFactory in
    # training/config.py) and has already run above, so we must not append a
    # second copy (it would find `prompt` already popped and raise).
    # For the tokenized Pi0FASTIncontextConfig variant, the model transforms
    # don't include an incontext tokenizer, so add it here.
    if not isinstance(model_config, _pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig):
        fast_tokenizer = _tokenizer.FASTTokenizer(
            max_len=getattr(model_config, "max_token_len", 256),
            fast_tokenizer_path=_maybe_fast_tokenizer_path(model_config),
        )
        inputs_layers.append(
            transforms.TokenizeFASTIncontextInputs(
                tokenizer=fast_tokenizer,
                max_incontext_steps=getattr(model_config, "sample_actions", 4),
            )
        )

    outputs_layers = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
        *repack_transforms.outputs,
    ]

    return inputs_layers, outputs_layers


def create_trained_policy_fast_incontext(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> PolicyFASTIncontext:
    if not isinstance(
        train_config.model, (Pi0FASTIncontextConfig, _pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig)
    ):
        raise TypeError(
            "create_trained_policy_fast_incontext requires a Pi0FASTIncontextConfig or Pi0FASTIncontextSeqConfig model."
        )

    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(params)

    if norm_stats is None:
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    input_layers, output_layers = _build_fast_incontext_transforms(
        train_config,
        repack_transforms=repack_transforms,
        default_prompt=default_prompt,
        norm_stats=norm_stats,
    )

    return PolicyFASTIncontext(
        model,
        transforms=input_layers,
        output_transforms=output_layers,
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
