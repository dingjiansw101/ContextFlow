import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoaderIncontext(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        # return _merge_params(loaded_params, params, missing_regex=".*")
        fallback_pattern = r".*(?:lora|llm.*_prompt_expert|demo_action_proj|demo_state_proj|img_proj|demo_track_proj|text_proj|image_compressor.*|state_compressor.*|action_compressor.*).*"
        # print(loaded_params['PaliGemma']['llm'].keys())
        # dict_keys(['embedder', 'final_norm', 'final_norm_1', 'layers'])
        # print(params['PaliGemma']['llm'].keys())
        # dict_keys(['embedder', 'final_norm', 'final_norm_1', 'final_norm_prompt_expert', 'layers'])
        return _merge_params(loaded_params, params, missing_regex=fallback_pattern)


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoaderIncontextV17(WeightLoader):
    """Loads weights from a checkpoint for v17 in-context models with future state prediction.

    This loader extends CheckpointWeightLoaderIncontext to support v17 models that include
    additional future state prediction layers. When loading from older checkpoints (e.g., pi0_base)
    that don't have these layers, they will be randomly initialized.

    New v17 layers that can be missing from checkpoint:
      - future_state_in_proj
      - state_time_mlp_in
      - state_time_mlp_out
      - future_state_out_proj
      - future_state_conditioning_proj

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Extended fallback pattern to include v17 future state prediction layers and state expert LLM layers
        fallback_pattern = r".*(?:lora|llm.*_prompt_expert|llm.*_state_expert|demo_action_proj|demo_state_proj|img_proj|demo_track_proj|text_proj|future_state.*|state_time_mlp.*).*"
        return _merge_params(loaded_params, params, missing_regex=fallback_pattern)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # TODO: check the llm keys of PaliGemma
        # import ipdb; ipdb.set_trace()
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class VisionEncoderOnlyLoader(WeightLoader):
    """Loads vision encoder and optionally embedder weights from the official PaliGemma checkpoint.

    This loader is useful when you want to initialize only the vision backbone from PaliGemma
    while randomly initializing the LLM layers. This avoids shape mismatch issues when using
    smaller LLM variants (e.g., gemma_300m) with the PaliGemma 3B checkpoint.

    The embedder can be safely loaded when the prompt_expert variant has width=2048 (e.g., gemma_300m_v2)
    which matches PaliGemma's Gemma-2B embedder dimension.

    Use cases:
    - Loading vision backbone from PaliGemma 3B into models with gemma_300m or gemma_300m_v2
    - Avoiding MLP dimension mismatches (PaliGemma: mlp_dim=16,384 vs gemma_300m: mlp_dim=4,096)
    - Starting with strong vision features while training LLM from scratch
    - Optionally loading pre-trained token embeddings

    Args:
        prefix: Parameter prefix for vision encoder (default: "PaliGemma/img")
        include_embedder: Whether to also load the embedder weights (default: False)
        verbose: Whether to log detailed loading information
    """

    prefix: str = "PaliGemma/img"
    include_embedder: bool = False
    verbose: bool = False

    def load(self, params: at.Params) -> at.Params:
        # Download the official PaliGemma checkpoint
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )

        # Load the checkpoint
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))

        # Unflatten to nested dict structure
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}

        # Flatten and filter vision encoder and optionally embedder parameters
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        if self.include_embedder:
            # Load both vision encoder and embedder
            filtered_params = {
                k: v for k, v in flat_loaded.items()
                if k.startswith(self.prefix) or "llm/embedder" in k
            }
        else:
            # Load only vision encoder
            filtered_params = {k: v for k, v in flat_loaded.items() if k.startswith(self.prefix)}

        if not filtered_params:
            raise ValueError(
                f"No parameters found with prefix '{self.prefix}' in PaliGemma checkpoint. "
                f"Available prefixes: {set(k.split('/')[0] for k in flat_loaded.keys())}"
            )

        # Count vision and embedder parameters separately for logging
        vision_count = sum(1 for k in filtered_params if k.startswith(self.prefix))
        embedder_count = sum(1 for k in filtered_params if "llm/embedder" in k)

        logger.info(
            f"[VisionEncoderOnlyLoader] Loaded {len(filtered_params)} parameters total"
        )
        logger.info(f"  Vision encoder: {vision_count} parameters")
        if self.include_embedder:
            logger.info(f"  Embedder: {embedder_count} parameters")

        if self.verbose:
            # Show vision encoder samples
            vision_keys = [k for k in sorted(filtered_params.keys()) if k.startswith(self.prefix)]
            if vision_keys:
                logger.info("  Vision encoder samples:")
                for k in vision_keys[:5]:
                    logger.info(f"    {k}: shape={filtered_params[k].shape}")
                if len(vision_keys) > 5:
                    logger.info(f"    ... and {len(vision_keys) - 5} more vision parameters")

            # Show embedder samples
            if self.include_embedder:
                embedder_keys = [k for k in sorted(filtered_params.keys()) if "llm/embedder" in k]
                if embedder_keys:
                    logger.info("  Embedder samples:")
                    for k in embedder_keys:
                        logger.info(f"    {k}: shape={filtered_params[k].shape}")

        # Unflatten and merge with reference parameters
        loaded_subset = flax.traverse_util.unflatten_dict(filtered_params, sep="/")

        # Use missing_regex=".*" to fill all non-loaded parameters from random initialization
        return _merge_params(loaded_subset, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class SelectiveVisionAndProjectionsLoader(WeightLoader):
    """Loads vision encoder, embedder, and basic projection layers from pi0_base checkpoint.

    This loader is designed for in-context learning models (like pi0_incontextv18) where you want to:
    - Load pre-trained vision encoder from pi0_base
    - Load pre-trained embedder from pi0_base
    - Load pre-trained basic projection layers (state_proj, action_*) from pi0_base
    - Randomly initialize all LLM layers (prompt expert and action expert)
    - Randomly initialize all in-context specific components (demo projections, compressors)

    This gives a fresh start for the LLM experts while leveraging pre-trained vision and action processing components.

    Components loaded from pi0_base:
    - Vision encoder: PaliGemma/img/*
    - Token embedder: PaliGemma/llm/embedder/*
    - Projections: state_proj, action_in_proj, action_time_mlp_in, action_time_mlp_out, action_out_proj

    Components randomly initialized (not loaded):
    - Prompt expert LLM layers: PaliGemma/llm/layers/*, PaliGemma/llm/final_norm*
    - Action expert LLM layers: PaliGemma/llm/layers/*_1, PaliGemma/llm/final_norm_1
    - LoRA weights (if any): *lora*
    - In-context projections: demo_action_proj, demo_state_proj, img_proj, text_proj, demo_track_proj
    - Compressors: image_compressor, state_compressor, action_compressor

    Args:
        params_path: Path to the pi0_base checkpoint (e.g., "s3://openpi-assets/checkpoints/pi0_base/params")
        verbose: Whether to log detailed loading information
    """

    params_path: str
    verbose: bool = False

    def load(self, params: at.Params) -> at.Params:
        # Load the pi0_base checkpoint
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        # Flatten to filter specific components
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")

        # Define what to load: vision encoder, embedder, and basic projection layers
        load_patterns = [
            r"^PaliGemma/img/.*",                    # Vision encoder
            r"^PaliGemma/llm/embedder/.*",           # Token embedder
            r"^state_proj/.*",                       # State projection
            r"^action_in_proj/.*",                   # Action input projection
            r"^action_time_mlp_in/.*",               # Action time MLP input
            r"^action_time_mlp_out/.*",              # Action time MLP output
            r"^action_out_proj/.*",                  # Action output projection
        ]

        # Compile patterns
        compiled_patterns = [re.compile(pattern) for pattern in load_patterns]

        # Filter loaded parameters to keep only matching components
        filtered_params = {}
        for k, v in flat_loaded.items():
            if any(pattern.match(k) for pattern in compiled_patterns):
                if k in flat_ref:  # Only load if the key exists in the reference params
                    filtered_params[k] = v

        # Count loaded parameters by category for logging
        vision_count = sum(1 for k in filtered_params if k.startswith("PaliGemma/img/"))
        embedder_count = sum(1 for k in filtered_params if k.startswith("PaliGemma/llm/embedder/"))
        projection_count = sum(1 for k in filtered_params if any(
            k.startswith(prefix) for prefix in ["state_proj/", "action_in_proj/", "action_time_mlp_in/",
                                                 "action_time_mlp_out/", "action_out_proj/"]
        ))

        logger.info(
            f"[SelectiveVisionAndProjectionsLoader] Loaded {len(filtered_params)} parameters from pi0_base:"
        )
        logger.info(f"  Vision encoder: {vision_count} parameters")
        logger.info(f"  Embedder: {embedder_count} parameters")
        logger.info(f"  Projections: {projection_count} parameters")

        if self.verbose:
            logger.info("  Loaded parameter samples:")
            for category, prefix in [("Vision", "PaliGemma/img/"), ("Embedder", "PaliGemma/llm/embedder/"),
                                     ("Projections", "state_proj/")]:
                category_keys = [k for k in sorted(filtered_params.keys()) if k.startswith(prefix)]
                if category_keys:
                    logger.info(f"  {category}:")
                    for k in category_keys[:3]:
                        logger.info(f"    {k}: shape={filtered_params[k].shape}")
                    if len(category_keys) > 3:
                        logger.info(f"    ... and {len(category_keys) - 3} more {category.lower()} parameters")

        # Unflatten the filtered parameters
        loaded_subset = flax.traverse_util.unflatten_dict(filtered_params, sep="/")

        # Define what should be randomly initialized (everything else)
        # This regex matches all parameters NOT in the load patterns above
        fallback_pattern = r".*(?:llm/(?:layers|final_norm)|lora|demo_action_proj|demo_state_proj|img_proj|text_proj|demo_track_proj|image_compressor|state_compressor|action_compressor).*"

        # Merge loaded parameters with random initialization
        result = _merge_params(loaded_subset, params, missing_regex=fallback_pattern)

        # Log what's being randomly initialized
        if self.verbose:
            randomly_init_keys = [k for k in flat_ref.keys() if k not in filtered_params]
            logger.info(f"  Randomly initialized: {len(randomly_init_keys)} parameters")
            logger.info("  Random initialization samples:")
            for k in sorted(randomly_init_keys)[:5]:
                logger.info(f"    {k}")
            if len(randomly_init_keys) > 5:
                logger.info(f"    ... and {len(randomly_init_keys) - 5} more randomly initialized parameters")

        return result


# TODO: write an empty weight loader that does not load any weights

def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype)

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


# XJ: for customized ViT variants
@dataclasses.dataclass(frozen=True)
class RemapSigLIPPrefixLoader(WeightLoader):
    # since openpi has rewrite the SigLIP/Vit source code: stack the encoderblocks to the following struction: (depth, ...)
    # we manually stack the loaded public ckp from https://github.com/google-research/vision_transformer 
    # and remap the embedding.kernel, pos_embedding, encoder_norm etc.
    """
    Loads vision encoder weights from a .npz file and remaps flat keys to a target prefix.
    The result is a nested dict like {PaliGemma: {img: ...}} ready for replacement.

    Example:
        npz keys like "block1/conv1/kernel" will be mapped to
        "PaliGemma/img/block1/conv1/kernel"
    """
    npz_path: str
    target_prefix: str = "PaliGemma/img"
    filter_regex: str = ".*"  
    pool_type: str = "none"
    # If verbose=True, print all loaded param keys and shapes for debugging
    verbose: bool = False
    
    # please check the the following comment for reasons of replacing key names
    remap_subkeys = {
        "MlpBlock_3": "MlpBlock_0",
        "MultiHeadDotProductAttention_1": "MultiHeadDotProductAttention_0",
        "LayerNorm_2": "LayerNorm_1"
    }

    def load(self, params: at.Params) -> at.Params:
        from openpi.shared import download
        import collections
        import jax.numpy as jnp

        path = download.maybe_download(self.npz_path)
        try:
            with path.open("rb") as f:
                raw = dict(np.load(f, allow_pickle=False))
        except Exception as e:
            raise ValueError(f"Failed to load .npz from {self.npz_path}: {e}. "
                 f"Make sure the file exists and is a valid .npz archive.")

        if self.verbose:
            logger.info(f"[RemapPrefixLoader] Listing raw .npz content from {self.npz_path}:")
            for k, v in raw.items():
                logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        pattern = re.compile(self.filter_regex)
        sub_params = {}
        stacked_params = collections.defaultdict(list)

        for k, v in raw.items():
            if not pattern.fullmatch(k):
                continue

            # keep record for encoderblock_i 
            m = re.match(r"Transformer/encoderblock_(\d+)/(.*)", k)
            if m:
                block_idx, subkey = m.groups()
                for old_subkey, new_subkey in self.remap_subkeys.items():
                    subkey = subkey.replace(old_subkey, new_subkey)
                final_key = f"{self.target_prefix}/Transformer/encoderblock/{subkey}"
                stacked_params[final_key].append((int(block_idx), jnp.array(v)))
            else:
                if k == "cls":
                    # ignore cls if pool_type=None
                    if self.pool_type=="none":
                        sub_params[k] = jnp.array(v)
                    else:
                        logger.warning("[RemapPrefixLoader] Found `cls` token in .npz, but model pool_type = None; skipping.")
                        continue
                elif k == "Transformer/posembed_input/pos_embedding":
                    # rename pos_embedding: since we need to skip cls token, we need to truncate the pos embedding of cls
                    v = v[:, 1:, :]
                    full_key = f"{self.target_prefix}/pos_embedding"
                    sub_params[full_key] = jnp.array(v)    
                elif k.startswith("head/"):
                    logger.warning(f"[RemapPrefixLoader] Skipping head param `{k}` due to shape mismatch or task-specific logic.")
                    continue
                else:
                    # normal remap
                    full_key = f"{self.target_prefix}/{k}"
                    sub_params[full_key] = jnp.array(v)

        # stack all encoderblock 
        for final_key, items in stacked_params.items():
            items.sort(key=lambda x: x[0])
            tensors = [v for _, v in items]
            stacked_tensor = jnp.stack(tensors, axis=0)
            sub_params[final_key] = stacked_tensor
            
            if self.verbose:
                logger.info(f"[RemapPrefixLoader] stacked {final_key}: shape={stacked_tensor.shape}, dtype={stacked_tensor.dtype}")

        if self.verbose:
            for full_key, value in sub_params.items():
                logger.info(f"[RemapPrefixLoader] {full_key}: shape={value.shape}, dtype={value.dtype}")
        
            logger.info(f"[RemapPrefixLoader] {len(sub_params)} parameters remapped under '{self.target_prefix}/'.")

        return flax.traverse_util.unflatten_dict(sub_params, sep="/")


# XJ: only load embedder
@dataclasses.dataclass(frozen=True)
class InputEmbedderLoader(WeightLoader):
    """
    Loads parameters containing 'llm/embedder' from a .npz checkpoint.
    Supports nested structures and recursively inspects all keys.
    """

    params_path: str
    substring: str = "llm/embedder"
    verbose: bool = False

    def load(self, params: at.Params) -> at.Params:
        import jax.numpy as jnp

        # Load checkpoint
        try:
            raw = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        except Exception as e:
            raise ValueError(f"Failed to load .npz from {self.params_path}: {e}")

        flat_params = flax.traverse_util.flatten_dict(raw, sep="/")
        flat_expected = flax.traverse_util.flatten_dict(params, sep="/")

        matched = {}
        for k, v in flat_params.items():
            if self.verbose:
                logger.info(f"[InputEmbedderLoader] Key: {k} \t Type: {type(v)}"
                            + (f" \t Shape: {getattr(v, 'shape', 'N/A')}" if isinstance(v, np.ndarray) else ""))
            if self.substring in k:
                expected_dtype = flat_expected.get(k, None)
                if expected_dtype is not None:
                    v = v.astype(expected_dtype.dtype)
                if not isinstance(v, np.ndarray):
                    raise TypeError(f"Matched key '{k}' has unexpected type: {type(v)}")
                matched[k] = jnp.asarray(v)

        if not matched:
            raise ValueError(f"[InputEmbedderLoader] No keys found containing substring '{self.substring}'.")

        if self.verbose:
            logger.info(f"[InputEmbedderLoader] Total matched keys: {len(matched)}")

        return flax.traverse_util.unflatten_dict(matched, sep="/")

# XJ: skip all 
@dataclasses.dataclass(frozen=True)
class EmptyLoader(WeightLoader):
    """
    Skip all parameter. Only preferred when image encoder, prompt & action experts, and embedder are customized
    """
    verbose: bool = False

    def load(self, params: at.Params) -> at.Params:

        matched = {}
        if self.verbose:
            logger.info(f"[InputEmbedderLoader] Total matched keys: {len(matched)}")

        return flax.traverse_util.unflatten_dict(matched, sep="/")
