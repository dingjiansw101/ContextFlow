import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_kvcache as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")
# This is a more clean version of v9

SIGLIP_OUTPUT_DIM = {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
}

def _host_assert(pred_scalar: jax.Array, msg: str, *debug_vals):
    """Convert a scalar JAX boolean into a host-side assertion without materializing a Tracer."""
    import numpy as np
    def _cb(p, *vals):
        if not bool(p):
            vals_np = tuple(np.array(v) for v in vals)
            raise AssertionError(msg.format(*vals_np))
    jax.debug.callback(_cb, pred_scalar, *debug_vals)

def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)

class AttnPoolOne(nnx.Module):
    """
    Pool [B, P, D] patch tokens down to a single [B, 1, D] image token.
    - A learnable query vector q scores each patch (dot product / sqrt(D)), then applies softmax weighting.
    - Optional mask: bool [B, P]; False entries are replaced with -inf.
    """
    def __init__(self, d_model: int, use_layernorm: bool = True, rngs: nnx.Rngs | None = None):
        dtype = jnp.bfloat16
        self.q = nnx.Param(jax.random.normal(rngs.params(), (d_model,), dtype=dtype))  # [D]
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nnx.LayerNorm(d_model, rngs=rngs) 

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        # x: [B, P, D]; mask: [B, P] (True=valid)
        if self.use_layernorm:
            x = self.ln(x)

        d = x.shape[-1]
        scores = (x @ (self.q / jnp.sqrt(d)))  # [B, P]

        if mask is not None:
            scores = jnp.where(mask, scores, -jnp.inf)

            # Prevent NaNs: if an example masks every position, force weights and pooled output to zero.
            all_masked = jnp.logical_not(jnp.any(mask, axis=1))          # [B]
            # Run the standard softmax first (may include -inf).
            w = jax.nn.softmax(scores, axis=1)                           # [B, P]
            # Zero out weights for fully masked examples.
            w = jnp.where(all_masked[:, None], jnp.zeros_like(w), w)
        else:
            w = jax.nn.softmax(scores, axis=1)

        pooled = jnp.sum(x * w[..., None], axis=1, keepdims=True)        # [B, 1, D]
        return pooled


    def pool_bt(self, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Pool each frame in [B, T, P, D] independently -> [B, T, 1, D].
        If provided, the mask should be [B, T] or [B, T, P] (a [B, T] mask broadcasts across patches).
        """
        B, T, P, D = x.shape
        x_bt = x.reshape(B*T, P, D)
        if mask is None:
            m_bt = None
        else:
            if mask.ndim == 2:
                # [B,T] -> [B,T,P]
                mask = einops.repeat(mask, "b t -> b t p", p=P)
            m_bt = mask.reshape(B*T, P)
        pooled_bt = self(x_bt, m_bt)                # [B*T, 1, D]
        return pooled_bt.reshape(B, T, 1, D)        # [B, T, 1, D]


@dataclasses.dataclass(frozen=True)
class Pi0LightIncontextConfigv14(_model.BaseModelConfig):
    # The version without using vlm
    dtype: str = "bfloat16"
    prompt_expert_variant: _gemma.Variant = "gemma_300m_v2"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    # XJ: set siglip_variant explicitly
    siglip_variant: str = "Ti/16"
    pool_type: str = "none"
    # XJ: freeze img encoder or not
    freeze_img_encoder: bool = False
    # XJ: freeze llm embedder or not (suggested when using customized gemma)
    freeze_llm_embedder: bool = False
    # XJ: customize embedder
    vocab_size: int | None = None
    
    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    # params for pi0 incontext
    sample_frames: int = 16
    sample_actions: int = 32
    random_select: bool = True

    avg_current_img: bool = True
    causal_attention: bool = False
    
    # XJ: for training sequence
    use_frame_sequence_transform: bool = False
    frame_sequence_length: int = 4

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_INCONTEXT

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0LightIncontextv14":
        return Pi0LightIncontextv14(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(
        self,
        *,
        batch_size: int = 1,
        keyframe_size: int = 16,
        max_len: int = 512,
    ) -> tuple[_model.ObservationIncontext, _model.Actions]:
        # TODO: rewrite this part
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        prompt_image_spec = jax.ShapeDtypeStruct([batch_size, keyframe_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        prompt_mask_spec = jax.ShapeDtypeStruct([batch_size, keyframe_size], jnp.bool_)
        with at.disable_typechecking():
            observation_spec = _model.ObservationIncontext(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                incontext_images={
                    "base_0_rgb": prompt_image_spec,
                    "left_wrist_0_rgb": prompt_image_spec,
                    "right_wrist_0_rgb": prompt_image_spec,
                },
                incontext_image_masks={
                    "base_0_rgb": prompt_mask_spec,
                    "left_wrist_0_rgb": prompt_mask_spec,
                    "right_wrist_0_rgb": prompt_mask_spec,
                },
                # TODO: add key_frames, and max_len to config
                incontext_states=jax.ShapeDtypeStruct([batch_size, max_len, self.action_dim], jnp.float32),
                incontext_state_masks=jax.ShapeDtypeStruct([batch_size, max_len], jnp.bool_),
                incontext_actions=jax.ShapeDtypeStruct([batch_size, max_len, self.action_dim], jnp.float32),
                incontext_action_masks=jax.ShapeDtypeStruct([batch_size, max_len], jnp.bool_),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec


    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        
        # XJ: Freeze the image encoder if enabled
        freeze_targets = []
        if self.freeze_img_encoder:
            logger.info("[Pi0Config] get_freeze_filter(): freeze_img_encoder:", self.freeze_img_encoder)
            freeze_targets.append(nnx_utils.PathRegex("PaliGemma/img/.*"))
            
        # XJ:  freeze only the embedder input_embedding
        if self.freeze_llm_embedder:
            logger.info("[Pi0Config] get_freeze_filter(): freeze_llm_embedder:", self.freeze_llm_embedder)
            freeze_targets.append(nnx_utils.PathRegex(".*llm/embedder.*"))
        
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.prompt_expert_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True
            
        # XJ: need to use Any (OR)
        if freeze_targets:
            filters.append(nnx.Any(*freeze_targets))
        # else:
        #     return nnx.Nothing

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
            
        return nnx.All(*filters) if filters else nnx.Nothing

'''

def get_freeze_filter(self) -> nnx.filterlib.Filter:
    """
    Freezing policy (union-of-inclusions, then exclusions):

    Inclusions (OR via Any):
    - If main expert has LoRA -> freeze its base weights: '.*llm.*'
    - If action expert has LoRA -> freeze its base weights: '.*llm.*_1.*'
    - If freeze_img_encoder -> freeze 'PaliGemma/img/.*'
    - If freeze_llm_embedder -> freeze '.*llm/embedder.*'

    Exclusions (AND via All):
    - Always keep LoRA adapters trainable: Not('.*lora.*')
    - If only the main expert has LoRA -> keep action base trainable: Not('.*llm.*_1.*')
    """
    inclusions: list[nnx.filterlib.Filter] = []
    exclusions: list[nnx.filterlib.Filter] = []

    # Optional toggles
    if getattr(self, "freeze_img_encoder", False):
        inclusions.append(nnx_utils.PathRegex(r"PaliGemma/img/.*"))
    if getattr(self, "freeze_llm_embedder", False):
        inclusions.append(nnx_utils.PathRegex(r".*llm/embedder.*"))

    # LoRA presence for the two experts (robust across pi0 vs incontext)
    main_variant = getattr(self, "paligemma_variant",
                    getattr(self, "prompt_expert_variant", None))
    act_variant = getattr(self, "action_expert_variant", None)
    main_has_lora = isinstance(main_variant, str) and ("lora" in main_variant)
    act_has_lora  = isinstance(act_variant, str) and ("lora" in act_variant)

    # Base regexes (match your model paths; see model_structure_debug JSON)
    main_base = nnx_utils.PathRegex(r".*llm.*")        # trunk / prompt expert
    act_base  = nnx_utils.PathRegex(r".*llm.*_1.*")    # action expert
    lora_any  = nnx_utils.PathRegex(r".*lora.*")       # any LoRA adapter

    # LoRA-driven inclusions
    if main_has_lora:
        inclusions.append(main_base)
        if not act_has_lora:
            exclusions.append(nnx.Not(act_base))  # keep non-LoRA action base trainable
    if act_has_lora:
        inclusions.append(act_base)

    # Always keep LoRA adapters trainable
    exclusions.append(nnx.Not(lora_any))

    # If nothing to freeze, return Nothing
    if not inclusions:
        return nnx.Nothing

    # Union-of-inclusions, intersected with exclusions
    return nnx.All(
        nnx.Any(*inclusions),
        *exclusions,
    )
'''


class Pi0LightIncontextv14(_model.BaseModel):
    def __init__(self, config: Pi0LightIncontextConfigv14, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        action_expert_config = _gemma.get_config(config.action_expert_variant, "action_expert")
        prompt_expert_config = _gemma.get_config(config.prompt_expert_variant, "prompt_expert")
        self.use_image_prompts = config.use_image_prompts
        self.use_text_prompts = config.use_text_prompts
        self.use_action_state_prompts = config.use_action_state_prompts
        self.avg_current_img = config.avg_current_img
        self.causal_attention = config.causal_attention
        self.freeze_img_encoder = config.freeze_img_encoder
        
        # XJ: for training sequence
        self.use_frame_sequence_transform = config.use_frame_sequence_transform
        self.frame_sequence_length = config.frame_sequence_length

        # TODO: rewrite gemma in NNX. For now, use bridge.
        gemma_kwargs = {
            "configs": [prompt_expert_config, action_expert_config],
            "embed_dtype": config.dtype,
            "use_text_prompts": config.use_text_prompts,
        }
        if config.vocab_size is not None:
            gemma_kwargs["voc_size"] = config.vocab_size

        llm = nnx_bridge.ToNNX(_gemma.Module(**gemma_kwargs)) 
        # llm = nnx_bridge.ToNNX(
        #     _gemma.Module(
        #         configs=[prompt_expert_config, action_expert_config],
        #         embed_dtype=config.dtype,
        #     )
        # )
        llm.lazy_init(rngs=rngs, method="init")

        # XJ
        siglip_variant = config.siglip_variant # "So400m/14", "Ti/16", "S/32",
        pool_type = config.pool_type
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=None,  # paligemma_config.width, None
                # we set the num_classes to None because we need to aligh the output of SigLIP with LLM
                # the public ckp. of SigLIP Ti/16 is used to predicted a different number of classes on ImageNet
                # therefore, we disable the last dense layer of SigLIP (siglip.py line 284)
                # and remove the last layer in the public ckp.
                # and create a trainable projection layer
                variant= siglip_variant,
                # TODO: XJ add a flag in Pi0Light config to determine the type of pooling
                # and associate with the weight_loader/cls token loadding/skipping
                pool_type=pool_type, # default is "none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # self.obs_img_proj = nnx.Linear(paligemma_config.width, prompt_expert_config.width, rngs=rngs)
        # self.text_proj = nnx.Linear(paligemma_config.width, prompt_expert_config.width, rngs=rngs)

        if self.use_action_state_prompts:
            self.demo_action_proj = nnx.Linear(config.action_dim, prompt_expert_config.width, rngs=rngs)
            self.demo_state_proj = nnx.Linear(config.action_dim, prompt_expert_config.width, rngs=rngs)

        # if self.use_image_prompts:
        #     self.img_proj = nnx.Linear(paligemma_config.width, prompt_expert_config.width, rngs=rngs)
            # TODO: add some layers to process in-context prompts
            
        # XJ: for SigLIP Ti/16
        siglip_key = siglip_variant.split("/")[0]
        try:
            siglip_output_dim = SIGLIP_OUTPUT_DIM[siglip_key]
        except KeyError:
            raise ValueError(f"Unknown SigLIP variant '{siglip_variant}' -- unable to determine output dim.")
        self.image_proj_promtp_expert = nnx.Linear(siglip_output_dim, prompt_expert_config.width, rngs=rngs)
        self.image_proj_action_expert = nnx.Linear(siglip_output_dim, action_expert_config.width, rngs=rngs)
        # self.image_proj_action_expert = self.image_proj_promtp_expert
        self.img_pool_prompt_expert = AttnPoolOne(prompt_expert_config.width, use_layernorm=True, rngs=rngs)
        self.img_pool_action_expert = AttnPoolOne(action_expert_config.width, use_layernorm=True, rngs=rngs)
        # self.img_pool_action_expert = self.img_pool_prompt_expert


    @at.typecheck
    def embed_midfix(
        self, obs: _model.ObservationIncontext
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:

        input_mask = []
        ar_mask = []
        tokens = []

        # add language (aka tokenized inputs)
        if self.use_text_prompts:
            if obs.tokenized_prompt is not None:
                tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
                # tokenized_inputs = self.text_proj(tokenized_inputs)
                tokens.append(tokenized_inputs)
                input_mask.append(obs.tokenized_prompt_mask)
                # full attention between image and language inputs
                ar_mask += [False] * tokenized_inputs.shape[1]

        # -------------------------------------------------------------------------
        # embed in-context images
        # TODO: set a ratio to randomly mask input or prompt
        if self.use_image_prompts:
            for name in obs.incontext_images:
                image_sequence = obs.incontext_images[name]
                if len(image_sequence.shape) == 6:
                    batch_size, episode_len, seq_len = image_sequence.shape[0], image_sequence.shape[1], image_sequence.shape[2]
                    image_sequence = image_sequence.reshape(
                        image_sequence.shape[0] * image_sequence.shape[1] * image_sequence.shape[2], *image_sequence.shape[3:]
                    )
                    image_sqeuence_tokens, _ = self.PaliGemma.img(image_sequence, train=False)
                    image_sqeuence_tokens = self.image_proj_promtp_expert(image_sqeuence_tokens)

                    # TODO: to organize multiple episode prompts in order
                    image_sqeuence_tokens = image_sqeuence_tokens.reshape(
                        batch_size, episode_len * seq_len, -1, image_sqeuence_tokens.shape[-1]
                    )
                    obs.incontext_image_masks[name] = obs.incontext_image_masks[name].reshape(batch_size, episode_len * seq_len)
                elif len(image_sequence.shape) == 5:
                    batch_size, seq_len = image_sequence.shape[0], image_sequence.shape[1]
                    image_sequence = image_sequence.reshape(
                        image_sequence.shape[0] * image_sequence.shape[1], *image_sequence.shape[2:]
                    )
                    image_sqeuence_tokens, _ = self.PaliGemma.img(image_sequence, train=False)
                    image_sqeuence_tokens = self.image_proj_promtp_expert(image_sqeuence_tokens)

                    image_sqeuence_tokens = image_sqeuence_tokens.reshape(
                        batch_size, seq_len, -1, image_sqeuence_tokens.shape[-1]
                    )

                # image_sqeuence_tokens = jnp.mean(image_sqeuence_tokens, axis=2)
                image_sqeuence_tokens = self.img_pool_prompt_expert.pool_bt(
                    image_sqeuence_tokens,
                    mask=obs.incontext_image_masks[name]  # [B, T] frame-level mask, broadcast across patches internally.
                ).squeeze(axis=2)  # [B, T, D]
                # image_sqeuence_tokens = self.img_proj(image_sqeuence_tokens)
                tokens.append(image_sqeuence_tokens)
                input_mask.append(obs.incontext_image_masks[name])
                ar_mask += [False] * image_sqeuence_tokens.shape[1]

        #------------------------------------------------------------------------
        # embed in-context states
        if self.use_action_state_prompts:
            if len(obs.incontext_states.shape) == 4:
                incontext_states_reshape = obs.incontext_states.reshape(obs.incontext_states.shape[0], -1, obs.incontext_states.shape[-1])
                dem_state_tokens = self.demo_state_proj(incontext_states_reshape)
                incontext_state_masks_input = obs.incontext_state_masks.reshape(obs.incontext_states.shape[0], -1)
            else:
                dem_state_tokens = self.demo_state_proj(obs.incontext_states)
                incontext_state_masks_input = obs.incontext_state_masks
            tokens.append(dem_state_tokens)
            input_mask.append(incontext_state_masks_input)
            ar_mask += [False] * dem_state_tokens.shape[1]

            #------------------------------------------------------------------------
            # embed in-context actions
            if len(obs.incontext_actions.shape) == 4:
                incontext_actions_reshape = obs.incontext_actions.reshape(obs.incontext_actions.shape[0], -1, obs.incontext_actions.shape[-1])
                dem_action_tokens = self.demo_action_proj(incontext_actions_reshape)
                incontext_action_masks_input = obs.incontext_action_masks.reshape(obs.incontext_actions.shape[0], -1)
            else:
                dem_action_tokens = self.demo_action_proj(obs.incontext_actions)
                incontext_action_masks_input = obs.incontext_action_masks
            tokens.append(dem_action_tokens)
            input_mask.append(incontext_action_masks_input)
            ar_mask += [False] * dem_action_tokens.shape[1]

        # ---------------------------------------------------------
        assert len(ar_mask) > 0
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)

        # ar_mask[0] = True
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask


    @at.typecheck
    def embed_suffix(
        self, obs: _model.ObservationIncontext, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            image_tokens = self.image_proj_action_expert(image_tokens)

            # image_tokens = self.obs_img_proj(image_tokens)
            if self.avg_current_img:
                # image_tokens = jnp.mean(image_tokens, axis=1, keepdims=True)
                image_tokens = self.img_pool_action_expert(image_tokens)
            tokens.append(image_tokens)  # image_tokens (32, 256, 2048)


            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [True] + ([False] * (image_tokens.shape[1] - 1))

        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    # @override
    # def compute_loss(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.ObservationIncontext,
    #     actions: _model.Actions,
    #     *,
    #     train: bool = False,
    # ) -> at.Float[at.Array, "*b ah"]:
    #     # jax.debug.print("observation = {} ", observation)
    #     # import ipdb; ipdb.set_trace()
    #     preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    #     observation = _model.preprocess_observation_incontext(preprocess_rng, observation, train=train)

    #     batch_shape = actions.shape[:-2]
    #     noise = jax.random.normal(noise_rng, actions.shape)
    #     time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    #     time_expanded = time[..., None, None]
    #     x_t = time_expanded * noise + (1 - time_expanded) * actions
    #     u_t = noise - actions

    #     midfix_tokens, midfix_mask, midfix_ar_mask = self.embed_midfix(observation)
    #     suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
    #     input_mask = jnp.concatenate([midfix_mask, suffix_mask], axis=1)
    #     ar_mask = jnp.concatenate([midfix_ar_mask, suffix_ar_mask], axis=0)
    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.ObservationIncontext,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:

        # ---------------------------
        # 0) Preprocess and detect multi-frame batches
        # ---------------------------
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        obs = _model.preprocess_observation_incontext_fused(preprocess_rng, observation, train=train)

        has_multi = (
            (obs.current_state_seq is not None) and
            (obs.current_images_seq is not None) and
            (obs.current_image_masks_seq is not None) and
            (obs.actions_seq is not None)
        )

        B = obs.state.shape[0]
        assert B > 0, "Batch size B must be > 0"

        if has_multi:
            # Validate expected shapes
            assert obs.actions_seq.ndim == 4, f"obs.actions_seq expected [B,N,H,A], got {obs.actions_seq.shape}"
            N = obs.current_state_seq.shape[1]
            H = obs.actions_seq.shape[2]
            A = obs.actions_seq.shape[3]
            assert N > 0 and H > 0 and A > 0, f"N/H/A must be > 0, got N={N}, H={H}, A={A}"

            actions_seq = obs.actions_seq
            obs_current_state_seq       = obs.current_state_seq
            obs_current_images_seq      = obs.current_images_seq
            obs_current_image_masks_seq = obs.current_image_masks_seq

        else:
            # Fallback to a single-frame path
            assert actions.ndim == 3, f"Single-frame mode expects actions=[B,H,A], got {actions.shape}"
            N = 1
            H, A = actions.shape[1], actions.shape[2]
            actions_seq = actions[:, None, :, :]
            obs_current_state_seq       = obs.state[:, None, :]
            obs_current_images_seq      = {k: v[:, None, ...] for k, v in obs.images.items()}
            obs_current_image_masks_seq = {k: v[:, None]      for k, v in obs.image_masks.items()}

        # Optional hard assertion to enforce the fused path during training
        # ---------------------------
        # 1) FM noise synthesis
        # ---------------------------
        noise = jax.random.normal(noise_rng, actions_seq.shape)                       # [B,N,H,A]
        t     = jax.random.beta(time_rng, 1.5, 1, (B, N)) * 0.999 + 0.001            # [B,N]
        x_t   = t[..., None, None] * noise + (1.0 - t[..., None, None]) * actions_seq
        u_t   = noise - actions_seq

        # ---------------------------
        # 2) midfix -> encode_pm_only (single pass)
        # ---------------------------
        midfix_tokens, midfix_mask, midfix_ar = self.embed_midfix(obs)               # [B, S_mid, D], [B, S_mid], [S_mid]
        midfix_attn = make_attn_mask(midfix_mask, midfix_ar)                         # [B,S_mid,S_mid]
        pos_midfix  = jnp.cumsum(midfix_mask, axis=1) - 1                             # [B,S_mid]

        pm_cache = self._llm_encode_pm_only(
            embedded_pm=[midfix_tokens, None],
            positions_pm=pos_midfix,
            mask_pm=midfix_attn,
            deterministic=True,
        )

        # ---------------------------
        # 3) Build the BN suffix and run decode_with_cache once
        # ---------------------------
        BN = B * N
        flat_images = {name: einops.rearrange(img, "b n h w c -> (b n) h w c")
                       for name, img in obs_current_images_seq.items()}
        flat_img_masks = {name: einops.rearrange(msk, "b n -> (b n)")
                          for name, msk in obs_current_image_masks_seq.items()}
        flat_states = einops.rearrange(obs_current_state_seq, "b n a -> (b n) a")
        flat_x_t    = einops.rearrange(x_t, "b n h a -> (b n) h a")
        flat_t      = einops.rearrange(t,   "b n -> (b n)")

        # Reuse the existing logic to construct the suffix tokens
        def inner_build_suffix_tokens_from_flat(flat_imgs, flat_img_masks, flat_states_, flat_x_t_, flat_t_):
            input_mask = []
            ar_mask = []
            tokens = []

            for name, img in flat_imgs.items():
                img_tokens, _ = self.PaliGemma.img(img, train=(train and not self.freeze_img_encoder))
                img_tokens = self.image_proj_action_expert(img_tokens)
                if self.avg_current_img:
                    # img_tokens = jnp.mean(img_tokens, axis=1, keepdims=True)
                    img_tokens = self.img_pool_action_expert(img_tokens)
                tokens.append(img_tokens)
                input_mask.append(einops.repeat(flat_img_masks[name], "bn -> bn s", s=img_tokens.shape[1]))
                ####### XJ: important bug fix: set an barrier between prompt expert & action expert!
                ar_mask += [True] + ([False] * (img_tokens.shape[1] - 1))

            state_token = self.state_proj(flat_states_)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((BN, 1), dtype=jnp.bool_))
            ar_mask += [True]

            time_emb = posemb_sincos(flat_t_, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
            action_tokens = self.action_in_proj(flat_x_t_)
            time_tokens   = einops.repeat(time_emb, "bn d -> bn h d", h=self.action_horizon)
            atoks = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            atoks = self.action_time_mlp_out(nnx.swish(self.action_time_mlp_in(atoks)))
            tokens.append(atoks)
            input_mask.append(jnp.ones(atoks.shape[:2], dtype=jnp.bool_))
            ar_mask += [True] + ([False] * (self.action_horizon - 1))

            tokens = jnp.concatenate(tokens, axis=1)
            imask  = jnp.concatenate(input_mask, axis=1)
            armask = jnp.array(ar_mask)
            return tokens, imask, armask

        suffix_tokens, suffix_mask, suffix_ar = inner_build_suffix_tokens_from_flat(
            flat_images, flat_img_masks, flat_states, flat_x_t, flat_t
        )

        suffix_attn = make_attn_mask(suffix_mask, suffix_ar)                          # [BN,Ts,Ts]
        midfix_seen = einops.repeat(midfix_mask, "b p -> (b n) s p", n=N, s=suffix_tokens.shape[1])
        # we only need the lower half of the full attention mask: [suffix_tokens, midfix_tokens+suffix_tokens]
        # It equals to: 1. midfix atten mask is broadcasted to [suffix_tokens, midfix_tokens]
        # 2. AND with a suffix->midfix attention mask (block-wise causal attention mask like), in this case, 1
        # midfix = jnp.broadcast_to(base_midfix, (BN, Ts, S_mid))            # [BN, Ts, S_mid]
        # midfix_seen = base_midfix & gate                                   
        full_mask   = jnp.concatenate([midfix_seen, suffix_attn], axis=-1)

        pos_offset = einops.repeat(jnp.sum(midfix_mask, axis=-1), "b -> (b n) 1", n=N)
        pos_suf    = pos_offset + jnp.cumsum(suffix_mask, axis=-1) - 1

        ep_index = einops.repeat(jnp.arange(B, dtype=jnp.int32), "b -> (b n)", n=N)
        outs_suf = self._llm_decode_with_cache(
            embedded_suf=[None, suffix_tokens],
            positions_suf=pos_suf,
            mask_suf=full_mask,
            pm_cache=pm_cache,
            ep_index=ep_index,
            deterministic=not train,
        )
        suffix_out = outs_suf[1]   # [BN,Ts,D]

        # ---------------------------
        # 4) Action positions and loss
        # ---------------------------
        action_start  = suffix_out.shape[1] - self.action_horizon
        assert action_start >= 0, f"Ts ({suffix_out.shape[1]}) must be >= action_horizon ({self.action_horizon})"
        action_hidden = suffix_out[:, action_start:, :]
        v_t = self.action_out_proj(action_hidden)  # [BN,H,A]
        v_t = einops.rearrange(v_t, " (b n) h a -> b n h a", b=B, n=N)

        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [B,N,H]

        # Final shape assertions to confirm we are truly in the fused path
        if has_multi:
        else:
        return loss if has_multi else loss[:, 0, :]


    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.ObservationIncontext,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation_incontext(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        midfix_tokens, midfix_mask, midfix_ar_mask = self.embed_midfix(observation)
        
        midfix_attn_mask = make_attn_mask(midfix_mask, midfix_ar_mask)
        positions = jnp.cumsum(midfix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([midfix_tokens, None], mask=midfix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)

            midfix_attn_mask = einops.repeat(midfix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([midfix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                midfix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(midfix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (midfix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert midfix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    # ---------------------------
    # Debug helpers (lightweight wrappers & assertions)
    # ---------------------------
        # Track whether encode_pm_only / decode_with_cache were actually invoked.

    def _llm_encode_pm_only(self, *, embedded_pm, positions_pm, mask_pm, deterministic):
        pm_cache = self.PaliGemma.llm(
            embedded_pm=embedded_pm,
            positions_pm=positions_pm,
            mask_pm=mask_pm,
            deterministic=deterministic,
            method="encode_pm_only",
        )
        return pm_cache

    def _llm_decode_with_cache(self, *, embedded_suf, positions_suf, mask_suf, pm_cache, ep_index, deterministic):
        # Ensure the reused pm_cache is populated.
        assert pm_cache is not None, "decode_with_cache expects a non-empty pm_cache but received None."
        return self.PaliGemma.llm(
            embedded_suf=embedded_suf,
            positions_suf=positions_suf,
            mask_suf=mask_suf,
            pm_cache=pm_cache,
            ep_index=ep_index,
            deterministic=deterministic,
            method="decode_with_cache",
        )

    @staticmethod
    def _assert_bool_mask(x, name: str):
        assert x.dtype == jnp.bool_, f"{name} must be a bool mask; current dtype={x.dtype}"

    @staticmethod
    def _assert_shape(x, shape, name: str):
        # Elements of shape may be None to indicate "any".
        assert len(x.shape) == len(shape), f"{name} rank mismatch: got {x.shape}, expect {shape}"
        for i, (got, exp) in enumerate(zip(x.shape, shape)):
            if exp is not None and got != exp:
                raise AssertionError(f"{name} dimension[{i}] mismatch: got {x.shape}, expect {shape}")
        
    @staticmethod
    def _assert_monotonic_positions(pos: jax.Array,
                                    mask: jax.Array,
                                    name: str,
                                    allow_offset: bool = False):
        """
        Verify that positions increment by one wherever the token mask is valid.
        - With allow_offset=False: positions must equal cumsum(valid) - 1.
        - With allow_offset=True: a batch-wise offset is allowed (e.g., suffix appended after midfix).
        Accepts masks shaped [B,T], [B,T,S], or [B,1,T,S].
        """
        # 1) Normalize the mask to token-level [B,T].
        if mask.ndim == 4:
            assert mask.shape[1] == 1, f"{name}: 4D mask dimension 1 should be 1, got {mask.shape}"
            mask = jnp.squeeze(mask, axis=1)        # [B,T,S]
        if mask.ndim == 3:
            token_mask = jnp.any(mask, axis=-1)     # [B,T]
        elif mask.ndim == 2:
            token_mask = mask                       # [B,T]
        else:
            raise ValueError(f"{name}: mask rank must be 2/3/4, got {mask.shape}")

        # 2) Validate shapes using host assertions to avoid materializing tracers.
        _host_assert(jnp.array(pos.ndim == 2), "{}: positions must be [B,T]", name)
        _host_assert(jnp.array((pos.shape[0] == token_mask.shape[0]) &
                            (pos.shape[1] == token_mask.shape[1])),
                    "{}: positions shape {} must equal token_mask shape {}",
                    name, pos.shape, token_mask.shape)

        # 3) Reference sequence
        ref_incr = jnp.cumsum(token_mask, axis=-1) - 1   # [B,T]

        if allow_offset:
            # Compute the first valid position per batch to use as an offset.
            big = jnp.iinfo(jnp.int32).max
            first_pos = jnp.min(jnp.where(token_mask, pos, big), axis=1)   # [B]
            # Handle the all-invalid case by replacing +inf with 0.
            first_pos = jnp.where(first_pos == big, 0, first_pos)
            ref = first_pos[:, None] + ref_incr
        else:
            ref = ref_incr

        ok = jnp.all(jnp.where(token_mask, pos == ref, True))
        _host_assert(ok, "{} is not a mask-aligned (allow_offset={}) monotonic position sequence", name, allow_offset)


    @staticmethod
    def _assert_block_ar_mask(ar_mask: jax.Array, H: int, name: str):
        """
        Convention: the final H suffix tokens form the action block and must follow:
        [True, False, False, ..., False]  (length H). Other positions are unconstrained.
        """
        m = jnp.asarray(ar_mask, dtype=bool).reshape(-1)     # [Ts]
        Ts = m.shape[0]                                      # Python int (static shape), safe to compare directly
        if Ts < H:
            raise AssertionError(f"{name}: len(ar_mask)={Ts} < action_horizon={H}")

        tail = m[-H:]                                        # [H]
        cond = jnp.logical_and(tail[0], jnp.all(jnp.logical_not(tail[1:])))
        _host_assert(cond, "{}: action block tail should be [True, False*(H-1)], actual tail={}", name, tail)


    ### XJ: for unit tests only.
    # =========================
    # Deterministic forward/loss variants with controllable noise/t
    # =========================

    def _build_suffix_tokens_from_flat(
        self,
        flat_images,
        flat_img_masks,
        flat_states,
        flat_x_t,
        flat_t,
    ):
        tokens = []
        input_mask = []
        ar_mask = []

        BN = flat_x_t.shape[0]

        for name, img in flat_images.items():
            img_tokens, _ = self.PaliGemma.img(img, train=False)
            img_tokens = self.image_proj_action_expert(img_tokens)
            if self.avg_current_img:
                img_tokens = self.img_pool_action_expert(img_tokens)
            tokens.append(img_tokens)
            input_mask.append(einops.repeat(flat_img_masks[name], "bn -> bn s", s=img_tokens.shape[1]))
            ar_mask += [True] + ([False] * (img_tokens.shape[1] - 1))

        state_token = self.state_proj(flat_states)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((BN, 1), dtype=jnp.bool_))
        ar_mask += [True]

        time_emb = posemb_sincos(flat_t, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        action_tokens = self.action_in_proj(flat_x_t)
        time_tokens   = einops.repeat(time_emb, "bn d -> bn h d", h=self.action_horizon)
        atoks = self.action_time_mlp_out(
            nnx.swish(
                self.action_time_mlp_in(jnp.concatenate([action_tokens, time_tokens], axis=-1))
            )
        )
        tokens.append(atoks)
        input_mask.append(jnp.ones(atoks.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        imask  = jnp.concatenate(input_mask, axis=1)
        armask = jnp.array(ar_mask)
        return tokens, imask, armask

    def forward_vt_sequence(
        self,
        observation: _model.ObservationIncontext,
        x_t: at.Float[at.Array, " b n h a"],
        t:  at.Float[at.Array, " b n"],
    ) -> at.Float[at.Array, " b n h a"]:
        """
        Flatten all N frames to BN, decode v_t with deterministic=True, without any sampling.
        Returns shape [B, N, H, A].
        """
        # Preprocess using the fused pipeline (no training randomness).
        obs = _model.preprocess_observation_incontext_fused(None, observation, train=False)

        # ---- Shape / has_multi detection matches compute_loss ----
        has_multi = (
            (obs.current_state_seq is not None) and
            (obs.current_images_seq is not None) and
            (obs.current_image_masks_seq is not None)
        )
        if not has_multi:
            # Single-frame fallback: pad current_* to N=1.
            B = obs.state.shape[0]
            x_t = x_t.reshape(B, 1, self.action_horizon, self.action_dim)
            t   = t.reshape(B, 1)
            obs_current_state_seq       = obs.state[:, None, :]
            obs_current_images_seq      = {k: v[:, None, ...] for k, v in obs.images.items()}
            obs_current_image_masks_seq = {k: v[:, None]      for k, v in obs.image_masks.items()}
            N = 1
        else:
            N = x_t.shape[1]
            obs_current_state_seq       = obs.current_state_seq
            obs_current_images_seq      = obs.current_images_seq
            obs_current_image_masks_seq = obs.current_image_masks_seq

        # ---- midfix -> encode_pm_only (deterministic=True) ----
        midfix_tokens, midfix_mask, midfix_ar = self.embed_midfix(obs)
        midfix_attn = make_attn_mask(midfix_mask, midfix_ar)
        pos_midfix  = jnp.cumsum(midfix_mask, axis=1) - 1
        pm_cache = self._llm_encode_pm_only(
            embedded_pm=[midfix_tokens, None],
            positions_pm=pos_midfix,
            mask_pm=midfix_attn,
            deterministic=True,
        )

        # ---- Flatten [B,N,...] to [BN,...], build suffix, decode once ----
        B = x_t.shape[0]
        H = x_t.shape[2]
        A = x_t.shape[3]
        BN = B * N

        flat_images = {name: einops.rearrange(img, "b n h w c -> (b n) h w c")
                       for name, img in obs_current_images_seq.items()}
        flat_img_masks = {name: einops.rearrange(msk, "b n -> (b n)")
                          for name, msk in obs_current_image_masks_seq.items()}
        flat_states = einops.rearrange(obs_current_state_seq, "b n a -> (b n) a")
        flat_x_t    = einops.rearrange(x_t, "b n h a -> (b n) h a")
        flat_t      = einops.rearrange(t,   "b n -> (b n)")

        suffix_tokens, suffix_mask, suffix_ar = self._build_suffix_tokens_from_flat(
            flat_images, flat_img_masks, flat_states, flat_x_t, flat_t
        )
        suffix_attn = make_attn_mask(suffix_mask, suffix_ar)
        midfix_seen = einops.repeat(midfix_mask, "b p -> (b n) s p", n=N, s=suffix_tokens.shape[1])
        full_mask   = jnp.concatenate([midfix_seen, suffix_attn], axis=-1)

        pos_offset = einops.repeat(jnp.sum(midfix_mask, axis=-1), "b -> (b n) 1", n=N)
        pos_suf    = pos_offset + jnp.cumsum(suffix_mask, axis=-1) - 1

        ep_index = einops.repeat(jnp.arange(B, dtype=jnp.int32), "b -> (b n)", n=N)
        outs_suf = self._llm_decode_with_cache(
            embedded_suf=[None, suffix_tokens],
            positions_suf=pos_suf,
            mask_suf=full_mask,
            pm_cache=pm_cache,
            ep_index=ep_index,
            deterministic=True,
        )
        suffix_out = outs_suf[1]  # [BN,Ts,D]

        action_start  = suffix_out.shape[1] - self.action_horizon
        action_hidden = suffix_out[:, action_start:, :].astype(jnp.float32)
        v_t = self.action_out_proj(action_hidden)  # [BN,H,A]
        v_t = einops.rearrange(v_t, " (b n) h a -> b n h a", b=B, n=N)
        return v_t


    def forward_vt_stepwise(
        self,
        observation: _model.ObservationIncontext,
        x_t: at.Float[at.Array, " b n h a"],
        t:  at.Float[at.Array, " b n"],
    ) -> at.Float[at.Array, " b n h a"]:
        """
        Decode one frame at a time (deterministic=True) and concatenate along N.
        Returns shape [B, N, H, A].
        """
        # Preprocess.
        obs = _model.preprocess_observation_incontext_fused(None, observation, train=False)

        # Determine whether we have multi-frame data.
        has_multi = (
            (obs.current_state_seq is not None) and
            (obs.current_images_seq is not None) and
            (obs.current_image_masks_seq is not None)
        )

        if not has_multi:
            B = obs.state.shape[0]
            x_t = x_t.reshape(B, 1, self.action_horizon, self.action_dim)
            t   = t.reshape(B, 1)
            obs_current_state_seq       = obs.state[:, None, :]
            obs_current_images_seq      = {k: v[:, None, ...] for k, v in obs.images.items()}
            obs_current_image_masks_seq = {k: v[:, None]      for k, v in obs.image_masks.items()}
            N = 1
        else:
            N = x_t.shape[1]
            obs_current_state_seq       = obs.current_state_seq
            obs_current_images_seq      = obs.current_images_seq
            obs_current_image_masks_seq = obs.current_image_masks_seq

        B = x_t.shape[0]

        # Shared midfix & cache.
        midfix_tokens, midfix_mask, midfix_ar = self.embed_midfix(obs)
        midfix_attn = make_attn_mask(midfix_mask, midfix_ar)
        pos_midfix  = jnp.cumsum(midfix_mask, axis=1) - 1
        pm_cache = self._llm_encode_pm_only(
            embedded_pm=[midfix_tokens, None],
            positions_pm=pos_midfix,
            mask_pm=midfix_attn,
            deterministic=True,
        )

        flat_images = {name: einops.rearrange(img, "b n h w c -> (b n) h w c")
                       for name, img in obs_current_images_seq.items()}
        flat_img_masks = {name: einops.rearrange(msk, "b n -> (b n)")
                          for name, msk in obs_current_image_masks_seq.items()}
        flat_states = einops.rearrange(obs_current_state_seq, "b n a -> (b n) a")
        flat_x_t    = einops.rearrange(x_t, "b n h a -> (b n) h a")
        flat_t      = einops.rearrange(t,   "b n -> (b n)")

        suffix_tokens_flat, suffix_mask_flat, suffix_ar = self._build_suffix_tokens_from_flat(
            flat_images, flat_img_masks, flat_states, flat_x_t, flat_t
        )
        suffix_tokens_bn = einops.rearrange(suffix_tokens_flat, "(b n) s d -> b n s d", b=B, n=N)
        suffix_mask_bn   = einops.rearrange(suffix_mask_flat, "(b n) s -> b n s", b=B, n=N)

        vt_list = []
        pos_offset = jnp.sum(midfix_mask, axis=-1)[:, None]   # [B,1]
        base_ep = jnp.arange(B, dtype=jnp.int32)              # [B]

        for n in range(N):
            suffix_tokens_n = suffix_tokens_bn[:, n, :, :]
            suffix_mask_n   = suffix_mask_bn[:, n, :]

            suffix_attn = make_attn_mask(suffix_mask_n, suffix_ar)
            midfix_seen = einops.repeat(midfix_mask, "b p -> b s p", s=suffix_tokens_n.shape[1])
            full_mask   = jnp.concatenate([midfix_seen, suffix_attn], axis=-1)

            pos_suf = pos_offset + jnp.cumsum(suffix_mask_n, axis=-1) - 1  # [B,Ts]

            outs_suf = self._llm_decode_with_cache(
                embedded_suf=[None, suffix_tokens_n],
                positions_suf=pos_suf,
                mask_suf=full_mask,
                pm_cache=pm_cache,
                ep_index=base_ep,
                deterministic=True,
            )
            suf_out = outs_suf[1]  # [B,Ts,D]

            action_start  = suf_out.shape[1] - self.action_horizon
            action_hidden = suf_out[:, action_start:, :].astype(jnp.float32)
            vt_n = self.action_out_proj(action_hidden)  # [B,H,A]
            vt_list.append(vt_n)

        v_t = jnp.stack(vt_list, axis=1)  # [B,N,H,A]
        return v_t

    def _mse_loss(vt, tgt, *, frame_mask=None):
        # Use consistent precision to avoid the ~1e-5 drift seen in practice.
        vt  = vt.astype(jnp.float32)   # [B, N, H, A]
        tgt = tgt.astype(jnp.float32)  # [B, N, H, A]

        err2   = jnp.square(vt - tgt)                      # [B, N, H, A]
        per_th = jnp.mean(err2, axis=-1, dtype=jnp.float32)  # [B, N, H]; mean over the action dimension.

        if frame_mask is None:
            return per_th

        # Frame-level mask: [B, N] -> broadcast to [B, N, 1] across H.
        fm = frame_mask.astype(jnp.float32)[..., None]      # [B, N, 1]
        num = (per_th * fm)                                 # [B, N, H]
        den = jnp.maximum(jnp.sum(fm, axis=-2, dtype=jnp.float32), 1.0)  # Count along N, shape [B, 1]
        return num / den[:, None, :]                        # Normalize per batch and horizon.

    # Helper used inside the model (shared by both compute_loss_* variants).
    def _merged_seq_mask(self, obs, image_keys=(
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )):
        ms = obs.current_image_masks_seq
        if ms is None:
            return None  # No weighting available.
        mlist = [jnp.asarray(ms[k]) for k in image_keys if k in ms]
        if len(mlist) == 0:
            return None
        # [B,N]: AND across selected camera views (matches test expectations).
        return jnp.logical_and.reduce(jnp.stack(mlist, axis=0), axis=0)

    def _loss_core(self, obs, actions, noise, t, vt_fn):
        # -- Normalize the source of action sequences --
        if obs.actions_seq is None:
            assert actions is not None, "Single-frame mode requires actions shaped [B,H,A]"
            a = actions[:, None, :, :]            # [B,1,H,A]
        else:
            a = obs.actions_seq                   # [B,N,H,A]

        # -- Align dtype and broadcast shapes --
        a     = a.astype(jnp.float32)
        eps   = noise.astype(jnp.float32)
        t     = t.astype(jnp.float32)
        t_exp = t[..., None, None]                # [B,N,1,1]
        one   = jnp.array(1.0, dtype=jnp.float32)

        # -- Construct x_t / u_t using identical expression trees --
        x_t = t_exp * eps + (one - t_exp) * a
        u_t = eps - a

        # -- The only difference is how vt is produced (sequence vs. stepwise) --
        v_t = vt_fn(obs, x_t, t)

        # -- MSE: sum across actions then divide to keep a consistent reduction order --
        diff = (v_t.astype(jnp.float32) - u_t)        # [B,N,H,A]
        err  = diff * diff                            # [B,N,H,A]
        A    = jnp.array(err.shape[-1], dtype=jnp.float32)
        loss = jnp.sum(err, axis=-1) / A              # [B,N,H]

        # -- Optional: apply sequence masks at a single location (needed for random masking) --
        m = self._merged_seq_mask(obs)
        if m is not None:
            w = m.astype(jnp.float32)[..., None]      # [B,N,1]
            loss = loss * w                           # [B,N,H]

        return loss


    def compute_loss_sequence(
        self, rng_unused, observation, actions, *, noise, t
    ):
        return self._loss_core(observation, actions, noise, t, vt_fn=self.forward_vt_sequence)

    def compute_loss_stepwise(
        self, rng_unused, observation, actions, *, noise, t
    ):
        return self._loss_core(observation, actions, noise, t, vt_fn=self.forward_vt_stepwise)

    
def _merge_current_frame_mask(obs, *, require_all=False):
    masks = list(obs.current_image_masks_seq.values())  # Each is [B,N] bool
    if not masks:
        B, N = obs.actions_seq.shape[:2]
        return jnp.ones((B, N), dtype=jnp.bool_)
    m = masks[0]
    for mm in masks[1:]:
        m = jnp.logical_and(m, mm) if require_all else jnp.logical_or(m, mm)
    return m

def _loss_from_v_and_u(v_t, u_t, frame_mask=None):
    v32 = v_t.astype(jnp.float32)
    u32 = u_t.astype(jnp.float32)
    per_th = jnp.mean(jnp.square(v32 - u32), axis=-1)  # [B,N,H]
    if frame_mask is None:
        return per_th
    fm = frame_mask.astype(jnp.float32)[..., None]     # [B,N,1] -> [B,N,H]
    # When you only need per-frame alignment (no aggregation over N), `per_th * fm` is sufficient.
    return per_th * fm  # Keeps identical shape and operations to the alternative path.
