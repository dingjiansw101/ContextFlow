import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")
# This is a more clean version of v9

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


@dataclasses.dataclass(frozen=True)
class Pi0IncontextConfigv12(_model.BaseModelConfig):
    # The version without using vlm
    dtype: str = "bfloat16"
    prompt_expert_variant: _gemma.Variant = "gemma_300m_v2"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    # params for pi0 incontext
    sample_frames: int = 16
    sample_actions: int = 32
    random_select: bool = True

    avg_current_img: bool = False
    causal_attention: bool = False

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_INCONTEXT

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Incontextv12":
        return Pi0Incontextv12(self, rngs=nnx.Rngs(rng))

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

    def get_freeze_filter_gpt(self) -> nnx.filterlib.Filter:
        """
        Freeze policy:

        • If *no* expert uses LoRA  →  freeze nothing.
        • If one or both experts use LoRA
            – freeze the shared PaLiGemma trunk (matches “.*llm.*”)
            – freeze the *base* weights of every LoRA-enabled expert
            – keep the base weights of non-LoRA experts trainable
            – always keep every `*.lora.*` adapter trainable
        """
        # ------------------------------------------------------------------ #
        # 1.  Regex atoms
        # ------------------------------------------------------------------ #
        gemma_base        = nnx_utils.PathRegex(r".*llm.*")                  # trunk (matches everything)
        action_base       = nnx_utils.PathRegex(r".*llm.*_1.*")              # action expert
        prompt_base       = nnx_utils.PathRegex(r".*llm.*_prompt_expert.*")  # prompt expert
        lora_any          = nnx_utils.PathRegex(r".*lora.*")                 # any LoRA adapter

        # Which experts carry LoRA?
        action_has_lora  = "lora" in self.action_expert_variant
        prompt_has_lora  = "lora" in self.prompt_expert_variant

        # ------------------------------------------------------------------ #
        # 2.  Early-out: nothing to freeze
        # ------------------------------------------------------------------ #
        if not (action_has_lora or prompt_has_lora):
            return nnx.Nothing             # → train the whole model

        # ------------------------------------------------------------------ #
        # 3.  Build the union-of-inclusions
        # ------------------------------------------------------------------ #
        inclusions: list[nnx.filterlib.Filter] = [gemma_base]                # always freeze the trunk
        if action_has_lora:
            inclusions.append(action_base)
        if prompt_has_lora:
            inclusions.append(prompt_base)

        # ------------------------------------------------------------------ #
        # 4.  Build the exclusions
        # ------------------------------------------------------------------ #
        exclusions: list[nnx.filterlib.Filter] = [nnx.Not(lora_any)]         # keep adapters trainable
        if not action_has_lora:
            exclusions.append(nnx.Not(action_base))                          # keep action base trainable
        if not prompt_has_lora:
            exclusions.append(nnx.Not(prompt_base))                          # keep prompt base trainable

        # ------------------------------------------------------------------ #
        # 5.  Assemble the composite filter
        # ------------------------------------------------------------------ #
        #   FROZEN =  ( G  ∪  A(lora)  ∪  P(lora) )   \   ( LoRA  ∪  non-LoRA bases )
        return nnx.All(
            nnx.Any(*inclusions),     # logical OR of everything we *may* freeze
            *exclusions,              # logical AND of every veto rule
        )


    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
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

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0Incontextv12(_model.BaseModel):
    def __init__(self, config: Pi0IncontextConfigv12, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        action_expert_config = _gemma.get_config(config.action_expert_variant, "action_expert")
        prompt_expert_config = _gemma.get_config(config.prompt_expert_variant, "prompt_expert")
        self.use_image_prompts = config.use_image_prompts
        self.use_text_prompts = config.use_text_prompts
        self.use_action_state_prompts = config.use_action_state_prompts
        self.avg_current_img = config.avg_current_img
        self.causal_attention = config.causal_attention
        # import ipdb; ipdb.set_trace()
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[prompt_expert_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=prompt_expert_config.width,
                variant="So400m/14",
                pool_type="none",
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

    @at.typecheck
    def embed_midfix_causal(
        self, obs: _model.ObservationIncontext
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        # TODO: this is hard-coded for 2 prompt images, need to be changed
        input_mask = []
        ar_mask = []
        tokens = []

        # -------------------------------------------------------------------------
        # embed in-context images
        # TODO: set a ratio to randomly mask input or prompt
        # import ipdb; ipdb.set_trace()
        assert self.use_image_prompts == True
        assert self.use_action_state_prompts == True

        if self.use_image_prompts:
            img_tokens_per_cam: dict[str, jnp.ndarray] = {}
            img_input_mask_per_cam: dict[str, jnp.ndarray] = {}

            for name in obs.incontext_images:
                image_sequence = obs.incontext_images[name]
                if len(image_sequence.shape) == 6:
                    batch_size, episode_len, seq_len = image_sequence.shape[0], image_sequence.shape[1], image_sequence.shape[2]
                    image_sequence = image_sequence.reshape(
                        image_sequence.shape[0] * image_sequence.shape[1] * image_sequence.shape[2], *image_sequence.shape[3:]
                    )
                    image_sqeuence_tokens, _ = self.PaliGemma.img(image_sequence, train=False)
                    # import ipdb; ipdb.set_trace()
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
                    image_sqeuence_tokens = image_sqeuence_tokens.reshape(
                        batch_size, seq_len, -1, image_sqeuence_tokens.shape[-1]
                    )

                image_sqeuence_tokens = jnp.mean(image_sqeuence_tokens, axis=2)
                # image_sqeuence_tokens = self.img_proj(image_sqeuence_tokens)
                img_tokens_per_cam[name] = image_sqeuence_tokens
                img_input_mask_per_cam[name] = obs.incontext_image_masks[name] 
                # tokens.append(image_sqeuence_tokens)
                # input_mask.append(obs.incontext_image_masks[name])
                # jax.debug.print("name = {}, obs.incontext_image_masks = {}", name, obs.incontext_image_masks[name])
                # ar_mask += [False] * image_sqeuence_tokens.shape[1]

        #------------------------------------------------------------------------
        # embed in-context states
        if self.use_action_state_prompts:
            # import ipdb; ipdb.set_trace()
            if len(obs.incontext_states.shape) == 4:
                incontext_states_reshape = obs.incontext_states.reshape(obs.incontext_states.shape[0], -1, obs.incontext_states.shape[-1])
                dem_state_tokens = self.demo_state_proj(incontext_states_reshape)
                incontext_state_masks_input = obs.incontext_state_masks.reshape(obs.incontext_states.shape[0], -1)
            else:
                dem_state_tokens = self.demo_state_proj(obs.incontext_states)
                incontext_state_masks_input = obs.incontext_state_masks
            # tokens.append(dem_state_tokens)
            # input_mask.append(incontext_state_masks_input)
            # ar_mask += [False] * dem_state_tokens.shape[1]

            #------------------------------------------------------------------------
            # embed in-context actions
            if len(obs.incontext_actions.shape) == 4:
                incontext_actions_reshape = obs.incontext_actions.reshape(obs.incontext_actions.shape[0], -1, obs.incontext_actions.shape[-1])
                dem_action_tokens = self.demo_action_proj(incontext_actions_reshape)
                incontext_action_masks_input = obs.incontext_action_masks.reshape(obs.incontext_actions.shape[0], -1)
            else:
                dem_action_tokens = self.demo_action_proj(obs.incontext_actions)
                incontext_action_masks_input = obs.incontext_action_masks
            # tokens.append(dem_action_tokens)
            # input_mask.append(incontext_action_masks_input)
            # ar_mask += [False] * dem_action_tokens.shape[1]
        
        batch_size_s, seq_len_s, dimension_s = dem_state_tokens.shape
        inter_tokens = jnp.stack([dem_state_tokens, dem_action_tokens], axis=2).reshape(batch_size_s, 2*seq_len_s, dimension_s)  # (32, 2, 2048)
        # jax.debug.print("inter_tokens shape = {}, dem_state_tokens shape = {}", inter_tokens.shape, dem_state_tokens.shape)
        # jax.debug.print("Are states aligned? {}", jnp.allclose(inter_tokens[:, ::2, :], dem_state_tokens))
        # import ipdb; ipdb.set_trace()

        inter_mask = jnp.stack([incontext_state_masks_input, incontext_action_masks_input], axis=2).reshape(batch_size_s, 2*seq_len_s)  

        # This part is hard coded
        for name in obs.incontext_images:    
            assert img_tokens_per_cam[name].shape[1] == 2
            tokens.append(img_tokens_per_cam[name][:, 0, :].reshape(batch_size_s, 1, dimension_s))
            input_mask.append(img_input_mask_per_cam[name][:, 0].reshape(batch_size_s, 1))

        tokens.append(inter_tokens[:, :-2, :])
        input_mask.append(inter_mask[:, :-2])
        for name in obs.incontext_images:    
            assert img_tokens_per_cam[name].shape[1] == 2
            tokens.append(img_tokens_per_cam[name][:, 1, :].reshape(batch_size_s, 1, dimension_s))
            input_mask.append(img_input_mask_per_cam[name][:, 1].reshape(batch_size_s, 1))

        tokens.append(inter_tokens[:, -2:, :])
        input_mask.append(inter_mask[:, -2:])

        # import ipdb; ipdb.set_trace()

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            # image_tokens = self.obs_img_proj(image_tokens)
            if self.avg_current_img:
                # import ipdb; ipdb.set_trace()
                image_tokens = jnp.mean(image_tokens, axis=1, keepdims=True)
            tokens.append(image_tokens)  # image_tokens (32, 256, 2048)
            # import ipdb; ipdb.set_trace()
            # jax.debug.print("obs.image_masks = {}", obs.image_masks[name])

            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            # ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if self.use_text_prompts:
            if obs.tokenized_prompt is not None:
                tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
                # tokenized_inputs = self.text_proj(tokenized_inputs)
                tokens.append(tokenized_inputs)
                input_mask.append(obs.tokenized_prompt_mask)
                # full attention between image and language inputs
                # ar_mask += [False] * tokenized_inputs.shape[1]

        # ---------------------------------------------------------
        tokens = jnp.concatenate(tokens, axis=1) # (32, num_tokens, 2048)
        input_mask = jnp.concatenate(input_mask, axis=1)

        # import ipdb; ipdb.set_trace()

        # ar_mask[0] = True
        ar_mask = [True] * tokens.shape[1]
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_midfix(
        self, obs: _model.ObservationIncontext
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:

        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            # image_tokens = self.obs_img_proj(image_tokens)
            if self.avg_current_img:
                # import ipdb; ipdb.set_trace()
                image_tokens = jnp.mean(image_tokens, axis=1, keepdims=True)
            tokens.append(image_tokens)  # image_tokens (32, 256, 2048)
            # import ipdb; ipdb.set_trace()
            # jax.debug.print("name = {}, obs.image_masks = {}", name, obs.image_masks[name])


            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

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
                    # import ipdb; ipdb.set_trace()
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
                    image_sqeuence_tokens = image_sqeuence_tokens.reshape(
                        batch_size, seq_len, -1, image_sqeuence_tokens.shape[-1]
                    )

                image_sqeuence_tokens = jnp.mean(image_sqeuence_tokens, axis=2)
                # image_sqeuence_tokens = self.img_proj(image_sqeuence_tokens)
                tokens.append(image_sqeuence_tokens)
                input_mask.append(obs.incontext_image_masks[name])
                ar_mask += [False] * image_sqeuence_tokens.shape[1]

        #------------------------------------------------------------------------
        # embed in-context states
        if self.use_action_state_prompts:
            # import ipdb; ipdb.set_trace()
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
        # import ipdb; ipdb.set_trace()
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

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.ObservationIncontext,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        # jax.debug.print("observation = {} ", observation)
        # import ipdb; ipdb.set_trace()
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation_incontext(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if self.causal_attention:
            midfix_tokens, midfix_mask, midfix_ar_mask = self.embed_midfix_causal(observation)    
        else:
            midfix_tokens, midfix_mask, midfix_ar_mask = self.embed_midfix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([midfix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([midfix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        # import ipdb; ipdb.set_trace()
        (midfix_out, suffix_out), _ = self.PaliGemma.llm(
            [midfix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.ObservationIncontext,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        # import ipdb; ipdb.set_trace()
        observation = _model.preprocess_observation_incontext(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if self.causal_attention:
            midfix_tokens, midfix_mask, midfix_ar_mask = self.embed_midfix_causal(observation)
        else:
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
