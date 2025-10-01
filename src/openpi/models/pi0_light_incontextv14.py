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
    """JAX 0-d bool -> host端断言，避免 Tracer 具体化。"""
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

    avg_current_img: bool = False
    causal_attention: bool = False
    
    # XJ: for training sequence
    use_frame_sequence_transform: bool = False
    frame_sequence_length: int = 4
    debug_fused_checks: bool = False

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
        self.debug_fused_checks = config.debug_fused_checks

        # import ipdb; ipdb.set_trace()
        # TODO: rewrite gemma in NNX. For now, use bridge.
        gemma_kwargs = {
            "configs": [prompt_expert_config, action_expert_config],
            "embed_dtype": config.dtype,
            "debug_checks":config.debug_fused_checks,
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
            raise ValueError(f"Unknown SigLIP variant '{siglip_variant}' — unable to determine output dim.")
        self.image_proj = nnx.Linear(siglip_output_dim, prompt_expert_config.width, rngs=rngs)

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
                    image_sqeuence_tokens = self.image_proj(image_sqeuence_tokens)

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
                    image_sqeuence_tokens = self.image_proj(image_sqeuence_tokens)

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

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            image_tokens = self.image_proj(image_tokens)

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
    #     attn_mask = make_attn_mask(input_mask, ar_mask)
    #     positions = jnp.cumsum(input_mask, axis=1) - 1
    #     # import ipdb; ipdb.set_trace()
    #     (midfix_out, suffix_out), _ = self.PaliGemma.llm(
    #         [midfix_tokens, suffix_tokens], mask=attn_mask, positions=positions
    #     )
    #     v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

    #     return jnp.mean(jnp.square(v_t - u_t), axis=-1)
    
    # @override
    # def compute_loss(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.ObservationIncontext,
    #     actions: _model.Actions,
    #     *,
    #     train: bool = False,
    # ) -> at.Float[at.Array, "*b ah"]:
    #     """
    #     方案B版 compute_loss：
    #     1) midfix 走一次前向，构建“可微”的 per-layer KV（deterministic=True，避免同迭代内抖动）
    #     2) suffix（含 state + [可选：当前帧图像] + H 个 action）一次性前向，复用 midfix KV
    #     3) 对最后 H 个 action 位置投线性头得到 v_t，做 FM 的 MSE(v_t, u_t)
    #     """

    #     # ---------------------------
    #     # 0) 预处理 + FM 合成
    #     # ---------------------------
    #     preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    #     obs = _model.preprocess_observation_incontext(preprocess_rng, observation, train=train)

    #     # 真值动作 x0、噪声 eps、采样时间 t（Single-t；对样本内 H 广播）
    #     noise = jax.random.normal(noise_rng, actions.shape)                          # [B,H,A]
    #     t = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001    # [B]
    #     t_exp = t[..., None, None]                                                   # [B,1,1]
    #     x_t = t_exp * noise + (1.0 - t_exp) * actions                                # [B,H,A]
    #     u_t = noise - actions                                                        # [B,H,A]

    #     # ---------------------------
    #     # 1) midfix → 可微 KV（一次）
    #     # ---------------------------
    #     # 约定：embed_midfix 仅包含 in-context 演示/文本等“提示侧”信息，不含“当前帧图像”
    #     midfix_tokens, midfix_mask, midfix_ar = self.embed_midfix(obs)               # tokens:[B,S_mid,D], mask:[B,S_mid]
    #     midfix_attn = make_attn_mask(midfix_mask, midfix_ar)                         # [B,S_mid,S_mid]
    #     pos_midfix = jnp.cumsum(midfix_mask, axis=1) - 1                             # [B,S_mid]

    #     # 关键：midfix 路径 deterministic=True，使得同一迭代内复用的 KV 不随 dropout 抖动
    #     pm_cache = self.PaliGemma.llm(
    #         [midfix_tokens, None],
    #         positions=pos_midfix,
    #         mask=midfix_attn,
    #         deterministic=True,
    #         method="encode_pm_only",
    #     )

    #     # ---------------------------
    #     # 2) suffix（一次性） + 复用 midfix KV
    #     # ---------------------------
    #     # 约定：embed_suffix 输出包含 [state] + [可选：当前帧图像 tokens] + [H 个 action tokens]
    #     suffix_tokens, suffix_mask, suffix_ar = self.embed_suffix(obs, x_t, t)       # tokens:[B,Ts,D], mask:[B,Ts]
    #     suffix_attn = make_attn_mask(suffix_mask, suffix_ar)                         # [B,Ts,Ts]

    #     # 让 suffix 的每个 query 都能看到 midfix 的所有 key/value
    #     midfix_seen = einops.repeat(midfix_mask, "b p -> b s p", s=suffix_tokens.shape[1])  # [B,Ts,S_mid]
    #     mask_full = jnp.concatenate([midfix_seen, suffix_attn], axis=-1)                   # [B,Ts,S_mid+Ts]

    #     # suffix 的绝对位置整体偏移到 midfix 之后
    #     pos_suf = jnp.sum(midfix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1  # [B,Ts]
        
    #     assert mask_full.shape[-1] == midfix_mask.shape[1] + suffix_mask.shape[1]
    #     assert pos_suf.shape == (suffix_tokens.shape[0], suffix_tokens.shape[1])
        
    #     # 只解码 action_expert 分支；复用上面得到的 pm_cache
    #     B = obs.state.shape[0]
    #     ep_index = einops.repeat(jnp.arange(B, dtype=jnp.int32), "b -> (b n)", n=1)
    #     outs_suf = self.PaliGemma.llm(
    #         [None, suffix_tokens],
    #         positions=pos_suf,
    #         mask=mask_full,
    #         pm_cache=pm_cache,
    #         ep_index=ep_index,
    #         deterministic=True,
    #         method="decode_with_cache",
    #     )
    #     suffix_out = outs_suf[1]   # action_expert 对应的输出，形状 [B,Ts,D]

    #     # ---------------------------
    #     # 3) 取出动作位、投头、计算 FM Loss
    #     # ---------------------------
    #     # 约定：suffix 的最后 H 个 token 是 action 位置（前面可能是 state / 当前帧图像）
    #     action_start = suffix_out.shape[1] - self.action_horizon
    #     action_hidden = suffix_out[:, action_start:, :]         # [B,H,D]
    #     v_t = self.action_out_proj(action_hidden)               # [B,H,A]

    #     # 对最后一维 A 求 MSE，返回 [B,H]（保持与你原实现一致）
    #     return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.ObservationIncontext,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:

        cfg_dbg = self.debug_fused_checks
        if cfg_dbg:
            self._dbg_reset_flags()

        # ---------------------------
        # 0) 预处理 & 判定是否多帧
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
        assert B > 0, "批大小 B 必须 > 0"

        if has_multi:
            # 形状期望检查
            assert obs.actions_seq.ndim == 4, f"obs.actions_seq 期望 [B,N,H,A]，got {obs.actions_seq.shape}"
            N = obs.current_state_seq.shape[1]
            H = obs.actions_seq.shape[2]
            A = obs.actions_seq.shape[3]
            assert N > 0 and H > 0 and A > 0, f"N/H/A 必须 > 0，got N={N}, H={H}, A={A}"

            actions_seq = obs.actions_seq
            obs_current_state_seq       = obs.current_state_seq
            obs_current_images_seq      = obs.current_images_seq
            obs_current_image_masks_seq = obs.current_image_masks_seq

            if cfg_dbg:
                # 所有相机视角的序列长度一致
                lens = [v.shape[1] for v in obs_current_images_seq.values()]
                assert all(L == N for L in lens), f"各相机 N 不一致: {lens}"
        else:
            # 单帧回退
            assert actions.ndim == 3, f"单帧模式期望 actions=[B,H,A]，got {actions.shape}"
            N = 1
            H, A = actions.shape[1], actions.shape[2]
            actions_seq = actions[:, None, :, :]
            obs_current_state_seq       = obs.state[:, None, :]
            obs_current_images_seq      = {k: v[:, None, ...] for k, v in obs.images.items()}
            obs_current_image_masks_seq = {k: v[:, None]      for k, v in obs.image_masks.items()}

        # 如果你“必须”强制 fused 路径（例如训练阶段），可以加这一条硬断言：
        if cfg_dbg and train:
            assert has_multi and N > 1, "训练时要求走 Fused-N 路径，但检测到非多帧（N<=1）。"

        # ---------------------------
        # 1) FM 噪声合成
        # ---------------------------
        noise = jax.random.normal(noise_rng, actions_seq.shape)                       # [B,N,H,A]
        t     = jax.random.beta(time_rng, 1.5, 1, (B, N)) * 0.999 + 0.001            # [B,N]
        x_t   = t[..., None, None] * noise + (1.0 - t[..., None, None]) * actions_seq
        u_t   = noise - actions_seq

        if cfg_dbg:
            self._assert_shape(x_t,   (B, N, H, A), "x_t")
            self._assert_shape(u_t,   (B, N, H, A), "u_t")
            self._assert_shape(t,     (B, N),       "t")

        # ---------------------------
        # 2) midfix → encode_pm_only（一次）
        # ---------------------------
        midfix_tokens, midfix_mask, midfix_ar = self.embed_midfix(obs)               # [B, S_mid, D], [B, S_mid], [S_mid]
        if cfg_dbg:
            self._assert_bool_mask(midfix_mask, "midfix_mask")
            assert midfix_ar.ndim == 1, f"midfix_ar 期望 [S_mid]，got {midfix_ar.shape}"

        midfix_attn = make_attn_mask(midfix_mask, midfix_ar)                         # [B,S_mid,S_mid]
        pos_midfix  = jnp.cumsum(midfix_mask, axis=1) - 1                             # [B,S_mid]

        if cfg_dbg:
            self._assert_shape(midfix_tokens, (B, None, None), "midfix_tokens")
            self._assert_shape(midfix_attn,   (B, midfix_tokens.shape[1], midfix_tokens.shape[1]), "midfix_attn")
            self._assert_shape(pos_midfix,    (B, midfix_tokens.shape[1]), "pos_midfix")
            self._assert_monotonic_positions(pos_midfix, midfix_mask, "pos_midfix", allow_offset=False)

        pm_cache = self._llm_encode_pm_only(
            embedded_pm=[midfix_tokens, None],
            positions_pm=pos_midfix,
            mask_pm=midfix_attn,
            deterministic=True,
        )

        if cfg_dbg:
            assert self._dbg_used_encode_pm_only, "未走 encode_pm_only 路径（包装器未被触发）"
            assert pm_cache is not None, "encode_pm_only 应返回 pm_cache，但得到 None"

        # ---------------------------
        # 3) 构造后缀 BN，一次 decode_with_cache
        # ---------------------------
        BN = B * N
        flat_images = {name: einops.rearrange(img, "b n h w c -> (b n) h w c")
                       for name, img in obs_current_images_seq.items()}
        flat_img_masks = {name: einops.rearrange(msk, "b n -> (b n)")
                          for name, msk in obs_current_image_masks_seq.items()}
        flat_states = einops.rearrange(obs_current_state_seq, "b n a -> (b n) a")
        flat_x_t    = einops.rearrange(x_t, "b n h a -> (b n) h a")
        flat_t      = einops.rearrange(t,   "b n -> (b n)")

        # 复用你已有的逻辑，构造 suffix
        def build_suffix_tokens_from_flat(flat_imgs, flat_img_masks, flat_states_, flat_x_t_, flat_t_):
            input_mask = []
            ar_mask = []
            tokens = []

            for name, img in flat_imgs.items():
                img_tokens, _ = self.PaliGemma.img(img, train=(train and not self.freeze_img_encoder))
                img_tokens = self.image_proj(img_tokens)
                if self.avg_current_img:
                    img_tokens = jnp.mean(img_tokens, axis=1, keepdims=True)
                tokens.append(img_tokens)
                input_mask.append(einops.repeat(flat_img_masks[name], "bn -> bn s", s=img_tokens.shape[1]))
                ar_mask += [False] * img_tokens.shape[1]

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

        suffix_tokens, suffix_mask, suffix_ar = build_suffix_tokens_from_flat(
            flat_images, flat_img_masks, flat_states, flat_x_t, flat_t
        )

        if cfg_dbg:
            self._assert_shape(suffix_tokens, (BN, None, None), "suffix_tokens")
            self._assert_bool_mask(suffix_mask, "suffix_mask")
            assert suffix_ar.ndim == 1, f"suffix_ar 期望 [Ts]，got {suffix_ar.shape}"
            # 检查动作块 AR 规则是否满足（尾部 H 个 token）
            self._assert_block_ar_mask(suffix_ar, self.action_horizon, "suffix_ar")

        suffix_attn = make_attn_mask(suffix_mask, suffix_ar)                          # [BN,Ts,Ts]
        midfix_seen = einops.repeat(midfix_mask, "b p -> (b n) s p", n=N, s=suffix_tokens.shape[1])
        full_mask   = jnp.concatenate([midfix_seen, suffix_attn], axis=-1)

        pos_offset = einops.repeat(jnp.sum(midfix_mask, axis=-1), "b -> (b n) 1", n=N)
        pos_suf    = pos_offset + jnp.cumsum(suffix_mask, axis=-1) - 1

        if cfg_dbg:
            self._assert_shape(suffix_attn, (BN, suffix_tokens.shape[1], suffix_tokens.shape[1]), "suffix_attn")
            self._assert_shape(full_mask,  (BN, suffix_tokens.shape[1], midfix_mask.shape[1] + suffix_tokens.shape[1]), "full_mask")
            self._assert_shape(pos_suf,    (BN, suffix_tokens.shape[1]), "pos_suf")
            self._assert_monotonic_positions(pos_suf, suffix_mask, "pos_suf", allow_offset=True)

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

        if cfg_dbg:
            assert self._dbg_used_decode_with_cache, "未走 decode_with_cache 路径（包装器未被触发）"
            # 软校验：suffix_out 的 batch 应为 BN
            self._assert_shape(suffix_out, (BN, None, None), "suffix_out")

        # ---------------------------
        # 4) 动作位 & Loss
        # ---------------------------
        action_start  = suffix_out.shape[1] - self.action_horizon
        assert action_start >= 0, f"Ts ({suffix_out.shape[1]}) 必须 ≥ action_horizon ({self.action_horizon})"
        action_hidden = suffix_out[:, action_start:, :]
        v_t = self.action_out_proj(action_hidden)  # [BN,H,A]
        v_t = einops.rearrange(v_t, " (b n) h a -> b n h a", b=B, n=N)

        if cfg_dbg:
            self._assert_shape(v_t, (B, N, H, A), "v_t")
            self._assert_shape(u_t, (B, N, H, A), "u_t")

        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [B,N,H]

        # 最后形状断言 & “确实为 fused”的强证据
        if has_multi:
            if cfg_dbg:
                assert loss.shape == (B, N, H), f"loss 期望 [B,N,H]，got {loss.shape}"
                assert N > 1, "has_multi=True 但 N<=1，不满足 fused-multi 的语义。"
        else:
            if cfg_dbg:
                assert loss.shape == (B, 1, H), f"单帧回退 loss 期望 [B,1,H]，got {loss.shape}"

        return loss if has_multi else loss[:, 0, :]



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
    # Debug helpers（轻量包装 & 断言工具）
    # ---------------------------
    def _dbg_reset_flags(self):
        # 记录是否真的调用了 encode_pm_only / decode_with_cache
        self._dbg_used_encode_pm_only = False
        self._dbg_used_decode_with_cache = False
        self._dbg_last_pm_cache = None

    def _llm_encode_pm_only(self, *, embedded_pm, positions_pm, mask_pm, deterministic):
        self._dbg_used_encode_pm_only = True
        pm_cache = self.PaliGemma.llm(
            embedded_pm=embedded_pm,
            positions_pm=positions_pm,
            mask_pm=mask_pm,
            deterministic=deterministic,
            method="encode_pm_only",
        )
        self._dbg_last_pm_cache = pm_cache
        return pm_cache

    def _llm_decode_with_cache(self, *, embedded_suf, positions_suf, mask_suf, pm_cache, ep_index, deterministic):
        # 确保复用的 pm_cache 非空
        assert pm_cache is not None, "decode_with_cache 期望传入非空 pm_cache，但得到 None。"
        self._dbg_used_decode_with_cache = True
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
        assert x.dtype == jnp.bool_, f"{name} 必须是 bool mask，当前 dtype={x.dtype}"

    @staticmethod
    def _assert_shape(x, shape, name: str):
        # shape 元素可用 None 表示“任意”
        assert len(x.shape) == len(shape), f"{name} 维度数不符：got {x.shape}, expect {shape}"
        for i, (got, exp) in enumerate(zip(x.shape, shape)):
            if exp is not None and got != exp:
                raise AssertionError(f"{name} 维度[{i}]不符：got {x.shape}, expect {shape}")
        
    @staticmethod
    def _assert_monotonic_positions(pos: jax.Array,
                                    mask: jax.Array,
                                    name: str,
                                    allow_offset: bool = False):
        """
        检查 positions 在“有效 token”处是否按步进 1 递增。
        - 当 allow_offset=False：要求等于 cumsum(valid)-1
        - 当 allow_offset=True ：允许整体加一个 batch-wise 偏移（例如 suffix 放在 midfix 之后）
        允许 mask 为 [B,T] / [B,T,S] / [B,1,T,S]。
        """
        # 1) 统一 token 级 mask: [B,T]
        if mask.ndim == 4:
            assert mask.shape[1] == 1, f"{name}: 4D mask 第二维应为 1, got {mask.shape}"
            mask = jnp.squeeze(mask, axis=1)        # [B,T,S]
        if mask.ndim == 3:
            token_mask = jnp.any(mask, axis=-1)     # [B,T]
        elif mask.ndim == 2:
            token_mask = mask                       # [B,T]
        else:
            raise ValueError(f"{name}: mask 维度必须是 2/3/4, got {mask.shape}")

        # 2) 形状一致（用 host 断言避免 Tracer 转换）
        _host_assert(jnp.array(pos.ndim == 2), "{}: positions 必须是 [B,T]", name)
        _host_assert(jnp.array((pos.shape[0] == token_mask.shape[0]) &
                            (pos.shape[1] == token_mask.shape[1])),
                    "{}: positions 形状 {} 必须等于 token_mask 形状 {}",
                    name, pos.shape, token_mask.shape)

        # 3) 参考序列
        ref_incr = jnp.cumsum(token_mask, axis=-1) - 1   # [B,T]

        if allow_offset:
            # 计算每个 batch 的“起始位置”作为 offset（第一个 valid 的 positions）
            big = jnp.iinfo(jnp.int32).max
            # 把无效处置为 +inf，再取最小值就等于“第一个有效位置”
            first_pos = jnp.min(jnp.where(token_mask, pos, big), axis=1)   # [B]
            # 防止全无效（极端情况），把 +inf 替回 0
            first_pos = jnp.where(first_pos == big, 0, first_pos)
            ref = first_pos[:, None] + ref_incr
        else:
            ref = ref_incr

        ok = jnp.all(jnp.where(token_mask, pos == ref, True))
        _host_assert(ok, "{} 不是基于 mask 的（允许偏移={}）单调累计位置", name, allow_offset)


    @staticmethod
    def _assert_block_ar_mask(ar_mask: jax.Array, H: int, name: str):
        """
        约定：后缀最后 H 个 token 是动作块，且模式应为：
        [True, False, False, ..., False]  # 长度 H
        其余位置不限。
        """
        m = jnp.asarray(ar_mask, dtype=bool).reshape(-1)     # [Ts]
        Ts = m.shape[0]                                      # 这是 Python int（静态 shape），可直接比较
        if Ts < H:
            # 这条在 trace 前就能跑到，保留原生 assert 也没问题
            raise AssertionError(f"{name}: len(ar_mask)={Ts} < action_horizon={H}")

        tail = m[-H:]                                        # [H]
        # cond: tail[0] == True 且 tail[1:] 全 False
        cond = jnp.logical_and(tail[0], jnp.all(jnp.logical_not(tail[1:])))
        _host_assert(cond, "{}: 动作块末尾 H 模式应为 [True, False×(H-1)]，实际 tail={}", name, tail)
