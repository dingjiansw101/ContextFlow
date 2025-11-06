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
class Pi0IncontextConfigv17(_model.BaseModelConfig):
    """Configuration for π₀ In-Context v17: 3-expert sequential diffusion model.

    Architecture:
    - Prompt Expert: Processes visual/textual prompts + in-context demonstrations
    - State Expert: Predicts future states using flow matching diffusion
    - Action Expert: Generates actions conditioned on predicted future states

    The model performs sequential diffusion:
    1. Generate future state trajectory from current observation
    2. Generate action sequence conditioned on predicted future states

    This decoupling allows the action expert to leverage temporal state information
    during action generation, improving long-horizon task performance.
    """

    dtype: str = "bfloat16"

    # THREE expert variants (default: prompt=full, state=LoRA, action=LoRA)
    prompt_expert_variant: _gemma.Variant = "gemma_300m_v2"
    state_expert_variant: _gemma.Variant = "gemma_300m_lora"
    action_expert_variant: _gemma.Variant = "gemma_300m_lora"

    # Model dimensions
    action_dim: int = 32
    state_dim: int = 32  # State dimension (often same as action_dim)
    action_horizon: int = 50
    max_token_len: int = 48

    # Future state prediction configuration
    future_state_downsample: int = 5  # Downsample factor for future states (action_horizon // future_state_downsample)
    state_loss_weight: float = 0.5  # Weight for future state prediction loss

    def __post_init__(self):
        """Validate config parameters."""
        if self.action_horizon % self.future_state_downsample != 0:
            raise ValueError(
                f"action_horizon ({self.action_horizon}) must be divisible by "
                f"future_state_downsample ({self.future_state_downsample})"
            )

    @property
    def future_state_horizon(self) -> int:
        """Compute future_state_horizon from action_horizon and downsample factor.

        Returns:
            Number of future states = action_horizon // future_state_downsample

        Note:
            Divisibility is validated in __post_init__.
        """
        return self.action_horizon // self.future_state_downsample

    @property
    def state_expert_width(self) -> int:
        """Hidden dimension of the state expert."""
        return _gemma.get_config(self.state_expert_variant).width

    @property
    def action_expert_width(self) -> int:
        """Hidden dimension of the action expert."""
        return _gemma.get_config(self.action_expert_variant).width

    def fake_obs(self, batch_size: int = 1) -> _model.ObservationIncontext:
        """Create fake observation with correct incontext dimensions.

        Overrides BaseModelConfig.fake_obs to ensure incontext_images and other
        incontext fields are created with proper 5-D/6-D shapes.
        """
        observation_spec, _ = self.inputs_spec(
            batch_size=batch_size,
            keyframe_size=self.sample_frames,
            max_len=64  # Default max sequence length for incontext states/actions
        )
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    # In-context learning params
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
    def create(self, rng: at.KeyArrayLike) -> "Pi0Incontextv17":
        return Pi0Incontextv17(self, rngs=nnx.Rngs(rng))

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
                future_states=jax.ShapeDtypeStruct([batch_size, self.future_state_horizon, self.state_dim], jnp.float32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter for 3-expert architecture.

        Vision encoder is TRAINABLE by default (not frozen).

        Freeze Policy:
        - Vision encoder (img): trainable (no freeze)
        - Expert with LoRA: freeze base weights, train LoRA adapters
        - Expert without LoRA: fully trainable (no freeze)
        - Projection layers: always trainable (no freeze)

        With default config (prompt=no lora, state=lora, action=lora):
          - Freezes: state_expert base, action_expert base
          - Trains: vision, prompt_expert, state LoRA, action LoRA, projections
        """
        filters = []

        # Define regex filters for each component
        prompt_expert_filter = nnx_utils.PathRegex(".*llm.*_prompt_expert.*")
        state_expert_filter = nnx_utils.PathRegex(".*llm.*_state_expert.*")
        action_expert_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        lora_filter = nnx_utils.PathRegex(".*lora.*")

        # Check which experts use LoRA
        prompt_has_lora = "lora" in self.prompt_expert_variant
        state_has_lora = "lora" in self.state_expert_variant
        action_has_lora = "lora" in self.action_expert_variant

        # Case 1: No LoRA anywhere - train everything (including vision)
        if not (prompt_has_lora or state_has_lora or action_has_lora):
            return nnx.Nothing

        # Case 2+: At least one expert uses LoRA

        # Freeze base weights of LoRA experts (union of selected experts)
        base_filters = []
        if prompt_has_lora:
            base_filters.append(prompt_expert_filter)
        if state_has_lora:
            base_filters.append(state_expert_filter)
        if action_has_lora:
            base_filters.append(action_expert_filter)

        if len(base_filters) == 1:
            filters.append(base_filters[0])
        else:
            filters.append(nnx.Any(*base_filters))

        # Unfreeze LoRA adapters (they should be trainable)
        filters.append(nnx.Not(lora_filter))

        # Unfreeze non-LoRA experts (they should be fully trainable)
        if not prompt_has_lora:
            filters.append(nnx.Not(prompt_expert_filter))
        if not state_has_lora:
            filters.append(nnx.Not(state_expert_filter))
        if not action_has_lora:
            filters.append(nnx.Not(action_expert_filter))

        # Combine all filters with logical AND
        return nnx.All(*filters)


class Pi0Incontextv17(_model.BaseModel):
    """π₀ In-Context v17: Sequential diffusion with 3 Gemma experts.

    This model extends the base π₀ architecture with:
    1. Three separate Gemma LLM experts (prompt, state, action)
    2. Sequential diffusion: state prediction → action generation
    3. In-context learning from demonstration images, states, and actions
    4. Future state conditioning for improved action generation

    Training:
    - Uses flow matching for both state and action diffusion
    - State and action losses are weighted and combined
    - Gradient flow is blocked from action expert to state predictions

    Inference:
    - First diffuses to predict future state trajectory (num_steps iterations)
    - Then diffuses to generate actions conditioned on predicted states
    - KV cache optimization avoids redundant prompt encoding
    """

    def __init__(self, config: Pi0IncontextConfigv17, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        prompt_expert_config = _gemma.get_config(config.prompt_expert_variant, "prompt_expert")
        state_expert_config = _gemma.get_config(config.state_expert_variant, "state_expert")
        action_expert_config = _gemma.get_config(config.action_expert_variant, "action_expert")
        self.use_image_prompts = config.use_image_prompts
        self.use_text_prompts = config.use_text_prompts
        self.use_action_state_prompts = config.use_action_state_prompts
        self.avg_current_img = config.avg_current_img
        self.causal_attention = config.causal_attention
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[prompt_expert_config, state_expert_config, action_expert_config],
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

        # Current state projection (shared by both state and action experts)
        self.state_proj = nnx.Linear(config.state_dim, state_expert_config.width, rngs=rngs)

        # STATE EXPERT projections (predicts future states)
        self.future_state_in_proj = nnx.Linear(config.state_dim, state_expert_config.width, rngs=rngs)
        self.state_time_mlp_in = nnx.Linear(2 * state_expert_config.width, state_expert_config.width, rngs=rngs)
        self.state_time_mlp_out = nnx.Linear(state_expert_config.width, state_expert_config.width, rngs=rngs)
        self.future_state_out_proj = nnx.Linear(state_expert_config.width, config.state_dim, rngs=rngs)

        # ACTION EXPERT projections (predicts actions)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Future state conditioning projection (for action expert)
        self.future_state_conditioning_proj = nnx.Linear(config.state_dim, action_expert_config.width, rngs=rngs)

        # Demo projections (from v12 - for in-context learning)
        if self.use_action_state_prompts:
            self.demo_action_proj = nnx.Linear(config.action_dim, prompt_expert_config.width, rngs=rngs)
            self.demo_state_proj = nnx.Linear(config.action_dim, prompt_expert_config.width, rngs=rngs)

        # Store config
        self.config = config

    @at.typecheck
    def embed_midfix(
        self, obs: _model.ObservationIncontext
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed shared prompt inputs for all experts.

        Processes:
        - Current observation images (via SigLIP vision encoder)
        - Text prompts (tokenized language)
        - In-context demonstration images (optional)
        - In-context demonstration states/actions (optional)

        Returns:
            tokens: Concatenated embeddings [batch, seq_len, embed_dim]
            input_mask: Valid token mask [batch, seq_len]
            ar_mask: Autoregressive mask pattern [seq_len]
                     False = bidirectional attention, True = causal block boundary
        """
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            if self.avg_current_img:
                image_tokens = jnp.mean(image_tokens, axis=1, keepdims=True)
            tokens.append(image_tokens)

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
        if self.use_text_prompts and obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
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
                # Handle different tensor shapes:
                # - 6D: [B, num_episodes, frames_per_episode, H, W, C] - multiple demo episodes
                # - 5D: [B, num_frames, H, W, C] - single demo episode
                if len(image_sequence.shape) == 6:
                    batch_size, episode_len, seq_len = (
                        image_sequence.shape[0],
                        image_sequence.shape[1],
                        image_sequence.shape[2],
                    )
                    image_sequence = image_sequence.reshape(
                        image_sequence.shape[0] * image_sequence.shape[1] * image_sequence.shape[2],
                        *image_sequence.shape[3:],
                    )
                    image_sequence_tokens, _ = self.PaliGemma.img(image_sequence, train=False)
                    # TODO: to organize multiple episode prompts in order
                    image_sequence_tokens = image_sequence_tokens.reshape(
                        batch_size, episode_len * seq_len, -1, image_sequence_tokens.shape[-1]
                    )
                    obs.incontext_image_masks[name] = obs.incontext_image_masks[name].reshape(
                        batch_size, episode_len * seq_len
                    )
                elif len(image_sequence.shape) == 5:
                    batch_size, seq_len = image_sequence.shape[0], image_sequence.shape[1]
                    image_sequence = image_sequence.reshape(
                        image_sequence.shape[0] * image_sequence.shape[1], *image_sequence.shape[2:]
                    )
                    image_sequence_tokens, _ = self.PaliGemma.img(image_sequence, train=False)
                    image_sequence_tokens = image_sequence_tokens.reshape(
                        batch_size, seq_len, -1, image_sequence_tokens.shape[-1]
                    )

                image_sequence_tokens = jnp.mean(image_sequence_tokens, axis=2)
                tokens.append(image_sequence_tokens)
                input_mask.append(obs.incontext_image_masks[name])
                ar_mask += [False] * image_sequence_tokens.shape[1]

        # ------------------------------------------------------------------------
        # embed in-context states
        if self.use_action_state_prompts:
            # Handle different tensor shapes:
            # - 4D: [B, num_episodes, seq_len, state_dim] - multiple demo episodes
            # - 3D: [B, seq_len, state_dim] - single demo episode
            if len(obs.incontext_states.shape) == 4:
                incontext_states_reshape = obs.incontext_states.reshape(
                    obs.incontext_states.shape[0], -1, obs.incontext_states.shape[-1]
                )
                dem_state_tokens = self.demo_state_proj(incontext_states_reshape)
                incontext_state_masks_input = obs.incontext_state_masks.reshape(obs.incontext_states.shape[0], -1)
            else:
                dem_state_tokens = self.demo_state_proj(obs.incontext_states)
                incontext_state_masks_input = obs.incontext_state_masks
            tokens.append(dem_state_tokens)
            input_mask.append(incontext_state_masks_input)
            ar_mask += [False] * dem_state_tokens.shape[1]

            # ------------------------------------------------------------------------
            # embed in-context actions
            # Handle different tensor shapes:
            # - 4D: [B, num_episodes, seq_len, action_dim] - multiple demo episodes
            # - 3D: [B, seq_len, action_dim] - single demo episode
            if len(obs.incontext_actions.shape) == 4:
                incontext_actions_reshape = obs.incontext_actions.reshape(
                    obs.incontext_actions.shape[0], -1, obs.incontext_actions.shape[-1]
                )
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
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix_state(
        self,
        obs: _model.ObservationIncontext,
        noisy_future_states: at.Float[at.Array, "b future_horizon state_dim"],
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed inputs for state prediction expert.

        Creates token sequence for the state expert to denoise future states.

        Args:
            obs: Current observation (only state is used)
            noisy_future_states: Flow-matched noisy future states [B, future_horizon, state_dim]
            timestep: Flow matching timestep t ∈ [0, 1] [B]

        Returns:
            tokens: [current_state_token | future_state+time_tokens]
            input_mask: All True (no padding)
            ar_mask: [True | False...] - current state starts new causal block,
                     future states have bidirectional attention
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # Current state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        ar_mask += [True]  # New causal block

        # Timestep embedding
        time_emb = posemb_sincos(timestep, self.future_state_in_proj.out_features, min_period=4e-3, max_period=4.0)

        # Future state + time tokens
        future_state_tokens = self.future_state_in_proj(noisy_future_states)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.config.future_state_horizon)
        state_time_tokens = jnp.concatenate([future_state_tokens, time_tokens], axis=-1)
        state_time_tokens = self.state_time_mlp_in(state_time_tokens)
        state_time_tokens = nnx.swish(state_time_tokens)
        state_time_tokens = self.state_time_mlp_out(state_time_tokens)

        tokens.append(state_time_tokens)
        input_mask.append(jnp.ones(state_time_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.config.future_state_horizon - 1))  # Bidirectional within block

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix_action(
        self,
        obs: _model.ObservationIncontext,
        future_states: at.Float[at.Array, "b future_horizon state_dim"],  # Conditioning
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed inputs for action generation expert.

        Creates token sequence for the action expert to denoise actions,
        conditioned on predicted future states.

        Args:
            obs: Current observation (only state is used)
            future_states: Predicted/GT future states as conditioning [B, future_horizon, state_dim]
            noisy_actions: Flow-matched noisy actions [B, action_horizon, action_dim]
            timestep: Flow matching timestep t ∈ [0, 1] [B]

        Returns:
            tokens: [current_state_token | future_state_conditioning | action+time_tokens]
            input_mask: All True (no padding)
            ar_mask: [True | False... | True, False...] - state starts causal block,
                     future states are bidirectional, actions have causal attention
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # Current state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        ar_mask += [True]  # New causal block

        # Future states as conditioning tokens (NO GRADIENTS during action expert training)
        future_state_conditioning = self.future_state_conditioning_proj(future_states)  # [B, future_horizon, emb]
        tokens.append(future_state_conditioning)
        input_mask.append(jnp.ones(future_state_conditioning.shape[:2], dtype=jnp.bool_))
        ar_mask += [False] * self.config.future_state_horizon  # Bidirectional within conditioning

        # Timestep embedding
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        # Action + time tokens
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)

        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))  # Causal

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.ObservationIncontext,
        actions: _model.Actions,  # [B, action_horizon, action_dim]
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        """Compute loss for both state prediction and action generation experts.

        The future_states are read from observation.future_states.
        Shape: [B, future_state_horizon, state_dim] (NOT action_horizon!)
        Dataloader is responsible for downsampling.
        """
        # Split RNG
        preprocess_rng, noise_rng_state, noise_rng_action, time_rng_state, time_rng_action = jax.random.split(rng, 5)

        # Preprocess observation
        observation = _model.preprocess_observation_incontext(preprocess_rng, observation, train=train)

        # === STAGE 1: Future State Prediction ===

        # Read future_states from observation (already downsampled by dataloader)
        future_states = observation.future_states  # [B, future_state_horizon, state_dim]

        # Sample noise and time for state expert
        batch_shape = future_states.shape[:-2]
        noise_state = jax.random.normal(noise_rng_state, future_states.shape)
        time_state = jax.random.beta(time_rng_state, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_state_expanded = time_state[..., None, None]

        # Flow matching: x_t = t * noise + (1-t) * data
        x_t_state = time_state_expanded * noise_state + (1 - time_state_expanded) * future_states
        u_t_state = noise_state - future_states  # Target velocity

        # === STAGE 2: Action Prediction ===

        # Sample noise and time for action expert
        noise_action = jax.random.normal(noise_rng_action, actions.shape)
        time_action = jax.random.beta(time_rng_action, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_action_expanded = time_action[..., None, None]

        # Flow matching for actions
        x_t_action = time_action_expanded * noise_action + (1 - time_action_expanded) * actions
        u_t_action = noise_action - actions  # Target velocity

        # === FORWARD PASS: All 3 Experts ===

        # Embed midfix (prompt expert input)
        midfix_tokens, midfix_mask, midfix_ar_mask = self.embed_midfix(observation)

        # Embed suffix for state expert
        suffix_state_tokens, suffix_state_mask, suffix_state_ar_mask = self.embed_suffix_state(
            observation, x_t_state, time_state
        )

        # Embed suffix for action expert (conditioned on GT future states)
        suffix_action_tokens, suffix_action_mask, suffix_action_ar_mask = self.embed_suffix_action(
            observation, future_states, x_t_action, time_action
        )

        # Concatenate all masks
        input_mask = jnp.concatenate([midfix_mask, suffix_state_mask, suffix_action_mask], axis=1)
        ar_mask = jnp.concatenate([midfix_ar_mask, suffix_state_ar_mask, suffix_action_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # TODO: add a unit test to verify the attention mask is correct
        # === NEW: Block action tokens from attending to state expert tokens ===
        midfix_len = midfix_mask.shape[1]
        state_len = suffix_state_mask.shape[1]

        state_start = midfix_len
        state_end = midfix_len + state_len
        action_start = midfix_len + state_len
        # action_end = midfix_len + state_len + action_len  # Not needed

        # Set attn_mask[action_positions, state_positions] = False
        # Shape: [B, T, S] - for all action tokens (dim 1), block state tokens (dim 2)
        attn_mask = attn_mask.at[:, action_start:, state_start:state_end].set(False)


        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Forward through all 3 experts
        (midfix_out, state_out, action_out), _ = self.PaliGemma.llm(
            [midfix_tokens, suffix_state_tokens, suffix_action_tokens],  # 3 experts!
            mask=attn_mask,
            positions=positions,
        )

        # === COMPUTE LOSSES ===

        # Future state prediction loss
        v_t_state = self.future_state_out_proj(state_out[:, -self.config.future_state_horizon :])
        loss_state = jnp.mean(jnp.square(v_t_state - u_t_state), axis=-1)

        # Action prediction loss
        v_t_action = self.action_out_proj(action_out[:, -self.action_horizon :])
        loss_action = jnp.mean(jnp.square(v_t_action - u_t_action), axis=-1)

        # Average over temporal dimension to get per-sample losses
        loss_state_per_sample = jnp.mean(loss_state, axis=-1)  # [B, future_state_horizon] -> [B]
        loss_action_per_sample = jnp.mean(loss_action, axis=-1)  # [B, action_horizon] -> [B]

        # Weighted combination
        return self.config.state_loss_weight * loss_state_per_sample + loss_action_per_sample

    def compute_loss_sequential(
        self,
        rng: at.KeyArrayLike,
        observation: _model.ObservationIncontext,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        """Sequential computation of state and action losses (for testing equivalence).

        This method implements the same computation as compute_loss but runs state and action
        experts sequentially rather than in a fused manner. It should produce identical results
        due to attention masking in the fused version.

        Returns identical loss shape as compute_loss: [B]
        """
        # Split RNG
        preprocess_rng, noise_rng_state, noise_rng_action, time_rng_state, time_rng_action = jax.random.split(rng, 5)

        # Preprocess observation
        observation = _model.preprocess_observation_incontext(preprocess_rng, observation, train=train)

        # === STAGE 1: Future State Prediction ===

        # Read future_states from observation
        future_states = observation.future_states  # [B, future_state_horizon, state_dim]

        # Sample noise and time for state expert
        batch_shape = future_states.shape[:-2]
        noise_state = jax.random.normal(noise_rng_state, future_states.shape)
        time_state = jax.random.beta(time_rng_state, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_state_expanded = time_state[..., None, None]

        # Flow matching: x_t = t * noise + (1-t) * data
        x_t_state = time_state_expanded * noise_state + (1 - time_state_expanded) * future_states
        u_t_state = noise_state - future_states  # Target velocity

        # === STAGE 2: Action Prediction ===

        # Sample noise and time for action expert
        noise_action = jax.random.normal(noise_rng_action, actions.shape)
        time_action = jax.random.beta(time_rng_action, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_action_expanded = time_action[..., None, None]

        # Flow matching for actions
        x_t_action = time_action_expanded * noise_action + (1 - time_action_expanded) * actions
        u_t_action = noise_action - actions  # Target velocity

        # === SEQUENTIAL FORWARD PASS ===

        # Embed midfix once (shared by both experts)
        midfix_tokens, midfix_mask, midfix_ar_mask = self.embed_midfix(observation)

        # --- Step 1: State Expert ---
        suffix_state_tokens, suffix_state_mask, suffix_state_ar_mask = self.embed_suffix_state(
            observation, x_t_state, time_state
        )

        # Concatenate midfix + state suffix
        input_mask_state = jnp.concatenate([midfix_mask, suffix_state_mask], axis=1)
        ar_mask_state = jnp.concatenate([midfix_ar_mask, suffix_state_ar_mask], axis=0)
        attn_mask_state = make_attn_mask(input_mask_state, ar_mask_state)
        positions_state = jnp.cumsum(input_mask_state, axis=1) - 1

        # Forward through prompt and state experts only
        (midfix_out_state, state_out, _), _ = self.PaliGemma.llm(
            [midfix_tokens, suffix_state_tokens, None],  # None for action expert
            mask=attn_mask_state,
            positions=positions_state,
        )

        # --- Step 2: Action Expert ---
        suffix_action_tokens, suffix_action_mask, suffix_action_ar_mask = self.embed_suffix_action(
            observation, future_states, x_t_action, time_action
        )

        # Concatenate midfix + action suffix
        input_mask_action = jnp.concatenate([midfix_mask, suffix_action_mask], axis=1)
        ar_mask_action = jnp.concatenate([midfix_ar_mask, suffix_action_ar_mask], axis=0)
        attn_mask_action = make_attn_mask(input_mask_action, ar_mask_action)
        positions_action = jnp.cumsum(input_mask_action, axis=1) - 1

        # Forward through prompt and action experts only
        (midfix_out_action, _, action_out), _ = self.PaliGemma.llm(
            [midfix_tokens, None, suffix_action_tokens],  # None for state expert
            mask=attn_mask_action,
            positions=positions_action,
        )

        # === COMPUTE LOSSES ===

        # Future state prediction loss
        v_t_state = self.future_state_out_proj(state_out[:, -self.config.future_state_horizon :])
        loss_state = jnp.mean(jnp.square(v_t_state - u_t_state), axis=-1)

        # Action prediction loss
        v_t_action = self.action_out_proj(action_out[:, -self.action_horizon :])
        loss_action = jnp.mean(jnp.square(v_t_action - u_t_action), axis=-1)

        # Average over temporal dimension to get per-sample losses
        loss_state_per_sample = jnp.mean(loss_state, axis=-1)  # [B, future_state_horizon] -> [B]
        loss_action_per_sample = jnp.mean(loss_action, axis=-1)  # [B, action_horizon] -> [B]

        # Weighted combination
        return self.config.state_loss_weight * loss_state_per_sample + loss_action_per_sample

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.ObservationIncontext,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        """Generate actions via sequential diffusion.

        Performs two-stage generation:
        1. State Expert: Diffuses from noise to predict future state trajectory
        2. Action Expert: Diffuses from noise to generate actions conditioned on predicted states

        Both stages use flow matching with num_steps iterations. The prompt embeddings
        are computed once and reused via KV cache for efficiency.

        Args:
            rng: JAX random key
            observation: Current observation with images, state, and optional demonstrations
            num_steps: Number of diffusion steps for both state and action generation

        Returns:
            Predicted action sequence [B, action_horizon, action_dim]
        """

        observation = _model.preprocess_observation_incontext(None, observation, train=False)
        rng_state, rng_action = jax.random.split(rng)
        batch_size = observation.state.shape[0]
        dt = -1.0 / num_steps

        # === STAGE 1: Generate Future State Trajectory ===

        # Prepare midfix embeddings (shared by all experts)
        midfix_tokens, midfix_mask, midfix_ar_mask = self.embed_midfix(observation)

        midfix_attn_mask = make_attn_mask(midfix_mask, midfix_ar_mask)
        positions_midfix = jnp.cumsum(midfix_mask, axis=1) - 1

        # Pre-compute KV cache for midfix (reused in all diffusion steps)
        _, kv_cache = self.PaliGemma.llm(
            [midfix_tokens, None, None],  # Only compute prompt expert
            mask=midfix_attn_mask,
            positions=positions_midfix,
        )

        # Initialize with noise
        noise_state = jax.random.normal(
            rng_state, (batch_size, self.config.future_state_horizon, self.config.state_dim)
        )

        # Diffusion loop for future states
        def step_state(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix_state(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )

            # Build attention mask
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            midfix_attn_mask_repeat = einops.repeat(midfix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([midfix_attn_mask_repeat, suffix_attn_mask], axis=-1)
            positions = jnp.sum(midfix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # Forward pass with KV cache
            (_, state_out, _), _ = self.PaliGemma.llm(
                [None, suffix_tokens, None],  # Only compute state expert
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
            )

            # Predict velocity
            v_t = self.future_state_out_proj(state_out[:, -self.config.future_state_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        future_states, _ = jax.lax.while_loop(cond, step_state, (noise_state, 1.0))

        # === STAGE 2: Generate Actions (conditioned on predicted future states) ===

        # Initialize with noise
        noise_action = jax.random.normal(rng_action, (batch_size, self.action_horizon, self.action_dim))

        # Diffusion loop for actions
        def step_action(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix_action(
                observation, future_states, x_t, jnp.broadcast_to(time, batch_size)
            )

            # Build attention mask
            # TODO: check if the attention mask is correct
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            midfix_attn_mask_repeat = einops.repeat(midfix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([midfix_attn_mask_repeat, suffix_attn_mask], axis=-1)
            positions = jnp.sum(midfix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # Forward pass with KV cache
            (_, _, action_out), _ = self.PaliGemma.llm(
                [None, None, suffix_tokens],  # Only compute action expert
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
            )

            # Predict velocity
            v_t = self.action_out_proj(action_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        actions, _ = jax.lax.while_loop(cond, step_action, (noise_action, 1.0))

        return actions  # Final output
