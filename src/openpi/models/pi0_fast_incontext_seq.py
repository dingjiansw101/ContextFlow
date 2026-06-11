import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
from openpi.models.pi0_fast_incontext import (
    PALIGEMMA_EOS_TOKEN,
    make_attn_mask,
    left_to_right_align,
    put_along_last_axis,
)
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class Pi0FASTIncontextSeqConfig(_model.BaseModelConfig):
    """FAST in-context config that keeps demo states/actions as continuous features."""

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"

    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 250
    state_dim: int = 32
    demo_action_dim: int | None = None
    demo_state_dim: int = 32

    siglip_variant: str = "So400m/14"
    pool_type: str = "none"
    freeze_img_encoder: bool = False

    sample_frames: int = 16
    sample_actions: int = 32
    random_select: bool = True
    avg_incontext_image_tokens: bool = False

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST_INCONTEXT

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FASTIncontextSeq":
        return Pi0FASTIncontextSeq(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.ObservationFASTIncontext, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.ObservationFASTIncontext(
                images={
                    "base_0_rgb": image_spec,
                    "base_1_rgb": image_spec,
                    "wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "base_1_rgb": image_mask_spec,
                    "wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.state_dim], jnp.float32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing


class Pi0FASTIncontextSeq(_model.BaseModel):
    def __init__(self, config: Pi0FASTIncontextSeqConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)

        self.use_image_prompts = config.sample_frames > 0
        self.use_action_state_prompts = True

        self.siglip_variant = config.siglip_variant
        self.pool_type = config.pool_type
        self.avg_incontext_image_tokens = config.avg_incontext_image_tokens

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant=self.siglip_variant,
                pool_type=self.pool_type,
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        self.demo_state_proj = nnx.Linear(config.demo_state_dim, paligemma_config.width, rngs=rngs)
        self.demo_action_proj = nnx.Linear(config.demo_action_dim, paligemma_config.width, rngs=rngs)

    def _project_sequence(
        self,
        values: jax.Array | None,
        mask: jax.Array | None,
        proj: nnx.Linear,
    ) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None]:
        if values is None:
            return None, None, None

        arr = jnp.asarray(values)
        if arr.ndim < 3:
            raise ValueError("In-context sequences must have at least 3 dimensions (B, T, D)")

        batch = arr.shape[0]
        feature_dim = arr.shape[-1]

        flat = arr.reshape(batch, -1, feature_dim)
        seq_len = flat.shape[1]
        projected = proj(flat.reshape(batch * seq_len, feature_dim)).reshape(batch, seq_len, -1)

        if mask is None:
            seq_mask = jnp.ones((batch, seq_len), dtype=jnp.bool_)
        else:
            mask_arr = jnp.asarray(mask)
            seq_mask = mask_arr.reshape(batch, seq_len)

        ar_mask = jnp.zeros(seq_mask.shape, dtype=jnp.int32)
        return projected, seq_mask, ar_mask

    @at.typecheck
    def embed_inputs(
        self, obs: _model.ObservationFASTIncontext
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        token_embeddings = []
        input_masks = []
        ar_masks = []

        if self.use_image_prompts and obs.incontext_images is not None:
            for name, image_sequence in obs.incontext_images.items():
                mask_sequence = None
                if obs.incontext_image_masks is not None and name in obs.incontext_image_masks:
                    mask_sequence = jnp.asarray(obs.incontext_image_masks[name], dtype=jnp.bool_)

                if image_sequence.ndim == 5:
                    batch_size, seq_len, height, width, channel = image_sequence.shape
                    flat_images = image_sequence.reshape(batch_size * seq_len, height, width, channel)
                    flat_tokens, _ = self.PaliGemma.img(flat_images, train=False)
                    token_len = flat_tokens.shape[1]

                    if mask_sequence is None:
                        frame_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
                    else:
                        frame_mask = mask_sequence.reshape(batch_size, seq_len)

                    if self.avg_incontext_image_tokens:
                        tokens = flat_tokens.reshape(batch_size, seq_len, token_len, flat_tokens.shape[-1])
                        tokens = jnp.mean(tokens, axis=2)
                        token_mask = frame_mask
                    else:
                        tokens = flat_tokens.reshape(batch_size, seq_len * token_len, flat_tokens.shape[-1])
                        token_mask = einops.repeat(frame_mask, "b t -> b (t tok)", tok=token_len)

                elif image_sequence.ndim == 6:
                    batch_size, episodes, seq_len, height, width, channel = image_sequence.shape
                    flat_images = image_sequence.reshape(batch_size * episodes * seq_len, height, width, channel)
                    flat_tokens, _ = self.PaliGemma.img(flat_images, train=False)
                    token_len = flat_tokens.shape[1]

                    if mask_sequence is None:
                        frame_mask = jnp.ones((batch_size, episodes * seq_len), dtype=jnp.bool_)
                    else:
                        frame_mask = mask_sequence.reshape(batch_size, episodes * seq_len)

                    if self.avg_incontext_image_tokens:
                        tokens = flat_tokens.reshape(batch_size, episodes * seq_len, token_len, flat_tokens.shape[-1])
                        tokens = jnp.mean(tokens, axis=2)
                        token_mask = frame_mask
                    else:
                        tokens = flat_tokens.reshape(batch_size, episodes * seq_len * token_len, flat_tokens.shape[-1])
                        token_mask = einops.repeat(frame_mask, "b m -> b (m tok)", tok=token_len)

                else:
                    raise ValueError(f"incontext image tensor '{name}' must be 5-D or 6-D, got {image_sequence.ndim}-D")

                token_embeddings.append(tokens)
                input_masks.append(token_mask)
                ar_masks.append(jnp.zeros(token_mask.shape, dtype=jnp.int32))

        # Continuous in-context states/actions
        state_tokens, state_mask, state_ar = self._project_sequence(
            obs.incontext_states,
            obs.incontext_state_masks,
            self.demo_state_proj,
        )
        if state_tokens is not None:
            token_embeddings.append(state_tokens)
            input_masks.append(state_mask)
            ar_masks.append(state_ar)

        action_tokens, action_mask, action_ar = self._project_sequence(
            obs.incontext_actions,
            obs.incontext_action_masks,
            self.demo_action_proj,
        )
        if action_tokens is not None:
            token_embeddings.append(action_tokens)
            input_masks.append(action_mask)
            ar_masks.append(action_ar)

        for name, image in obs.images.items():
            image_token_embeddings, _ = self.PaliGemma.img(image, train=False)
            token_embeddings.append(image_token_embeddings)
            image_mask = einops.repeat(
                obs.image_masks[name],
                "b -> b s",
                s=image_token_embeddings.shape[1],
            )
            input_masks.append(image_mask)
            ar_masks.append(jnp.zeros(image_mask.shape, dtype=jnp.int32))

        if obs.tokenized_prompt is None or obs.tokenized_prompt_mask is None or obs.token_ar_mask is None:
            raise ValueError("Tokenized prompt information is required for FAST models.")

        prompt_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(prompt_embeddings)
        input_masks.append(obs.tokenized_prompt_mask)
        ar_masks.append(obs.token_ar_mask)

        embeddings = jnp.concatenate(token_embeddings, axis=1)
        masks = jnp.concatenate([jnp.asarray(mask, dtype=jnp.bool_) for mask in input_masks], axis=1)
        ar = jnp.concatenate([jnp.asarray(mask, dtype=jnp.int32) for mask in ar_masks], axis=1)

        return embeddings, masks, ar

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.ObservationFASTIncontext,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        observation = _model.preprocess_observation_incontext_fast(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
        attn_mask = make_attn_mask(input_mask, ar_mask)

        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.PaliGemma.llm.module.vocab_size,
        )
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )

        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1] :],
        )
        logp = jax.nn.log_softmax(logits, axis=-1)

        loss_mask = observation.token_loss_mask[:, 1:]
        token_pplx = jnp.sum(targets * logp, axis=-1)
        return -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.ObservationFASTIncontext,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:
        observation = _model.preprocess_observation_incontext_fast(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
        )

        last_logit = prefix_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

        def step(carry):
            last_logit, output_tokens, cache, _, step = carry
            if temperature > 0.0:
                last_logit = last_logit / temperature
                token = jax.random.categorical(rng, last_logit, axis=-1)
            else:
                token = jnp.argmax(last_logit, axis=-1)
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
            )

            return last_logit, output_tokens, cache, all_eos, step + 1

        def cond(carry):
            _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        _, output_tokens, _, _, _ = jax.lax.while_loop(cond, step, (last_logit, output_tokens, kv_cache, False, 0))
        return output_tokens
