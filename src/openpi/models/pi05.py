# openpi/models/pi05.py
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import gemma_unified as _gemma
from openpi.models import siglip as _siglip
from openpi.shared import array_typing as at

# openpi/models/_mask_and_posenc.py
import jax
import jax.numpy as jnp
import openpi.shared.array_typing as at

@at.typecheck
def make_attn_mask(input_mask: at.Bool[at.Array, "b n"], mask_ar: at.Int[at.Array, "b n"]) -> at.Bool[at.Array, "b n n"]:
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)       # [B,N]
    csum    = jnp.cumsum(mask_ar, axis=1)                       # [B,N]
    attn    = csum[:, None, :] <= csum[:, :, None]              # [B,N,N]
    valid   = input_mask[:, None, :] & input_mask[:, :, None]   # [B,N,N]
    return attn & valid

@at.typecheck
def posemb_sincos(
    pos: at.Float[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b d"]:
    if embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be even")
    frac = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** frac
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


PALIGEMMA_VOCAB_SIZE = _gemma.PALIGEMMA_VOCAB_SIZE  # 257_152

class Pi05(_model.BaseModel):
    """π0.5: Single Gemma (multi-expert) unified training:
       - Prefix (image + text): FAST / language token prediction (CE with token_loss_mask M_ℓ)
       - Suffix (action expert): Flow-Matching (MSE with optional action mask M_act)
       - Attention kernel: KI (Eq.(5)/(6)) already enabled in gemma.Attention
       - Time conditioning: AdaRMS used only on the action side (not concatenated with action tokens)
    """

    def __init__(self, config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        # Load two configs (1st = backbone / 2nd = action expert)
        paligemma_cfg = _gemma.get_config(config.paligemma_variant)
        action_cfg    = _gemma.get_config(config.action_expert_variant)

        # 1) Gemma (multi-expert shared attention; KI is implemented inside gemma.Attention)
        self.llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_cfg, action_cfg],
                embed_dtype=config.dtype,
                adarms=True,     # π0.5: enable AdaRMS only for the action side
            )
        )
        # Use AdaRMS conditioning only for the action side; backbone does not use it
        self.llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])

        # 2) SigLIP encodes images into patch tokens (width matches backbone width)
        self.img_enc = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_cfg.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        self.img_enc.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        # 3) Action-side input/output projections + time conditioning MLP (as AdaRMS cond)
        self.action_in_proj  = nnx.Linear(config.action_dim, action_cfg.width, rngs=rngs)  # AD -> D_act
        self.time_mlp_in     = nnx.Linear(action_cfg.width, action_cfg.width, rngs=rngs)
        self.time_mlp_out    = nnx.Linear(action_cfg.width, action_cfg.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_cfg.width, config.action_dim, rngs=rngs)

        # 4) Language LM head (by default not tied to the embedding weights;
        #    if weight tying is needed, see notes below)
        self.lm_head = nnx.Linear(paligemma_cfg.width, PALIGEMMA_VOCAB_SIZE, rngs=rngs)

        self.dtype = config.dtype
        self.deterministic = True

    # ----------------- Prefix: image + text (backbone expert) -----------------
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b p d"], at.Bool[at.Array, "b p"], at.Int[at.Array, "b p"], int]:
        tokens    = []
        in_masks  = []
        ar_masks  = []

        # a) Image → SigLIP → patch tokens
        for name, img in obs.images.items():
            img_tokens, _ = self.img_enc(img, train=False)        # [B, S_img, D_vlm]
            tokens.append(img_tokens)
            in_masks.append(einops.repeat(obs.image_masks[name], "b -> b s", s=img_tokens.shape[1]))
            ar_masks.append(jnp.zeros_like(in_masks[-1]))         # intra-image (bidirectional)

        # b) Text → Gemma.embed
        if obs.tokenized_prompt is not None:
            text_emb = self.llm(obs.tokenized_prompt, method="embed")    # [B, L, D_vlm]
            tokens.append(text_emb)
            in_masks.append(obs.tokenized_prompt_mask)                   # [B, L]
            ar_masks.append(obs.token_ar_mask.astype(jnp.int32))         # [B, L] (0=same block,1=new block)

        # Record text length (used to slice the text segment from prefix_out)
        L_txt = obs.tokenized_prompt.shape[1] if obs.tokenized_prompt is not None else 0

        return (
            jnp.concatenate(tokens, axis=1),      # prefix_tokens: [B, P, D_vlm]
            jnp.concatenate(in_masks, axis=1),    # prefix_mask:   [B, P]
            jnp.concatenate(ar_masks, axis=1),    # prefix_ar:     [B, P]
            L_txt,
        )

    # ----------------- Suffix: action expert (continuous Flow-Matching) -----------------
    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        x_t: at.Float[at.Array, "b ah ad"],
        t:   at.Float[at.Array, " b"],
    ) -> tuple[at.Float[at.Array, "b s d"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"], at.Float[at.Array, "b d"]]:
        B = x_t.shape[0]
        action_tokens = self.action_in_proj(x_t)                      # [B, AH, D_act]

        # Time → sin/cos → MLP → used as AdaRMS conditioning (not concatenated with actions)
        time_emb = posemb_sincos(t, action_tokens.shape[-1], min_period=4e-3, max_period=4.0)  # [B,D_act]
        time_emb = nnx.swish(self.time_mlp_in(time_emb))
        time_emb = nnx.swish(self.time_mlp_out(time_emb))
        adarms_cond = time_emb                                        # [B, D_act]

        # Suffix mask: first token starts a new block (blocking prefix from attending to suffix),
        # rest are in the same block (bidirectional inside the block)
        suffix_tokens = action_tokens                                 # [B, AH, D_act]
        suffix_mask   = jnp.ones((B, self.action_horizon), dtype=jnp.bool_)   # [B, AH]
        suffix_ar     = jnp.concatenate(
            [jnp.ones((B, 1), dtype=jnp.int32), jnp.zeros((B, self.action_horizon - 1), dtype=jnp.int32)], axis=1
        )  # [B, AH]

        return suffix_tokens, suffix_mask, suffix_ar, adarms_cond



    # ----------------- Training: joint loss (FAST NLL + FM MSE) -----------------
    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # Flow-Matching noise and time
        B = actions.shape[0]
        # 1) Sample noise/time and build x_t and u_t (same as before)
        noise = jax.random.normal(noise_rng, actions.shape)                     # [B, AH, AD]
        t     = jax.random.beta(time_rng, 1.5, 1, (B,)) * 0.999 + 0.001         # [B]
        x_t   = t[:,None,None] * noise + (1-t[:,None,None]) * actions           # [B, AH, AD]
        u_t   = noise - actions                                                 # [B, AH, AD]

        # 2) Prefix (image + text); record text length L_txt
        prefix_tokens, prefix_mask, prefix_ar, L_txt = self.embed_prefix(observation)

        # 3) Suffix (action expert)
        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(observation, x_t, t)

        # 4) Concatenate masks/positions and make a single LLM forward pass
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)        # [B, N]
        ar_mask    = jnp.concatenate([prefix_ar,  suffix_ar],  axis=1)          # [B, N]
        attn_mask  = make_attn_mask(input_mask, ar_mask)                        # [B, N, N]
        positions  = jnp.cumsum(input_mask, axis=1) - 1                         # [B, N]

        (prefix_out, suffix_out), _ = self.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        # 5) FAST / language NLL — use the text slice from prefix_out
        h_text = prefix_out[:, -L_txt:, :]                                      # [B, L, D_vlm]
        logits = self.lm_head(h_text[:, :-1, :])                                # [B, L-1, V]
        logp   = jax.nn.log_softmax(logits, axis=-1)
        
        raise NotImplementedError
        targets   = 
        loss_mask = 
        
        loss_fast = -(jnp.sum(targets * logp, axis=-1) * loss_mask).sum() / jnp.clip(loss_mask.sum(), 1)

        # 6) Flow-Matching — use the action slice from suffix_out
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:, :])     # [B, AH, AD]
        return jnp.mean((v_t - u_t) ** 2, axis=-1) + loss_fast
        # per = jnp.mean((v_t - u_t) ** 2, axis=-1)                               # [B, AH]
        # if getattr(observation, "action_mask", None) is not None:
        #     M_act = observation.action_mask                                     # [B]
        #     loss_fm = (per.mean(axis=-1) * M_act).sum() / jnp.clip(M_act.sum(), 1)
        # else:
        #     loss_fm = per.mean()

        # # 7) Total loss (paper Eq.(4); α≈1)
        # loss_total = loss_fast + loss_fm

        # # 8) If your trainer expects [B, AH], return per-step MSE plus the broadcasted mean NLL
        # return per + (loss_fast / self.action_horizon)[:, None]

    # ----------------- Inference: same as pi0 (Euler step integration) -----------------
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)

        dt = -1.0 / num_steps
        B = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (B, self.action_horizon, self.action_dim))

        # First, prefill KV cache with prefix (backbone expert only)
        prefix_tokens, prefix_mask, prefix_ar, _ = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar)               # [B,P,P]
        positions = jnp.cumsum(prefix_mask, axis=1) - 1                         # [B,P]
        _, kv_cache = self.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, B)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar)                   # [B,S,S]
            prefix_seen = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn = jnp.concatenate([prefix_seen, suffix_attn_mask], axis=-1)       # [B,S,P+S]
            pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.llm(
                [None, suffix_tokens],
                mask=full_attn,
                positions=pos,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:, :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    # ----------------- FAST / language NLL (computed only on text segment) -----------------
    @at.typecheck
    def token_nll(
        self,
        prefix_tokens: at.Float[at.Array, "b p d"],
        prefix_mask:   at.Bool[at.Array,  "b p"],
        prefix_ar:     at.Int[at.Array,   "b p"],
        obs: _model.Observation,
        L_txt: int,
    ) -> at.Float[at.Array, ""]:
        """Run one forward pass with prefix (image+text), slice out the text segment,
        and perform next-token prediction there."""
        attn_mask = make_attn_mask(prefix_mask, prefix_ar)                  # [B, P, P]
        positions = jnp.cumsum(prefix_mask, axis=1) - 1                     # [B, P]

        # Run backbone expert only: suffix = None
        (prefix_out, _), _ = self.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        # Slice out the "text segment": take last L_txt tokens
        h_text = prefix_out[:, -L_txt:, :]                                  # [B, L, D_vlm]

        # Predict "next token": inputs 0..L-2, targets 1..L-1
        logits = self.lm_head(h_text[:, :-1, :])                            # [B, L-1, V]
        logp   = jax.nn.log_softmax(logits, axis=-1)

        targets   = jax.nn.one_hot(obs.tokenized_prompt[:, 1:], PALIGEMMA_VOCAB_SIZE)  # [B, L-1, V]
        loss_mask = obs.token_loss_mask[:, 1:]                               # [B, L-1]
        nll = -jnp.sum(targets * logp, axis=-1)                              # [B, L-1]
        return (nll * loss_mask).sum() / jnp.clip(loss_mask.sum(), 1)

        # If you want "weight tying" instead of lm_head:
        # You can add a method='decode' inside gemma.Module to call embedder.decode,
        # or access self.llm.module.embedder parameters here
        # (requires correct access via NNX bridge).
        # Both are equivalent in functionality, training performance is usually similar.
