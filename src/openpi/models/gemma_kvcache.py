# Copyright 2024 Big Vision Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias
from functools import partial

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding


PALIGEMMA_VOCAB_SIZE = 257_152

# Top-level KVCache (before scan): [L,B,T,K,H]
KVCache: TypeAlias = tuple[
    at.Float[at.Array, "l b _t _k _h"],  # K: [L, B, T, K, H]
    at.Float[at.Array, "l b _t _v _h"],  # V: [L, B, T, V, H]
]

# -------- Debug helpers --------
def _host_assert(pred_scalar: jax.Array, msg: str, *debug_vals: jax.Array):
    """Convert a JAX 0-d bool to a host-side assertion. Prints debug values robustly and remains replay-safe."""
    import numpy as np

    def _cb(p, *vals):
        if bool(p):
            return
        try:
            vals_fmt = [np.asarray(v).tolist() for v in vals]
            # Prefer formatting placeholders first
            if "{}" in msg:
                text = msg.format(*vals_fmt)
            else:
                text = msg
        except Exception:
            # Provide readable debug info even if vals are dropped
            try:
                vals_fmt = [np.asarray(v).tolist() for v in vals]
            except Exception:
                vals_fmt = ["<unavailable>"]
            text = f"{msg} | debug={vals_fmt}"
        # Append debug values even when formatting succeeds to aid replay diagnostics
        try:
            extra = [np.asarray(v).tolist() for v in vals]
            if extra:
                text = f"{text} | debug={extra}"
        except Exception:
            pass
        raise AssertionError(text)

    jax.debug.callback(_cb, pred_scalar, *debug_vals)


def _assert_block_ar_mask(ar_mask: jax.Array, H: int, _name: str):
    """
    Convention: the last H tokens mark the action block with pattern:
      [True, False, False, ..., False]  # length H
    Other positions are unrestricted.
    """
    m = jnp.asarray(ar_mask, dtype=bool).reshape(-1)     # [Ts]
    Ts = m.shape[0]
    if Ts < H:
        raise AssertionError(f"{_name}: len(ar_mask)={Ts} < action_horizon={H}")
    tail = m[-H:]                                        # [H]
    cond = jnp.logical_and(tail[0], jnp.all(jnp.logical_not(tail[1:])))
    _host_assert(cond, "tail pattern invalid: {}", tail.astype(jnp.int32))

# -------- Online pm-cache type --------
@dataclasses.dataclass
class PMCache:
    # Stack (K, V) per layer (same as the top-level KVCache layout)
    kv: KVCache  # tuple(k, v) with k: [L,B,Tpm,K,H], v: [L,B,Tpm,K,H]

# -------- Config --------
@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)
    expert_name: str | None = None

Variant = Literal[
    "dummy",
    "gemma_300m",
    "gemma_2b",
    "gemma_2b_lora",
    "gemma_A",
    "gemma_B",
    "gemma_132m",
    "gemma_66m",
    "gemma_43m",
    "gemma_52m",
    "gemma_85m_d8",
    "gemma_10m_d8",
    "gemma_300m_v2",
    "gemma_300m_lora",
]

def get_config(variant: Variant, expert_name: str | None = None) -> Config:
    if variant == "dummy":
        return Config(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16, expert_name=expert_name)
    if variant == "gemma_300m":
        return Config(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256, expert_name=expert_name)
    if variant == "gemma_300m_v2":
        return Config(width=2048, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256, expert_name=expert_name)
    if variant == "gemma_2b":
        return Config(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256, expert_name=expert_name)
    if variant == "gemma_2b_lora":
        return Config(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256,
                      lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0),
                                    "ffn":  lora.LoRAConfig(rank=16, alpha=16.0)},
                      expert_name=expert_name)
    if variant == "gemma_300m_lora":
        return Config(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256,
                      lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0),
                                    "ffn":  lora.LoRAConfig(rank=32, alpha=32.0)},
                      expert_name=expert_name)
    if variant == "gemma_132m":
        return Config(width=2048, depth=6, mlp_dim=2048, num_heads=8, num_kv_heads=1, head_dim=256, expert_name=expert_name)
    if variant == "gemma_66m":
        return Config(width=1024, depth=6, mlp_dim=2048, num_heads=8, num_kv_heads=1, head_dim=256, expert_name=expert_name)
    if variant == "gemma_43m":
        # ~43.26M
        return Config(
            width=640, depth=8, mlp_dim=2048,
            num_heads=8, num_kv_heads=1, head_dim=128,
            expert_name=expert_name
        )
    if variant == "gemma_52m":
        # ~51.92M
        return Config(
            width=704, depth=8, mlp_dim=2304,
            num_heads=8, num_kv_heads=1, head_dim=128,
            expert_name=expert_name
        )

    if variant == "gemma_85m_d8":
        return Config(
            width=768,
            depth=8,
            mlp_dim=3840,
            num_heads=8,
            num_kv_heads=1,
            head_dim=128,
            expert_name=expert_name
        )

    if variant == "gemma_10m_d8":
        return Config(
          width=320,
          depth=8,
          mlp_dim=512,
          num_heads=8,
          num_kv_heads=1,
          head_dim=128,
          expert_name=expert_name
        )
    
    if variant == "gemma_A":
        return Config(width=512, depth=6, mlp_dim=768, num_heads=8, num_kv_heads=1, head_dim=256, expert_name=expert_name)
    if variant == "gemma_B":
        return Config(width=512, depth=6, mlp_dim=1024, num_heads=8, num_kv_heads=1, head_dim=256, expert_name=expert_name)
    raise ValueError(f"Unknown variant: {variant}")

# -------- RMSNorm / Embedder --------
@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs.astype(dtype)

@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""
    vocab_size: int
    embed_dim: int
    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"),
            (self.vocab_size, self.embed_dim),
        )
    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x
    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)

# -------- Attention --------
@at.typecheck
class Attention(nn.Module):
    """Attention module."""
    configs: Sequence[Config]
    allow_bn_broadcast: bool = True

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache, ep_index=None):
        # Ensure model config dimensions align (lightweight check worth keeping)
        assert all(cfg.head_dim     == self.configs[0].head_dim     for cfg in self.configs)
        assert all(cfg.num_heads    == self.configs[0].num_heads    for cfg in self.configs)
        assert all(cfg.num_kv_heads == self.configs[0].num_kv_heads for cfg in self.configs)
        assert (self.configs[0].num_heads % self.configs[0].num_kv_heads) == 0, \
            "num_heads must be divisible by num_kv_heads."

        # === 1) Basic dtype/shape validation ===
        assert isinstance(xs, (list, tuple)) and any(x is not None for x in xs), "xs must contain at least one non-None branch"
        x0 = next(x for x in xs if x is not None)
        Bq = int(x0.shape[0])
        dtype = x0.dtype

        # === 2) Per-expert projections to gather q/k/v ===
        qkvs = []
        _nm = partial(_namev2, expert_names=[cfg.expert_name for cfg in self.configs]) \
            if self.configs[0].expert_name is not None else _namev2

        for i, (x, cfg) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if cfg.num_kv_heads == cfg.num_heads:
                qkv_e = lora.Einsum(
                    shape=(3, cfg.num_heads, cfg.width, cfg.head_dim),
                    name=_nm("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=cfg.lora_configs.get("attn"),
                )
                q_e, k_e, v_e = qkv_e("BSD,3KDH->3BSKH", x)  # [3,B,Ti,K,H]
            else:
                q_einsum = lora.Einsum(
                    shape=(cfg.num_heads, cfg.width, cfg.head_dim),
                    name=_nm("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=cfg.lora_configs.get("attn"),
                )
                q_e = q_einsum("BTD,NDH->BTNH", x)  # [B,Ti,N,H]
                kv_einsum = lora.Einsum(
                    shape=(2, cfg.num_kv_heads, cfg.width, cfg.head_dim),
                    name=_nm("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=cfg.lora_configs.get("attn"),
                )
                k_e, v_e = kv_einsum("BSD,2KDH->2BSKH", x)  # [B,Ti,K,H]
            qkvs.append((q_e, k_e, v_e))

        # Concatenate along the time axis
        q = jnp.concatenate([q for q, _, _ in qkvs], axis=1)
        k_cur = jnp.concatenate([k for _, k, _ in qkvs], axis=1)
        v_cur = jnp.concatenate([v for _, _, v in qkvs], axis=1)
        Tq = int(q.shape[1])
        Tcur = int(k_cur.shape[1])

        # Verify positions against Tq before applying RoPE
        # RoPE + scaling (cached K/V already had RoPE applied during encode; only apply to current tokens here)
        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5
        k_cur = _apply_rope(k_cur, positions=positions)

        # === 4) Broadcast KV cache across batch and concatenate time (per-layer view inside Attention: cache is [B,Tpm,K,H]) ===
        cache_k, cache_v = kv_cache  # [Bcache, Tpm, K, H]

        B = int(q.shape[0])                  # Use B consistently for the current query batch size
        Bc, Tpm = int(cache_k.shape[0]), int(cache_k.shape[1])
        if ep_index is not None:
            # With ep_index: each query row points to an episode (0..Bcache-1) so BN shuffling stays aligned.
            ep = jnp.asarray(ep_index)
            # Gather per-sample cache entries according to ep_index (regular gather, not uniform repeat).
            cache_k = cache_k[ep, :, :, :]    # [B, Tpm, K, H]
            cache_v = cache_v[ep, :, :, :]
        else:
            # Legacy path: if B and Bc differ but divide evenly, do uniform repeat (used when BN expands in contiguous blocks).
            if B != Bc:
                if (B % Bc) != 0:
                    raise ValueError(f"kv_cache batch {Bc} not dividing current batch {B}. "
                                     "If samples in the batch come from different episodes, pass ep_index.")
                if not self.allow_bn_broadcast:
                    raise ValueError("Batch mismatch but broadcasting disabled.")
                N = B // Bc
                cache_k = jax.lax.broadcast_in_dim(
                    cache_k, shape=(Bc, N, Tpm, cache_k.shape[2], cache_k.shape[3]),
                    broadcast_dimensions=(0, 2, 3, 4),
                ).reshape(B, Tpm, cache_k.shape[2], cache_k.shape[3])
                cache_v = jax.lax.broadcast_in_dim(
                    cache_v, shape=(Bc, N, Tpm, cache_v.shape[2], cache_v.shape[3]),
                    broadcast_dimensions=(0, 2, 3, 4),
                ).reshape(B, Tpm, cache_v.shape[2], cache_v.shape[3])

        # Align K/H dimensions
        # Concatenate cached tokens first, then current tokens
        k = jnp.concatenate([cache_k, k_cur], axis=1)  # [Bq, Tpm+Tcur, K, H]
        v = jnp.concatenate([cache_v, v_cur], axis=1)
        # === 5) Attention computation plus mask validation ===
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        big_neg = -2.3819763e38  # Match the original Gemma behavior
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        # === 6) Write back each expert's time slice
        out = []
        start = 0
        for i, (x, cfg) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                Ti = int(x.shape[1])
                end = start + Ti
                out_e = lora.Einsum(
                    shape=(cfg.num_heads, cfg.head_dim, cfg.width),
                    name=_nm("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=cfg.lora_configs.get("attn"),
                )("BTNH,NHD->BTD", encoded[:, start:end])
                out.append(out_e)
                start = end
            else:
                out.append(None)

        return out, (k, v)

# -------- FeedForward / Block --------
@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""
    features: int
    hidden_dim: int
    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)
        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1
        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        return outputs

@at.typecheck
class Block(nn.Module):
    """Transformer block."""
    configs: Sequence[Config]
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, ep_index=None, deterministic=True):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        if self.configs[0].expert_name is not None:
            _name = partial(_namev2, expert_names=[config.expert_name for config in self.configs])
        else:
            _name = _namev2

        pre_attn = []
        for i, x in enumerate(xs):
            if x is not None:
                x = RMSNorm(name=_name("pre_attention_norm", i))(x)
            pre_attn.append(x)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache, ep_index=ep_index)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = jax.tree.map(lambda x, y: x + y, xs, post_attn)
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x = RMSNorm(name=_name("pre_ffw_norm", i))(x)
                x = lora.FeedForward(
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = jax.tree.map(lambda x, y: x + y, xs, out)
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache

# -------- Module --------
@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""
    configs: Sequence[Config]
    embed_dtype: str
    voc_size: int = PALIGEMMA_VOCAB_SIZE

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    use_text_prompts: bool = True

    def _zero_kv(self, B: int, dtype) -> KVCache:
        L = self.configs[0].depth
        K = self.configs[0].num_kv_heads
        H = self.configs[0].head_dim
        k0 = jnp.zeros((L, B, 0, K, H), dtype)
        v0 = jnp.zeros((L, B, 0, K, H), dtype)
        return (k0, v0)

    @staticmethod
    def _assert_monotonic_positions(pos: jax.Array, mask: jax.Array, name: str):
        """Check positions == cumsum(valid)-1 on valid tokens. mask: [B,T]/[B,T,S]/[B,1,T,S]."""
        if mask.ndim == 4:
            _host_assert(jnp.array(mask.shape[1] == 1), f"{name}: 4D mask second axis must be 1, got {{}}", jnp.array(mask.shape[1]))
            mask = jnp.squeeze(mask, axis=1)
        if mask.ndim == 3:
            token_mask = jnp.any(mask, axis=-1)        # [B,T,S] -> [B,T]
        elif mask.ndim == 2:
            token_mask = mask                           # [B,T]
        else:
            raise ValueError(f"{name}: mask rank must be 2/3/4, got {mask.shape}")

        _host_assert(jnp.array(pos.ndim == 2), f"{name}: positions must be 2D, ndim={{}}", jnp.array(pos.ndim))
        _host_assert(
            jnp.array((pos.shape[0] == token_mask.shape[0]) & (pos.shape[1] == token_mask.shape[1])),
            f"{name}: positions shape must match token_mask (pos: [{{}}, {{}}], mask: [{{}}, {{}}])",
            jnp.array(pos.shape[0]), jnp.array(pos.shape[1]),
            jnp.array(token_mask.shape[0]), jnp.array(token_mask.shape[1]),
        )
        ref = jnp.cumsum(token_mask, axis=-1) - 1
        ok = jnp.all(jnp.where(token_mask, pos == ref, True))
        _host_assert(ok, f"{name} is not the mask-based monotonic cumulative position (should equal cumsum(valid)-1)", jnp.array(1))

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        if self.use_text_prompts:
            self.embedder = Embedder(vocab_size=self.voc_size, embed_dim=self.configs[0].width, name="embedder")

        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(6,),  # 0=self, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        # Note: in_axes applies to arguments after the carry: (kv_cache, positions, mask, ep_index, deterministic)
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            # <<<<<< forwarded to Block
        )

        if self.configs[0].expert_name is not None:
            _name = partial(_namev2, expert_names=[config.expert_name for config in self.configs])
        else:
            _name = _namev2
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        if self.use_text_prompts:
            return self.embedder.encode(tokens).astype(self.embed_dtype)
        else:
            raise ValueError(f"use_text_prompts is set to False but gemma.Module.embed is called")

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        *,
        kv_cache: KVCache | None = None,
        ep_index: at.Int[at.Array,"b"] | None = None, 
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype) if e is not None else None, embedded)
        mask_3d = jnp.asarray(mask)  # [B,T,S]

        # Extract batch size and dtype
        e_list = [e for e in embedded if e is not None]
        _host_assert(jnp.array(len(e_list) > 0), "embedded must include at least one non-None branch", jnp.array(len(e_list)))
        B = int(e_list[0].shape[0])
        dt = jnp.dtype(self.embed_dtype)

        kv_arg = kv_cache if kv_cache is not None else self._zero_kv(B, dt)

        # Expand to [B,1,T,S] before entering scan/Attention
        mask_4d = mask_3d[:, None, :, :]

        embedded, kv_out = self.layers(embedded, kv_arg, positions, mask_4d, ep_index, deterministic)
        out = [f(e) if e is not None else e for f, e in zip(self.final_norms, embedded, strict=True)]

        return out, kv_out

    def init(self):
        if self.use_text_prompts:
            self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
        )

    # ------------ Encode prefix/midfix and collect per-layer KV ------------
    @at.typecheck
    def encode_pm_only(
        self,
        embedded_pm: Sequence[at.Float[at.Array, "b t d"] | None],
        positions_pm: at.Int[at.Array, "b t"],
        mask_pm: at.Bool[at.Array, "b t s"],
        *,
        deterministic: bool = True,
    ) -> PMCache:
        embedded_pm = jax.tree.map(lambda e: e.astype(self.embed_dtype) if e is not None else None, embedded_pm)
        mask_pm_b = jnp.asarray(mask_pm)[:, None, :, :]  # [B,1,T,S]
        e0 = next(e for e in embedded_pm if e is not None)
        B = int(e0.shape[0])
        dt = jnp.dtype(self.embed_dtype)

        _, kv_cache = self.layers(embedded_pm, self._zero_kv(B, dt), positions_pm, mask_pm_b, None, deterministic)

        return PMCache(kv_cache)

    # ------------ Decode suffix using pm-cache ------------
    @at.typecheck
    def decode_with_cache(
        self,
        embedded_suf: Sequence[at.Float[at.Array, "b _t _d"] | None],  # b is actually the BN dimension
        positions_suf: at.Int[at.Array, "b t"],
        mask_suf: at.Bool[at.Array, "b t s"],  # s = Tpm+Ts
        *,
        pm_cache: PMCache,
        ep_index: at.Int[at.Array,"b"] | None = None, 
        deterministic: bool = True,
    ) -> Sequence[at.Float[at.Array, "b _t _d"] | None]:
        embedded_suf = jax.tree.map(lambda e: e.astype(self.embed_dtype) if e is not None else None, embedded_suf)
        mask_suf_b = jnp.asarray(mask_suf)[:, None, :, :]  # [BN,1,Ts,Tpm+Ts]

        L = self.configs[0].depth
        outputs, _ = self.layers(embedded_suf, pm_cache.kv, positions_suf, mask_suf_b, ep_index, deterministic)
        outs = [f(e) if e is not None else e for f, e in zip(self.final_norms, outputs, strict=True)]

        return outs

# -------- Utils --------
def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return res.astype(x.dtype)

def _name(name, i):
    if i == 0:
        return name
    return f"{name}_{i}"

def _namev2(name, i, expert_names=["paligemma", "action_expert"]):
    assert expert_names[i] in ["paligemma", "action_expert", "prompt_expert"]
    if expert_names[i] == "paligemma":
        return name
    elif expert_names[i] == "action_expert":
        return f"{name}_{1}"
    return f"{name}_{expert_names[i]}"
