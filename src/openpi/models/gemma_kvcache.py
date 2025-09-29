# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

from functools import partial

PALIGEMMA_VOCAB_SIZE = 257_152

KVCache: TypeAlias = tuple[
    at.Float[at.Array, "l b _t _k _h"],  # K: [L, B, T, K, H]
    at.Float[at.Array, "l b _t _v _h"],  # V: [L, B, T, V, H]  # K==V heads
]


### XJ: for online kv caching type
@dataclasses.dataclass
class PMCache:
    # (K, V) stacked per-layer as in KVCache; kept identical to simplify passing into self.layers
    kv: KVCache  # tuple[k, v] with k: [L,B,Tpm,K,H], v: [L,B,Tpm,K,H]



@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)
    # added
    expert_name: str | None = None     


Variant = Literal["dummy", "gemma_300m", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant, expert_name: str | None = None) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
            expert_name=expert_name,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            expert_name=expert_name,
        )
    if variant == "gemma_300m_v2":
        # 311M params
        return Config(
            width=2048,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            expert_name=expert_name,
        )
    
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            expert_name=expert_name,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
            expert_name=expert_name,
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
            expert_name=expert_name,
        )
    # XJ: customizer gemma mini models
    if variant == "gemma_132m":
        # 132M params (without embedder)
        return Config(
            width=2048,
            depth=6,
            mlp_dim=2048,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            expert_name=expert_name,
        )    
    if variant == "gemma_66m":
        # 66M params
        return Config(
            width=1024,
            depth=6,
            mlp_dim=2048,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            expert_name=expert_name,
        )   
    # XJ: customizer gemma mini models with smaller embedder
    if variant == "gemma_A":
        # 132M params (without embedder)
        return Config(
            width=512,
            depth=6,
            mlp_dim=768,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            expert_name=expert_name,
        )    
    if variant == "gemma_B":
        # 66M params
        return Config(
            width=512,
            depth=6,
            mlp_dim=1024,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            expert_name=expert_name,
        )       
    
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        normed_inputs = normed_inputs * (
            1 + scale
        )  # scale by learned parameter in float32 (matches Flax implementation)
        return normed_inputs.astype(dtype)  # return in original dtype


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            # nn.initializers.normal(),
            nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]
    allow_bn_broadcast: bool = True


    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
        assert (self.configs[0].num_heads % self.configs[0].num_kv_heads) == 0, \
            "num_heads must be divisible by num_kv_heads."

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        if self.configs[0].expert_name is not None:
            _name = partial(_namev2, expert_names=[config.expert_name for config in self.configs])
        else:
            _name = _namev2
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )                    
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype


        ### XJ: KV-cache support in batch
        '''
        当当前批次的 q/k/v 的 batch 大小是 BxN,
        而 cache 的 batch 是 B 时，
        用 lax.broadcast_in_dim 把 cache 在“隐含的 N 维”上广播，
        再视图式 reshape成 [B*N, ...] 与当前 k,v 对齐。
        '''
        if kv_cache is not None:
            cache_k, cache_v = kv_cache  # [Bc, Tpm, K, H]
            Bq = q.shape[0]
            Bc = cache_k.shape[0]
            if Bq != Bc:
                if (Bq % Bc) != 0:
                    raise ValueError(f"kv_cache batch {Bc} not dividing current batch {Bq}.")
                if not self.allow_bn_broadcast:
                    raise ValueError("Batch mismatch but broadcasting disabled.")
                N = Bq // Bc
                cache_k = jax.lax.broadcast_in_dim(
                    cache_k,  # [B, Tpm, K, H]
                    shape=(Bc, N, cache_k.shape[1], cache_k.shape[2], cache_k.shape[3]),
                    broadcast_dimensions=(0, 2, 3, 4),
                ).reshape(Bq, cache_k.shape[1], cache_k.shape[2], cache_k.shape[3])
                cache_v = jax.lax.broadcast_in_dim(
                    cache_v,
                    shape=(Bc, N, cache_v.shape[1], cache_v.shape[2], cache_v.shape[3]),
                    broadcast_dimensions=(0, 2, 3, 4),
                ).reshape(Bq, cache_v.shape[1], cache_v.shape[2], cache_v.shape[3])
            # 把 midfix 的 KV 接到前面
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

            

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
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
        assert outputs.dtype == dtype
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: Sequence[Config]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, deterministic=True):  # noqa: FBT002
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
                x = RMSNorm(name=_name("pre_attention_norm", i))(x)  # noqa: PLW2901
            pre_attn.append(x)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = jax.tree.map(lambda x, y: x + y, xs, post_attn)
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x = RMSNorm(name=_name("pre_ffw_norm", i))(x)  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
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




@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str
    voc_size: int = PALIGEMMA_VOCAB_SIZE

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=self.voc_size,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            ### XJ: delete decode in Block.__call__, so 4 is deterministic
            static_argnums=(4,),  # 0=self, 4=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            ### XJ: self.layers 的前 4 个位置参数依次是：embedded, kv_cache, positions, mask
            # embedded 设成了 0（意味着它自带一个“层维”要扫描），但 embedded 实际并没有层维，这里应该 broadcast
            in_axes=(nn.broadcast, 0, nn.broadcast, nn.broadcast),
            # in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast),  # 0=kv_cache, 1=positions, 2=mask, 3=decode
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        
        if self.configs[0].expert_name is not None:
            _name = partial(_namev2, expert_names=[config.expert_name for config in self.configs])
        else:
            _name = _namev2

        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree_map(lambda e: e.astype(self.embed_dtype) if e is not None else None, embedded)
        mask = jnp.asarray(mask)[:, None, :, :]

        # ★ NEW：scan 的第二个参数我们设了 in_axes=0，因此需要有“层维”。
        L = self.configs[0].depth
        kv_arg = kv_cache if kv_cache is not None else [None] * L

        embedded, kv_cache = self.layers(embedded, kv_arg, positions, mask, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)
        return [f(e) if e is not None else e for f, e in zip(self.final_norms, embedded, strict=True)], kv_cache

    def init(self):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
        )
    
    ### XJ: midfix KV cache per layer
    '''
    这两个方法只是把你已有的 self.layers 包了一层，完全复用现有 scan 堆叠出的逐层 K/V 返回机制；
    decode_with_cache 的关键在于：传入的 pm_cache 是 [L,B,...]，而 embedded_suf/mask_suf 用的是 [B*N, ...];
    在 Attention 内部会自动把 cache 从 B 广播到 B*N(Patch 1)
    '''  
    # ------------ NEW: helper to encode prefix/midfix and collect per-layer K/V ------------
    @at.typecheck
    def encode_pm_only(
        self,
        embedded_pm: Sequence[at.Float[at.Array, "b t d"] | None],
        positions_pm: at.Int[at.Array, "b t"],
        mask_pm: at.Bool[at.Array, "b t s"],
        *,
        deterministic: bool = True,
    ) -> PMCache:
        """
        Run only the prefix/midfix tokens to build per-layer KV cache.
        Returns PMCache with shapes:
          K: [L, B, Tpm, num_kv_heads, head_dim]
          V: [L, B, Tpm, num_kv_heads, head_dim]
        """
        embedded_pm = jax.tree.map(lambda e: e.astype(self.embed_dtype) if e is not None else None, embedded_pm)
        mask_pm_b = jnp.asarray(mask_pm)[:, None, :, :]  # [B,1,T,S] as expected by Attention
        # Pass kv_cache=None; scan will return stacked per-layer (k,v)
        L = self.configs[0].depth
        _, kv_cache = self.layers(embedded_pm, [None] * L, positions_pm, mask_pm_b, deterministic)
        ### XJ: for clarification: using prefix-midfix KV cahce
        return PMCache(kv_cache)
    
    # ------------ NEW: helper to decode suffix using a given per-layer KV cache ------------
    @at.typecheck
    def decode_with_cache(
        self,
        embedded_suf: Sequence[at.Float[at.Array, "b t d"] | None],  # b 实际是 BN
        positions_suf: at.Int[at.Array, "b t"],
        mask_suf: at.Bool[at.Array, "b t s"],  # s = Tpm+Ts
        *,
        pm_cache: PMCache,
        deterministic: bool = True,
    ) -> Sequence[at.Float[at.Array, "b t d"] | None]:
        """
        Run suffix tokens (batch flattened to B*N) while reusing prefix/midfix KV cache (batch=B).
        Thanks to Patch 1, the cache will be auto-broadcast from B -> (B*N) in-graph with no physical repeat.
        """
        embedded_suf = jax.tree.map(lambda e: e.astype(self.embed_dtype) if e is not None else None, embedded_suf)
        mask_suf_b = jnp.asarray(mask_suf)[:, None, :, :]  # [BN,1,Ts,Tpm+Ts]
        outputs, _ = self.layers(embedded_suf, pm_cache.kv, positions_suf, mask_suf_b, deterministic)
        # final_norms are already applied inside __call__; we mimic that behavior here for parity
        assert all(e is None or e.dtype == jnp.dtype(self.embed_dtype) for e in outputs)
        return [f(e) if e is not None else e for f, e in zip(self.final_norms, outputs, strict=True)]



def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"

def _namev2(name, i, expert_names=["paligemma", "action_expert"]):
    # we want the names of paligemma and action experts to be the same as pre-trained checkpoint names, so that they can be loaded
    assert expert_names[i] in ["paligemma", "action_expert", "prompt_expert"]
    if expert_names[i] == "paligemma":
        return name
    elif expert_names[i] == "action_expert":
        return f"{name}_{1}"
    return f"{name}_{expert_names[i]}"