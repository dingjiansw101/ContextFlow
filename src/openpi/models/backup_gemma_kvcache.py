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

def _host_assert(pred_scalar: jax.Array, msg: str, *debug_vals):
    """原始 host 断言（会走 jax.debug.callback）"""
    import numpy as np
    def _cb(p, *vals):
        if not bool(p):
            vals_np = tuple(np.array(v) for v in vals)
            raise AssertionError(msg.format(*vals_np))
    jax.debug.callback(_cb, pred_scalar, *debug_vals)

def _dbg_assert(enable: bool, pred_scalar: jax.Array, msg: str, *debug_vals):
    """仅在 enable=True 时才触发 host 断言；否则完全 no-op。"""
    if enable:
        _host_assert(pred_scalar, msg, *debug_vals)
  
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
    debug_checks: bool = False

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # --- 头部一致性（模型配置维） ---
        assert all(cfg.head_dim     == self.configs[0].head_dim     for cfg in self.configs)
        assert all(cfg.num_heads    == self.configs[0].num_heads    for cfg in self.configs)
        assert all(cfg.num_kv_heads == self.configs[0].num_kv_heads for cfg in self.configs)
        assert (self.configs[0].num_heads % self.configs[0].num_kv_heads) == 0, \
            "num_heads must be divisible by num_kv_heads."

        if self.debug_checks:
            # === 1) 基础 dtype/shape 校验（允许各 expert 的 T 不同；Tq=∑T_i） ===
            assert isinstance(xs, (list, tuple)) and any(x is not None for x in xs), "xs 至少应有一个非 None 分支"
            assert positions.ndim == 2, f"positions 期望 [B,T]，got {positions.shape}"
            assert attn_mask.ndim == 4, f"attn_mask 期望 [B,1,T,S]，got {attn_mask.shape}"

        # batch 与总时间长度
        x0 = next(x for x in xs if x is not None)
        Bq = int(x0.shape[0])
        if self.debug_checks:
            assert all((x is None) or (int(x.shape[0]) == Bq) for x in xs), "多 expert 的 batch 不一致"
        t_list = [int(x.shape[1]) for x in xs if x is not None]
        if self.debug_checks:
            assert len(t_list) > 0
        Tq = int(sum(t_list))  # ★ 拼接后的总 query 长度

        dtype = x0.dtype  # half-precision 也保持

        # === 2) 逐 expert 线性映射，收集 q/k/v，然后在时间维拼接 ===
        qkvs = []
        _nm = partial(_namev2, expert_names=[cfg.expert_name for cfg in self.configs]) \
            if self.configs[0].expert_name is not None else _namev2

        for i, (x, cfg) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if cfg.num_kv_heads == cfg.num_heads:
                # K==N 情况：一次性投出 q/k/v
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

        # 按时间维拼接
        q = jnp.concatenate([q for q, _, _ in qkvs], axis=1)  # [B,Tq, N or K, H]
        k_cur = jnp.concatenate([k for _, k, _ in qkvs], axis=1)  # [B,Tcur,K,H]
        v_cur = jnp.concatenate([v for _, _, v in qkvs], axis=1)  # [B,Tcur,K,H]
        Tcur = int(k_cur.shape[1])

        if self.debug_checks:
            # Rope 之前核对 positions 与 Tq
            assert positions.shape == (Bq, Tq), f"positions 与 q 形状不匹配：pos={positions.shape}, Tq={Tq}"

        # === 3) RoPE & 缩放（cache 中的 K/V 已在 encode 阶段做过 RoPE，这里只对当前 tokens） ===
        q = _apply_rope(q, positions=positions); q *= self.configs[0].head_dim ** -0.5
        k_cur = _apply_rope(k_cur, positions=positions)
        
        if self.debug_checks:
            assert q.dtype == k_cur.dtype == v_cur.dtype == dtype

        # === 4) KV-cache 批广播 + 时间拼接（先拿 Tpm，再拼接） ===
        cache_k, cache_v = kv_cache  # 期望 [Bcache, Tpm, K, H]；Tpm 可能为 0
        if self.debug_checks:
            assert cache_k.ndim == 4 and cache_v.ndim == 4, "kv_cache 期望 [B,Tpm,K,H]"
        Bc, Tpm = int(cache_k.shape[0]), int(cache_k.shape[1])

        if Bq != Bc:
            if (Bq % Bc) != 0:
                raise ValueError(f"kv_cache batch {Bc} not dividing current batch {Bq}.")
            if not self.allow_bn_broadcast:
                raise ValueError("Batch mismatch but broadcasting disabled.")
            N = Bq // Bc
            cache_k = jax.lax.broadcast_in_dim(
                cache_k, shape=(Bc, N, Tpm, cache_k.shape[2], cache_k.shape[3]),
                broadcast_dimensions=(0, 2, 3, 4),
            ).reshape(Bq, Tpm, cache_k.shape[2], cache_k.shape[3])
            cache_v = jax.lax.broadcast_in_dim(
                cache_v, shape=(Bc, N, Tpm, cache_v.shape[2], cache_v.shape[3]),
                broadcast_dimensions=(0, 2, 3, 4),
            ).reshape(Bq, Tpm, cache_v.shape[2], cache_v.shape[3])

        # 拼接：先 cache 再当前
        k = jnp.concatenate([cache_k, k_cur], axis=1)  # [Bq, Tpm+Tcur, K, H]
        v = jnp.concatenate([cache_v, v_cur], axis=1)
        if self.debug_checks:
            assert k.shape[1] == Tpm + Tcur and v.shape[1] == Tpm + Tcur, \
                f"KV 时间维拼接异常：Tpm={Tpm}, Tcur={Tcur}, got K={k.shape}, V={v.shape}"

        # === 5) 注意力计算 + 掩码校验 ===
        # 将 q 视作 (K*G) 个头，按 K 分组
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)  # [B,K,G,Tq,S]

        expected_mask_shape = (Bq, 1, Tq, Tpm + Tcur)
        if attn_mask.shape != expected_mask_shape:
            raise ValueError(f"Attention mask shape {attn_mask.shape} != expected {expected_mask_shape} "
                            f"(q={q.shape}, k={k.shape})")

        big_neg = -2.3819763e38  # 与原 Gemma 对齐
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)         # [B,Tq,K,G,H]
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")  # [B,Tq,N,H]

        # === 6) 回写到各 expert 的时间切片（不越界） ===
        out = []
        start = 0
        for i, (x, cfg) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                Ti = int(x.shape[1])
                end = start + Ti
                assert end <= encoded.shape[1], "encoded T 维切片越界"
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
    debug_checks: bool = True

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, deterministic=True):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn", debug_checks=self.debug_checks)

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
    debug_checks: bool = True
    
    def _zero_kv(self, B: int, dtype) -> KVCache:
        L = self.configs[0].depth
        K = self.configs[0].num_kv_heads
        H = self.configs[0].head_dim
        k0 = jnp.zeros((L, B, 0, K, H), dtype)
        v0 = jnp.zeros((L, B, 0, K, H), dtype)
        return (k0, v0)

    # --- 小工具：monotonic 位置校验（仅在 True mask 上） ---


    @staticmethod
    def _assert_monotonic_positions(pos: jax.Array, mask: jax.Array, name: str):
        """
        检查：在“有效 token”位置，positions 是否等于 cumsum(valid)-1。
        允许 mask 为 [B,T] / [B,T,S] / [B,1,T,S]。
        """
        # 1) 统一成 token 级 mask: [B,T]
        if mask.ndim == 4:
            # [B,1,T,S] -> [B,T,S]
            assert mask.shape[1] == 1, f"{name}: 4D mask 第二维应为 1，got {mask.shape}"
            mask = jnp.squeeze(mask, axis=1)
        if mask.ndim == 3:
            token_mask = jnp.any(mask, axis=-1)        # [B,T,S] -> [B,T]
        elif mask.ndim == 2:
            token_mask = mask                           # [B,T]
        else:
            raise ValueError(f"{name}: mask 维度必须是 2/3/4，got {mask.shape}")

        # 2) 形状对齐
        _dbg_assert(True, jnp.array(pos.ndim == 2 and pos.shape == token_mask.shape),
                    f"{name}: positions 形状 {pos.shape} 必须等于 token_mask 形状 {token_mask.shape}")

        # 3) 参考位置：有效 token 的累计计数 - 1
        ref = jnp.cumsum(token_mask, axis=-1) - 1      # [B,T]

        # 4) 仅在有效 token 处比较
        ok = jnp.all(jnp.where(token_mask, pos == ref, True))
        _dbg_assert(True, ok, f"{name} 不是基于 mask 的单调累计位置（应等于 cumsum(valid)-1）")

        
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
            # Arguments after the carry (`xs`) are: kv_cache, positions, mask, deterministic.
            # `kv_cache` carries the per-layer axis, so slice axis 0 for both K and V while broadcasting the rest.
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            debug_checks=self.debug_checks,
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
        # 统一 dtype
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype) if e is not None else None, embedded)
        mask_3d = jnp.asarray(mask)  # 保留为 [B,T,S] 做断言；稍后再扩一维给 Attention

        # 取 batch 大小 & dtype
        e_list = [e for e in embedded if e is not None]
        assert len(e_list) > 0, "embedded 至少要有一个非 None 分支"
        B = int(e_list[0].shape[0])
        dt = jnp.dtype(self.embed_dtype)

        # ★ 关键：没有 cache 时传“零长 cache”（带层维 L）
        kv_arg = kv_cache if kv_cache is not None else self._zero_kv(B, dt)

        # 扩成 [B,1,T,S] 再进入 scan/Attention
        mask_4d = mask_3d[:, None, :, :]

        embedded, kv_out = self.layers(embedded, kv_arg, positions, mask_4d, deterministic)
        out = [f(e) if e is not None else e for f, e in zip(self.final_norms, embedded, strict=True)]

        if self.debug_checks:
            # _host_assert(jnp.array(positions.dtype in (jnp.int32, jnp.int16, jnp.int64)),
            #  "positions dtype 应为整数，got {}", positions.dtype)
            # _host_assert(jnp.array(attn_mask.dtype == jnp.bool_), 
            #             "attn_mask 必须为 bool，got {}", attn_mask.dtype)
            assert isinstance(kv_out, tuple) and len(kv_out) == 2
            L = self.configs[0].depth
            assert kv_out[0].shape[0] == L and kv_out[1].shape[0] == L, \
                f"L 不匹配：{kv_out[0].shape} / {kv_out[1].shape}"

        return out, kv_out

            
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
    这两个方法只是把你已有的 self.layers 包了一层,完全复用现有 scan 堆叠出的逐层 K/V 返回机制;
    decode_with_cache 的关键在于：传入的 pm_cache 是 [L,B,...],而 embedded_suf/mask_suf 用的是 [B*N, ...];
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
        e0 = next(e for e in embedded_pm if e is not None)
        B  = int(e0.shape[0])
        dt = jnp.dtype(self.embed_dtype)

        _, kv_cache = self.layers(embedded_pm, self._zero_kv(B, dt), positions_pm, mask_pm_b, deterministic)

        if self.debug_checks:
            # 1) 位置与 mask 的一致性（已修过的版本，OK）
            self._assert_monotonic_positions(positions_pm, mask_pm.astype(bool), "positions_pm")

            # 2) pm_cache 的 T 维 与 mask 有效 token 数一致（逐 batch）
            k_all = kv_cache[0]                       # [L,B,Tpm,K,H]
            Tpm_cache = k_all.shape[2]                # 缓存里每层的 K/V 序列长度
            Tpm_input = positions_pm.shape[1]         # 传入的总 token 数（∑ 各分支长度）

            _dbg_assert(self.debug_checks, 
                jnp.array(Tpm_cache == Tpm_input),
                "pm_cache T ({}) 应等于 positions_pm.shape[1] ({})",
                Tpm_cache, Tpm_input,
            )

            # 可选：如果你仍想监控“有效 token 数”，只能做下界/上界式的 sanity，不要等号：
            valid_pm = jnp.any(mask_pm.astype(bool), axis=-1)  # [B,Tpm]
            Tpm_valid_min = jnp.min(jnp.sum(valid_pm, axis=-1))  # 每 batch 的最少有效数
            _dbg_assert(self.debug_checks, 
                jnp.array((Tpm_valid_min <= Tpm_cache) & (Tpm_valid_min >= 0)),
                "mask_pm 有效 token 数的最小值 {} 不应超过 pm_cache T ({})",
                Tpm_valid_min, Tpm_cache,
            )
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
        
        L = self.configs[0].depth
        k0, v0 = pm_cache.kv
        assert k0.shape[0] == L and v0.shape[0] == L, f"pm_cache L 不匹配：{k0.shape} / {v0.shape}"
        outputs, _ = self.layers(embedded_suf, pm_cache.kv, positions_suf, mask_suf_b, deterministic)
        outs = [f(e) if e is not None else e for f, e in zip(self.final_norms, outputs, strict=True)]
        if self.debug_checks:
            # 输出 dtype 与 embed_dtype 一致
            assert all((e is None) or (e.dtype == jnp.dtype(self.embed_dtype)) for e in outs)
        return outs



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