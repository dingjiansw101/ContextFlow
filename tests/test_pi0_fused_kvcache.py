# tests/test_pi0_fused_kvcache.py
import types
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import pytest

# ====== 你项目内的导入 ======
from openpi.models.pi0_incontext_v14 import Pi0LightIncontextConfigv14, Pi0LightIncontextv14, SIGLIP_OUTPUT_DIM
from openpi.models import model as _model
from openpi.shared import array_typing as at

# ---------- 小工具 ----------
def randn(shape, key=0):
    return jax.random.normal(jax.random.PRNGKey(key), shape, dtype=jnp.float32)

def zeros(shape, dtype=jnp.bfloat16):
    return jnp.zeros(shape, dtype=dtype)

# ---------- Stub: LLM ----------
class StubPMCache:
    def __init__(self, kv):
        self.kv = kv  # tuple(k,v) with shapes [L,B,Tpm,K,H]

class StubLLM:
    """
    只实现 encode_pm_only / decode_with_cache 两个方法，模拟 KV 形状与返回。
    - depth=6, num_kv_heads=1, head_dim=16：足够小、跑得快
    - 输出直接把输入 embedded 原样返回（identity），便于 Pi0 后续线性头取到动作位
    """
    def __init__(self, depth=6, num_kv_heads=1, head_dim=16, width=128, embed_dtype="bfloat16"):
        self.depth = depth
        self.K = num_kv_heads
        self.H = head_dim
        self.width = width
        self.embed_dtype = embed_dtype

    # Pi0 在 encode 上用的是 method="encode_pm_only" 的包装
    def __call__(self, *args, **kwargs):
        # 兼容旧（联合 __call__）路径，不在本测试使用
        raise NotImplementedError

    # === 真正被 Pi0 调的 ===
    def encode_pm_only(self, *, embedded_pm, positions_pm, mask_pm, deterministic=True):
        # embedded_pm: [pm_branch, ae_branch(None)]
        e0 = next(e for e in embedded_pm if e is not None)
        B, Tpm, D = e0.shape
        # 构造每层 KV 为全零（只要形状正确即可）
        k = jnp.zeros((self.depth, B, Tpm, self.K, self.H), dtype=jnp.bfloat16)
        v = jnp.zeros((self.depth, B, Tpm, self.K, self.H), dtype=jnp.bfloat16)
        return StubPMCache((k, v))

    def decode_with_cache(self, *, embedded_suf, positions_suf, mask_suf, pm_cache, ep_index, deterministic=True):
        # embedded_suf: [None, suffix_tokens]，我们返回原样（identity）
        out = []
        for e in embedded_suf:
            if e is None:
                out.append(None)
            else:
                out.append(e.astype(jnp.dtype(self.embed_dtype)))
        return out

# ---------- Stub: IMG (SigLIP) ----------
class StubIMG:
    """
    模拟 SigLIP 图像编码输出 patch tokens。
    只要输出最后维度等于 SIGLIP_OUTPUT_DIM[variant_key]，Pi0 的线性投影就不会报 shape。
    """
    def __init__(self, siglip_variant="Ti/16"):
        key = siglip_variant.split("/")[0]
        self.out_dim = SIGLIP_OUTPUT_DIM[key]

    def __call__(self, x, train=False):
        # x: [B, H, W, C]
        B = x.shape[0]
        # 设定一个很小的 patch 数（例如 4 个 token），加速测试
        P = 4
        tokens = jnp.zeros((B, P, self.out_dim), dtype=jnp.bfloat16)
        return tokens, None

# ---------- Mock: 预处理 fused ----------
@dataclasses.dataclass
class _FusedObs:
    # Pi0 fused 路径里会访问的字段
    state: jnp.ndarray                # [B,A]
    current_state_seq: jnp.ndarray    # [B,N,A]
    current_images_seq: dict          # name -> [B,N,H,W,C]
    current_image_masks_seq: dict     # name -> [B,N] bool
    actions_seq: jnp.ndarray          # [B,N,H,A]
    # midfix 所需（启用 image prompts 的话）
    incontext_images: dict            # name -> [B,T,H,W,C]
    incontext_image_masks: dict       # name -> [B,T] bool
    # 以下字段 embed_midfix 里会检查，但我们让 text/state/action prompts 关闭，不用给

def _mock_preprocess_observation_incontext_fused(rng, observation, train=False):
    # 测试里 observation 直接就是我们构造的 _FusedObs，原样返回即可
    return observation

# ---------- 构造一个最小可运行的 Pi0 实例 ----------
def make_minimal_pi0(B=2, N=3, H=4, A=6, Tpm=5, siglip_variant="Ti/16"):
    # 配置：减小宽度/深度，加快测试（我们 stub 了 Gemma，本处只影响 Linear 尺寸）
    cfg = Pi0LightIncontextConfigv14(
        action_dim=A,
        action_horizon=H,
        sample_frames=N,          # 仅影响 inputs_spec；真正用的是我们 mock 的 fused obs
        max_token_len=8,
        siglip_variant=siglip_variant,
        pool_type="none",
        debug_fused_checks=True,  # 打开 debug 断言，帮助发现问题
    )
    model = cfg.create(jax.random.PRNGKey(0))

    # 关闭会增加 midfix 复杂度的文本/状态/动作演示，只保留图像演示以满足 embed_midfix 的非空断言
    model.use_text_prompts = False
    model.use_action_state_prompts = False
    model.use_image_prompts = True
    model.avg_current_img = True
    model.causal_attention = False

    # 注入 Stub LLM 与 IMG
    model.PaliGemma["llm"] = StubLLM(depth=6, num_kv_heads=1, head_dim=16, width=128, embed_dtype="bfloat16")
    model.PaliGemma["img"] = StubIMG(siglip_variant)

    return model, cfg

# ---------- 构造一个 fused 观测（带 incontext 图像以驱动 midfix） ----------
def make_fused_observation(B=2, N=3, H=4, A=6, Tpm=5, cams=("base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb")):
    H_img, W_img, C = 64, 64, 3
    # current 序列（用于 suffix）
    current_images_seq = {name: jnp.zeros((B, N, H_img, W_img, C), dtype=jnp.float32) for name in cams}
    current_image_masks_seq = {name: jnp.ones((B, N), dtype=jnp.bool_) for name in cams}
    current_state_seq = jnp.zeros((B, N, A), dtype=jnp.float32)
    actions_seq = jnp.zeros((B, N, H, A), dtype=jnp.float32)

    # incontext 图像（用于 midfix）
    inctx_images = {name: jnp.zeros((B, Tpm, H_img, W_img, C), dtype=jnp.float32) for name in cams}
    inctx_masks  = {name: jnp.ones((B, Tpm), dtype=jnp.bool_) for name in cams}

    obs = _FusedObs(
        state=jnp.zeros((B, A), dtype=jnp.float32),
        current_state_seq=current_state_seq,
        current_images_seq=current_images_seq,
        current_image_masks_seq=current_image_masks_seq,
        actions_seq=actions_seq,
        incontext_images=inctx_images,
        incontext_image_masks=inctx_masks,
    )
    return obs, actions_seq  # actions_seq 在 compute_loss 内只用来取形状（已封装入 obs）

# ===================== 测试用例 =====================

def test_make_attn_mask_shapes_and_values():
    from openpi.models.pi0_incontext_v14 import make_attn_mask
    B, T = 2, 6
    input_mask = jnp.array([[1,1,1,1,1,1],[1,1,1,1,1,1]], dtype=bool)
    # 构造一个“前缀 + 因果”的 ar_mask：前三个共享，后三个因果
    mask_ar = jnp.array([0,0,0,1,1,1], dtype=bool)
    attn = make_attn_mask(input_mask, mask_ar)
    assert attn.shape == (B, T, T)
    # 检查前缀段互相可见、后缀因果
    assert bool(attn[0, 1, 0]) is True   # 1 能看 0（同前缀）
    assert bool(attn[0, 4, 5]) is False  # 因果方向不允许看未来

def test_fused_compute_loss_calls_kv_paths_and_shapes(monkeypatch):
    # 1) 构造最小模型 + mock 预处理
    model, cfg = make_minimal_pi0(B=2, N=3, H=4, A=6, Tpm=5)
    monkeypatch.setattr(_model, "preprocess_observation_incontext_fused", _mock_preprocess_observation_incontext_fused)

    # 2) 构造 fused 观测与动作
    obs, acts_seq = make_fused_observation(B=2, N=3, H=4, A=6, Tpm=5)

    # 3) 运行 compute_loss（训练模式）
    rng = jax.random.PRNGKey(42)
    loss = model.compute_loss(rng, obs, jnp.zeros((2, 4, 6), dtype=jnp.float32), train=True)

    # 4) 断言：形状应该是 [B,N,H]
    assert loss.shape == (2, 3, 4)

    # 5) 断言：确实走了 encode_pm_only 与 decode_with_cache（debug 标志位）
    assert getattr(model, "_dbg_used_encode_pm_only", False), "encode_pm_only 未被触发"
    assert getattr(model, "_dbg_used_decode_with_cache", False), "decode_with_cache 未被触发"

def test_positions_monotonicity_and_suffix_tail_mask(monkeypatch):
    # 目标：触发并通过 Pi0 内部的两类断言
    #  - _assert_monotonic_positions：midfix/suffix 的 positions 必须基于 mask 严格单调（允许整体偏移）
    #  - _assert_block_ar_mask：suffix 尾部 H 段的 AR 模式应为 [True, False*(H-1)]
    model, cfg = make_minimal_pi0(B=1, N=2, H=3, A=4, Tpm=4)
    monkeypatch.setattr(_model, "preprocess_observation_incontext_fused", _mock_preprocess_observation_incontext_fused)
    obs, _ = make_fused_observation(B=1, N=2, H=3, A=4, Tpm=4)
    rng = jax.random.PRNGKey(0)

    # 不抛异常即通过（内部会在 debug_fused_checks=True 下做断言）
    _ = model.compute_loss(rng, obs, jnp.zeros((1, 3, 4), dtype=jnp.float32), train=True)

def test_deterministic_flag_flow(monkeypatch):
    """
    训练（train=True）时 suffix 的 deterministic 应为 False；
    推理（train=False）时应为 True。
    这里通过 Pi0 的包装函数被调用与否来侧向验证（encode_pm_only 一直用 True）。
    """
    model, cfg = make_minimal_pi0(B=1, N=2, H=2, A=3, Tpm=3)
    monkeypatch.setattr(_model, "preprocess_observation_incontext_fused", _mock_preprocess_observation_incontext_fused)
    obs, _ = make_fused_observation(B=1, N=2, H=2, A=3, Tpm=3)
    rng = jax.random.PRNGKey(0)

    # 训练：会走 decode_with_cache(deterministic=False)
    _ = model.compute_loss(rng, obs, jnp.zeros((1, 2, 3), dtype=jnp.float32), train=True)
    assert model._dbg_used_encode_pm_only and model._dbg_used_decode_with_cache

    # 推理：同样会走 decode_with_cache，但 deterministic=True（我们这里不直接拿到 flag，
    # 仅确保推理路径也能稳定复用 pm_cache，不抛异常）
    _ = model.compute_loss(rng, obs, jnp.zeros((1, 2, 3), dtype=jnp.float32), train=False)

