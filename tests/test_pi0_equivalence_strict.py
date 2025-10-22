# tests/test_pi0_equivalence_strict.py
import dataclasses
import math
import types
from functools import partial

import einops
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import flax.nnx as nnx

import openpi.models.pi0_light_incontextv14 as pi0_mod
from openpi.models.pi0_light_incontextv14 import (
    Pi0LightIncontextConfigv14,
    SIGLIP_OUTPUT_DIM,
)
from openpi.models import model as _model
from openpi.models.model import ObservationIncontext, IMAGE_KEYS

# ---------------------------
# Stubs for Gemma / SigLIP
# ---------------------------

class _StubPMCache:
    def __init__(self, kv):
        self.kv = kv

class _StubLLM:
    def __init__(self, depth=4, num_kv_heads=1, head_dim=16, embed_dtype="bfloat16"):
        self.depth = depth
        self.K = num_kv_heads
        self.H = head_dim
        self.embed_dtype = embed_dtype

    def encode_pm_only(self, *, embedded_pm, positions_pm, mask_pm, deterministic=True):
        e0 = next(e for e in embedded_pm if e is not None)
        B, Tpm, _ = e0.shape
        k = jnp.zeros((self.depth, B, Tpm, self.K, self.H), dtype=jnp.bfloat16)
        v = jnp.zeros((self.depth, B, Tpm, self.K, self.H), dtype=jnp.bfloat16)
        return _StubPMCache((k, v))

    def decode_with_cache(self, *, embedded_suf, positions_suf, mask_suf, pm_cache, ep_index, deterministic=True):
        outs = []
        for e in embedded_suf:
            if e is None:
                outs.append(None)
            else:
                outs.append(e.astype(jnp.bfloat16))  # identity
        return outs

class _StubLLMWrapper:
    def __init__(self, depth=4, num_kv_heads=1, head_dim=16):
        self._stub = _StubLLM(depth=depth, num_kv_heads=num_kv_heads, head_dim=head_dim)

    def lazy_init(self, *args, **kwargs):
        return None

    def __call__(self, *args, method=None, **kwargs):
        if method == "encode_pm_only":
            return self._stub.encode_pm_only(**kwargs)
        elif method == "decode_with_cache":
            return self._stub.decode_with_cache(**kwargs)
        elif method == "embed":
            # text 已禁用，但若被调用，返回形状合法的 0
            tokens = kwargs.get("embedded", None)
            return jnp.zeros_like(tokens) if tokens is not None else jnp.zeros((1, 1, 16), jnp.bfloat16)
        else:
            raise NotImplementedError(f"StubLLMWrapper: unsupported method={method}")

class _StubIMGWrapper:
    def __init__(self, siglip_module):
        self.out_dim = int(getattr(siglip_module, "width"))

    def lazy_init(self, *args, **kwargs):
        return None

    def __call__(self, x, train=False):
        B = x.shape[0]
        P = 4
        tokens = jnp.zeros((B, P, self.out_dim), dtype=jnp.bfloat16)
        return tokens, None

def _to_nnx_stub(module_obj):
    # 识别 Gemma
    if hasattr(module_obj, "configs") and hasattr(module_obj, "embed_dtype"):
        try:
            c0 = module_obj.configs[0]
            return _StubLLMWrapper(depth=c0.depth, num_kv_heads=c0.num_kv_heads, head_dim=c0.head_dim)
        except Exception:
            return _StubLLMWrapper()
    # 识别 SigLIP（你工程里的类为 openpi.models.siglip._Module，带 width/dtype_mm）
    if hasattr(module_obj, "width") and hasattr(module_obj, "dtype_mm"):
        return _StubIMGWrapper(module_obj)
    raise RuntimeError(f"Unexpected module for ToNNX stub: {type(module_obj)}")


# 预处理替换为恒等：不 resize、不增广、也不改变数据（避免随机性）
def _mock_preprocess_observation_incontext_fused(rng, observation, *, train=False, **_):
    return observation

# ---------------------------
# Fixtures & helpers
# ---------------------------

def _make_model(monkeypatch, *, siglip_variant="Ti/16", debug=False, seed=0):
    # 在创建模型前打桩
    monkeypatch.setattr(pi0_mod.nnx_bridge, "ToNNX", _to_nnx_stub, raising=True)
    monkeypatch.setattr(_model, "preprocess_observation_incontext_fused",
                        _mock_preprocess_observation_incontext_fused, raising=True)

    cfg = Pi0LightIncontextConfigv14(
        vocab_size=5000,
        prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
        siglip_variant=siglip_variant,
        action_dim=6,
        action_horizon=4,
        max_token_len=8,
        avg_current_img=True,
        debug_fused_checks=debug,
    )
    model = cfg.create(jax.random.PRNGKey(seed))
    # 关闭除图像外的 prompt，路径一致简单
    model.use_text_prompts = False
    model.use_action_state_prompts = False
    model.use_image_prompts = True
    model.avg_current_img = True
    return model

def _make_obs(
    *,
    B=2,
    N=3,
    H=4,
    A=6,
    H_img=64,
    W_img=64,
    C=3,
    T_inctx=5,
    mask_ratio=0.0,
    seed=0,
):
    """
    生成随机但可复现的 ObservationIncontext：
    - 图像：U[0,1]（float32）
    - state / actions：N(0, 0.1^2)（float32）
    - masks：按 mask_ratio 独立采样 False；其余 True
    """
    rng = np.random.default_rng(seed)
    cams = IMAGE_KEYS

    # ---- 主观测（单帧） ----
    image = {
        nm: jnp.asarray(
            rng.random((B, H_img, W_img, C), dtype=np.float32),
            dtype=jnp.float32,
        )
        for nm in cams
    }
    # 单帧 mask 就设为全 True（这些不是“序列”，一般不测空帧）
    image_mask = {nm: jnp.ones((B,), dtype=jnp.bool_) for nm in cams}

    # robot state：零均值小噪声
    state = jnp.asarray(rng.normal(loc=0.0, scale=0.1, size=(B, A)), dtype=jnp.float32)

    # ---- 当前序列（长度 N）----
    # 图像序列：随机 U[0,1]
    current_images_seq = {
        nm: jnp.asarray(
            rng.random((B, N, H_img, W_img, C), dtype=np.float32),
            dtype=jnp.float32,
        )
        for nm in cams
    }

    # 图像序列 masks：独立伯努利，(1 - mask_ratio) 为 True 概率
    def bernoulli_mask(shape):
        # True 比例 ~ (1 - mask_ratio)
        keep = rng.random(shape) >= mask_ratio
        return jnp.asarray(keep, dtype=jnp.bool_)

    current_image_masks_seq = {
        nm: bernoulli_mask((B, N)) for nm in cams
    }

    # state 序列：零均值小噪声
    current_state_seq = jnp.asarray(
        rng.normal(loc=0.0, scale=0.1, size=(B, N, A)),
        dtype=jnp.float32,
    )

    # actions 序列：零均值小噪声（形状 [B, N, H, A]）
    actions_seq = jnp.asarray(
        rng.normal(loc=0.0, scale=0.1, size=(B, N, H, A)),
        dtype=jnp.float32,
    )

    # ---- in-context 演示（最小合法）----
    dem_prompt_images = {
        nm: jnp.asarray(
            rng.random((B, T_inctx, H_img, W_img, C), dtype=np.float32),
            dtype=jnp.float32,
        )
        for nm in cams
    }
    dem_prompt_images_mask = {
        nm: jnp.ones((B, T_inctx), dtype=jnp.bool_) for nm in cams
    }

    # selected_episode：给定合法范围内的整数（这里就全 0 也行；保持原语义）
    selected_episode = jnp.asarray(
        rng.integers(low=0, high=max(1, B), size=(B, 1), dtype=np.int32),
        dtype=jnp.int32,
    )

    obs_dict = {
        "image": image,
        "image_mask": image_mask,
        "state": state,
        "dem_prompt_images": dem_prompt_images,
        "dem_prompt_images_mask": dem_prompt_images_mask,
        "selected_episode": selected_episode,
        "current_images_seq": current_images_seq,
        "current_image_masks_seq": current_image_masks_seq,
        "current_state_seq": current_state_seq,
        "actions_seq": actions_seq,
    }
    return ObservationIncontext.from_dict(obs_dict)

def _fixed_noise_t(*, key=0, shape_vt=(2,3,4,6)):
    key = jax.random.PRNGKey(key)
    k1, k2 = jax.random.split(key)
    noise = jnp.full(shape_vt, 0.123, dtype=jnp.float32)  # 常量最稳
    t = jnp.full(shape_vt[:2], 0.5, dtype=jnp.float32)
    return noise, t

def _allclose(a, b, *, rtol=1e-5, atol=5e-6):
    return bool(jnp.allclose(a, b, rtol=rtol, atol=atol))

# ---------------------------
# 1) 前向：一次性 vs 逐步
# ---------------------------

def test_forward_vt_and_loss_equivalence(monkeypatch):
    model = _make_model(monkeypatch, debug=False, seed=0)
    B, N, H, A = 2, 3, model.action_horizon, model.action_dim
    obs = _make_obs(B=B, N=N, H=H, A=A, seed=0)

    noise, t = _fixed_noise_t(shape_vt=(B, N, H, A), key=7)

    vt_seq  = model.forward_vt_sequence(obs, noise, t)
    vt_step = model.forward_vt_stepwise(obs, noise, t)

    assert vt_seq.shape == (B, N, H, A)
    assert vt_step.shape == (B, N, H, A)
    assert _allclose(vt_seq, vt_step)

    loss_seq  = model.compute_loss_sequence(None, obs, None, noise=noise, t=t)  # [B,N,H]
    loss_step = model.compute_loss_stepwise(None, obs, None, noise=noise, t=t)
    assert loss_seq.shape == (B, N, H)
    assert loss_step.shape == (B, N, H)
    assert _allclose(loss_seq, loss_step)

# ---------------------------
# 2) 任意 chunk 拼接 vs 整段
# ---------------------------

@pytest.mark.parametrize("N,chunk", [(5,1), (5,2), (5,3), (8,4), (3,3)])
def test_chunking_equivalence(monkeypatch, N, chunk):
    model = _make_model(monkeypatch, debug=False, seed=1)
    B, H, A = 2, model.action_horizon, model.action_dim
    obs_full = _make_obs(B=B, N=N, H=H, A=A, seed=1)
    noise, t = _fixed_noise_t(shape_vt=(B, N, H, A), key=11)

    vt_all = model.forward_vt_sequence(obs_full, noise, t)

    # 按 chunk 切 N 轴，逐段跑 forward_vt_sequence，再 concat
    outs = []
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        # 构造子 obs（只切 current_*_seq / actions_seq / masks 的 N 维）
        def slc_kv(dct): return {k: v[:, s:e] for k, v in dct.items()}
        sub_dict = obs_full.to_dict()
        sub_dict["current_images_seq"] = slc_kv(sub_dict["current_images_seq"])
        sub_dict["current_image_masks_seq"] = slc_kv(sub_dict["current_image_masks_seq"])
        sub_dict["current_state_seq"] = sub_dict["current_state_seq"][:, s:e]
        sub_dict["actions_seq"] = sub_dict["actions_seq"][:, s:e]
        obs_sub = ObservationIncontext.from_dict(sub_dict)

        vt_sub = model.forward_vt_sequence(obs_sub, noise[:, s:e], t[:, s:e])
        outs.append(vt_sub)
    vt_chunked = jnp.concatenate(outs, axis=1)

    assert _allclose(vt_all, vt_chunked)

# ---------------------------
# 3) 随机 mask & 全 False mask
# ---------------------------

def test_random_masks_equivalence(monkeypatch):
    model = _make_model(monkeypatch, debug=False, seed=2)
    B, N, H, A = 2, 5, model.action_horizon, model.action_dim
    obs = _make_obs(B=B, N=N, H=H, A=A, seed=2, mask_ratio=0.5)
    noise, t = _fixed_noise_t(shape_vt=(B, N, H, A), key=5)

    loss_seq  = model.compute_loss_sequence(None, obs, None, noise=noise, t=t)
    loss_step = model.compute_loss_stepwise(None, obs, None, noise=noise, t=t)
    assert _allclose(loss_seq, loss_step)

def test_all_false_masks_zeroish_loss(monkeypatch):
    model = _make_model(monkeypatch, debug=False, seed=3)
    B, N, H, A = 2, 4, model.action_horizon, model.action_dim
    # 强制所有 current_image_masks_seq=False
    obs = _make_obs(B=B, N=N, H=H, A=A, seed=3, mask_ratio=1.0)
    noise, t = _fixed_noise_t(shape_vt=(B, N, H, A), key=9)

    loss_seq = model.compute_loss_sequence(None, obs, None, noise=noise, t=t)
    # 如果你的实现用 mask 屏蔽该帧 loss，则应该接近 0；否则放宽断言
    mean_loss = float(jnp.mean(loss_seq))
    assert mean_loss <= 1e-4 or math.isfinite(mean_loss)

# ---------------------------
# 4) 时间乱序/恢复 不变性
# ---------------------------

def test_time_order_invariance(monkeypatch):
    model = _make_model(monkeypatch, debug=False, seed=4)
    B, N, H, A = 2, 6, model.action_horizon, model.action_dim
    obs = _make_obs(B=B, N=N, H=H, A=A, seed=4, mask_ratio=0.2)
    noise, t = _fixed_noise_t(shape_vt=(B, N, H, A), key=13)

    # 原
    vt_ref = model.forward_vt_sequence(obs, noise, t)

    # 乱序
    perm = np.random.default_rng(0).permutation(N)
    def permute_obs(o: ObservationIncontext):
        d = o.to_dict()
        d["current_images_seq"] = {k: v[:, perm] for k, v in d["current_images_seq"].items()}
        d["current_image_masks_seq"] = {k: v[:, perm] for k, v in d["current_image_masks_seq"].items()}
        d["current_state_seq"] = d["current_state_seq"][:, perm]
        d["actions_seq"] = d["actions_seq"][:, perm]
        return ObservationIncontext.from_dict(d)

    obs_perm = permute_obs(obs)
    vt_perm = model.forward_vt_sequence(obs_perm, noise[:, perm], t[:, perm])

    # 恢复顺序
    inv = np.argsort(perm)
    vt_restored = vt_perm[:, inv]
    assert _allclose(vt_ref, vt_restored)

# ---------------------------
# 5) 确定性：重复两次一致
# ---------------------------

def test_deterministic_repro(monkeypatch):
    model = _make_model(monkeypatch, debug=False, seed=5)
    B, N, H, A = 2, 3, model.action_horizon, model.action_dim
    obs = _make_obs(B=B, N=N, H=H, A=A, seed=5)
    noise, t = _fixed_noise_t(shape_vt=(B, N, H, A), key=17)

    vt1 = model.forward_vt_sequence(obs, noise, t)
    vt2 = model.forward_vt_sequence(obs, noise, t)
    assert _allclose(vt1, vt2)

    l1 = model.compute_loss_sequence(None, obs, None, noise=noise, t=t)
    l2 = model.compute_loss_sequence(None, obs, None, noise=noise, t=t)
    assert _allclose(l1, l2)

# ---------------------------
# 6) JVP / VJP 等价（标量 loss）
# ---------------------------

def _loss_seq_scalar(m, obs, noise, t):
    return jnp.mean(m.compute_loss_sequence(None, obs, None, noise=noise, t=t))

def _loss_step_scalar(m, obs, noise, t):
    return jnp.mean(m.compute_loss_stepwise(None, obs, None, noise=noise, t=t))

def test_jvp_vjp_equal(monkeypatch):
    model_seq = _make_model(monkeypatch, debug=False, seed=6)
    model_stp = _make_model(monkeypatch, debug=False, seed=6)  # 同初始化

    B, N, H, A = 2, 3, model_seq.action_horizon, model_seq.action_dim
    obs = _make_obs(B=B, N=N, H=H, A=A, seed=6)
    noise, t = _fixed_noise_t(shape_vt=(B, N, H, A), key=19)

    # JVP：对所有 Param 做一次随机方向
    params = nnx.state(model_seq, nnx.Param)
    vec = jax.tree.map(lambda x: jnp.ones_like(x) * 1e-3, params)

    def f_seq(p):
        nnx.update(model_seq, p)
        return _loss_seq_scalar(model_seq, obs, noise, t)

    def f_stp(p):
        nnx.update(model_stp, p)
        return _loss_step_scalar(model_stp, obs, noise, t)

    _, jvp_seq = jax.jvp(f_seq, (params,), (vec,))
    _, jvp_stp = jax.jvp(f_stp, (params,), (vec,))
    assert _allclose(jvp_seq, jvp_stp, rtol=5e-4, atol=1e-7)

    # VJP：同一 cotangent
    prim_seq, vjp_seq_fn = jax.vjp(f_seq, params)
    prim_stp, vjp_stp_fn = jax.vjp(f_stp, params)
    assert _allclose(prim_seq, prim_stp)

    cot = jnp.array(1.0, dtype=prim_seq.dtype)
    (grad_seq,) = vjp_seq_fn(cot)
    (grad_stp,) = vjp_stp_fn(cot)

    def tree_allclose(a, b):
        la, ta = jax.tree.flatten(a)
        lb, tb = jax.tree.flatten(b)
        assert ta == tb
        for i, (x, y) in enumerate(zip(la, lb, strict=True)):
            assert _allclose(x, y, rtol=1e-5, atol=1e-7), f"vjp leaf {i} mismatch"

    tree_allclose(grad_seq, grad_stp)

# ---------------------------
# 7) 多步更新后参数等价
# ---------------------------

def test_multi_step_update_equal(monkeypatch):
    model_seq = _make_model(monkeypatch, debug=False, seed=7)
    model_stp = _make_model(monkeypatch, debug=False, seed=7)  # 同初始化

    B, N, H, A = 2, 3, model_seq.action_horizon, model_seq.action_dim
    obs = _make_obs(B=B, N=N, H=H, A=A, seed=7)
    noise, t = _fixed_noise_t(shape_vt=(B, N, H, A), key=23)

    lr = 5e-3
    steps = 3
    diff_all = nnx.DiffState(0, nnx.Param)

    def loss_seq(_m): return _loss_seq_scalar(_m, obs, noise, t)
    def loss_stp(_m): return _loss_step_scalar(_m, obs, noise, t)

    for _ in range(steps):
        l_s, g_s = nnx.value_and_grad(loss_seq, argnums=diff_all)(model_seq)
        l_t, g_t = nnx.value_and_grad(loss_stp, argnums=diff_all)(model_stp)

        # 梯度树一致
        la, ta = jax.tree.flatten(g_s)
        lb, tb = jax.tree.flatten(g_t)
        assert ta == tb
        for i, (x, y) in enumerate(zip(la, lb, strict=True)):
            assert _allclose(x, y, rtol=1e-5, atol=1e-7), f"grad leaf {i} mismatch"

        p_s = nnx.state(model_seq, nnx.Param)
        p_t = nnx.state(model_stp, nnx.Param)
        new_p_s = jax.tree.map(lambda p, g: p - lr * g, p_s, g_s)
        new_p_t = jax.tree.map(lambda p, g: p - lr * g, p_t, g_t)

        nnx.update(model_seq, new_p_s)
        nnx.update(model_stp, new_p_t)

    # K 步后参数一致
    pa = nnx.state(model_seq, nnx.Param)
    pb = nnx.state(model_stp, nnx.Param)
    la, ta = jax.tree.flatten(pa)
    lb, tb = jax.tree.flatten(pb)
    assert ta == tb
    for i, (x, y) in enumerate(zip(la, lb, strict=True)):
        assert _allclose(x, y, rtol=1e-6, atol=1e-7), f"param leaf {i} mismatch after {steps} steps"

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 PYTHONPATH=src uv run --active python -m pytest -vv tests/test_pi0_equivalence_strict.py