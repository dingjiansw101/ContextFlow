# tests/test_pi0_bp_sequence_vs_stepwise.py
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest

from openpi.models.pi0_light_incontextv14 import (
    Pi0LightIncontextConfigv14,
    SIGLIP_OUTPUT_DIM,
)
from openpi.models import model as _model
from openpi.models.model import ObservationIncontext, IMAGE_KEYS
import openpi.models.pi0_light_incontextv14 as pi0_mod


# ---------- Stubs（与前一个测试一致，但把 dtype 统一成 float32，减小数值误差） ----------
class StubPMCache:
    def __init__(self, kv):
        self.kv = kv

class StubLLM:
    def __init__(self, depth=4, num_kv_heads=1, head_dim=16, embed_dtype="bfloat16"):
        self.depth = depth
        self.K = num_kv_heads
        self.H = head_dim
        self.embed_dtype = embed_dtype
    def encode_pm_only(self, *, embedded_pm, positions_pm, mask_pm, deterministic=True):
        e0 = next(e for e in embedded_pm if e is not None)
        B, Tpm, _ = e0.shape
        # 为了数值稳定，返回 float32
        k = jnp.zeros((self.depth, B, Tpm, self.K, self.H), dtype=jnp.float32)
        v = jnp.zeros((self.depth, B, Tpm, self.K, self.H), dtype=jnp.float32)
        return StubPMCache((k, v))
    def decode_with_cache(self, *, embedded_suf, positions_suf, mask_suf, pm_cache, ep_index, deterministic=True):
        outs = []
        for e in embedded_suf:
            outs.append(None if e is None else e.astype(jnp.float32))  # 走 float32
        return outs

class StubLLMWrapper:
    def __init__(self, depth=4, num_kv_heads=1, head_dim=16):
        self._stub = StubLLM(depth=depth, num_kv_heads=num_kv_heads, head_dim=head_dim)
    def lazy_init(self, *args, **kwargs): return None
    def __call__(self, *args, method=None, **kwargs):
        if method == "encode_pm_only":
            return self._stub.encode_pm_only(**kwargs)
        elif method == "decode_with_cache":
            return self._stub.decode_with_cache(**kwargs)
        elif method == "embed":
            tokens = kwargs.get("embedded", None)
            return jnp.zeros_like(tokens) if tokens is not None else jnp.zeros((1,1,16), jnp.float32)
        else:
            raise NotImplementedError(method)

class StubIMGWrapper:
    def __init__(self, siglip_module):
        self.out_dim = int(getattr(siglip_module, "width"))
    def lazy_init(self, *args, **kwargs): return None
    def __call__(self, x, train=False):
        B = x.shape[0]
        P = 4
        # 走 float32，减少误差
        tokens = jnp.zeros((B, P, self.out_dim), dtype=jnp.float32)
        return tokens, None

def _to_nnx_stub(module_obj):
    # Gemma: 有 configs + embed_dtype
    if hasattr(module_obj, "configs") and hasattr(module_obj, "embed_dtype"):
        try:
            c0 = module_obj.configs[0]
            return StubLLMWrapper(depth=c0.depth, num_kv_heads=c0.num_kv_heads, head_dim=c0.head_dim)
        except Exception:
            return StubLLMWrapper()
    # SigLIP: 有 width + dtype_mm
    if hasattr(module_obj, "width") and hasattr(module_obj, "dtype_mm"):
        return StubIMGWrapper(module_obj)
    raise RuntimeError(f"Unexpected module for ToNNX stub: {type(module_obj)}")


# ---------- 预处理 mock（保持原样） ----------
def _mock_preprocess_observation_incontext_fused(rng, observation, train=False):
    return observation


# ---------- 构造 ObservationIncontext ----------
def make_obs(B=2, N=3, H=4, A=6, H_img=64, W_img=64, C=3, T_inctx=5):
    cams = IMAGE_KEYS
    image = {nm: jnp.zeros((B, H_img, W_img, C), dtype=jnp.float32) for nm in cams}
    image_mask = {nm: jnp.ones((B,), dtype=jnp.bool_) for nm in cams}
    state = jnp.zeros((B, A), dtype=jnp.float32)

    current_images_seq = {nm: jnp.zeros((B, N, H_img, W_img, C), dtype=jnp.float32) for nm in cams}
    current_image_masks_seq = {nm: jnp.ones((B, N), dtype=jnp.bool_) for nm in cams}
    current_state_seq = jnp.zeros((B, N, A), dtype=jnp.float32)
    actions_seq = jnp.zeros((B, N, H, A), dtype=jnp.float32)

    dem_prompt_images = {nm: jnp.zeros((B, T_inctx, H_img, W_img, C), dtype=jnp.float32) for nm in cams}
    dem_prompt_images_mask = {nm: jnp.ones((B, T_inctx), dtype=jnp.bool_) for nm in cams}
    selected_episode = jnp.zeros((B, 1), dtype=jnp.int32)

    return ObservationIncontext.from_dict({
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
    })


# ---------- 两份相同初始化的模型（注意 ToNNX 在 create 前已被 stub） ----------
def make_model(seed=0, siglip_variant="Ti/16"):
    cfg = Pi0LightIncontextConfigv14(
        vocab_size=5000,
        prompt_expert_variant="gemma_A",
        action_expert_variant="gemma_B",
        siglip_variant=siglip_variant,
        action_dim=6,
        action_horizon=4,
        max_token_len=8,
        avg_current_img=True,
        debug_fused_checks=False,
    )
    m1 = cfg.create(jax.random.PRNGKey(seed))
    m2 = cfg.create(jax.random.PRNGKey(seed))
    # 关闭不需要的分支
    for m in (m1, m2):
        m.use_text_prompts = False
        m.use_action_state_prompts = False
        m.use_image_prompts = True
        m.avg_current_img = True
    return m1, m2, cfg


# ---------- 核心测试 ----------
def test_backprop_sequence_equals_stepwise(monkeypatch):
    # 1) 在 create 之前 stub ToNNX
    monkeypatch.setattr(pi0_mod.nnx_bridge, "ToNNX", _to_nnx_stub, raising=True)
    # 2) 预处理打桩
    monkeypatch.setattr(_model, "preprocess_observation_incontext_fused", _mock_preprocess_observation_incontext_fused, raising=True)

    # 两份相同初始化的模型
    model_seq, model_step, _ = make_model(seed=123)

    # 固定数据 & 噪声/时间
    B, N, H, A = 2, 3, model_seq.action_horizon, model_seq.action_dim
    obs = make_obs(B, N, H, A)
    noise = jnp.ones((B, N, H, A), dtype=jnp.float32) * 0.123
    t     = jnp.ones((B, N), dtype=jnp.float32) * 0.5

    # 定义两条损失（标量）
    def loss_seq_fn(m):
        return jnp.mean(m.compute_loss_sequence(None, obs, None, noise=noise, t=t))
    def loss_step_fn(m):
        return jnp.mean(m.compute_loss_stepwise(None, obs, None, noise=noise, t=t))

    # 计算梯度（对全部 nnx.Param）
    diff_all = nnx.DiffState(0, nnx.Param)
    loss_s, grads_s = nnx.value_and_grad(loss_seq_fn, argnums=diff_all)(model_seq)
    loss_t, grads_t = nnx.value_and_grad(loss_step_fn, argnums=diff_all)(model_step)

    # 梯度树结构一致
    leaves_a, treedef_a = jax.tree.flatten(grads_s)
    leaves_b, treedef_b = jax.tree.flatten(grads_t)
    assert treedef_a == treedef_b, "梯度树结构不同"

    # 逐叶比较（容忍极小误差；若你把 stub 改回 bfloat16，可把阈值放宽到 rtol=1e-4, atol=1e-6）
    for i, (ga, gb) in enumerate(zip(leaves_a, leaves_b, strict=True)):
        assert jnp.allclose(ga, gb, rtol=1e-5, atol=1e-6), f"梯度第 {i} 个叶子不一致"

    # 用简单 SGD 做一次更新后，再比较参数
    lr = 1e-2
    params_s = nnx.state(model_seq, nnx.Param)
    params_t = nnx.state(model_step, nnx.Param)
    new_params_s = jax.tree.map(lambda p, g: p - lr * g, params_s, grads_s)
    new_params_t = jax.tree.map(lambda p, g: p - lr * g, params_t, grads_t)
    nnx.update(model_seq, new_params_s)
    nnx.update(model_step, new_params_t)

    params_s_after = nnx.state(model_seq, nnx.Param)
    params_t_after = nnx.state(model_step, nnx.Param)

    la, ta = jax.tree.flatten(params_s_after)
    lb, tb = jax.tree.flatten(params_t_after)
    assert ta == tb, "参数树结构不同"
    for i, (pa, pb) in enumerate(zip(la, lb, strict=True)):
        assert jnp.allclose(pa, pb, rtol=1e-6, atol=1e-7), f"参数第 {i} 个叶子在一步更新后不一致"

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 PYTHONPATH=src uv run --active python -m pytest -vv tests/test_pi0_bp_sequence_vs_stepwise.py 