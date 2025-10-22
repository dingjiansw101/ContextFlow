# tests/test_pi0_sequence_vs_stepwise.py
import jax
import jax.numpy as jnp
import pytest

from openpi.models.pi0_light_incontextv14 import Pi0LightIncontextConfigv14, SIGLIP_OUTPUT_DIM
import openpi.models.pi0_light_incontextv14 as pi0_mod
from openpi.models import model as _model
from openpi.models.model import ObservationIncontext, IMAGE_KEYS


# ----------------- 简单 stubs -----------------
class StubPMCache:
    def __init__(self, kv): self.kv = kv

class StubLLM:
    def __init__(self, depth=4, num_kv_heads=1, head_dim=16, embed_dtype="bfloat16"):
        self.depth = depth; self.K = num_kv_heads; self.H = head_dim; self.embed_dtype = embed_dtype
    def encode_pm_only(self, *, embedded_pm, positions_pm, mask_pm, deterministic=True):
        e0 = next(e for e in embedded_pm if e is not None)
        B, Tpm, _ = e0.shape
        k = jnp.zeros((self.depth, B, Tpm, self.K, self.H), dtype=jnp.bfloat16)
        v = jnp.zeros((self.depth, B, Tpm, self.K, self.H), dtype=jnp.bfloat16)
        return StubPMCache((k, v))
    def decode_with_cache(self, *, embedded_suf, positions_suf, mask_suf, pm_cache, ep_index, deterministic=True):
        outs = []
        for e in embedded_suf:
            outs.append(None if e is None else e.astype(jnp.float32))
        return outs

class StubLLMWrapper:
    def __init__(self, depth=4, num_kv_heads=1, head_dim=16):
        self._stub = StubLLM(depth=depth, num_kv_heads=num_kv_heads, head_dim=head_dim)
    def lazy_init(self, *args, **kwargs): return None
    def __call__(self, *args, method=None, **kwargs):
        if method == "encode_pm_only":  return self._stub.encode_pm_only(**kwargs)
        if method == "decode_with_cache": return self._stub.decode_with_cache(**kwargs)
        if method == "embed":
            tokens = kwargs.get("embedded", None)
            return jnp.zeros_like(tokens) if tokens is not None else jnp.zeros((1,1,16), jnp.float32)
        raise NotImplementedError(f"StubLLMWrapper: unsupported method={method}")

class StubIMGWrapper:
    def __init__(self, siglip_module):
        self.out_dim = int(getattr(siglip_module, "width"))
    def lazy_init(self, *args, **kwargs): return None
    def __call__(self, x, train=False):
        B = x.shape[0]; P = 4
        return jnp.zeros((B, P, self.out_dim), dtype=jnp.float32), None

def _to_nnx_stub(module_obj):
    # Gemma: linen module 上有 configs + embed_dtype
    if hasattr(module_obj, "configs") and hasattr(module_obj, "embed_dtype"):
        try:
            c0 = module_obj.configs[0]
            return StubLLMWrapper(depth=c0.depth, num_kv_heads=c0.num_kv_heads, head_dim=c0.head_dim)
        except Exception:
            return StubLLMWrapper()
    # SigLIP: 我们的 _Module 有 width/dtype_mm
    if hasattr(module_obj, "width") and hasattr(module_obj, "dtype_mm"):
        return StubIMGWrapper(module_obj)
    raise RuntimeError(f"Unexpected module for ToNNX stub: {type(module_obj)}")


# 预处理 mock：恒等返回，避免 resize/aug 带来差异
def _mock_preprocess_observation_incontext_fused(rng, observation, *, train=False, image_keys=IMAGE_KEYS, image_resolution=(224,224)):
    return observation


# ----------------- 构造模型/观测 -----------------
def make_model(siglip_variant="Ti/16", debug=False):
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
    # 注意：ToNNX 已在测试里 monkeypatch，因此这里直接 create
    model = cfg.create(jax.random.PRNGKey(0))
    # 只保留图像提示，简化路径
    model.use_text_prompts = False
    model.use_action_state_prompts = False
    model.use_image_prompts = True
    model.avg_current_img = True
    return model, cfg


def make_obs(B=2, N=3, H=4, A=6, H_img=64, W_img=64, C=3, T_inctx=5):
    key = jax.random.PRNGKey(0)

    # 拆分随机键
    key, k_state, k_actions = jax.random.split(key, 3)
    cams = IMAGE_KEYS
    # image = {nm: jnp.zeros((B, H_img, W_img, C), dtype=jnp.float32) for nm in cams}
    image = {nm: jax.random.uniform(k_state, (B, H_img, W_img, C), minval=0.0, maxval=1.0, dtype=jnp.float32) for nm in cams}

    image_mask = {nm: jnp.ones((B,), dtype=jnp.bool_) for nm in cams}
    # state = jnp.zeros((B, A), dtype=jnp.float32)
    state = jax.random.uniform(k_state, (B, A), minval=0.0, maxval=1.0, dtype=jnp.float32)

    current_images_seq = {nm: jax.random.uniform(k_actions, (B, N, H_img, W_img, C), minval=0.0, maxval=1.0, dtype=jnp.float32) for nm in cams}
    # current_images_seq = {nm: jnp.zeros((B, N, H_img, W_img, C), dtype=jnp.float32) for nm in cams}
    current_image_masks_seq = {nm: jnp.ones((B, N), dtype=jnp.bool_) for nm in cams}
    current_state_seq = jax.random.uniform(k_actions, (B, N, A), minval=0.0, maxval=1.0, dtype=jnp.float32)
    actions_seq = jax.random.uniform(k_actions, (B, N, H, A), minval=0.0, maxval=1.0, dtype=jnp.float32)
    # current_state_seq = jnp.zeros((B, N, A), dtype=jnp.float32)
    # actions_seq = jnp.zeros((B, N, H, A), dtype=jnp.float32)

    # dem_prompt_images = {nm: jnp.zeros((B, T_inctx, H_img, W_img, C), dtype=jnp.float32) for nm in cams}
    dem_prompt_images = {nm: jax.random.uniform(k_actions, (B, T_inctx, H_img, W_img, C), minval=0.0, maxval=1.0, dtype=jnp.float32) for nm in cams}
    dem_prompt_images_mask = {nm: jnp.ones((B, T_inctx), dtype=jnp.bool_) for nm in cams}
    selected_episode = jnp.zeros((B, 1), dtype=jnp.int32)

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


# ----------------- 测试 -----------------
def test_sequence_vs_stepwise_forward_and_loss(monkeypatch):
    # 先把 ToNNX 和预处理打桩，再 create 模型
    monkeypatch.setattr(pi0_mod.nnx_bridge, "ToNNX", _to_nnx_stub, raising=True)
    monkeypatch.setattr(_model, "preprocess_observation_incontext_fused",
                        _mock_preprocess_observation_incontext_fused, raising=True)

    model, _ = make_model(debug=False)

    # 两条轨迹
    B, N, H, A = 2, 3, model.action_horizon, model.action_dim
    obs = make_obs(B=B, N=N, H=H, A=A)

    # 固定 noise / t
    key = jax.random.PRNGKey(7)
    key_n, key_t = jax.random.split(key)
    noise = jnp.full((B, N, H, A), 0.123, dtype=jnp.float32)
    t = jnp.full((B, N), 0.5, dtype=jnp.float32)

    # v_t 对比
    vt_seq  = model.forward_vt_sequence(obs, noise, t)
    vt_loop = model.forward_vt_stepwise(obs, noise, t)
    assert vt_seq.shape  == (B, N, H, A)
    assert vt_loop.shape == (B, N, H, A)
    assert jnp.allclose(vt_seq, vt_loop, rtol=1e-5, atol=1e-6)
    print("vt_seq:",vt_seq.shape,vt_seq)

    # loss 对比
    loss_seq  = model.compute_loss_sequence(None, obs, None, noise=noise, t=t)
    loss_loop = model.compute_loss_stepwise(None, obs, None, noise=noise, t=t)
    assert loss_seq.shape  == (B, N, H)
    assert loss_loop.shape == (B, N, H)
    assert jnp.allclose(loss_seq, loss_loop, rtol=1e-5, atol=1e-6)
    print("loss_loop:", loss_loop)
    print("obs:", obs.actions_seq.shape, obs.actions_seq)


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 PYTHONPATH=src uv run --active python -m pytest -vv tests/test_pi0_sequence_vs_stepwise.py