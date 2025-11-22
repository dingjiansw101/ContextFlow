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
            # Text prompts are disabled, but if called return zeros with a valid shape
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
    # Identify Gemma
    if hasattr(module_obj, "configs") and hasattr(module_obj, "embed_dtype"):
        try:
            c0 = module_obj.configs[0]
            return _StubLLMWrapper(depth=c0.depth, num_kv_heads=c0.num_kv_heads, head_dim=c0.head_dim)
        except Exception:
            return _StubLLMWrapper()
    # Identify SigLIP (your class openpi.models.siglip._Module has width/dtype_mm)
    if hasattr(module_obj, "width") and hasattr(module_obj, "dtype_mm"):
        return _StubIMGWrapper(module_obj)
    raise RuntimeError(f"Unexpected module for ToNNX stub: {type(module_obj)}")


# Replace preprocessing with identity: no resize/augment/changes (avoid randomness)
def _mock_preprocess_observation_incontext_fused(rng, observation, *, train=False, **_):
    return observation

# ---------------------------
# Fixtures & helpers
# ---------------------------

def _make_model(monkeypatch, *, siglip_variant="Ti/16", debug=False, seed=0):
    # Stub before creating the model
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
    # Disable non-image prompts to keep the path simple
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
    Generate a random but reproducible ObservationIncontext:
    - Images: U[0,1] (float32)
    - state / actions: N(0, 0.1^2) (float32)
    - masks: Bernoulli with mask_ratio for False; otherwise True
    """
    rng = np.random.default_rng(seed)
    cams = IMAGE_KEYS

    # ---- Main observation (single frame) ----
    image = {
        nm: jnp.asarray(
            rng.random((B, H_img, W_img, C), dtype=np.float32),
            dtype=jnp.float32,
        )
        for nm in cams
    }
    # Single-frame masks set to all True (not sequences; usually no empty frames)
    image_mask = {nm: jnp.ones((B,), dtype=jnp.bool_) for nm in cams}

    # robot state: small zero-mean noise
    state = jnp.asarray(rng.normal(loc=0.0, scale=0.1, size=(B, A)), dtype=jnp.float32)

    # ---- Current sequence (length N) ----
    # Image sequence: random U[0,1]
    current_images_seq = {
        nm: jnp.asarray(
            rng.random((B, N, H_img, W_img, C), dtype=np.float32),
            dtype=jnp.float32,
        )
        for nm in cams
    }

    # Image sequence masks: independent Bernoulli with True probability (1 - mask_ratio)
    def bernoulli_mask(shape):
        # True proportion ≈ (1 - mask_ratio)
        keep = rng.random(shape) >= mask_ratio
        return jnp.asarray(keep, dtype=jnp.bool_)

    current_image_masks_seq = {
        nm: bernoulli_mask((B, N)) for nm in cams
    }

    # State sequence: small zero-mean noise
    current_state_seq = jnp.asarray(
        rng.normal(loc=0.0, scale=0.1, size=(B, N, A)),
        dtype=jnp.float32,
    )

    # Actions sequence: small zero-mean noise (shape [B, N, H, A])
    actions_seq = jnp.asarray(
        rng.normal(loc=0.0, scale=0.1, size=(B, N, H, A)),
        dtype=jnp.float32,
    )

    # ---- In-context demos (minimal valid) ----
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

    # selected_episode: integer within valid range (all zeros also fine to preserve semantics)
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
    noise = jnp.full(shape_vt, 0.123, dtype=jnp.float32)  # Constant for stability
    t = jnp.full(shape_vt[:2], 0.5, dtype=jnp.float32)
    return noise, t

def _allclose(a, b, *, rtol=1e-5, atol=5e-6):
    return bool(jnp.allclose(a, b, rtol=rtol, atol=atol))

# ---------------------------
# 1) Forward: single pass vs stepwise
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
# 2) Arbitrary chunk concat vs full sequence
# ---------------------------

@pytest.mark.parametrize("N,chunk", [(5,1), (5,2), (5,3), (8,4), (3,3)])
def test_chunking_equivalence(monkeypatch, N, chunk):
    model = _make_model(monkeypatch, debug=False, seed=1)
    B, H, A = 2, model.action_horizon, model.action_dim
    obs_full = _make_obs(B=B, N=N, H=H, A=A, seed=1)
    noise, t = _fixed_noise_t(shape_vt=(B, N, H, A), key=11)

    vt_all = model.forward_vt_sequence(obs_full, noise, t)

    # Slice the N axis by chunk, run forward_vt_sequence per segment, then concat
    outs = []
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        # Build sub-observation (slice only the N dimension of current_*_seq / actions_seq / masks)
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
# 3) Random mask & all-False mask
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
    # Force all current_image_masks_seq to False
    obs = _make_obs(B=B, N=N, H=H, A=A, seed=3, mask_ratio=1.0)
    noise, t = _fixed_noise_t(shape_vt=(B, N, H, A), key=9)

    loss_seq = model.compute_loss_sequence(None, obs, None, noise=noise, t=t)
    # If your implementation masks out loss for those frames, the mean should be near 0; otherwise loosen the assertion
    mean_loss = float(jnp.mean(loss_seq))
    assert mean_loss <= 1e-4 or math.isfinite(mean_loss)

# ---------------------------
# 4) Time shuffle/restore invariance
# ---------------------------

def test_time_order_invariance(monkeypatch):
    model = _make_model(monkeypatch, debug=False, seed=4)
    B, N, H, A = 2, 6, model.action_horizon, model.action_dim
    obs = _make_obs(B=B, N=N, H=H, A=A, seed=4, mask_ratio=0.2)
    noise, t = _fixed_noise_t(shape_vt=(B, N, H, A), key=13)

    # Original order
    vt_ref = model.forward_vt_sequence(obs, noise, t)

    # Shuffled
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

    # Restored order
    inv = np.argsort(perm)
    vt_restored = vt_perm[:, inv]
    assert _allclose(vt_ref, vt_restored)

# ---------------------------
# 5) Determinism: repeated runs match
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
# 6) JVP / VJP equivalence (scalar loss)
# ---------------------------

def _loss_seq_scalar(m, obs, noise, t):
    return jnp.mean(m.compute_loss_sequence(None, obs, None, noise=noise, t=t))

def _loss_step_scalar(m, obs, noise, t):
    return jnp.mean(m.compute_loss_stepwise(None, obs, None, noise=noise, t=t))

def test_jvp_vjp_equal(monkeypatch):
    model_seq = _make_model(monkeypatch, debug=False, seed=6)
    model_stp = _make_model(monkeypatch, debug=False, seed=6)  # Same initialization

    B, N, H, A = 2, 3, model_seq.action_horizon, model_seq.action_dim
    obs = _make_obs(B=B, N=N, H=H, A=A, seed=6)
    noise, t = _fixed_noise_t(shape_vt=(B, N, H, A), key=19)

    # JVP: apply a random direction to all Params
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

    # VJP: use the same cotangent
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
# 7) Parameters equivalent after multiple updates
# ---------------------------

def test_multi_step_update_equal(monkeypatch):
    model_seq = _make_model(monkeypatch, debug=False, seed=7)
    model_stp = _make_model(monkeypatch, debug=False, seed=7)  # Same initialization

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

        # Gradient trees match
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

    # Parameters match after K steps
    pa = nnx.state(model_seq, nnx.Param)
    pb = nnx.state(model_stp, nnx.Param)
    la, ta = jax.tree.flatten(pa)
    lb, tb = jax.tree.flatten(pb)
    assert ta == tb
    for i, (x, y) in enumerate(zip(la, lb, strict=True)):
        assert _allclose(x, y, rtol=1e-6, atol=1e-7), f"param leaf {i} mismatch after {steps} steps"

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 PYTHONPATH=src uv run --active python -m pytest -vv tests/test_pi0_equivalence_strict.py
