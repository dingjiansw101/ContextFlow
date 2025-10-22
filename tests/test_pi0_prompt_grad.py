import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_light_incontextv14 as pi0_mod
from openpi.models.model import IMAGE_KEYS, ObservationIncontext
import openpi.shared.nnx_utils as nnx_utils


class _KVCacheLLMStub:
    """Minimal stub returned by nnx_bridge.ToNNX for Gemma."""

    def lazy_init(self, *args, **kwargs):
        return None

    def __call__(self, *args, method=None, **kwargs):
        raise RuntimeError(f"Unexpected llm call with method={method}")


class _KVCacheIMGStub:
    """Minimal stub for SigLIP image encoder that produces non-zero tokens."""

    def __init__(self, out_dim: int):
        self.out_dim = int(out_dim)

    def lazy_init(self, *args, **kwargs):
        return None

    def __call__(self, x, train=False):
        batch = x.shape[0]
        tokens = jnp.ones((batch, 1, self.out_dim), dtype=jnp.float32)
        return tokens, None


def _to_nnx_grad_stub(module_obj):
    # Gemma module (prompt/action expert)
    if hasattr(module_obj, "configs") and hasattr(module_obj, "embed_dtype"):
        return _KVCacheLLMStub()
    # SigLIP module
    if hasattr(module_obj, "width"):
        return _KVCacheIMGStub(module_obj.width)
    raise RuntimeError(f"Unexpected module for ToNNX stub: {type(module_obj)}")





def make_random_obs(*, B=2, N=2, H=2, A=2, H_img=32, W_img=32, C=3, T_inctx=2, mask_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    cams = tuple(IMAGE_KEYS)

    def rand_float(shape):
        return jnp.asarray(rng.normal(loc=0.0, scale=1.0, size=shape), dtype=jnp.float32)

    image = {nm: rand_float((B, H_img, W_img, C)) for nm in cams}
    image_mask = {nm: jnp.asarray(rng.random((B,)) > mask_ratio, dtype=jnp.bool_) for nm in cams}
    state = jnp.asarray(rng.normal(loc=0.0, scale=0.5, size=(B, A)), dtype=jnp.float32)

    current_images_seq = {nm: rand_float((B, N, H_img, W_img, C)) for nm in cams}
    current_image_masks_seq = {nm: jnp.asarray(rng.random((B, N)) > mask_ratio, dtype=jnp.bool_) for nm in cams}
    current_state_seq = jnp.asarray(rng.normal(loc=0.0, scale=0.5, size=(B, N, A)), dtype=jnp.float32)
    actions_seq = jnp.asarray(rng.normal(loc=0.0, scale=0.5, size=(B, N, H, A)), dtype=jnp.float32)

    dem_prompt_images = {nm: rand_float((B, T_inctx, H_img, W_img, C)) for nm in cams}
    dem_prompt_images_mask = {nm: jnp.asarray(rng.random((B, T_inctx)) > mask_ratio, dtype=jnp.bool_) for nm in cams}

    selected_episode = jnp.asarray(rng.integers(low=0, high=max(1, B), size=(B, 1)), dtype=jnp.int32)

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


def _install_fake_kvcache(model):
    def fake_encode(self, *, embedded_pm, positions_pm, mask_pm, deterministic):
        self._dbg_used_encode_pm_only = True
        pm_tokens = next(tok for tok in embedded_pm if tok is not None)
        prompt_mean = jnp.mean(pm_tokens, axis=1)  # [B, D]
        self._dbg_last_pm_cache = prompt_mean
        return prompt_mean

    def fake_decode(self, *, embedded_suf, positions_suf, mask_suf, pm_cache, ep_index, deterministic):
        self._dbg_used_decode_with_cache = True
        suffix_tokens = embedded_suf[1]
        prompt_vecs = pm_cache[ep_index]  # [batch, D]
        prompt_vecs = prompt_vecs[:, None, :]
        prompt_broadcast = jnp.broadcast_to(prompt_vecs, suffix_tokens.shape)
        return [None, prompt_broadcast]

    model._llm_encode_pm_only = types.MethodType(fake_encode, model)
    model._llm_decode_with_cache = types.MethodType(fake_decode, model)


def _install_blocked_decode(model):
    def fake_encode(self, *, embedded_pm, positions_pm, mask_pm, deterministic):
        self._dbg_used_encode_pm_only = True
        pm_tokens = next(tok for tok in embedded_pm if tok is not None)
        prompt_mean = jnp.mean(pm_tokens, axis=1)
        self._dbg_last_pm_cache = prompt_mean
        return prompt_mean

    def fake_decode(self, *, embedded_suf, positions_suf, mask_suf, pm_cache, ep_index, deterministic):
        self._dbg_used_decode_with_cache = True
        suffix_tokens = embedded_suf[1]
        zeros = jnp.zeros_like(suffix_tokens)
        return [None, zeros]

    model._llm_encode_pm_only = types.MethodType(fake_encode, model)
    model._llm_decode_with_cache = types.MethodType(fake_decode, model)


def build_test_model(monkeypatch):
    monkeypatch.setattr(nnx_utils, "ToNNX", None, raising=False)
    monkeypatch.setattr(pi0_mod.nnx_bridge, "ToNNX", _to_nnx_grad_stub, raising=True)
    monkeypatch.setattr(pi0_mod._model, "preprocess_observation_incontext_fused",
                         lambda rng, observation, **kwargs: observation, raising=True)

    cfg = pi0_mod.Pi0LightIncontextConfigv14(
        action_dim=2,
        action_horizon=2,
        max_token_len=1,
        prompt_expert_variant="dummy",
        action_expert_variant="dummy",
        siglip_variant="mu/16",
        avg_current_img=True,
        use_image_prompts=True,
        use_text_prompts=False,
        use_action_state_prompts=False,
        sample_frames=1,
        sample_actions=1,
        random_select=False,
        debug_fused_checks=False,
    )
    model = cfg.create(jax.random.PRNGKey(0))
    return model, cfg


def compute_prompt_grad(model, obs, noise, t):
    params = nnx.state(model, nnx.Param)

    def loss_fn(p):
        nnx.update(model, p)
        loss = model.compute_loss_sequence(None, obs, None, noise=noise, t=t)
        return jnp.sum(loss)

    grad = jax.grad(loss_fn)(params)
    prompt_kernel = grad["image_proj_promtp_expert"]["kernel"].value
    return jnp.linalg.norm(prompt_kernel)


def test_prompt_expert_grad_depends_on_kvcache(monkeypatch):
    # Model with kv-cache contributing to suffix decode
    model_kv, _ = build_test_model(monkeypatch)
    _install_fake_kvcache(model_kv)

    obs = make_random_obs(seed=0)
    B, N, H, A = obs.actions_seq.shape
    noise_key, time_key = jax.random.split(jax.random.PRNGKey(1))
    noise = jax.random.normal(noise_key, (B, N, H, A), dtype=jnp.float32)
    t = jax.random.beta(time_key, a=1.5, b=1.0, shape=(B, N), dtype=jnp.float32) * 0.999 + 0.001

    grad_norm_kv = compute_prompt_grad(model_kv, obs, noise, t)
    assert float(grad_norm_kv) > 0.0

    # Identical model but block pm_cache influence
    model_blocked, _ = build_test_model(monkeypatch)
    _install_blocked_decode(model_blocked)

    grad_norm_blocked = compute_prompt_grad(model_blocked, obs, noise, t)
    np.testing.assert_allclose(grad_norm_blocked, 0.0, atol=1e-7)

# PYTHONPATH=src uv run --active python -m pytest tests/test_pi0_prompt_grad.py

