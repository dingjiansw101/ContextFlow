# tests/test_pi0_fused_kvcache.py
import types
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import pytest

# ====== Project imports ======
from openpi.models.pi0_light_incontextv14 import Pi0LightIncontextConfigv14, Pi0LightIncontextv14, SIGLIP_OUTPUT_DIM
from openpi.models import model as _model
from openpi.shared import array_typing as at

# ---------- Utilities ----------
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
    Implements only encode_pm_only / decode_with_cache to mimic KV shapes and outputs.
    - depth=6, num_kv_heads=1, head_dim=16: small and fast
    - Outputs return the embedded inputs as-is (identity) so Pi0's later linear head can read action tokens
    """
    def __init__(self, depth=6, num_kv_heads=1, head_dim=16, width=128, embed_dtype="bfloat16"):
        self.depth = depth
        self.K = num_kv_heads
        self.H = head_dim
        self.width = width
        self.embed_dtype = embed_dtype

    # Pi0 uses the method="encode_pm_only" wrapper on encode
    def __call__(self, *args, **kwargs):
        # Compatibility with legacy combined __call__ path; not used in this test
        raise NotImplementedError

    # === Methods actually invoked by Pi0 ===
    def encode_pm_only(self, *, embedded_pm, positions_pm, mask_pm, deterministic=True):
        # embedded_pm: [pm_branch, ae_branch(None)]
        e0 = next(e for e in embedded_pm if e is not None)
        B, Tpm, D = e0.shape
        # Build per-layer KV as zeros (shape correctness only)
        k = jnp.zeros((self.depth, B, Tpm, self.K, self.H), dtype=jnp.bfloat16)
        v = jnp.zeros((self.depth, B, Tpm, self.K, self.H), dtype=jnp.bfloat16)
        return StubPMCache((k, v))

    def decode_with_cache(self, *, embedded_suf, positions_suf, mask_suf, pm_cache, ep_index, deterministic=True):
        # embedded_suf: [None, suffix_tokens]; return as-is (identity)
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
    Simulate SigLIP image encoding output patch tokens.
    As long as the last dimension equals SIGLIP_OUTPUT_DIM[variant_key], Pi0's linear projection will match shapes.
    """
    def __init__(self, siglip_variant="Ti/16"):
        key = siglip_variant.split("/")[0]
        self.out_dim = SIGLIP_OUTPUT_DIM[key]

    def __call__(self, x, train=False):
        # x: [B, H, W, C]
        B = x.shape[0]
        # Use a small patch count (e.g., 4 tokens) to speed tests
        P = 4
        tokens = jnp.zeros((B, P, self.out_dim), dtype=jnp.bfloat16)
        return tokens, None

# ---------- Mock: fused preprocessing ----------
@dataclasses.dataclass
class _FusedObs:
    # Fields accessed by the Pi0 fused path
    state: jnp.ndarray                # [B,A]
    current_state_seq: jnp.ndarray    # [B,N,A]
    current_images_seq: dict          # name -> [B,N,H,W,C]
    current_image_masks_seq: dict     # name -> [B,N] bool
    actions_seq: jnp.ndarray          # [B,N,H,A]
    # Needed for midfix when image prompts are enabled
    incontext_images: dict            # name -> [B,T,H,W,C]
    incontext_image_masks: dict       # name -> [B,T] bool
    # embed_midfix would check the following fields, but text/state/action prompts are disabled so we omit them

def _mock_preprocess_observation_incontext_fused(rng, observation, train=False):
    # In tests the observation is the constructed _FusedObs; return as-is
    return observation

# ---------- Build a minimal runnable Pi0 instance ----------
def make_minimal_pi0(B=2, N=3, H=4, A=6, Tpm=5, siglip_variant="Ti/16"):
    # Config: shrink width/depth to speed tests (Gemma is stubbed; only affects Linear sizes)
    cfg = Pi0LightIncontextConfigv14(
        action_dim=A,
        action_horizon=H,
        sample_frames=N,          # Only affects inputs_spec; we actually use mocked fused obs
        max_token_len=8,
        siglip_variant=siglip_variant,
        pool_type="none",
        debug_fused_checks=True,  # Enable debug assertions to help catch issues
    )
    model = cfg.create(jax.random.PRNGKey(0))

    # Disable text/state/action prompts that complicate midfix; keep image prompts to satisfy embed_midfix non-empty assertion
    model.use_text_prompts = False
    model.use_action_state_prompts = False
    model.use_image_prompts = True
    model.avg_current_img = True
    model.causal_attention = False

    # Inject stub LLM and IMG
    model.PaliGemma["llm"] = StubLLM(depth=6, num_kv_heads=1, head_dim=16, width=128, embed_dtype="bfloat16")
    model.PaliGemma["img"] = StubIMG(siglip_variant)

    return model, cfg

# ---------- Build a fused observation (with in-context images to drive midfix) ----------
def make_fused_observation(B=2, N=3, H=4, A=6, Tpm=5, cams=("base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb")):
    H_img, W_img, C = 64, 64, 3
    # Current sequences (used for suffix)
    current_images_seq = {name: jnp.zeros((B, N, H_img, W_img, C), dtype=jnp.float32) for name in cams}
    current_image_masks_seq = {name: jnp.ones((B, N), dtype=jnp.bool_) for name in cams}
    current_state_seq = jnp.zeros((B, N, A), dtype=jnp.float32)
    actions_seq = jnp.zeros((B, N, H, A), dtype=jnp.float32)

    # In-context images (for midfix)
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
    return obs, actions_seq  # actions_seq only used for shape inside compute_loss (already wrapped in obs)

# ===================== Test cases =====================

def test_make_attn_mask_shapes_and_values():
    from openpi.models.pi0_light_incontextv14 import make_attn_mask
    B, T = 2, 6
    input_mask = jnp.array([[1,1,1,1,1,1],[1,1,1,1,1,1]], dtype=bool)
    # Build a "prefix + causal" ar_mask: first three shared, last three causal
    mask_ar = jnp.array([0,0,0,1,1,1], dtype=bool)
    attn = make_attn_mask(input_mask, mask_ar)
    assert attn.shape == (B, T, T)
    # Check prefix tokens see each other; suffix is causal
    assert bool(attn[0, 1, 0]) is True   # 1 can see 0 (same prefix)
    assert bool(attn[0, 4, 5]) is False  # Causality forbids seeing the future

def test_fused_compute_loss_calls_kv_paths_and_shapes(monkeypatch):
    # 1) Build minimal model + mock preprocessing
    model, cfg = make_minimal_pi0(B=2, N=3, H=4, A=6, Tpm=5)
    monkeypatch.setattr(_model, "preprocess_observation_incontext_fused", _mock_preprocess_observation_incontext_fused)

    # 2) Build fused observation and actions
    obs, acts_seq = make_fused_observation(B=2, N=3, H=4, A=6, Tpm=5)

    # 3) Run compute_loss (training mode)
    rng = jax.random.PRNGKey(42)
    loss = model.compute_loss(rng, obs, jnp.zeros((2, 4, 6), dtype=jnp.float32), train=True)

    # 4) Assert shape is [B,N,H]
    assert loss.shape == (2, 3, 4)

    # 5) Assert encode_pm_only and decode_with_cache were used (debug flags)
    assert getattr(model, "_dbg_used_encode_pm_only", False), "encode_pm_only was not triggered"
    assert getattr(model, "_dbg_used_decode_with_cache", False), "decode_with_cache was not triggered"

def test_positions_monotonicity_and_suffix_tail_mask(monkeypatch):
    # Goal: trigger and pass Pi0's two internal assertions
    #  - _assert_monotonic_positions: midfix/suffix positions must be strictly monotonic based on mask (allow global shift)
    #  - _assert_block_ar_mask: suffix tail of length H should have AR pattern [True, False*(H-1)]
    model, cfg = make_minimal_pi0(B=1, N=2, H=3, A=4, Tpm=4)
    monkeypatch.setattr(_model, "preprocess_observation_incontext_fused", _mock_preprocess_observation_incontext_fused)
    obs, _ = make_fused_observation(B=1, N=2, H=3, A=4, Tpm=4)
    rng = jax.random.PRNGKey(0)

    # Passing means no exception (internal assertions active with debug_fused_checks=True)
    _ = model.compute_loss(rng, obs, jnp.zeros((1, 3, 4), dtype=jnp.float32), train=True)

def test_deterministic_flag_flow(monkeypatch):
    """
    Training (train=True) should set suffix deterministic to False; inference should set it to True.
    We indirectly verify by whether Pi0's wrapper functions get called (encode_pm_only always True).
    """
    model, cfg = make_minimal_pi0(B=1, N=2, H=2, A=3, Tpm=3)
    monkeypatch.setattr(_model, "preprocess_observation_incontext_fused", _mock_preprocess_observation_incontext_fused)
    obs, _ = make_fused_observation(B=1, N=2, H=2, A=3, Tpm=3)
    rng = jax.random.PRNGKey(0)

    # Training: calls decode_with_cache(deterministic=False)
    _ = model.compute_loss(rng, obs, jnp.zeros((1, 2, 3), dtype=jnp.float32), train=True)
    assert model._dbg_used_encode_pm_only and model._dbg_used_decode_with_cache

    # Inference: also calls decode_with_cache but deterministic=True (flag not directly exposed here,
    # just ensure inference path reuses pm_cache without exceptions)
    _ = model.compute_loss(rng, obs, jnp.zeros((1, 2, 3), dtype=jnp.float32), train=False)
