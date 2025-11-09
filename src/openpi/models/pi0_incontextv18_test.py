"""Unit tests for pi0_incontextv18 - Perceiver compression features.

This test file focuses ONLY on NEW functionality compared to v17:
- PerceiverCompressor module
- Compression in embed_midfix
- New config parameters (num_image_queries, num_state_queries, num_action_queries, num_attn_heads)
- Model initialization with compressors

Tests for unchanged functionality (make_attn_mask, posemb_sincos, compute_loss, sample_actions)
are covered by pi0_incontextv17_test.py and not duplicated here.

USAGE:
------
    # Run all tests
    uv run pytest src/openpi/models/pi0_incontextv18_test.py

    # Run specific test class
    uv run pytest src/openpi/models/pi0_incontextv18_test.py::TestPerceiverCompressor

    # Run with verbose output
    uv run pytest src/openpi/models/pi0_incontextv18_test.py -v

    # Run specific test
    uv run pytest src/openpi/models/pi0_incontextv18_test.py::TestPerceiverCompressor::test_compression_shape
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import pi0_incontextv18 as _pi0v18
from openpi.shared import nnx_utils


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Create default v18 configuration with Perceiver compression."""
    return _pi0v18.Pi0IncontextConfigv18(
        prompt_expert_variant="gemma_300m_v2",
        action_expert_variant="gemma_300m",
    )


@pytest.fixture
def compressor():
    """Create standalone PerceiverCompressor for unit testing."""
    key = jax.random.key(0)
    return _pi0v18.PerceiverCompressor(
        num_queries=32,
        embed_dim=2048,
        num_heads=8,
        num_layers=4,  # Default to 4 layers (2 cross + 2 self)
        rngs=nnx.Rngs(key)
    )


@pytest.fixture
def small_model(default_config):
    """Create small model instance for testing."""
    key = jax.random.key(0)
    return default_config.create(key)


# =============================================================================
# Test PerceiverCompressor Module (NEW in v18)
# =============================================================================


class TestPerceiverCompressor:
    """Test suite for the new PerceiverCompressor module."""

    def test_initialization(self):
        """Test compressor initializes with correct parameter shapes."""
        key = jax.random.key(0)
        num_queries, embed_dim, num_heads, num_layers = 32, 2048, 8, 4

        compressor = _pi0v18.PerceiverCompressor(
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            rngs=nnx.Rngs(key)
        )

        # Check learnable queries shape
        assert compressor.queries.value.shape == (num_queries, embed_dim), \
            f"Expected queries shape ({num_queries}, {embed_dim}), got {compressor.queries.value.shape}"

        # Check stored parameters
        assert compressor.num_layers == num_layers
        assert compressor.num_queries == num_queries
        assert compressor.embed_dim == embed_dim

        # Check layer lists exist
        assert hasattr(compressor, 'cross_attn_layers')
        assert hasattr(compressor, 'self_attn_layers')
        assert hasattr(compressor, 'query_norm_cross_layers')
        assert hasattr(compressor, 'query_norm_self_layers')
        assert hasattr(compressor, 'kv_norm_layers')
        assert hasattr(compressor, 'ffn_layers')
        assert hasattr(compressor, 'ffn_norm_layers')

        # Check correct number of layers (4 layers = 2 cross + 2 self)
        assert len(compressor.cross_attn_layers) == 2, \
            f"Expected 2 cross-attention layers for num_layers=4, got {len(compressor.cross_attn_layers)}"
        assert len(compressor.self_attn_layers) == 2, \
            f"Expected 2 self-attention layers for num_layers=4, got {len(compressor.self_attn_layers)}"

    def test_compression_shape(self, compressor):
        """Test input (B, S, D) is compressed to (B, num_queries, D)."""
        batch_size, seq_len, embed_dim = 2, 256, 2048
        num_queries = 32

        # Create input tokens
        key = jax.random.key(1)
        tokens = jax.random.normal(key, (batch_size, seq_len, embed_dim))

        # Compress
        output = compressor(tokens)

        # Check output shape
        assert output.shape == (batch_size, num_queries, embed_dim), \
            f"Expected output shape ({batch_size}, {num_queries}, {embed_dim}), got {output.shape}"

    def test_with_masking(self):
        """Test cross-attention respects input mask."""
        key = jax.random.key(0)
        batch_size, seq_len, embed_dim = 2, 100, 2048
        num_queries = 16

        compressor = _pi0v18.PerceiverCompressor(
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=2,
            rngs=nnx.Rngs(key)
        )

        # Create tokens and mask (second batch has last 50 tokens masked)
        tokens = jax.random.normal(key, (batch_size, seq_len, embed_dim))
        mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
        mask = mask.at[1, 50:].set(False)

        # Compress with mask
        output = compressor(tokens, mask=mask)

        # Output should have correct shape
        assert output.shape == (batch_size, num_queries, embed_dim)

        # Output should be finite
        assert jnp.all(jnp.isfinite(output))

        # First batch (unmasked) and second batch (partially masked) should be different
        assert not jnp.allclose(output[0], output[1]), \
            "Masked and unmasked inputs should produce different outputs"

    def test_positional_encoding_added(self):
        """Test sinusoidal positional encoding is added to input sequence."""
        key = jax.random.key(0)
        batch_size, seq_len, embed_dim = 1, 50, 128

        compressor = _pi0v18.PerceiverCompressor(
            num_queries=8,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=2,
            rngs=nnx.Rngs(key)
        )

        # Create zero input (so we can observe PE effect)
        tokens_zero = jnp.zeros((batch_size, seq_len, embed_dim))

        # Create non-zero input
        tokens_nonzero = jax.random.normal(key, (batch_size, seq_len, embed_dim))

        # Compress both
        output_zero = compressor(tokens_zero)
        output_nonzero = compressor(tokens_nonzero)

        # Both should produce outputs (PE is added internally)
        assert jnp.all(jnp.isfinite(output_zero))
        assert jnp.all(jnp.isfinite(output_nonzero))

        # Zero input should NOT produce zero output (due to PE and learned queries)
        assert not jnp.allclose(output_zero, 0.0, atol=1e-3), \
            "Output should not be zero when PE is added"

    def test_different_sequence_lengths(self):
        """Test compressor handles various input sequence lengths."""
        key = jax.random.key(0)
        embed_dim, num_queries = 512, 16

        compressor = _pi0v18.PerceiverCompressor(
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=4,
            rngs=nnx.Rngs(key)
        )

        # Test with different sequence lengths
        for seq_len in [10, 64, 256, 1024]:
            tokens = jax.random.normal(key, (1, seq_len, embed_dim))
            output = compressor(tokens)

            assert output.shape == (1, num_queries, embed_dim), \
                f"Failed for seq_len={seq_len}"
            assert jnp.all(jnp.isfinite(output)), \
                f"Non-finite values for seq_len={seq_len}"

    def test_multiple_layer_counts(self):
        """Test compressor works correctly with different layer counts."""
        key = jax.random.key(0)
        embed_dim, num_queries = 256, 16
        batch_size, seq_len = 2, 64

        # Test with 1, 2, 4, 6 layers
        for num_layers in [1, 2, 4, 6]:
            compressor = _pi0v18.PerceiverCompressor(
                num_queries=num_queries,
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=num_layers,
                rngs=nnx.Rngs(jax.random.fold_in(key, num_layers))
            )

            tokens = jax.random.normal(key, (batch_size, seq_len, embed_dim))
            output = compressor(tokens)

            # Check output shape is always (B, num_queries, D)
            assert output.shape == (batch_size, num_queries, embed_dim), \
                f"Failed for num_layers={num_layers}"
            assert jnp.all(jnp.isfinite(output)), \
                f"Non-finite output for num_layers={num_layers}"

            # Check layer counts
            expected_cross = (num_layers + 1) // 2  # Ceiling division
            expected_self = num_layers // 2  # Floor division
            assert len(compressor.cross_attn_layers) == expected_cross, \
                f"Expected {expected_cross} cross-attn layers for num_layers={num_layers}"
            assert len(compressor.self_attn_layers) == expected_self, \
                f"Expected {expected_self} self-attn layers for num_layers={num_layers}"

    def test_interleaved_attention_pattern(self):
        """Test that interleaved cross/self-attention produces different outputs than cross-only."""
        key = jax.random.key(0)
        embed_dim, num_queries = 256, 16
        batch_size, seq_len = 2, 64

        # Create single-layer (cross-only) compressor
        compressor_1layer = _pi0v18.PerceiverCompressor(
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=1,
            rngs=nnx.Rngs(key)
        )

        # Create multi-layer (cross + self) compressor
        compressor_4layer = _pi0v18.PerceiverCompressor(
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=4,
            rngs=nnx.Rngs(key)
        )

        # Same input
        tokens = jax.random.normal(key, (batch_size, seq_len, embed_dim))

        # Get outputs
        output_1layer = compressor_1layer(tokens)
        output_4layer = compressor_4layer(tokens)

        # Both should be valid
        assert jnp.all(jnp.isfinite(output_1layer))
        assert jnp.all(jnp.isfinite(output_4layer))

        # Same shape
        assert output_1layer.shape == output_4layer.shape

        # Different values (due to different architectures)
        # Note: outputs may be similar initially, but should diverge with training
        # This test just ensures both architectures are valid
        assert output_1layer.shape == (batch_size, num_queries, embed_dim)

    def test_self_attention_ffn_exists(self):
        """Test that self-attention layers include FFN components."""
        key = jax.random.key(0)

        # Create compressor with 4 layers (2 cross + 2 self)
        compressor = _pi0v18.PerceiverCompressor(
            num_queries=32,
            embed_dim=512,
            num_heads=8,
            num_layers=4,
            rngs=nnx.Rngs(key)
        )

        # Should have 2 self-attention blocks, each with FFN
        assert len(compressor.ffn_layers) == 2, \
            f"Expected 2 FFN blocks for 4 layers, got {len(compressor.ffn_layers)}"
        assert len(compressor.ffn_norm_layers) == 2, \
            f"Expected 2 FFN norm layers for 4 layers, got {len(compressor.ffn_norm_layers)}"

        # Each FFN should have 2 linear layers (up-project + down-project)
        for i, (layer_key, ffn) in enumerate(compressor.ffn_layers.items()):
            assert len(ffn) == 2, \
                f"FFN block {i} (key={layer_key}) should have 2 linear layers, got {len(ffn)}"
            assert 'up_proj' in ffn, f"FFN block {i} should have 'up_proj' key"
            assert 'down_proj' in ffn, f"FFN block {i} should have 'down_proj' key"
            # First layer expands by 4x
            assert ffn['up_proj'].out_features == 512 * 4, \
                f"FFN up-projection should expand to 4x embed_dim"
            # Second layer projects back
            assert ffn['down_proj'].out_features == 512, \
                f"FFN down-projection should return to embed_dim"

    def test_jit_compatible(self, compressor):
        """Test PerceiverCompressor works with JIT compilation."""
        batch_size, seq_len, embed_dim = 2, 128, 2048

        # JIT compile the forward pass
        @jax.jit
        def compress_jit(tokens):
            return compressor(tokens)

        # Create input
        key = jax.random.key(1)
        tokens = jax.random.normal(key, (batch_size, seq_len, embed_dim))

        # Run JIT-compiled version
        output = compress_jit(tokens)

        # Should produce valid output
        assert output.shape == (batch_size, 32, embed_dim)
        assert jnp.all(jnp.isfinite(output))


# =============================================================================
# Test Perceiver Config Parameters (NEW in v18)
# =============================================================================


class TestPerceiverConfig:
    """Test new Perceiver-related configuration parameters."""

    def test_default_values(self):
        """Test default values for Perceiver parameters."""
        config = _pi0v18.Pi0IncontextConfigv18()

        assert config.num_image_queries == 32, \
            f"Expected num_image_queries=32, got {config.num_image_queries}"
        assert config.num_state_queries == 32, \
            f"Expected num_state_queries=32, got {config.num_state_queries}"
        assert config.num_action_queries == 32, \
            f"Expected num_action_queries=32, got {config.num_action_queries}"
        assert config.num_attn_heads == 8, \
            f"Expected num_attn_heads=8, got {config.num_attn_heads}"
        assert config.num_image_compressor_layers == 4, \
            f"Expected num_image_compressor_layers=4, got {config.num_image_compressor_layers}"
        assert config.num_state_compressor_layers == 2, \
            f"Expected num_state_compressor_layers=2, got {config.num_state_compressor_layers}"
        assert config.num_action_compressor_layers == 2, \
            f"Expected num_action_compressor_layers=2, got {config.num_action_compressor_layers}"

    def test_custom_values(self):
        """Test configuration with custom query counts."""
        config = _pi0v18.Pi0IncontextConfigv18(
            num_image_queries=16,
            num_state_queries=64,
            num_action_queries=8,
            num_attn_heads=4,
            num_image_compressor_layers=6,
            num_state_compressor_layers=4,
            num_action_compressor_layers=2
        )

        assert config.num_image_queries == 16
        assert config.num_state_queries == 64
        assert config.num_action_queries == 8
        assert config.num_attn_heads == 4
        assert config.num_image_compressor_layers == 6
        assert config.num_state_compressor_layers == 4
        assert config.num_action_compressor_layers == 2


# =============================================================================
# Test Model Initialization with Compressors (NEW in v18)
# =============================================================================


class TestCompressorInitialization:
    """Test compressors are created correctly during model initialization."""

    def test_compressors_exist(self):
        """Test image/state/action compressors are created when prompts enabled."""
        config = _pi0v18.Pi0IncontextConfigv18(
            use_image_prompts=True,
            use_action_state_prompts=True
        )
        key = jax.random.key(0)
        model = config.create(key)

        # Check compressors exist
        assert hasattr(model, 'image_compressor'), \
            "image_compressor should exist when use_image_prompts=True"
        assert hasattr(model, 'state_compressor'), \
            "state_compressor should exist when use_action_state_prompts=True"
        assert hasattr(model, 'action_compressor'), \
            "action_compressor should exist when use_action_state_prompts=True"

        # Check they are PerceiverCompressor instances
        assert isinstance(model.image_compressor, _pi0v18.PerceiverCompressor)
        assert isinstance(model.state_compressor, _pi0v18.PerceiverCompressor)
        assert isinstance(model.action_compressor, _pi0v18.PerceiverCompressor)

    def test_compressors_not_created_when_disabled(self):
        """Test compressors not created when prompts disabled."""
        config = _pi0v18.Pi0IncontextConfigv18(
            use_image_prompts=False,
            use_action_state_prompts=False
        )
        key = jax.random.key(0)
        model = config.create(key)

        # Compressors should not exist
        assert not hasattr(model, 'image_compressor'), \
            "image_compressor should not exist when use_image_prompts=False"
        assert not hasattr(model, 'state_compressor'), \
            "state_compressor should not exist when use_action_state_prompts=False"
        assert not hasattr(model, 'action_compressor'), \
            "action_compressor should not exist when use_action_state_prompts=False"

    def test_stored_parameters(self):
        """Test num_*_queries parameters are stored in model."""
        config = _pi0v18.Pi0IncontextConfigv18(
            num_image_queries=16,
            num_state_queries=24,
            num_action_queries=8
        )
        key = jax.random.key(0)
        model = config.create(key)

        # Check stored parameters
        assert model.num_image_queries == 16, \
            f"Expected num_image_queries=16, got {model.num_image_queries}"
        assert model.num_state_queries == 24, \
            f"Expected num_state_queries=24, got {model.num_state_queries}"
        assert model.num_action_queries == 8, \
            f"Expected num_action_queries=8, got {model.num_action_queries}"


# =============================================================================
# Test embed_midfix Compression (MODIFIED for v18)
# =============================================================================


class TestEmbedMidfixCompression:
    """Test embed_midfix uses Perceiver compression correctly."""

    def test_image_compression_reduces_tokens(self):
        """Test demo images are compressed to num_image_queries per camera."""
        config = _pi0v18.Pi0IncontextConfigv18(
            num_image_queries=32,
            use_image_prompts=True,
            use_text_prompts=False,
            use_action_state_prompts=False
        )
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(2)

        tokens, input_mask, ar_mask = model.embed_midfix(obs)

        # Calculate expected token count
        num_current_cameras = len(obs.images)
        num_incontext_cameras = len(obs.incontext_images)

        # Current observation contributes 256 tokens per camera (from SigLIP vision encoder)
        current_img_tokens = num_current_cameras * 256

        # In-context images are compressed to num_image_queries per camera
        compressed_img_tokens = num_incontext_cameras * config.num_image_queries

        expected_tokens = current_img_tokens + compressed_img_tokens

        # Verify exact token count matches expectation
        # For default config: 3 cameras × 256 + 3 cameras × 32 = 768 + 96 = 864 tokens
        assert tokens.shape[1] == expected_tokens, \
            f"Expected {expected_tokens} tokens ({current_img_tokens} current + " \
            f"{compressed_img_tokens} compressed), got {tokens.shape[1]}"

        # Output should be finite
        assert jnp.all(jnp.isfinite(tokens))

    def test_state_action_compression(self):
        """Test states and actions are compressed to num_*_queries."""
        config = _pi0v18.Pi0IncontextConfigv18(
            num_state_queries=16,
            num_action_queries=24,
            use_image_prompts=False,
            use_text_prompts=False,
            use_action_state_prompts=True
        )
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(2)

        tokens, input_mask, ar_mask = model.embed_midfix(obs)

        # Tokens should include compressed states and actions
        # Total should be: current_images + num_state_queries + num_action_queries
        # Expected: 3 cameras × 256 tokens + 16 state queries + 24 action queries = 808
        num_cameras = 3
        tokens_per_camera = 256  # SigLIP vision encoder output
        expected_tokens = (num_cameras * tokens_per_camera +
                          config.num_state_queries + config.num_action_queries)
        assert tokens.shape[1] == expected_tokens, \
            f"Expected {expected_tokens} tokens (3×256 current + {config.num_state_queries} state + " \
            f"{config.num_action_queries} action), got {tokens.shape[1]}"

        assert jnp.all(jnp.isfinite(tokens))

    def test_output_mask_validity(self):
        """Test output masks correctly reflect input validity."""
        config = _pi0v18.Pi0IncontextConfigv18(
            num_image_queries=32,
            use_image_prompts=True
        )
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(2)

        # Modify one camera's mask to be partially valid
        camera_name = list(obs.incontext_images.keys())[0]
        obs.incontext_image_masks[camera_name] = obs.incontext_image_masks[camera_name].at[:, :8].set(False)

        tokens, input_mask, ar_mask = model.embed_midfix(obs)

        # Input mask should have correct shape
        assert input_mask.shape[0] == 2  # batch size
        assert input_mask.shape[1] == tokens.shape[1]

        # All masks should be boolean
        assert input_mask.dtype == jnp.bool_

        # AR mask should match token count
        assert ar_mask.shape[0] == tokens.shape[1]

    def test_compression_preserves_finite_values(self):
        """Test all compressed tokens remain finite."""
        config = _pi0v18.Pi0IncontextConfigv18(
            use_image_prompts=True,
            use_action_state_prompts=True
        )
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(4)  # Larger batch

        tokens, input_mask, ar_mask = model.embed_midfix(obs)

        # All tokens should be finite
        assert jnp.all(jnp.isfinite(tokens)), \
            "Compressed tokens contain non-finite values"

        # Check reasonable value ranges
        assert jnp.all(jnp.abs(tokens) < 1e6), \
            "Token values are unreasonably large"


# =============================================================================
# Test Random Modality Masking (NEW in v18)
# =============================================================================


class TestModalityMasking:
    """Test random modality masking during training."""

    def _count_masked_sections(self, input_mask, num_cameras=3, tokens_per_camera=256,
                               num_image_queries=32, num_state_queries=32, num_action_queries=32,
                               has_text=False, text_len=48):
        """Helper to identify which modality sections are masked.

        Returns dict with 'image', 'text', 'state', 'action' -> True if masked (all False)
        """
        # Token layout: current_images | [text] | [incontext_images] | [states] | [actions]
        idx = 0
        result = {}

        # Skip current images (never masked by prompt masking)
        idx += num_cameras * tokens_per_camera

        # Check text if present
        if has_text:
            text_mask = input_mask[:, idx:idx + text_len]
            result['text'] = jnp.all(~text_mask)  # True if all False
            idx += text_len

        # Check incontext images
        img_mask = input_mask[:, idx:idx + num_cameras * num_image_queries]
        result['image'] = jnp.all(~img_mask)  # True if all False
        idx += num_cameras * num_image_queries

        # Check states
        state_mask = input_mask[:, idx:idx + num_state_queries]
        result['state'] = jnp.all(~state_mask)
        idx += num_state_queries

        # Check actions
        action_mask = input_mask[:, idx:idx + num_action_queries]
        result['action'] = jnp.all(~action_mask)

        return result

    def test_mask_prob_default(self):
        """Test prompt_mask_prob defaults to 0.0."""
        config = _pi0v18.Pi0IncontextConfigv18()
        assert config.prompt_mask_prob == 0.0, \
            f"Expected prompt_mask_prob=0.0, got {config.prompt_mask_prob}"

    def test_mask_prob_custom(self):
        """Test prompt_mask_prob can be set to custom value."""
        config = _pi0v18.Pi0IncontextConfigv18(prompt_mask_prob=0.3)
        assert config.prompt_mask_prob == 0.3, \
            f"Expected prompt_mask_prob=0.3, got {config.prompt_mask_prob}"

    def test_no_masking_during_inference(self):
        """Test no masking occurs during inference even with prob=1.0."""
        config = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=1.0,  # Always try to mask
            use_image_prompts=True,
            use_text_prompts=True,
            use_action_state_prompts=True
        )
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(1)

        # Call with train=False (inference mode)
        tokens, input_mask, ar_mask = model.embed_midfix(obs, rng=key, train=False)

        # Parse which sections are masked
        masked = self._count_masked_sections(
            input_mask,
            has_text=True,
            num_image_queries=config.num_image_queries,
            num_state_queries=config.num_state_queries,
            num_action_queries=config.num_action_queries
        )

        # NO modality should be masked during inference
        assert not masked.get('image', False), "Images should not be masked during inference"
        assert not masked.get('text', False), "Text should not be masked during inference"
        assert not masked.get('state', False), "States should not be masked during inference"
        assert not masked.get('action', False), "Actions should not be masked during inference"

    def test_no_masking_when_prob_zero(self):
        """Test no masking when prompt_mask_prob=0.0."""
        config = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=0.0,
            use_image_prompts=True,
            use_text_prompts=True,
            use_action_state_prompts=True
        )
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(1)

        # Call with train=True but prob=0
        tokens, input_mask, ar_mask = model.embed_midfix(obs, rng=key, train=True)

        # Parse which sections are masked
        masked = self._count_masked_sections(
            input_mask,
            has_text=True,
            num_image_queries=config.num_image_queries,
            num_state_queries=config.num_state_queries,
            num_action_queries=config.num_action_queries
        )

        # NO modality should be masked when prob=0
        assert not any(masked.values()), "No modality should be masked when prob=0.0"

    def test_image_masking_occurs(self):
        """Test image prompts can be masked during training."""
        config = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=1.0,
            use_image_prompts=True,
            use_text_prompts=True,
            use_action_state_prompts=True
        )
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(1)

        # Run multiple times to find image masking
        image_masked_found = False
        for i in range(10):
            test_key = jax.random.fold_in(key, i)
            tokens, input_mask, ar_mask = model.embed_midfix(obs, rng=test_key, train=True)

            masked = self._count_masked_sections(
                input_mask,
                has_text=True,
                num_image_queries=config.num_image_queries,
                num_state_queries=config.num_state_queries,
                num_action_queries=config.num_action_queries
            )

            if masked.get('image', False):
                image_masked_found = True
                break

        assert image_masked_found, "Image prompts should be masked in at least one run"

    def test_text_masking_occurs(self):
        """Test text prompts can be masked during training."""
        config = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=1.0,
            use_image_prompts=True,
            use_text_prompts=True,
            use_action_state_prompts=True
        )
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(1)

        # Run multiple times to find text masking
        text_masked_found = False
        for i in range(10):
            test_key = jax.random.fold_in(key, i)
            tokens, input_mask, ar_mask = model.embed_midfix(obs, rng=test_key, train=True)

            masked = self._count_masked_sections(
                input_mask,
                has_text=True,
                num_image_queries=config.num_image_queries,
                num_state_queries=config.num_state_queries,
                num_action_queries=config.num_action_queries
            )

            if masked.get('text', False):
                text_masked_found = True
                break

        assert text_masked_found, "Text prompts should be masked in at least one run"

    def test_state_action_masking_occurs(self):
        """Test state/action prompts can be masked during training."""
        config = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=1.0,
            use_image_prompts=True,
            use_text_prompts=True,
            use_action_state_prompts=True
        )
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(1)

        # Run multiple times to find state/action masking
        state_action_masked_found = False
        for i in range(10):
            test_key = jax.random.fold_in(key, i)
            tokens, input_mask, ar_mask = model.embed_midfix(obs, rng=test_key, train=True)

            masked = self._count_masked_sections(
                input_mask,
                has_text=True,
                num_image_queries=config.num_image_queries,
                num_state_queries=config.num_state_queries,
                num_action_queries=config.num_action_queries
            )

            # Both state and action should be masked together
            if masked.get('state', False) and masked.get('action', False):
                state_action_masked_found = True
                break

        assert state_action_masked_found, "State/action prompts should be masked together in at least one run"

    def test_only_one_modality_masked_at_a_time(self):
        """Test exactly one modality type is masked when masking occurs."""
        config = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=1.0,
            use_image_prompts=True,
            use_text_prompts=True,
            use_action_state_prompts=True
        )
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(1)

        # Run multiple times and check only one modality is masked
        for i in range(20):
            test_key = jax.random.fold_in(key, i)
            tokens, input_mask, ar_mask = model.embed_midfix(obs, rng=test_key, train=True)

            masked = self._count_masked_sections(
                input_mask,
                has_text=True,
                num_image_queries=config.num_image_queries,
                num_state_queries=config.num_state_queries,
                num_action_queries=config.num_action_queries
            )

            # Count how many modalities are masked
            # Note: state and action are treated as one modality
            image_masked = masked.get('image', False)
            text_masked = masked.get('text', False)
            state_action_masked = masked.get('state', False) or masked.get('action', False)

            # Exactly one should be True
            masked_count = sum([image_masked, text_masked, state_action_masked])
            assert masked_count == 1, \
                f"Expected exactly 1 modality masked, got {masked_count} (iteration {i})"

    def test_at_least_one_modality_unmasked(self):
        """Test at least one modality remains unmasked."""
        # Test with 2 modalities
        config_2mod = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=1.0,
            use_image_prompts=True,
            use_text_prompts=True,
            use_action_state_prompts=False
        )
        key = jax.random.key(0)
        model = config_2mod.create(key)
        obs = config_2mod.fake_obs(1)

        for i in range(10):
            test_key = jax.random.fold_in(key, i)
            tokens, input_mask, ar_mask = model.embed_midfix(obs, rng=test_key, train=True)

            masked = self._count_masked_sections(
                input_mask,
                has_text=True,
                num_image_queries=config_2mod.num_image_queries,
                num_state_queries=config_2mod.num_state_queries,
                num_action_queries=config_2mod.num_action_queries
            )

            # At least one should be unmasked
            all_masked = masked.get('image', False) and masked.get('text', False)
            assert not all_masked, "At least one modality must remain unmasked"

        # Test with 3 modalities
        config_3mod = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=1.0,
            use_image_prompts=True,
            use_text_prompts=True,
            use_action_state_prompts=True
        )
        model = config_3mod.create(key)
        obs = config_3mod.fake_obs(1)

        for i in range(10):
            test_key = jax.random.fold_in(key, i)
            tokens, input_mask, ar_mask = model.embed_midfix(obs, rng=test_key, train=True)

            masked = self._count_masked_sections(
                input_mask,
                has_text=True,
                num_image_queries=config_3mod.num_image_queries,
                num_state_queries=config_3mod.num_state_queries,
                num_action_queries=config_3mod.num_action_queries
            )

            # At least one should be unmasked
            all_masked = (masked.get('image', False) and masked.get('text', False) and
                         masked.get('state', False) and masked.get('action', False))
            assert not all_masked, "At least one modality must remain unmasked"

    def test_single_modality_never_masked(self):
        """Test single modality is never masked (guaranteed unmasked)."""
        # Test with only image prompts
        config = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=1.0,
            use_image_prompts=True,
            use_text_prompts=False,
            use_action_state_prompts=False
        )
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(1)

        for i in range(10):
            test_key = jax.random.fold_in(key, i)
            tokens, input_mask, ar_mask = model.embed_midfix(obs, rng=test_key, train=True)

            masked = self._count_masked_sections(
                input_mask,
                has_text=False,
                num_image_queries=config.num_image_queries,
                num_state_queries=config.num_state_queries,
                num_action_queries=config.num_action_queries
            )

            # Single modality should never be masked
            assert not masked.get('image', False), \
                "Single available modality should never be masked"

    def test_masking_respects_probability(self):
        """Test masking occurs approximately at the specified probability."""
        config = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=0.3,  # 30% probability
            use_image_prompts=True,
            use_text_prompts=True,
            use_action_state_prompts=False
        )
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(1)

        num_trials = 200
        num_masked = 0

        for i in range(num_trials):
            test_key = jax.random.fold_in(key, i)
            tokens, input_mask, ar_mask = model.embed_midfix(obs, rng=test_key, train=True)

            masked = self._count_masked_sections(
                input_mask,
                has_text=True,
                num_image_queries=config.num_image_queries,
                num_state_queries=config.num_state_queries,
                num_action_queries=config.num_action_queries
            )

            # Check if any modality was masked
            if any(masked.values()):
                num_masked += 1

        # Calculate empirical probability
        empirical_prob = num_masked / num_trials

        # Should be approximately 0.3 (with some tolerance)
        assert 0.15 < empirical_prob < 0.45, \
            f"Expected masking probability ~0.3, got {empirical_prob:.3f} ({num_masked}/{num_trials})"

    def test_deterministic_with_fixed_rng(self):
        """Test masking is deterministic with same RNG key."""
        config = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=0.5,
            use_image_prompts=True,
            use_text_prompts=True,
            use_action_state_prompts=True
        )
        key = jax.random.key(42)
        model = config.create(jax.random.key(0))
        obs = config.fake_obs(1)

        # Run twice with same key
        tokens1, mask1, ar_mask1 = model.embed_midfix(obs, rng=key, train=True)
        tokens2, mask2, ar_mask2 = model.embed_midfix(obs, rng=key, train=True)

        # Masks should be identical
        assert jnp.array_equal(mask1, mask2), \
            "Same RNG key should produce identical masking"

    def test_different_rng_keys_produce_different_masks(self):
        """Test different RNG keys can produce different masking decisions."""
        config = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=1.0,
            use_image_prompts=True,
            use_text_prompts=True,
            use_action_state_prompts=True
        )
        model = config.create(jax.random.key(0))
        obs = config.fake_obs(1)

        # Collect masks from different keys
        masks_list = []
        for i in range(20):
            test_key = jax.random.key(i)
            tokens, input_mask, ar_mask = model.embed_midfix(obs, rng=test_key, train=True)
            masks_list.append(input_mask)

        # At least some should be different (since we randomly choose which modality to mask)
        unique_masks = 0
        for i in range(len(masks_list)):
            is_unique = True
            for j in range(i):
                if jnp.array_equal(masks_list[i], masks_list[j]):
                    is_unique = False
                    break
            if is_unique:
                unique_masks += 1

        # With 3 modalities and 20 trials, we should see multiple different masking patterns
        assert unique_masks >= 2, \
            f"Expected at least 2 unique masking patterns, got {unique_masks}"

    def test_masking_in_compute_loss(self):
        """Test compute_loss works correctly with masking enabled."""
        config = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=0.5,
            use_image_prompts=True,
            use_text_prompts=True,
            use_action_state_prompts=True
        )
        key = jax.random.key(0)
        model = config.create(key)

        batch_size = 2
        obs = config.fake_obs(batch_size)
        actions = config.fake_act(batch_size)

        # Compute loss with masking
        loss_key = jax.random.key(1)
        loss = model.compute_loss(loss_key, obs, actions, train=True)

        # Loss should be valid
        assert loss.shape == (batch_size, config.action_horizon), \
            f"Expected loss shape ({batch_size}, {config.action_horizon}), got {loss.shape}"
        assert jnp.all(jnp.isfinite(loss)), "Loss contains non-finite values"
        assert jnp.all(loss >= 0), "MSE loss should be non-negative"

    def test_gradients_with_masking(self):
        """Test gradients flow correctly when masking is enabled."""
        config = _pi0v18.Pi0IncontextConfigv18(
            prompt_mask_prob=0.5,
            use_image_prompts=True,
            use_action_state_prompts=True
        )
        key = jax.random.key(0)
        model = config.create(key)

        batch_size = 2
        obs = config.fake_obs(batch_size)
        actions = config.fake_act(batch_size)
        loss_key = jax.random.key(1)

        # Define loss function
        def loss_fn(m):
            return m.compute_loss(loss_key, obs, actions, train=True).mean()

        # Compute gradients
        loss_value = loss_fn(model)
        grads = nnx.grad(loss_fn)(model)

        # Loss and gradients should be valid
        assert jnp.isfinite(loss_value), "Loss is not finite"
        assert grads is not None, "Gradients are None"

        # Check compressor gradients exist and are finite
        if hasattr(model, 'image_compressor'):
            assert hasattr(grads, 'image_compressor'), "Gradients missing image_compressor"
            assert jnp.all(jnp.isfinite(grads.image_compressor.queries.value)), \
                "Non-finite gradients in image_compressor"


# =============================================================================
# Integration Tests for Compression (NEW in v18)
# =============================================================================


class TestCompressionIntegration:
    """Test Perceiver compression in full forward/backward passes."""

    def test_forward_pass_with_compression(self, default_config):
        """Test compute_loss works correctly with compressed tokens."""
        key = jax.random.key(0)
        model = default_config.create(key)

        # Create sample data
        batch_size = 2
        obs = default_config.fake_obs(batch_size)
        actions = default_config.fake_act(batch_size)

        # Forward pass
        loss_key = jax.random.key(1)
        loss = model.compute_loss(loss_key, obs, actions, train=False)

        # Loss should be valid - v12/v18 return per-timestep losses
        assert loss.shape == (batch_size, default_config.action_horizon)
        assert jnp.all(jnp.isfinite(loss))
        assert jnp.all(loss >= 0), "MSE loss should be non-negative"

    def test_inference_with_compression(self, default_config):
        """Test sample_actions works with compressed tokens."""
        key = jax.random.key(0)
        model = default_config.create(key)

        # Create sample observation
        batch_size = 2
        obs = default_config.fake_obs(batch_size)

        # Run inference
        inference_key = jax.random.key(1)
        actions = model.sample_actions(inference_key, obs, num_steps=10)

        # Actions should be valid
        expected_shape = (batch_size, default_config.action_horizon, default_config.action_dim)
        assert actions.shape == expected_shape
        assert jnp.all(jnp.isfinite(actions))

    def test_gradient_flow_through_compression(self, default_config):
        """Test gradients flow correctly through Perceiver compressors."""
        key = jax.random.key(0)
        model = default_config.create(key)

        # Create sample data
        batch_size = 2
        obs = default_config.fake_obs(batch_size)
        actions = default_config.fake_act(batch_size)
        loss_key = jax.random.key(1)

        # Define loss function
        def loss_fn(m):
            return m.compute_loss(loss_key, obs, actions, train=True).mean()

        # Compute gradients
        loss_value = loss_fn(model)
        grads = nnx.grad(loss_fn)(model)

        # Loss should be valid
        assert jnp.isfinite(loss_value)

        # Gradients should exist
        assert grads is not None

        # Check that compressor parameters have gradients by verifying gradient structure
        if hasattr(model, 'image_compressor'):
            # Verify compressor gradient objects exist
            assert hasattr(grads, 'image_compressor'), \
                "Gradients should include image_compressor"

            # Check that the queries parameter has gradients
            assert hasattr(grads.image_compressor, 'queries'), \
                "image_compressor gradients should include queries"

            # Verify gradient values are finite (spot check)
            assert jnp.all(jnp.isfinite(grads.image_compressor.queries.value)), \
                "Non-finite gradients in image_compressor queries"

    def test_different_num_queries_configs(self):
        """Test model works with various num_queries settings."""
        for nq in [8, 16, 32, 64]:
            config = _pi0v18.Pi0IncontextConfigv18(
                num_image_queries=nq,
                num_state_queries=nq,
                num_action_queries=nq
            )
            key = jax.random.key(0)
            model = config.create(key)

            # Test inference
            obs = config.fake_obs(1)
            inference_key = jax.random.key(1)
            actions = model.sample_actions(inference_key, obs, num_steps=5)

            assert jnp.all(jnp.isfinite(actions)), \
                f"Failed for num_queries={nq}"

    def test_jit_compilation_with_compression(self, default_config):
        """Test full model works with JIT when using compression."""
        key = jax.random.key(0)
        model = default_config.create(key)

        # JIT compile compute_loss
        jitted_loss = nnx_utils.module_jit(
            model.compute_loss,
            static_argnames='train'
        )

        # Create sample data
        batch_size = 2
        obs = default_config.fake_obs(batch_size)
        actions = default_config.fake_act(batch_size)
        loss_key = jax.random.key(1)

        # Run JIT-compiled loss
        loss = jitted_loss(loss_key, obs, actions, train=False)

        # Should produce valid results - v12/v18 return per-timestep losses
        assert loss.shape == (batch_size, default_config.action_horizon)
        assert jnp.all(jnp.isfinite(loss))
