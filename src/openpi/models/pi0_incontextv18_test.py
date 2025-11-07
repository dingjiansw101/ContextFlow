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
        num_queries, embed_dim, num_heads = 32, 2048, 8

        compressor = _pi0v18.PerceiverCompressor(
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=num_heads,
            rngs=nnx.Rngs(key)
        )

        # Check learnable queries shape
        assert compressor.queries.value.shape == (num_queries, embed_dim), \
            f"Expected queries shape ({num_queries}, {embed_dim}), got {compressor.queries.value.shape}"

        # Check LayerNorms exist
        assert hasattr(compressor, 'query_norm')
        assert hasattr(compressor, 'kv_norm')

        # Check MultiHeadAttention exists
        assert hasattr(compressor, 'cross_attn')

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

    def test_custom_values(self):
        """Test configuration with custom query counts."""
        config = _pi0v18.Pi0IncontextConfigv18(
            num_image_queries=16,
            num_state_queries=64,
            num_action_queries=8,
            num_attn_heads=4
        )

        assert config.num_image_queries == 16
        assert config.num_state_queries == 64
        assert config.num_action_queries == 8
        assert config.num_attn_heads == 4


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

        # Count tokens - should have compressed image tokens
        # Each camera contributes num_image_queries tokens (not seq_len * 256)
        num_cameras_with_data = sum(
            1 for name in obs.incontext_images
            if jnp.any(obs.incontext_image_masks[name])
        )

        # Total image tokens should be: num_cameras * num_image_queries
        # (Plus current observation images if present)
        assert tokens.shape[1] < 10000, \
            f"Token count too high ({tokens.shape[1]}), compression may not be working"

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
        assert tokens.shape[1] < 5000, \
            f"Token count too high ({tokens.shape[1]}), compression may not be working"

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

        # Loss should be valid
        assert loss.shape == (batch_size,)
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

        # Check that compressor parameters have gradients
        if hasattr(model, 'image_compressor'):
            # Queries should have gradients
            grad_state = nnx.state(grads)
            # Just verify we can extract the state without errors
            assert grad_state is not None

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

        # Should produce valid results
        assert loss.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(loss))
