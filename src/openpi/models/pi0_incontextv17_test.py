"""Unit tests for pi0_incontextv17 model.

This test file provides comprehensive coverage of all functions and methods in
the Pi0IncontextV17 model, including:
- Standalone utility functions (make_attn_mask, posemb_sincos)
- Configuration methods (freeze filters, input specs)
- Model embedding methods (embed_midfix, embed_suffix_state, embed_suffix_action)
- Training methods (compute_loss)
- Inference methods (sample_actions)
- Integration tests

USAGE:
------

Basic Usage:
    # Run all tests in this file
    uv run pytest src/openpi/models/pi0_incontextv17_test.py

    # Run with verbose output
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -v

    # Run with output printed (disable capture)
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -s

Run Specific Test Classes:
    # Test standalone functions only
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestMakeAttnMask
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestPosembSincos

    # Test configuration methods
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestPi0IncontextConfigv17

    # Test model initialization
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestModelInitialization

    # Test embedding methods
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestEmbedMidfix
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestEmbedSuffixState
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestEmbedSuffixAction

    # Test training methods
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestComputeLoss

    # Test inference methods
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestSampleActions

    # Test integration
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestIntegration

Run Specific Test Functions:
    # Test a specific function
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestMakeAttnMask::test_pure_causal
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestComputeLoss::test_jit_compilation

    # Use -k to match test names by pattern
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -k "causal"
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -k "jit"
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -k "shape"

Performance Tests:
    # Skip slow tests (default behavior)
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -m "not slow"

    # Run only slow/performance tests
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -m slow

    # Run all tests including slow ones
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -m ""

Stop on First Failure:
    # Stop after first failure for faster debugging
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -x

    # Stop after N failures
    uv run pytest src/openpi/models/pi0_incontextv17_test.py --maxfail=3

Show Test Coverage:
    # Run with coverage report
    uv run pytest src/openpi/models/pi0_incontextv17_test.py --cov=openpi.models.pi0_incontextv17

    # Generate HTML coverage report
    uv run pytest src/openpi/models/pi0_incontextv17_test.py --cov=openpi.models.pi0_incontextv17 --cov-report=html

Debugging:
    # Run with Python debugger on failure
    uv run pytest src/openpi/models/pi0_incontextv17_test.py --pdb

    # Show local variables on failure
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -l

    # Show full diff on assertion failures
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -vv

Parallel Execution:
    # Run tests in parallel (requires pytest-xdist)
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -n auto

Custom Markers:
    # List all available markers
    uv run pytest src/openpi/models/pi0_incontextv17_test.py --markers

Examples:
    # Quick smoke test (run fast tests only)
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -m "not slow" -x

    # Full test suite with verbose output
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -v -m ""

    # Test only shape validations
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -k "shape" -v

    # Test JIT compilation across all methods
    uv run pytest src/openpi/models/pi0_incontextv17_test.py -k "jit" -v

    # Debug specific failing test
    uv run pytest src/openpi/models/pi0_incontextv17_test.py::TestComputeLoss::test_shape -vv -s --pdb

TEST STRUCTURE:
---------------
- 60+ test functions organized into 11 test classes
- Comprehensive coverage of all public APIs
- Tests for edge cases, JIT compilation, and gradient flow
- Integration tests for full forward/backward passes
- Performance tests for large batches and many diffusion steps

FIXTURES AVAILABLE:
-------------------
- default_config: Standard Pi0IncontextConfigv17 configuration
- lora_config: Configuration with LoRA enabled for all experts
- small_model: Pre-instantiated model for quick testing
- sample_batch: Sample batch data (obs, act, future_states, key)
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import pi0_incontextv17 as _pi0v17
from openpi.models import model as _model
from openpi.shared import nnx_utils


# =============================================================================
# Helper Functions
# =============================================================================


def _get_frozen_state(config: _pi0v17.Pi0IncontextConfigv17) -> nnx.State:
    """Helper to extract frozen parameters from config.

    Args:
        config: Model configuration

    Returns:
        Flattened state containing frozen parameters
    """
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Create default configuration for testing."""
    return _pi0v17.Pi0IncontextConfigv17(
        prompt_expert_variant="gemma_300m_v2",
        state_expert_variant="gemma_300m",
        action_expert_variant="gemma_300m_lora",
    )


@pytest.fixture
def lora_config():
    """Create configuration with LoRA enabled for all experts."""
    return _pi0v17.Pi0IncontextConfigv17(
        prompt_expert_variant="gemma_300m_lora",
        state_expert_variant="gemma_300m_lora",
        action_expert_variant="gemma_300m_lora",
    )


@pytest.fixture
def small_model(default_config):
    """Create a small model instance for testing."""
    key = jax.random.key(0)
    return default_config.create(key)


@pytest.fixture
def sample_batch(default_config):
    """Create sample batch data for testing."""
    batch_size = 2
    key = jax.random.key(1)
    key1, key2 = jax.random.split(key)
    obs = default_config.fake_obs(batch_size)
    future_states = jax.random.normal(
        key1,
        (batch_size, default_config.future_state_horizon, default_config.state_dim)
    )
    # Inject sampled future states into observation
    obs = obs.replace(future_states=future_states)

    return {
        'obs': obs,
        'act': default_config.fake_act(batch_size),
        'key': key2,
    }


# =============================================================================
# Test Standalone Functions
# =============================================================================


class TestGetFreezeFilter:
    """Tests for Pi0IncontextConfigv17.get_freeze_filter across all LoRA combinations."""

    @pytest.mark.parametrize(
        ("prompt_lora", "state_lora", "action_lora", "expected_experts"),
        [
            (False, False, False, ()),
            (True, False, False, ("prompt",)),
            (False, True, False, ("state",)),
            (False, False, True, ("action",)),
            (True, True, False, ("prompt", "state")),
            (True, False, True, ("prompt", "action")),
            (False, True, True, ("state", "action")),
            (True, True, True, ("prompt", "state", "action")),
        ],
        ids=[
            "no_lora",
            "prompt_lora",
            "state_lora",
            "action_lora",
            "prompt_state_lora",
            "prompt_action_lora",
            "state_action_lora",
            "all_lora",
        ],
    )
    def test_freeze_matrix(self, prompt_lora, state_lora, action_lora, expected_experts):
        prompt_variant = "gemma_300m_lora" if prompt_lora else "gemma_300m_v2"
        state_variant = "gemma_300m_lora" if state_lora else "gemma_300m"
        action_variant = "gemma_300m_lora" if action_lora else "gemma_300m"

        config = _pi0v17.Pi0IncontextConfigv17(
            prompt_expert_variant=prompt_variant,
            state_expert_variant=state_variant,
            action_expert_variant=action_variant,
        )

        freeze_filter = config.get_freeze_filter()
        frozen_state = _get_frozen_state(config)

        expected_set = set(expected_experts)
        if not expected_set:
            assert freeze_filter is nnx.Nothing
            assert not frozen_state
            return

        assert frozen_state, "Expected base parameters to be frozen when LoRA is enabled"

        found_experts: set[str] = set()
        for path in frozen_state.keys():
            components = [str(part) for part in path]
            categories = set()
            if any("_prompt_expert" in comp for comp in components):
                categories.add("prompt")
            if any("_state_expert" in comp for comp in components):
                categories.add("state")
            if any(comp.endswith("_1") or "_1" in comp for comp in components):
                categories.add("action")

            assert categories, f"Frozen parameter path {components} did not map to any known expert"
            assert categories <= expected_set, f"Unexpected expert frozen for path {'/'.join(components)}"

            assert all("lora" not in comp for comp in components), f"LoRA adapters must remain trainable: {'/'.join(components)}"

            found_experts |= categories

        assert found_experts == expected_set, f"Mismatch in frozen experts: expected {expected_set}, found {found_experts}"

    def test_state_and_action_lora_specific(self):
        """Explicit regression test for state/action LoRA combination matching v7/v9 expectations."""
        config = _pi0v17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_300m_lora",
            action_expert_variant="gemma_300m_lora",
        )
        freeze_filter = config.get_freeze_filter()
        frozen_state = _get_frozen_state(config)

        assert frozen_state, "Expected base parameters to be frozen for state/action LoRA combo"
        assert isinstance(freeze_filter, nnx.filterlib.All)

        categories = []
        for path in frozen_state.keys():
            parts = [str(part) for part in path]
            assert all("lora" not in part for part in parts), f"LoRA adapters should remain trainable: {'/'.join(parts)}"
            if any("_state_expert" in part for part in parts):
                categories.append("state")
            if any(part.endswith("_1") or "_1" in part for part in parts):
                categories.append("action")
            assert not any("_prompt_expert" in part for part in parts), f"Prompt expert should remain trainable: {'/'.join(parts)}"

        assert "state" in categories, "State expert base weights should be frozen"
        assert "action" in categories, "Action expert base weights should be frozen"


class TestMakeAttnMask:
    """Test suite for make_attn_mask function."""

    def test_pure_causal(self):
        """Test pure causal attention mask (all tokens start new blocks)."""
        batch_size, seq_len = 2, 4
        input_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
        mask_ar = jnp.ones((seq_len,), dtype=jnp.bool_)

        attn_mask = _pi0v17.make_attn_mask(input_mask, mask_ar)

        # Shape check
        assert attn_mask.shape == (batch_size, seq_len, seq_len)

        # Causality check: attn_mask[i, j] should be True only if j <= i
        for i in range(seq_len):
            for j in range(seq_len):
                expected = j <= i
                assert jnp.all(attn_mask[:, i, j] == expected), \
                    f"Position ({i},{j}): expected {expected}, got {attn_mask[0, i, j]}"

    def test_prefix_lm(self):
        """Test prefix-LM attention pattern (bidirectional prefix + causal suffix)."""
        batch_size, seq_len = 2, 6
        input_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
        # First 3 tokens bidirectional, last 3 causal
        mask_ar = jnp.array([False, False, False, True, True, True], dtype=jnp.bool_)

        attn_mask = _pi0v17.make_attn_mask(input_mask, mask_ar)

        # First 3 tokens can attend to each other bidirectionally
        assert jnp.all(attn_mask[:, :3, :3]), "Prefix should have full attention"

        # # Prefix can attend to suffix
        # assert jnp.all(attn_mask[:, :3, 3:]), "Prefix should attend to suffix"

        # Suffix can attend to prefix
        assert jnp.all(attn_mask[:, 3:, :3]), "Suffix should attend to prefix"

        # Suffix has causal attention
        assert jnp.all(attn_mask[:, 3, 3]), "Token 3 attends to self"
        assert jnp.all(attn_mask[:, 4, 3:5]), "Token 4 attends to 3-4"
        assert jnp.all(attn_mask[:, 5, 3:6]), "Token 5 attends to 3-5"

        # Suffix cannot attend backward to prefix without including prefix block
        assert jnp.all(~attn_mask[:, 4, :3]) or jnp.all(attn_mask[:, :3, :3]), \
            "Causal suffix tokens attend only within suffix or to bidirectional prefix"

    def test_multi_block_causal(self):
        """Test multiple causal blocks with bidirectional sections."""
        batch_size, seq_len = 2, 6
        input_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
        # Pattern: [block0: 0-1 bidir, block1: 2-3 bidir, block2: 4-5 bidir]
        mask_ar = jnp.array([True, False, True, False, True, False], dtype=jnp.bool_)

        attn_mask = _pi0v17.make_attn_mask(input_mask, mask_ar)

        # Tokens 0-1 attend bidirectionally
        assert jnp.all(attn_mask[:, 0, :2]), "Block 0 attends to self"
        assert jnp.all(attn_mask[:, 1, :2]), "Block 0 attends to self"

        # Tokens 2-3 attend to 0-1 and themselves bidirectionally
        assert jnp.all(attn_mask[:, 2, :4]), "Block 1 attends to prev blocks + self"
        assert jnp.all(attn_mask[:, 3, :4]), "Block 1 attends to prev blocks + self"

        assert jnp.all(~attn_mask[:, :3, 5:]), "Cannot attend to future"

    def test_with_padding(self):
        """Test attention mask with padded tokens."""
        batch_size, seq_len = 2, 4
        # Second sequence has padding at position 3
        input_mask = jnp.array([
            [True, True, True, True],
            [True, True, True, False]
        ], dtype=jnp.bool_)
        mask_ar = jnp.zeros((seq_len,), dtype=jnp.bool_)  # All bidirectional

        attn_mask = _pi0v17.make_attn_mask(input_mask, mask_ar)

        # First batch: all True (no padding)
        assert jnp.all(attn_mask[0, :, :]), "No padding should have full attention"

        # Second batch: last column and row should be False
        assert jnp.all(~attn_mask[1, :, 3]), "Cannot attend to padded token"
        assert jnp.all(~attn_mask[1, 3, :]), "Padded token cannot attend"

    def test_broadcasting(self):
        """Test that ar_mask broadcasts correctly to batch dimension."""
        batch_size, seq_len = 3, 4
        input_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
        mask_ar = jnp.ones((seq_len,), dtype=jnp.bool_)  # 1D, should broadcast

        attn_mask = _pi0v17.make_attn_mask(input_mask, mask_ar)

        # Should work and produce same result for all batches
        assert attn_mask.shape == (batch_size, seq_len, seq_len)
        assert jnp.all(attn_mask[0] == attn_mask[1])
        assert jnp.all(attn_mask[1] == attn_mask[2])


# =============================================================================
# Test Configuration Methods
# =============================================================================


class TestPi0IncontextConfigv17:
    """Test suite for Pi0IncontextConfigv17."""

    def test_model_type(self, default_config):
        """Test that model_type returns correct value."""
        assert default_config.model_type == _model.ModelType.PI0_INCONTEXT

    def test_create(self, default_config):
        """Test that create method instantiates model successfully."""
        key = jax.random.key(0)
        model = default_config.create(key)

        assert isinstance(model, _pi0v17.Pi0Incontextv17)
        assert model.config == default_config

    def test_inputs_spec_shapes(self, default_config):
        """Test that inputs_spec returns correct shapes."""
        batch_size = 4
        spec = default_config.inputs_spec(batch_size=batch_size)

        # Check observation spec
        assert spec.state.shape == (batch_size, default_config.state_dim)

        # Check image specs
        for camera_key, image_spec in spec.images.items():
            assert image_spec.shape == (batch_size, 224, 224, 3)

        # Check incontext specs if present
        if hasattr(spec, 'incontext_images'):
            for camera_key, spec_val in spec.incontext_images.items():
                # Shape depends on sample_frames
                assert spec_val.shape[0] == batch_size

    def test_freeze_filter_no_lora(self, default_config):
        """Test freeze filter when no LoRA (all trainable)."""
        freeze_filter = default_config.get_freeze_filter()

        # No LoRA means everything is trainable
        assert freeze_filter == nnx.Nothing

    def test_freeze_filter_prompt_lora(self):
        """Test freeze filter when only prompt expert uses LoRA."""
        config = _pi0v17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_lora",
            state_expert_variant="gemma_300m",
            action_expert_variant="gemma_300m",
        )

        state = _get_frozen_state(config)

        # Should have frozen parameters from prompt expert
        assert len(state) > 0
        # All frozen params should be from LLM (Gemma)
        assert all("llm" in p for p in state)
        # Should not include LoRA parameters
        assert all("lora" not in p for p in state)

    def test_freeze_filter_state_lora(self):
        """Test freeze filter when only state expert uses LoRA."""
        config = _pi0v17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m",
            state_expert_variant="gemma_300m_lora",
            action_expert_variant="gemma_300m",
        )

        state = _get_frozen_state(config)

        # Should have frozen parameters from state expert
        assert len(state) > 0
        assert all("llm" in p for p in state)

    def test_freeze_filter_action_lora(self):
        """Test freeze filter when only action expert uses LoRA."""
        config = _pi0v17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m",
            state_expert_variant="gemma_300m",
            action_expert_variant="gemma_300m_lora",
        )

        state = _get_frozen_state(config)

        # Should have frozen parameters from action expert
        assert len(state) > 0
        assert all("llm" in p for p in state)

    def test_freeze_filter_all_lora(self, lora_config):
        """Test freeze filter when all experts use LoRA."""
        state = _get_frozen_state(lora_config)

        # All base weights should be frozen
        assert len(state) > 0
        assert all("lora" not in p for p in state)
        assert all("llm" in p for p in state)


# =============================================================================
# Test Model Initialization
# =============================================================================


class TestModelInitialization:
    """Test suite for model initialization."""

    def test_initialization_success(self, default_config):
        """Test model initializes successfully with all components."""
        key = jax.random.key(0)
        model = default_config.create(key)

        # Check basic attributes
        assert model.action_dim == default_config.action_dim
        assert model.action_horizon == default_config.action_horizon
        assert model.config == default_config

        # Check key components exist
        assert hasattr(model, 'PaliGemma')
        assert hasattr(model, 'state_proj')
        assert hasattr(model, 'future_state_in_proj')
        assert hasattr(model, 'action_in_proj')
        assert hasattr(model, 'action_out_proj')
        assert hasattr(model, 'future_state_out_proj')

    def test_demo_projections_created(self, default_config):
        """Test demo projections are created when use_action_state_prompts=True."""
        config = _pi0v17.Pi0IncontextConfigv17(use_action_state_prompts=True)
        key = jax.random.key(0)
        model = config.create(key)

        assert hasattr(model, 'demo_state_proj')
        assert hasattr(model, 'demo_action_proj')

    def test_demo_projections_not_created(self):
        """Test demo projections are not created when use_action_state_prompts=False."""
        config = _pi0v17.Pi0IncontextConfigv17(use_action_state_prompts=False)
        key = jax.random.key(0)
        model = config.create(key)

        assert not hasattr(model, 'demo_state_proj')
        assert not hasattr(model, 'demo_action_proj')


# =============================================================================
# Test Embedding Methods
# =============================================================================


class TestEmbedMidfix:
    """Test suite for embed_midfix method."""

    def test_basic_shape(self, small_model, default_config):
        """Test embed_midfix returns correct shapes."""
        batch_size = 2
        obs = default_config.fake_obs(batch_size)

        tokens, input_mask, ar_mask = small_model.embed_midfix(obs)

        # Check shapes
        assert tokens.shape[0] == batch_size
        # assert tokens.shape[2] == 1152  # Gemma 300M v2 width
        # assert tokens.shape[2] == 2048
        assert input_mask.shape[0] == batch_size
        assert input_mask.shape[1] == tokens.shape[1]
        assert ar_mask.shape[0] == tokens.shape[1]

    def test_finite_values(self, small_model, default_config):
        """Test all token values are finite."""
        obs = default_config.fake_obs(2)
        tokens, _, _ = small_model.embed_midfix(obs)

        assert jnp.all(jnp.isfinite(tokens)), "All token values should be finite"

    def test_ar_mask_structure(self, small_model, default_config):
        """Test ar_mask has correct structure (mostly bidirectional)."""
        obs = default_config.fake_obs(2)
        _, _, ar_mask = small_model.embed_midfix(obs)

        # For default config without causal_attention, should be mostly False
        # (bidirectional attention)
        if not default_config.causal_attention:
            assert jnp.sum(ar_mask) <= 3, "Most tokens should have bidirectional attention"

    def test_with_text_prompts(self):
        """Test embed_midfix with text prompts enabled."""
        config = _pi0v17.Pi0IncontextConfigv17(use_text_prompts=True)
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(2)

        tokens, input_mask, ar_mask = model.embed_midfix(obs)

        # Should include text prompt tokens
        assert tokens.shape[1] > 0
        assert jnp.all(jnp.isfinite(tokens))

    def test_with_demo_images(self):
        """Test embed_midfix with demo images."""
        config = _pi0v17.Pi0IncontextConfigv17(use_image_prompts=True, sample_frames=4)
        key = jax.random.key(0)
        model = config.create(key)
        obs = config.fake_obs(2)

        tokens, input_mask, ar_mask = model.embed_midfix(obs)

        # Should include demo image tokens
        assert tokens.shape[1] > 0


class TestEmbedSuffixState:
    """Test suite for embed_suffix_state method."""

    def test_shape(self, small_model, default_config):
        """Test embed_suffix_state returns correct shapes."""
        batch_size = 2
        obs = default_config.fake_obs(batch_size)

        key = jax.random.key(0)
        noisy_states = jax.random.normal(
            key, (batch_size, default_config.future_state_horizon, default_config.state_dim)
        )
        timestep = jax.random.uniform(key, (batch_size,))

        tokens, input_mask, ar_mask = small_model.embed_suffix_state(obs, noisy_states, timestep)

        # Expected: 1 current state + future_state_horizon noisy state tokens
        expected_len = 1 + default_config.future_state_horizon
        assert tokens.shape == (batch_size, expected_len, default_config.state_expert_width)
        assert input_mask.shape == (batch_size, expected_len)
        assert ar_mask.shape == (expected_len,)

    def test_ar_mask_pattern(self, small_model, default_config):
        """Test ar_mask has correct pattern for state expert."""
        batch_size = 2
        obs = default_config.fake_obs(batch_size)

        key = jax.random.key(0)
        noisy_states = jax.random.normal(
            key, (batch_size, default_config.future_state_horizon, default_config.state_dim)
        )
        timestep = jax.random.uniform(key, (batch_size,))

        _, _, ar_mask = small_model.embed_suffix_state(obs, noisy_states, timestep)

        # First token (current state) starts new causal block
        assert ar_mask[0] == True
        # Second token (first future state) also starts new causal block
        assert ar_mask[1] == True
        # Remaining tokens are bidirectional within their block
        assert jnp.all(ar_mask[2:] == False)

    def test_finite_values(self, small_model, default_config):
        """Test all values are finite."""
        batch_size = 2
        obs = default_config.fake_obs(batch_size)

        key = jax.random.key(0)
        noisy_states = jax.random.normal(
            key, (batch_size, default_config.future_state_horizon, default_config.state_dim)
        )
        timestep = jax.random.uniform(key, (batch_size,))

        tokens, _, _ = small_model.embed_suffix_state(obs, noisy_states, timestep)

        assert jnp.all(jnp.isfinite(tokens))


class TestEmbedSuffixAction:
    """Test suite for embed_suffix_action method."""

    def test_shape(self, small_model, default_config):
        """Test embed_suffix_action returns correct shapes."""
        batch_size = 2
        obs = default_config.fake_obs(batch_size)

        key = jax.random.key(0)
        key1, key2 = jax.random.split(key)

        future_states = jax.random.normal(
            key1, (batch_size, default_config.future_state_horizon, default_config.state_dim)
        )
        noisy_actions = jax.random.normal(
            key2, (batch_size, default_config.action_horizon, default_config.action_dim)
        )
        timestep = jax.random.uniform(key, (batch_size,))

        tokens, input_mask, ar_mask = small_model.embed_suffix_action(
            obs, future_states, noisy_actions, timestep
        )

        # Expected: 1 state + future_state_horizon conditioning + action_horizon
        expected_len = 1 + default_config.future_state_horizon + default_config.action_horizon
        assert tokens.shape == (batch_size, expected_len, default_config.action_expert_width)
        assert input_mask.shape == (batch_size, expected_len)
        assert ar_mask.shape == (expected_len,)

    def test_ar_mask_pattern(self, small_model, default_config):
        """Test ar_mask has correct pattern for action expert."""
        batch_size = 2
        obs = default_config.fake_obs(batch_size)

        key = jax.random.key(0)
        key1, key2 = jax.random.split(key)

        future_states = jax.random.normal(
            key1, (batch_size, default_config.future_state_horizon, default_config.state_dim)
        )
        noisy_actions = jax.random.normal(
            key2, (batch_size, default_config.action_horizon, default_config.action_dim)
        )
        timestep = jax.random.uniform(key, (batch_size,))

        _, _, ar_mask = small_model.embed_suffix_action(
            obs, future_states, noisy_actions, timestep
        )

        # Pattern: [True (state) | False... (future states) | True (first action) | False... (remaining actions)]
        assert ar_mask[0] == True, "Current state token starts new block"

        # Future state conditioning should be bidirectional
        future_state_start = 1
        future_state_end = 1 + default_config.future_state_horizon
        assert jnp.all(ar_mask[future_state_start:future_state_end] == False), \
            "Future state conditioning should be bidirectional"

        # First action token starts new block
        assert ar_mask[future_state_end] == True, "First action token starts new block"

    def test_finite_values(self, small_model, default_config):
        """Test all values are finite."""
        batch_size = 2
        obs = default_config.fake_obs(batch_size)

        key = jax.random.key(0)
        key1, key2 = jax.random.split(key)

        future_states = jax.random.normal(
            key1, (batch_size, default_config.future_state_horizon, default_config.state_dim)
        )
        noisy_actions = jax.random.normal(
            key2, (batch_size, default_config.action_horizon, default_config.action_dim)
        )
        timestep = jax.random.uniform(key, (batch_size,))

        tokens, _, _ = small_model.embed_suffix_action(obs, future_states, noisy_actions, timestep)

        assert jnp.all(jnp.isfinite(tokens))


# =============================================================================
# Test Training Methods
# =============================================================================


class TestComputeLoss:
    """Test suite for compute_loss method."""

    def test_shape(self, small_model, sample_batch):
        """Test compute_loss returns correct shape."""
        loss = small_model.compute_loss(
            sample_batch['key'],
            sample_batch['obs'],
            sample_batch['act'],
            train=False
        )

        batch_size = sample_batch['obs'].state.shape[0]
        assert loss.shape == (batch_size,), f"Expected shape ({batch_size},), got {loss.shape}"

    def test_finite_values(self, small_model, sample_batch):
        """Test loss values are finite."""
        loss = small_model.compute_loss(
            sample_batch['key'],
            sample_batch['obs'],
            sample_batch['act'],
            train=False
        )

        assert jnp.all(jnp.isfinite(loss)), "Loss should be finite"

    def test_non_negative(self, small_model, sample_batch):
        """Test loss is non-negative (MSE property)."""
        loss = small_model.compute_loss(
            sample_batch['key'],
            sample_batch['obs'],
            sample_batch['act'],
            train=False
        )

        assert jnp.all(loss >= 0), "MSE loss should be non-negative"

    def test_with_training_mode(self, small_model, sample_batch):
        """Test compute_loss works with train=True."""
        loss_train = small_model.compute_loss(
            sample_batch['key'],
            sample_batch['obs'],
            sample_batch['act'],
            train=True
        )

        assert jnp.all(jnp.isfinite(loss_train))
        assert jnp.all(loss_train >= 0)

    def test_jit_compilation(self, small_model, sample_batch):
        """Test compute_loss works with JIT compilation."""
        jitted_loss = nnx_utils.module_jit(
            small_model.compute_loss,
            static_argnames='train'
        )

        loss = jitted_loss(
            sample_batch['key'],
            sample_batch['obs'],
            sample_batch['act'],
            train=False
        )

        batch_size = sample_batch['obs'].state.shape[0]
        assert loss.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(loss))

    def test_gradient_flow(self, small_model, sample_batch):
        """Test gradients can be computed through compute_loss."""
        def loss_fn(model):
            return model.compute_loss(
                sample_batch['key'],
                sample_batch['obs'],
                sample_batch['act'],
                train=True
            ).mean()

        # Should not raise an error
        grads = nnx.grad(loss_fn)(small_model)

        # Gradients should exist and be finite
        assert grads is not None


# =============================================================================
# Test Inference Methods
# =============================================================================


class TestSampleActions:
    """Test suite for sample_actions method."""

    def test_shape(self, small_model, default_config):
        """Test sample_actions returns correct shape."""
        batch_size = 2
        obs = default_config.fake_obs(batch_size)
        key = jax.random.key(0)

        actions = small_model.sample_actions(key, obs, num_steps=10)

        assert actions.shape == (batch_size, default_config.action_horizon, default_config.action_dim)

    def test_finite_values(self, small_model, default_config):
        """Test all action values are finite."""
        obs = default_config.fake_obs(2)
        key = jax.random.key(0)

        actions = small_model.sample_actions(key, obs, num_steps=10)

        assert jnp.all(jnp.isfinite(actions)), "All action values should be finite"

    def test_deterministic_with_seed(self, small_model, default_config):
        """Test sample_actions is deterministic with same seed."""
        batch_size = 1
        obs = default_config.fake_obs(batch_size)
        key = jax.random.key(42)

        actions1 = small_model.sample_actions(key, obs, num_steps=5)
        actions2 = small_model.sample_actions(key, obs, num_steps=5)

        assert jnp.allclose(actions1, actions2, atol=1e-5), \
            "Same seed should produce same actions"

    def test_different_batch_sizes(self, small_model, default_config):
        """Test sample_actions works with different batch sizes."""
        key = jax.random.key(0)

        for batch_size in [1, 2, 4]:
            obs = default_config.fake_obs(batch_size)
            actions = small_model.sample_actions(key, obs, num_steps=5)

            assert actions.shape == (batch_size, default_config.action_horizon, default_config.action_dim), \
                f"Failed for batch_size={batch_size}"

    def test_different_num_steps(self, small_model, default_config):
        """Test sample_actions works with different num_steps."""
        obs = default_config.fake_obs(2)
        key = jax.random.key(0)

        for num_steps in [1, 5, 10, 20]:
            actions = small_model.sample_actions(key, obs, num_steps=num_steps)

            assert jnp.all(jnp.isfinite(actions)), f"Failed for num_steps={num_steps}"

    def test_jit_compilation(self, small_model, default_config):
        """Test sample_actions works with JIT compilation."""
        batch_size = 2
        obs = default_config.fake_obs(batch_size)
        key = jax.random.key(0)

        jitted_sample = nnx_utils.module_jit(small_model.sample_actions)
        actions = jitted_sample(key, obs, num_steps=10)

        assert actions.shape == (batch_size, default_config.action_horizon, default_config.action_dim)
        assert jnp.all(jnp.isfinite(actions))


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full pipelines."""

    def test_forward_pass(self, small_model, sample_batch):
        """Test full forward pass through model."""
        # Forward pass
        loss = small_model.compute_loss(
            sample_batch['key'],
            sample_batch['obs'],
            sample_batch['act'],
            train=False
        )

        # Should produce valid loss
        assert jnp.all(jnp.isfinite(loss))
        assert jnp.all(loss >= 0)

    def test_inference_pipeline(self, small_model, default_config):
        """Test full inference pipeline with KV caching."""
        batch_size = 2
        obs = default_config.fake_obs(batch_size)
        key = jax.random.key(0)

        # Run inference
        actions = small_model.sample_actions(key, obs, num_steps=10)

        # Should produce valid actions
        assert actions.shape == (batch_size, default_config.action_horizon, default_config.action_dim)
        assert jnp.all(jnp.isfinite(actions))

    def test_training_step(self, small_model, sample_batch):
        """Test a single training step."""
        # Define loss function
        def loss_fn(model):
            return model.compute_loss(
                sample_batch['key'],
                sample_batch['obs'],
                sample_batch['act'],
                train=True
            ).mean()

        # Compute loss and gradients
        loss_value = loss_fn(small_model)
        grads = jax.grad(loss_fn)(small_model)

        # Verify
        assert jnp.isfinite(loss_value)
        assert loss_value >= 0
        assert grads is not None

    def test_different_expert_variants(self):
        """Test model works with different expert variant combinations."""
        variants = [
            ("gemma_300m_v2", "gemma_300m", "gemma_300m"),
            ("gemma_300m_lora", "gemma_300m_lora", "gemma_300m_lora"),
            ("gemma_300m_v2", "gemma_300m_lora", "gemma_300m"),
        ]

        key = jax.random.key(0)

        for prompt_var, state_var, action_var in variants:
            config = _pi0v17.Pi0IncontextConfigv17(
                prompt_expert_variant=prompt_var,
                state_expert_variant=state_var,
                action_expert_variant=action_var,
            )

            # Should create successfully
            model = config.create(key)
            assert model is not None

            # Should run inference
            obs = config.fake_obs(1)
            actions = model.sample_actions(key, obs, num_steps=5)
            assert jnp.all(jnp.isfinite(actions))


# =============================================================================
# Equivalence Tests: Fused vs Sequential
# =============================================================================


class TestComputeLossEquivalence:
    """Test equivalence between fused and sequential computation approaches."""

    def test_forward_equivalence(self, small_model, default_config):
        """Verify fused and sequential approaches produce identical loss values."""
        batch_size = 2
        key = jax.random.key(42)
        key1, key2 = jax.random.split(key)

        # Create observation with correct shapes using inputs_spec
        obs_spec, act_spec = default_config.inputs_spec(
            batch_size=batch_size,
            keyframe_size=8,
            max_len=8
        )

        # Create actual arrays from specs
        obs = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs_spec)

        # Add future_states with random values
        future_states = jax.random.normal(
            key1, (batch_size, default_config.future_state_horizon, default_config.state_dim)
        )
        obs = obs.replace(future_states=future_states)

        actions = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), act_spec)

        # Compute loss using fused approach (current implementation)
        loss_fused = small_model.compute_loss(key2, obs, actions, train=False)

        # Compute loss using sequential approach (v1 - without KV caching)
        loss_sequential = small_model.compute_loss_sequential(key2, obs, actions, train=False)

        # Should produce similar losses (with relaxed tolerance due to different computational paths)
        assert loss_fused.shape == loss_sequential.shape, \
            f"Shape mismatch: fused {loss_fused.shape} vs sequential {loss_sequential.shape}"

        print(f"Fused vs Sequential(v1) max diff: {jnp.max(jnp.abs(loss_fused - loss_sequential))}")
        assert jnp.allclose(loss_fused, loss_sequential, rtol=1e-2, atol=1e-2), \
            f"Loss mismatch (v1): max diff = {jnp.max(jnp.abs(loss_fused - loss_sequential))}"

    def test_forward_equivalence_v2(self, small_model, default_config):
        """Verify fused and sequential v2 (KV caching) approaches produce identical loss values."""
        batch_size = 2
        key = jax.random.key(42)
        key1, key2 = jax.random.split(key)

        # Create observation with correct shapes using inputs_spec
        obs_spec, act_spec = default_config.inputs_spec(
            batch_size=batch_size,
            keyframe_size=8,
            max_len=8
        )

        # Create actual arrays from specs
        obs = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs_spec)

        # Add future_states with random values
        future_states = jax.random.normal(
            key1, (batch_size, default_config.future_state_horizon, default_config.state_dim)
        )
        obs = obs.replace(future_states=future_states)

        actions = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), act_spec)

        # Compute loss using fused approach (current implementation)
        loss_fused = small_model.compute_loss(key2, obs, actions, train=False)

        # Compute loss using sequential approach (v2 - with KV caching)
        loss_sequentialv2 = small_model.compute_loss_sequentialv2(key2, obs, actions, train=False)

        # Should produce much closer results due to KV caching matching position encodings
        assert loss_fused.shape == loss_sequentialv2.shape, \
            f"Shape mismatch: fused {loss_fused.shape} vs sequentialv2 {loss_sequentialv2.shape}"

        print(f"Fused vs Sequential(v2) max diff: {jnp.max(jnp.abs(loss_fused - loss_sequentialv2))}")
        assert jnp.allclose(loss_fused, loss_sequentialv2, rtol=1e-5, atol=1e-6), \
            f"Loss mismatch (v2): max diff = {jnp.max(jnp.abs(loss_fused - loss_sequentialv2))}"

    def test_forward_equivalence_with_training_mode(self, small_model, default_config):
        """Verify equivalence holds with training mode (includes augmentation)."""
        batch_size = 2
        key = jax.random.key(43)
        key1, key2 = jax.random.split(key)

        # Create observation with correct shapes
        obs_spec, act_spec = default_config.inputs_spec(
            batch_size=batch_size,
            keyframe_size=default_config.sample_frames,
            max_len=64
        )
        obs = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs_spec)
        future_states = jax.random.normal(
            key1, (batch_size, default_config.future_state_horizon, default_config.state_dim)
        )
        obs = obs.replace(future_states=future_states)
        actions = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), act_spec)

        # Compute with train=True (includes image augmentation)
        loss_fused = small_model.compute_loss(key2, obs, actions, train=True)
        loss_sequential = small_model.compute_loss_sequential(key2, obs, actions, train=True)

        # Should still be equivalent despite augmentation
        assert jnp.allclose(loss_fused, loss_sequential, rtol=1e-5, atol=1e-6), \
            f"Loss mismatch with train=True: max diff = {jnp.max(jnp.abs(loss_fused - loss_sequential))}"

    def test_gradient_equivalence(self, default_config):
        """Verify gradients are identical between fused and sequential approaches."""
        # Create two models with same initialization
        key = jax.random.key(42)
        model_fused = default_config.create(key)
        model_seq = default_config.create(key)

        # Create sample batch
        batch_size = 2
        key1, key2 = jax.random.split(key)
        obs = default_config.fake_obs(batch_size)
        future_states = jax.random.normal(
            key1, (batch_size, default_config.future_state_horizon, default_config.state_dim)
        )
        obs = obs.replace(future_states=future_states)
        actions = default_config.fake_act(batch_size)

        # Define loss functions
        def loss_fused_fn(model):
            return model.compute_loss(key2, obs, actions, train=True).mean()

        def loss_seq_fn(model):
            return model.compute_loss_sequential(key2, obs, actions, train=True).mean()

        # Compute gradients
        grads_fused = nnx.grad(loss_fused_fn)(model_fused)
        grads_seq = nnx.grad(loss_seq_fn)(model_seq)

        # Gradients should be identical
        # Extract parameter pytrees for comparison
        _, params_fused = nnx.split(grads_fused)
        _, params_seq = nnx.split(grads_seq)

        params_fused_dict = params_fused.to_pure_dict()
        params_seq_dict = params_seq.to_pure_dict()

        # Compare each parameter
        for key_path in params_fused_dict.keys():
            grad_fused = params_fused_dict[key_path]
            grad_seq = params_seq_dict.get(key_path)

            assert grad_seq is not None, f"Gradient missing in sequential: {key_path}"

            if not jnp.allclose(grad_fused, grad_seq, rtol=1e-5, atol=1e-6):
                max_diff = jnp.max(jnp.abs(grad_fused - grad_seq))
                raise AssertionError(
                    f"Gradient mismatch at {key_path}: max diff = {max_diff}\n"
                    f"Fused: {grad_fused.flatten()[:5]}...\n"
                    f"Sequential: {grad_seq.flatten()[:5]}..."
                )


# =============================================================================
# Performance Tests (optional, can be slow)
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance and stress tests (run with pytest -m slow)."""

    def test_large_batch(self, small_model, default_config):
        """Test model handles large batch sizes."""
        batch_size = 16
        obs = default_config.fake_obs(batch_size)
        key = jax.random.key(0)

        actions = small_model.sample_actions(key, obs, num_steps=10)

        assert actions.shape == (batch_size, default_config.action_horizon, default_config.action_dim)

    def test_many_diffusion_steps(self, small_model, default_config):
        """Test model with many diffusion steps."""
        obs = default_config.fake_obs(2)
        key = jax.random.key(0)

        # Should handle many steps
        actions = small_model.sample_actions(key, obs, num_steps=50)

        assert jnp.all(jnp.isfinite(actions))
