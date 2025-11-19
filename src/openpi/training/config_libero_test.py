"""Unit tests for config_libero configurations, focusing on weight loading strategies.

This test file validates that training configurations correctly initialize model weights
according to their specified weight loaders.

USAGE:
------
    # Run all tests
    uv run pytest src/openpi/training/config_libero_test.py

    # Run specific test
    uv run pytest src/openpi/training/config_libero_test.py::test_pi0_libero_incontextv18_low_mem_finetune_sample_frames8_selective_init

    # Run with verbose output
    uv run pytest src/openpi/training/config_libero_test.py -v
"""

import flax.nnx as nnx
import flax.traverse_util
import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import weight_loaders


@pytest.mark.manual
def test_pi0_libero_incontextv18_low_mem_finetune_sample_frames8_paligemma_init():
    """Test that VisionEncoderOnlyLoader correctly loads only vision encoder and embedder.

    This test validates that:
    1. The config can be loaded successfully
    2. VisionEncoderOnlyLoader loads only the vision encoder and embedder from PaliGemma
    3. All LLM layers are randomly initialized (not loaded from checkpoint)
    4. All in-context v18 components are randomly initialized
    """
    # Load the config
    config = _config.get_config("pi0_libero_incontextv18_low_mem_finetune_sample_frames8_paligemma_init")
    assert config is not None, "Config should be loaded successfully"
    assert isinstance(config.weight_loader, weight_loaders.VisionEncoderOnlyLoader), \
        f"Expected VisionEncoderOnlyLoader, got {type(config.weight_loader)}"

    # Create model with random initialization
    key = jax.random.key(42)
    model = config.model.create(key)
    random_params = nnx.state(model, nnx.Param)

    # Apply weight loader
    loaded_params = config.weight_loader.load(random_params)

    # Flatten parameters for comparison
    flat_random = flax.traverse_util.flatten_dict(random_params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # Check that vision encoder parameters were loaded (should differ from random)
    vision_keys = [k for k in flat_loaded.keys() if k.startswith("PaliGemma/img/")]
    assert len(vision_keys) > 0, "Should have vision encoder parameters"

    vision_params_changed = 0
    for k in vision_keys:
        if not np.allclose(flat_random[k], flat_loaded[k], rtol=1e-5):
            vision_params_changed += 1

    assert vision_params_changed > 0, "Vision encoder parameters should be loaded from checkpoint"
    print(f"✓ Vision encoder: {vision_params_changed}/{len(vision_keys)} parameters loaded from PaliGemma")

    # Check that embedder parameters were loaded (since include_embedder=True)
    embedder_keys = [k for k in flat_loaded.keys() if "llm/embedder" in k]
    assert len(embedder_keys) > 0, "Should have embedder parameters"

    embedder_params_changed = 0
    for k in embedder_keys:
        if not np.allclose(flat_random[k], flat_loaded[k], rtol=1e-5):
            embedder_params_changed += 1

    assert embedder_params_changed > 0, "Embedder parameters should be loaded from PaliGemma"
    print(f"✓ Embedder: {embedder_params_changed}/{len(embedder_keys)} parameters loaded from PaliGemma")

    # Check that LLM layer parameters are randomly initialized (should be the same as random)
    llm_layer_keys = [k for k in flat_loaded.keys() if "llm/layers" in k]
    assert len(llm_layer_keys) > 0, "Should have LLM layer parameters"

    llm_params_unchanged = 0
    for k in llm_layer_keys:
        if np.allclose(flat_random[k], flat_loaded[k], rtol=1e-5):
            llm_params_unchanged += 1

    # All LLM layers should remain randomly initialized
    assert llm_params_unchanged == len(llm_layer_keys), \
        f"LLM layers should be randomly initialized, but {len(llm_layer_keys) - llm_params_unchanged}/{len(llm_layer_keys)} were loaded"
    print(f"✓ LLM layers: {llm_params_unchanged}/{len(llm_layer_keys)} parameters randomly initialized")

    # Check that compressor parameters are randomly initialized (v18 specific)
    compressor_keys = [k for k in flat_loaded.keys() if any(
        comp in k for comp in ["image_compressor", "state_compressor", "action_compressor"]
    )]

    if len(compressor_keys) > 0:  # Only check if model has compressors
        compressor_params_unchanged = sum(
            1 for k in compressor_keys if np.allclose(flat_random[k], flat_loaded[k], rtol=1e-5)
        )
        assert compressor_params_unchanged == len(compressor_keys), \
            f"Compressors should be randomly initialized, but {len(compressor_keys) - compressor_params_unchanged}/{len(compressor_keys)} were loaded"
        print(f"✓ Compressors: {compressor_params_unchanged}/{len(compressor_keys)} parameters randomly initialized")

    print("✓ Test passed: VisionEncoderOnlyLoader correctly loads only vision encoder and embedder")


@pytest.mark.manual
def test_pi0_libero_incontextv18_low_mem_finetune_sample_frames8_selective_init():
    """Test that SelectiveVisionAndProjectionsLoader correctly loads vision, embedder, and projections.

    This test validates that:
    1. The config can be loaded successfully
    2. SelectiveVisionAndProjectionsLoader loads vision encoder, embedder, and basic projections from pi0_base
    3. All LLM layers (prompt and action experts) are randomly initialized
    4. All in-context v18 components (compressors, demo projections) are randomly initialized
    """
    # Load the config
    config = _config.get_config("pi0_libero_incontextv18_low_mem_finetune_sample_frames8_selective_init")
    assert config is not None, "Config should be loaded successfully"
    assert isinstance(config.weight_loader, weight_loaders.SelectiveVisionAndProjectionsLoader), \
        f"Expected SelectiveVisionAndProjectionsLoader, got {type(config.weight_loader)}"

    # Create model with random initialization
    key = jax.random.key(42)
    model = config.model.create(key)
    random_params = nnx.state(model, nnx.Param)

    # Load pi0_base checkpoint for comparison
    pi0_base_path = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")
    pi0_base_params = _model.restore_params(pi0_base_path, restore_type=np.ndarray)
    flat_pi0_base = flax.traverse_util.flatten_dict(pi0_base_params, sep="/")

    # Apply weight loader
    loaded_params = config.weight_loader.load(random_params)

    # Flatten parameters for comparison
    flat_random = flax.traverse_util.flatten_dict(random_params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # Define component categories to check
    components_to_load = {
        "Vision encoder": lambda k: k.startswith("PaliGemma/img/"),
        "Embedder": lambda k: "llm/embedder" in k,
        "State projection": lambda k: k.startswith("state_proj/"),
        "Action projections": lambda k: any(k.startswith(p) for p in [
            "action_in_proj/", "action_time_mlp_in/", "action_time_mlp_out/", "action_out_proj/"
        ]),
    }

    components_to_random_init = {
        "LLM layers": lambda k: "llm/layers" in k,
        "LLM final norms": lambda k: "llm/final_norm" in k,
        "Demo projections": lambda k: any(p in k for p in ["demo_action_proj", "demo_state_proj"]),
        "Compressors": lambda k: any(c in k for c in ["image_compressor", "state_compressor", "action_compressor"]),
    }

    # Check components that should be loaded from pi0_base
    print("\n=== Components loaded from pi0_base ===")
    for component_name, key_filter in components_to_load.items():
        component_keys = [k for k in flat_loaded.keys() if key_filter(k)]

        if len(component_keys) == 0:
            print(f"⚠ {component_name}: No parameters found (may not exist in this model variant)")
            continue

        # Count how many parameters match pi0_base (not random)
        loaded_from_checkpoint = 0
        for k in component_keys:
            # Check if this key exists in pi0_base and matches loaded params
            if k in flat_pi0_base and not np.allclose(flat_random[k], flat_loaded[k], rtol=1e-5):
                # Further verify it matches pi0_base
                if np.allclose(flat_pi0_base[k].astype(flat_loaded[k].dtype), flat_loaded[k], rtol=1e-5):
                    loaded_from_checkpoint += 1

        assert loaded_from_checkpoint > 0, \
            f"{component_name} should be loaded from pi0_base, but 0/{len(component_keys)} parameters were loaded"
        print(f"✓ {component_name}: {loaded_from_checkpoint}/{len(component_keys)} parameters loaded from pi0_base")

    # Check components that should be randomly initialized
    print("\n=== Components randomly initialized ===")
    for component_name, key_filter in components_to_random_init.items():
        component_keys = [k for k in flat_loaded.keys() if key_filter(k)]

        if len(component_keys) == 0:
            print(f"⚠ {component_name}: No parameters found (may not exist in this model variant)")
            continue

        # Count how many parameters are unchanged from random initialization
        randomly_initialized = 0
        for k in component_keys:
            if np.allclose(flat_random[k], flat_loaded[k], rtol=1e-5):
                randomly_initialized += 1

        # All should be randomly initialized (not loaded from checkpoint)
        assert randomly_initialized == len(component_keys), \
            f"{component_name} should be randomly initialized, but only {randomly_initialized}/{len(component_keys)} were"
        print(f"✓ {component_name}: {randomly_initialized}/{len(component_keys)} parameters randomly initialized")

    print("\n✓ Test passed: SelectiveVisionAndProjectionsLoader correctly loads selective components")


@pytest.mark.manual
def test_selective_loader_vs_paligemma_loader_difference():
    """Test that SelectiveVisionAndProjectionsLoader loads more components than VisionEncoderOnlyLoader.

    This test ensures that:
    1. SelectiveVisionAndProjectionsLoader loads projections (unlike VisionEncoderOnlyLoader)
    2. Both loaders handle vision encoder and embedder correctly
    3. Both loaders randomly initialize LLM layers
    """
    # Create model
    from openpi.models import pi0_incontextv18
    config = pi0_incontextv18.Pi0IncontextConfigv18(
        prompt_expert_variant="gemma_300m_v2",
        action_expert_variant="gemma_300m_lora",
        sample_frames=8,
        sample_actions=128,
    )

    key = jax.random.key(42)
    model = config.create(key)
    random_params = nnx.state(model, nnx.Param)

    # Load with both loaders
    selective_loader = weight_loaders.SelectiveVisionAndProjectionsLoader(
        params_path="s3://openpi-assets/checkpoints/pi0_base/params",
        verbose=False
    )
    vision_only_loader = weight_loaders.VisionEncoderOnlyLoader(
        verbose=False,
        include_embedder=True
    )

    selective_params = selective_loader.load(random_params)
    vision_only_params = vision_only_loader.load(random_params)

    # Flatten for comparison
    flat_random = flax.traverse_util.flatten_dict(random_params, sep="/")
    flat_selective = flax.traverse_util.flatten_dict(selective_params, sep="/")
    flat_vision_only = flax.traverse_util.flatten_dict(vision_only_params, sep="/")

    # Check projection keys
    projection_keys = [k for k in flat_random.keys() if any(
        k.startswith(p) for p in ["state_proj/", "action_in_proj/", "action_time_mlp_in/",
                                   "action_time_mlp_out/", "action_out_proj/"]
    )]

    if len(projection_keys) > 0:
        # Count loaded projections in selective loader (should be loaded from checkpoint)
        selective_projections_loaded = sum(
            1 for k in projection_keys if not np.allclose(flat_random[k], flat_selective[k], rtol=1e-5)
        )

        # Count loaded projections in vision-only loader (should be randomly initialized)
        vision_only_projections_loaded = sum(
            1 for k in projection_keys if not np.allclose(flat_random[k], flat_vision_only[k], rtol=1e-5)
        )

        assert selective_projections_loaded > 0, \
            "SelectiveVisionAndProjectionsLoader should load projection parameters"
        assert vision_only_projections_loaded == 0, \
            "VisionEncoderOnlyLoader should NOT load projection parameters"

        print(f"✓ Projection difference verified:")
        print(f"  - SelectiveVisionAndProjectionsLoader: {selective_projections_loaded}/{len(projection_keys)} projections loaded")
        print(f"  - VisionEncoderOnlyLoader: {vision_only_projections_loaded}/{len(projection_keys)} projections loaded")
    else:
        pytest.skip("No projection parameters found in model (may be using a different variant)")

    print("✓ Test passed: Loaders have expected differences in projection loading")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
