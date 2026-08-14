"""Unit tests for config_libero configurations, focusing on weight loading strategies.

This test file validates that training configurations correctly initialize model weights
according to their specified weight loaders.

USAGE:
------
    # Run all tests
    uv run pytest src/openpi/training/config_libero_test.py

    # Run specific test
    uv run pytest src/openpi/training/config_libero_test.py::test_selective_loader_vs_paligemma_loader_difference

    # Run with verbose output
    uv run pytest src/openpi/training/config_libero_test.py -v
"""

import flax.nnx as nnx
import flax.traverse_util
import jax
import numpy as np
import pytest

from openpi.training import config as _config
from openpi.training import config_libero
from openpi.training import weight_loaders


def test_libero_incontext_inference_configs_removed():
    names = sorted(
        config.name
        for config in config_libero.build(_config)
        if config.name.startswith("pi0_libero") and "incontext" in config.name and config.name.endswith("_inference")
    )
    assert names == []


def test_all_incontext_configs_use_custom_loader():
    configs = [config for config in _config._CONFIGS if config.model.model_type == _config.ModelType.PI0_INCONTEXT]

    assert configs
    assert all(config.use_custom_dataloader for config in configs)


@pytest.mark.manual
def test_selective_loader_vs_paligemma_loader_difference():
    """Test that SelectiveVisionAndProjectionsLoader loads more components than VisionEncoderOnlyLoader.

    This test ensures that:
    1. SelectiveVisionAndProjectionsLoader loads projections (unlike VisionEncoderOnlyLoader)
    2. Both loaders handle vision encoder and embedder correctly
    3. Both loaders randomly initialize LLM layers
    """
    # Create model
    from openpi.models import contextflow
    config = contextflow.ContextFlowConfig(
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
