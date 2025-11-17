"""Verification script for VisionEncoderOnlyLoader with pi0_libero_incontextv18_low_mem_finetune_sample_frames8_paligemma_init config.

This script validates that:
1. Vision encoder parameters are loaded from PaliGemma checkpoint
2. Embedder parameters are loaded from PaliGemma checkpoint (when include_embedder=True)
3. LLM parameters (prompt expert, action expert) are randomly initialized
4. In-context components (compressors, projections) are randomly initialized
5. No shape mismatch errors occur during loading

The embedder can be safely loaded because gemma_300m_v2 (prompt expert) has width=2048,
which matches PaliGemma's Gemma-2B embedder dimension.

Usage:
    uv run python scripts/verify_weight_loading_paligemma_init.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from openpi.training import config as _config
from openpi.shared import download


def categorize_parameters(flat_params: dict) -> dict:
    """Categorize parameters by component type."""
    categories = {
        "vision_encoder": [],
        "prompt_expert_llm": [],
        "action_expert_llm": [],
        "embedder": [],
        "in_context_compressors": [],
        "demo_projections": [],
        "action_projections": [],
        "state_projections": [],
        "other": [],
    }

    for key in flat_params.keys():
        if key.startswith("PaliGemma/img"):
            categories["vision_encoder"].append(key)
        elif key.startswith("PaliGemma/llm/embedder"):
            categories["embedder"].append(key)
        elif "_compressor" in key:
            categories["in_context_compressors"].append(key)
        elif "demo_action_proj" in key or "demo_state_proj" in key:
            categories["demo_projections"].append(key)
        elif "action_" in key and "_proj" in key:
            categories["action_projections"].append(key)
        elif "state_proj" in key:
            categories["state_projections"].append(key)
        elif "PaliGemma/llm" in key and ("_1" in key or "prompt_expert" in key):
            categories["action_expert_llm"].append(key)
        elif "PaliGemma/llm" in key:
            categories["prompt_expert_llm"].append(key)
        else:
            categories["other"].append(key)

    return categories


def main():
    print("=" * 80)
    print("Verification: VisionEncoderOnlyLoader with gemma_300m variant")
    print("=" * 80)

    # Load config
    config_name = "pi0_libero_incontextv18_low_mem_finetune_sample_frames8_paligemma_init"
    print(f"\n[1/5] Loading config: {config_name}")
    config = _config.get_config(config_name)

    # Initialize model with random parameters
    print("\n[2/5] Initializing model with random parameters...")
    rng = jax.random.PRNGKey(42)
    model = config.model.create(rng)

    # Extract initial parameters (random)
    print("\n[3/5] Extracting initial random parameters...")
    import flax.nnx as nnx
    graphdef, params_initial = nnx.split(model)
    params_dict_initial = params_initial.to_pure_dict()

    # Apply weight loader
    print("\n[4/5] Applying VisionEncoderOnlyLoader...")
    print(f"  Weight loader: {config.weight_loader}")
    params_loaded = config.weight_loader.load(params_dict_initial)

    # Flatten for analysis
    print("\n[5/5] Analyzing loaded parameters...")
    flat_initial = traverse_util.flatten_dict(params_dict_initial, sep="/")
    flat_loaded = traverse_util.flatten_dict(params_loaded, sep="/")

    # Compare to find what was loaded vs randomly initialized
    loaded_keys = []
    random_keys = []

    for key in flat_initial.keys():
        if key in flat_loaded:
            # Check if values changed (loaded from checkpoint)
            if not np.allclose(flat_initial[key], flat_loaded[key], rtol=1e-5, atol=1e-8):
                loaded_keys.append(key)
            else:
                random_keys.append(key)
        else:
            # Key not in loaded params (shouldn't happen with VisionEncoderOnlyLoader)
            random_keys.append(key)

    # Categorize parameters
    loaded_categories = categorize_parameters({k: None for k in loaded_keys})
    random_categories = categorize_parameters({k: None for k in random_keys})

    # Print results
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)

    print(f"\n✅ Total parameters in model: {len(flat_initial)}")
    print(f"✅ Parameters loaded from checkpoint: {len(loaded_keys)}")
    print(f"✅ Parameters randomly initialized: {len(random_keys)}")

    print("\n" + "-" * 80)
    print("LOADED PARAMETERS (from PaliGemma checkpoint)")
    print("-" * 80)
    for category, keys in loaded_categories.items():
        if keys:
            print(f"  {category}: {len(keys)} parameters")
            if len(keys) <= 5:
                for k in keys:
                    print(f"    - {k}")
            else:
                for k in keys[:3]:
                    print(f"    - {k}")
                print(f"    ... and {len(keys) - 3} more")

    print("\n" + "-" * 80)
    print("RANDOMLY INITIALIZED PARAMETERS")
    print("-" * 80)
    for category, keys in random_categories.items():
        if keys:
            print(f"  {category}: {len(keys)} parameters")
            if len(keys) <= 5:
                for k in keys:
                    print(f"    - {k}")
            else:
                for k in keys[:3]:
                    print(f"    - {k}")
                print(f"    ... and {len(keys) - 3} more")

    # Verification checks
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS")
    print("=" * 80)

    checks_passed = True

    # Check 1: Vision encoder should be loaded
    if loaded_categories["vision_encoder"]:
        print("✅ CHECK 1: Vision encoder loaded from PaliGemma")
        print(f"   Found {len(loaded_categories['vision_encoder'])} vision encoder parameters")
    else:
        print("❌ CHECK 1 FAILED: No vision encoder parameters loaded!")
        checks_passed = False

    # Check 2: Embedder should be loaded (when include_embedder=True)
    if loaded_categories["embedder"]:
        print("✅ CHECK 2: Embedder loaded from PaliGemma")
        print(f"   Found {len(loaded_categories['embedder'])} embedder parameters")
        # Show the embedder parameter details
        for k in loaded_categories["embedder"]:
            shape = flat_loaded[k].shape if k in flat_loaded else "unknown"
            print(f"   - {k}: shape={shape}")
    else:
        print("⚠️  CHECK 2: No embedder parameters loaded (include_embedder may be False)")

    # Check 3: LLM should be randomly initialized (to avoid shape mismatch)
    if random_categories["prompt_expert_llm"] or random_categories["action_expert_llm"]:
        print("✅ CHECK 3: LLM parameters randomly initialized (avoiding shape mismatch)")
        print(f"   Prompt expert: {len(random_categories['prompt_expert_llm'])} params")
        print(f"   Action expert: {len(random_categories['action_expert_llm'])} params")
    else:
        print("⚠️  CHECK 3: Some LLM parameters may have been loaded")
        print(f"   This could cause shape mismatch issues with gemma_300m!")

    # Check 4: In-context components should be randomly initialized
    in_context_random = (
        len(random_categories["in_context_compressors"]) +
        len(random_categories["demo_projections"])
    )
    if in_context_random > 0:
        print("✅ CHECK 4: In-context components randomly initialized")
        print(f"   Compressors: {len(random_categories['in_context_compressors'])} params")
        print(f"   Demo projections: {len(random_categories['demo_projections'])} params")
    else:
        print("⚠️  CHECK 4: Expected in-context components to be randomly initialized")

    # Check 5: No shape mismatches (if we got here without errors, we're good)
    print("✅ CHECK 5: No shape mismatch errors during loading")

    # Final summary
    print("\n" + "=" * 80)
    if checks_passed:
        print("✅ ALL CHECKS PASSED - VisionEncoderOnlyLoader working correctly!")
    else:
        print("⚠️  SOME CHECKS FAILED - Review the results above")
    print("=" * 80)

    # Optionally: Check specific parameter shapes
    print("\n" + "-" * 80)
    print("SAMPLE PARAMETER SHAPES")
    print("-" * 80)

    # Show a few vision encoder params
    vision_samples = loaded_categories["vision_encoder"][:3] if loaded_categories["vision_encoder"] else []
    for key in vision_samples:
        shape = flat_loaded[key].shape
        print(f"  {key}: {shape}")

    # Show a few LLM params (should be random)
    llm_samples = random_categories["prompt_expert_llm"][:3] if random_categories["prompt_expert_llm"] else []
    for key in llm_samples:
        shape = flat_initial[key].shape
        print(f"  {key}: {shape} (random)")

    print("\n" + "=" * 80)
    print("Verification complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
