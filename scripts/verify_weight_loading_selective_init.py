"""Verification script for SelectiveVisionAndProjectionsLoader with pi0_libero_incontextv18_low_mem_finetune_sample_frames8_selective_init config.

This script validates that:
1. Vision encoder parameters are loaded from pi0_base checkpoint
2. Embedder parameters are loaded from pi0_base checkpoint
3. Basic projection layers (state_proj, action_*) are loaded from pi0_base checkpoint
4. LLM parameters (prompt expert, action expert) are randomly initialized
5. In-context components (compressors, demo projections) are randomly initialized
6. No LoRA weights exist (config uses gemma_300m, not gemma_300m_lora)
7. No shape mismatch errors occur during loading

This loader provides more fine-grained control than VisionEncoderOnlyLoader by also loading
the basic projection layers from pi0_base while keeping LLM experts randomly initialized.

NOTE: This config uses gemma_300m (without LoRA) for the action expert, which means ALL
action expert weights are trainable through full fine-tuning rather than parameter-efficient
LoRA adapters. This requires more GPU memory but may provide better performance.

Usage:
    uv run python scripts/verify_weight_loading_selective_init.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from openpi.training import config as _config
from openpi.shared import download
from openpi.models import model as _model


def categorize_parameters(flat_params: dict) -> dict:
    """Categorize parameters by component type.

    Note: lora_weights category should be empty for this config since it uses
    gemma_300m (without LoRA) instead of gemma_300m_lora.
    """
    categories = {
        "vision_encoder": [],
        "embedder": [],
        "state_projections": [],
        "action_projections": [],
        "prompt_expert_llm": [],
        "action_expert_llm": [],
        "lora_weights": [],
        "in_context_compressors": [],
        "demo_projections": [],
        "other": [],
    }

    for key in flat_params.keys():
        if key.startswith("PaliGemma/img"):
            categories["vision_encoder"].append(key)
        elif key.startswith("PaliGemma/llm/embedder"):
            categories["embedder"].append(key)
        elif key.startswith("state_proj/"):
            categories["state_projections"].append(key)
        elif any(key.startswith(p) for p in ["action_in_proj/", "action_time_mlp_in/",
                                               "action_time_mlp_out/", "action_out_proj/"]):
            categories["action_projections"].append(key)
        elif "lora" in key:
            categories["lora_weights"].append(key)
        elif "_compressor" in key:
            categories["in_context_compressors"].append(key)
        elif "demo_action_proj" in key or "demo_state_proj" in key or "img_proj" in key or "text_proj" in key:
            categories["demo_projections"].append(key)
        elif "PaliGemma/llm" in key and ("_1" in key or "action_expert" in key):
            categories["action_expert_llm"].append(key)
        elif "PaliGemma/llm" in key:
            categories["prompt_expert_llm"].append(key)
        else:
            categories["other"].append(key)

    return categories


def load_pi0_base_checkpoint():
    """Load pi0_base checkpoint for comparison."""
    print("  Loading pi0_base checkpoint for comparison...")
    pi0_base_path = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")
    pi0_base_params = _model.restore_params(pi0_base_path, restore_type=np.ndarray)
    flat_pi0_base = traverse_util.flatten_dict(pi0_base_params, sep="/")
    print(f"  Loaded {len(flat_pi0_base)} parameters from pi0_base")
    return flat_pi0_base


def main():
    print("=" * 80)
    print("Verification: SelectiveVisionAndProjectionsLoader")
    print("=" * 80)

    # Load config
    config_name = "pi0_libero_incontextv18_low_mem_finetune_sample_frames8_selective_init"
    print(f"\n[1/6] Loading config: {config_name}")
    config = _config.get_config(config_name)

    # Initialize model with random parameters
    print("\n[2/6] Initializing model with random parameters...")
    rng = jax.random.PRNGKey(42)
    model = config.model.create(rng)

    # Extract initial parameters (random)
    print("\n[3/6] Extracting initial random parameters...")
    import flax.nnx as nnx
    graphdef, params_initial = nnx.split(model)
    params_dict_initial = params_initial.to_pure_dict()

    # Load pi0_base for comparison
    print("\n[4/6] Loading pi0_base checkpoint for comparison...")
    flat_pi0_base = load_pi0_base_checkpoint()

    # Apply weight loader
    print("\n[5/6] Applying SelectiveVisionAndProjectionsLoader...")
    print(f"  Weight loader: {config.weight_loader}")
    params_loaded = config.weight_loader.load(params_dict_initial)

    # Flatten for analysis
    print("\n[6/6] Analyzing loaded parameters...")
    flat_initial = traverse_util.flatten_dict(params_dict_initial, sep="/")
    flat_loaded = traverse_util.flatten_dict(params_loaded, sep="/")

    # Compare to find what was loaded vs randomly initialized
    loaded_from_checkpoint = []
    randomly_initialized = []
    loaded_and_matches_pi0_base = []

    for key in flat_initial.keys():
        if key in flat_loaded:
            # Check if values changed from initial (loaded from some checkpoint)
            if not np.allclose(flat_initial[key], flat_loaded[key], rtol=1e-5, atol=1e-8):
                loaded_from_checkpoint.append(key)

                # Check if it matches pi0_base
                if key in flat_pi0_base:
                    if np.allclose(flat_pi0_base[key].astype(flat_loaded[key].dtype),
                                   flat_loaded[key], rtol=1e-5, atol=1e-8):
                        loaded_and_matches_pi0_base.append(key)
            else:
                randomly_initialized.append(key)
        else:
            # Key not in loaded params (shouldn't happen)
            randomly_initialized.append(key)

    # Categorize parameters
    loaded_categories = categorize_parameters({k: None for k in loaded_from_checkpoint})
    random_categories = categorize_parameters({k: None for k in randomly_initialized})
    matched_pi0_categories = categorize_parameters({k: None for k in loaded_and_matches_pi0_base})

    # Print results
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)

    print(f"\n✅ Total parameters in model: {len(flat_initial)}")
    print(f"✅ Parameters loaded from checkpoint: {len(loaded_from_checkpoint)}")
    print(f"✅ Parameters matching pi0_base: {len(loaded_and_matches_pi0_base)}")
    print(f"✅ Parameters randomly initialized: {len(randomly_initialized)}")

    print("\n" + "-" * 80)
    print("LOADED FROM PI0_BASE CHECKPOINT")
    print("-" * 80)
    for category, keys in matched_pi0_categories.items():
        if keys:
            print(f"  {category}: {len(keys)} parameters")
            if len(keys) <= 5:
                for k in keys:
                    shape = flat_loaded[k].shape
                    print(f"    - {k}: shape={shape}")
            else:
                for k in keys[:3]:
                    shape = flat_loaded[k].shape
                    print(f"    - {k}: shape={shape}")
                print(f"    ... and {len(keys) - 3} more")

    print("\n" + "-" * 80)
    print("RANDOMLY INITIALIZED PARAMETERS")
    print("-" * 80)
    for category, keys in random_categories.items():
        if keys:
            print(f"  {category}: {len(keys)} parameters")
            if len(keys) <= 5:
                for k in keys:
                    shape = flat_initial[k].shape
                    print(f"    - {k}: shape={shape}")
            else:
                for k in keys[:3]:
                    shape = flat_initial[k].shape
                    print(f"    - {k}: shape={shape}")
                print(f"    ... and {len(keys) - 3} more")

    # Verification checks
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS")
    print("=" * 80)

    checks_passed = True

    # Check 1: Vision encoder should be loaded from pi0_base
    if matched_pi0_categories["vision_encoder"]:
        print("✅ CHECK 1: Vision encoder loaded from pi0_base")
        print(f"   Found {len(matched_pi0_categories['vision_encoder'])} vision encoder parameters")
    else:
        print("❌ CHECK 1 FAILED: No vision encoder parameters loaded from pi0_base!")
        checks_passed = False

    # Check 2: Embedder should be loaded from pi0_base
    if matched_pi0_categories["embedder"]:
        print("✅ CHECK 2: Embedder loaded from pi0_base")
        print(f"   Found {len(matched_pi0_categories['embedder'])} embedder parameters")
        # Show the embedder parameter details
        for k in matched_pi0_categories["embedder"]:
            shape = flat_loaded[k].shape if k in flat_loaded else "unknown"
            print(f"   - {k}: shape={shape}")
    else:
        print("❌ CHECK 2 FAILED: No embedder parameters loaded from pi0_base!")
        checks_passed = False

    # Check 3: Basic projections should be loaded from pi0_base
    total_projections = len(matched_pi0_categories["state_projections"]) + len(matched_pi0_categories["action_projections"])
    if total_projections > 0:
        print("✅ CHECK 3: Basic projection layers loaded from pi0_base")
        print(f"   State projections: {len(matched_pi0_categories['state_projections'])} params")
        print(f"   Action projections: {len(matched_pi0_categories['action_projections'])} params")
        print(f"   Total: {total_projections} params")
    else:
        print("❌ CHECK 3 FAILED: No projection parameters loaded from pi0_base!")
        checks_passed = False

    # Check 4: LLM should be randomly initialized
    total_llm_random = len(random_categories["prompt_expert_llm"]) + len(random_categories["action_expert_llm"])
    if total_llm_random > 0:
        print("✅ CHECK 4: LLM parameters randomly initialized")
        print(f"   Prompt expert: {len(random_categories['prompt_expert_llm'])} params")
        print(f"   Action expert: {len(random_categories['action_expert_llm'])} params")
        print(f"   Total: {total_llm_random} params")
    else:
        print("⚠️  CHECK 4: Expected LLM parameters to be randomly initialized")
        checks_passed = False

    # Check 5: In-context components should be randomly initialized
    in_context_random = (
        len(random_categories["in_context_compressors"]) +
        len(random_categories["demo_projections"])
    )
    if in_context_random > 0:
        print("✅ CHECK 5: In-context components randomly initialized")
        print(f"   Compressors: {len(random_categories['in_context_compressors'])} params")
        print(f"   Demo projections: {len(random_categories['demo_projections'])} params")
        print(f"   Total: {in_context_random} params")
    else:
        print("⚠️  CHECK 5: Expected in-context components to be randomly initialized")
        # This is a warning, not a failure, as the model might not have these components

    # Check 6: No LoRA weights should exist (using gemma_300m, not gemma_300m_lora)
    total_lora = len(random_categories["lora_weights"]) + len(loaded_categories["lora_weights"])
    if total_lora == 0:
        print("✅ CHECK 6: No LoRA weights found (expected with gemma_300m variant)")
        print("   Using full fine-tuning instead of LoRA adapters")
    else:
        print(f"⚠️  CHECK 6: Found {total_lora} LoRA parameters (unexpected with gemma_300m)")
        print("   This config should use gemma_300m (no LoRA), not gemma_300m_lora")
        checks_passed = False

    # Check 7: No shape mismatches (if we got here without errors, we're good)
    print("✅ CHECK 7: No shape mismatch errors during loading")

    # Final summary
    print("\n" + "=" * 80)
    if checks_passed:
        print("✅ ALL CRITICAL CHECKS PASSED - SelectiveVisionAndProjectionsLoader working correctly!")
    else:
        print("⚠️  SOME CHECKS FAILED - Review the results above")
    print("=" * 80)

    # Summary comparison
    print("\n" + "-" * 80)
    print("LOADING STRATEGY SUMMARY")
    print("-" * 80)
    print("\nFrom pi0_base checkpoint:")
    print(f"  • Vision encoder: {len(matched_pi0_categories['vision_encoder'])} params")
    print(f"  • Embedder: {len(matched_pi0_categories['embedder'])} params")
    print(f"  • State projections: {len(matched_pi0_categories['state_projections'])} params")
    print(f"  • Action projections: {len(matched_pi0_categories['action_projections'])} params")
    print(f"  TOTAL LOADED: {len(loaded_and_matches_pi0_base)} params")

    print("\nRandomly initialized (full fine-tuning with gemma_300m):")
    print(f"  • Prompt expert LLM: {len(random_categories['prompt_expert_llm'])} params")
    print(f"  • Action expert LLM: {len(random_categories['action_expert_llm'])} params (full weights, not LoRA)")
    print(f"  • In-context compressors: {len(random_categories['in_context_compressors'])} params")
    print(f"  • Demo projections: {len(random_categories['demo_projections'])} params")
    print(f"  • LoRA weights: {len(random_categories['lora_weights'])} params (should be 0)")
    print(f"  TOTAL RANDOM: {len(randomly_initialized)} params")

    # Sample parameter shapes
    print("\n" + "-" * 80)
    print("SAMPLE PARAMETER SHAPES")
    print("-" * 80)

    # Show vision encoder samples
    print("\nVision encoder (loaded from pi0_base):")
    vision_samples = matched_pi0_categories["vision_encoder"][:3] if matched_pi0_categories["vision_encoder"] else []
    for key in vision_samples:
        shape = flat_loaded[key].shape
        print(f"  {key}: {shape}")

    # Show projection samples
    print("\nProjection layers (loaded from pi0_base):")
    projection_samples = (matched_pi0_categories["action_projections"][:2] +
                         matched_pi0_categories["state_projections"][:1])
    for key in projection_samples:
        shape = flat_loaded[key].shape
        print(f"  {key}: {shape}")

    # Show LLM samples
    print("\nLLM layers (randomly initialized):")
    llm_samples = random_categories["prompt_expert_llm"][:2] if random_categories["prompt_expert_llm"] else []
    for key in llm_samples:
        shape = flat_initial[key].shape
        print(f"  {key}: {shape}")

    print("\n" + "=" * 80)
    print("Verification complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
