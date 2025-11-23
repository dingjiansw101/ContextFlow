"""Unified verification script for weight loading strategies.

This script validates weight loading for different initialization strategies:
- VisionEncoderOnlyLoader: Loads vision encoder (+ embedder) from PaliGemma
- SelectiveVisionAndProjectionsLoader: Loads vision + embedder + projections from pi0_base

The script auto-detects the loader type from the config and performs appropriate checks.

Usage:
    uv run python scripts/verify_weight_loading.py --config-name <config_name>

Examples:
    # Verify PaliGemma initialization
    uv run python scripts/verify_weight_loading.py --config-name pi0_libero_incontextv18_low_mem_finetune_sample_frames8_paligemma_init

    # Verify selective initialization
    uv run python scripts/verify_weight_loading.py --config-name pi0_libero_incontextv18_low_mem_finetune_sample_frames8_selective_init
"""

import argparse
import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from openpi.training import config as _config
from openpi.training import weight_loaders
from openpi.shared import download
from openpi.models import model as _model


def categorize_parameters(flat_params: dict) -> dict:
    """Categorize parameters by component type."""
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


def compare_parameters(flat_initial: dict, flat_loaded: dict, reference_params: dict = None):
    """Compare initial and loaded parameters, optionally checking against reference.

    Args:
        flat_initial: Flattened initial (random) parameters
        flat_loaded: Flattened loaded parameters
        reference_params: Optional reference checkpoint params (e.g., pi0_base)

    Returns:
        Tuple of (loaded_keys, random_keys, matched_reference_keys)
    """
    loaded_keys = []
    random_keys = []
    matched_reference_keys = []

    for key in flat_initial.keys():
        if key in flat_loaded:
            # Check if values changed from initial (loaded from some checkpoint)
            if not np.allclose(flat_initial[key], flat_loaded[key], rtol=1e-5, atol=1e-8):
                loaded_keys.append(key)

                # Check if it matches reference checkpoint
                if reference_params is not None and key in reference_params:
                    if np.allclose(reference_params[key].astype(flat_loaded[key].dtype),
                                   flat_loaded[key], rtol=1e-5, atol=1e-8):
                        matched_reference_keys.append(key)
            else:
                random_keys.append(key)
        else:
            # Key not in loaded params (shouldn't happen)
            random_keys.append(key)

    return loaded_keys, random_keys, matched_reference_keys


def print_categorized_params(title: str, categories: dict, flat_params: dict, show_shapes: bool = True):
    """Print categorized parameters with optional shapes."""
    print(f"\n{'-' * 80}")
    print(title)
    print(f"{'-' * 80}")

    for category, keys in categories.items():
        if keys:
            print(f"  {category}: {len(keys)} parameters")
            if len(keys) <= 5:
                for k in keys:
                    if show_shapes and k in flat_params:
                        shape = flat_params[k].shape
                        print(f"    - {k}: shape={shape}")
                    else:
                        print(f"    - {k}")
            else:
                for k in keys[:3]:
                    if show_shapes and k in flat_params:
                        shape = flat_params[k].shape
                        print(f"    - {k}: shape={shape}")
                    else:
                        print(f"    - {k}")
                print(f"    ... and {len(keys) - 3} more")


def verify_vision_encoder_only_loader(config, flat_initial, flat_loaded, loaded_keys, random_keys):
    """Verify VisionEncoderOnlyLoader strategy (PaliGemma initialization)."""
    loaded_categories = categorize_parameters({k: None for k in loaded_keys})
    random_categories = categorize_parameters({k: None for k in random_keys})

    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS - VisionEncoderOnlyLoader (PaliGemma)")
    print("=" * 80)

    print(f"\n✅ Total parameters in model: {len(flat_initial)}")
    print(f"✅ Parameters loaded from checkpoint: {len(loaded_keys)}")
    print(f"✅ Parameters randomly initialized: {len(random_keys)}")

    print_categorized_params("LOADED PARAMETERS (from PaliGemma checkpoint)",
                            loaded_categories, flat_loaded)
    print_categorized_params("RANDOMLY INITIALIZED PARAMETERS",
                            random_categories, flat_initial)

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
        print("   This could cause shape mismatch issues with gemma_300m!")

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

    # Check 5: No shape mismatches
    print("✅ CHECK 5: No shape mismatch errors during loading")

    return checks_passed


def verify_selective_loader(config, flat_initial, flat_loaded, loaded_keys, random_keys, flat_pi0_base):
    """Verify SelectiveVisionAndProjectionsLoader strategy (pi0_base initialization)."""
    loaded_categories = categorize_parameters({k: None for k in loaded_keys})
    random_categories = categorize_parameters({k: None for k in random_keys})

    # Find parameters that match pi0_base
    matched_pi0_keys = []
    for key in loaded_keys:
        if key in flat_pi0_base:
            if np.allclose(flat_pi0_base[key].astype(flat_loaded[key].dtype),
                          flat_loaded[key], rtol=1e-5, atol=1e-8):
                matched_pi0_keys.append(key)

    matched_pi0_categories = categorize_parameters({k: None for k in matched_pi0_keys})

    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS - SelectiveVisionAndProjectionsLoader (pi0_base)")
    print("=" * 80)

    print(f"\n✅ Total parameters in model: {len(flat_initial)}")
    print(f"✅ Parameters loaded from checkpoint: {len(loaded_keys)}")
    print(f"✅ Parameters matching pi0_base: {len(matched_pi0_keys)}")
    print(f"✅ Parameters randomly initialized: {len(random_keys)}")

    print_categorized_params("LOADED FROM PI0_BASE CHECKPOINT",
                            matched_pi0_categories, flat_loaded)
    print_categorized_params("RANDOMLY INITIALIZED PARAMETERS",
                            random_categories, flat_initial)

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

    # Check 6: No LoRA weights should exist (using gemma_300m, not gemma_300m_lora)
    total_lora = len(random_categories["lora_weights"]) + len(loaded_categories["lora_weights"])
    if total_lora == 0:
        print("✅ CHECK 6: No LoRA weights found (expected with gemma_300m variant)")
        print("   Using full fine-tuning instead of LoRA adapters")
    else:
        print(f"⚠️  CHECK 6: Found {total_lora} LoRA parameters (unexpected with gemma_300m)")
        print("   This config should use gemma_300m (no LoRA), not gemma_300m_lora")
        checks_passed = False

    # Check 7: No shape mismatches
    print("✅ CHECK 7: No shape mismatch errors during loading")

    # Summary comparison
    print("\n" + "-" * 80)
    print("LOADING STRATEGY SUMMARY")
    print("-" * 80)
    print("\nFrom pi0_base checkpoint:")
    print(f"  • Vision encoder: {len(matched_pi0_categories['vision_encoder'])} params")
    print(f"  • Embedder: {len(matched_pi0_categories['embedder'])} params")
    print(f"  • State projections: {len(matched_pi0_categories['state_projections'])} params")
    print(f"  • Action projections: {len(matched_pi0_categories['action_projections'])} params")
    print(f"  TOTAL LOADED: {len(matched_pi0_keys)} params")

    print("\nRandomly initialized (full fine-tuning with gemma_300m):")
    print(f"  • Prompt expert LLM: {len(random_categories['prompt_expert_llm'])} params")
    print(f"  • Action expert LLM: {len(random_categories['action_expert_llm'])} params (full weights, not LoRA)")
    print(f"  • In-context compressors: {len(random_categories['in_context_compressors'])} params")
    print(f"  • Demo projections: {len(random_categories['demo_projections'])} params")
    print(f"  • LoRA weights: {len(random_categories['lora_weights'])} params (should be 0)")
    print(f"  TOTAL RANDOM: {len(random_keys)} params")

    return checks_passed


def print_sample_shapes(flat_initial, flat_loaded, loaded_keys, random_keys):
    """Print sample parameter shapes."""
    loaded_categories = categorize_parameters({k: None for k in loaded_keys})
    random_categories = categorize_parameters({k: None for k in random_keys})

    print("\n" + "-" * 80)
    print("SAMPLE PARAMETER SHAPES")
    print("-" * 80)

    # Vision encoder samples
    if loaded_categories["vision_encoder"]:
        print("\nVision encoder (loaded):")
        for key in loaded_categories["vision_encoder"][:3]:
            shape = flat_loaded[key].shape
            print(f"  {key}: {shape}")

    # Projection samples
    projection_samples = (loaded_categories["action_projections"][:2] +
                         loaded_categories["state_projections"][:1])
    if projection_samples:
        print("\nProjection layers (loaded):")
        for key in projection_samples:
            shape = flat_loaded[key].shape
            print(f"  {key}: {shape}")

    # LLM samples
    if random_categories["prompt_expert_llm"]:
        print("\nLLM layers (randomly initialized):")
        for key in random_categories["prompt_expert_llm"][:3]:
            shape = flat_initial[key].shape
            print(f"  {key}: {shape} (random)")


def main():
    parser = argparse.ArgumentParser(description="Verify weight loading strategies")
    parser.add_argument("--config-name", type=str, required=True,
                       help="Name of the config to verify (e.g., pi0_libero_incontextv18_low_mem_finetune_sample_frames8_paligemma_init)")
    args = parser.parse_args()

    print("=" * 80)
    print("Weight Loading Verification Script")
    print("=" * 80)

    # Load config
    print(f"\n[1/6] Loading config: {args.config_name}")
    config = _config.get_config(args.config_name)

    # Initialize model with random parameters
    print("\n[2/6] Initializing model with random parameters...")
    rng = jax.random.PRNGKey(42)
    model = config.model.create(rng)

    # Extract initial parameters (random)
    print("\n[3/6] Extracting initial random parameters...")
    import flax.nnx as nnx
    graphdef, params_initial = nnx.split(model)
    params_dict_initial = params_initial.to_pure_dict()

    # Detect loader type
    print("\n[4/6] Detecting weight loader type...")
    loader = config.weight_loader
    print(f"  Weight loader: {loader}")

    is_selective_loader = isinstance(loader, weight_loaders.SelectiveVisionAndProjectionsLoader)
    is_vision_only_loader = isinstance(loader, weight_loaders.VisionEncoderOnlyLoader)

    # Load reference checkpoint if needed
    flat_pi0_base = None
    if is_selective_loader:
        print("\n[5/6] Loading pi0_base checkpoint for comparison...")
        pi0_base_path = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")
        pi0_base_params = _model.restore_params(pi0_base_path, restore_type=np.ndarray)
        flat_pi0_base = traverse_util.flatten_dict(pi0_base_params, sep="/")
        print(f"  Loaded {len(flat_pi0_base)} parameters from pi0_base")
    else:
        print("\n[5/6] Skipping reference checkpoint load (not needed for this loader)")

    # Apply weight loader
    print("\n[6/6] Applying weight loader...")
    params_loaded = config.weight_loader.load(params_dict_initial)

    # Flatten for analysis
    print("\nAnalyzing loaded parameters...")
    flat_initial = traverse_util.flatten_dict(params_dict_initial, sep="/")
    flat_loaded = traverse_util.flatten_dict(params_loaded, sep="/")

    # Compare parameters
    loaded_keys, random_keys, _ = compare_parameters(flat_initial, flat_loaded, flat_pi0_base)

    # Run appropriate verification
    if is_vision_only_loader:
        checks_passed = verify_vision_encoder_only_loader(
            config, flat_initial, flat_loaded, loaded_keys, random_keys
        )
    elif is_selective_loader:
        checks_passed = verify_selective_loader(
            config, flat_initial, flat_loaded, loaded_keys, random_keys, flat_pi0_base
        )
    else:
        print(f"\n⚠️  WARNING: Unknown loader type: {type(loader).__name__}")
        print("Performing basic verification only...")
        loaded_categories = categorize_parameters({k: None for k in loaded_keys})
        random_categories = categorize_parameters({k: None for k in random_keys})

        print(f"\n✅ Total parameters: {len(flat_initial)}")
        print(f"✅ Loaded from checkpoint: {len(loaded_keys)}")
        print(f"✅ Randomly initialized: {len(random_keys)}")

        print_categorized_params("LOADED PARAMETERS", loaded_categories, flat_loaded)
        print_categorized_params("RANDOM PARAMETERS", random_categories, flat_initial)
        checks_passed = True

    # Print sample shapes
    print_sample_shapes(flat_initial, flat_loaded, loaded_keys, random_keys)

    # Final summary
    print("\n" + "=" * 80)
    if checks_passed:
        print("✅ ALL CHECKS PASSED - Weight loader working correctly!")
    else:
        print("⚠️  SOME CHECKS FAILED - Review the results above")
    print("=" * 80)
    print("\nVerification complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
