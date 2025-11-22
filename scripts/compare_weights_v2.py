import numpy as np
from flax import traverse_util
import flax.core

from openpi.models import model as _model
from openpi.shared import download
from openpi.training import weight_loaders
# compare the weights of vision encoder of pi0 and fine-tuned in-context models
# Note: restore_params loads np.ndarray by default in the loader
pi0_params_path = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")
pi0_params = _model.restore_params(pi0_params_path, restore_type=np.ndarray)

# # Load PaliGemma params
# v7_params_path = download.maybe_download("checkpoints/pi0_libero_incontextv7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/pi0_libero_incontextv7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/19999/params")
# v7_params = _model.restore_params(v7_params_path, restore_type=np.ndarray)

# Load PaliGemma params
paligemma_loader = weight_loaders.PaliGemmaWeightLoader()
paligemma_npz_path = download.maybe_download(
    "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
)
with paligemma_npz_path.open("rb") as f:
    flat_params = dict(np.load(f, allow_pickle=False))
# Note: The original PaliGemma checkpoint has an extra 'params' level
paligemma_params = {"PaliGemma": traverse_util.unflatten_dict(flat_params, sep="/")["params"]}


pi0_libero_params_path = download.maybe_download("checkpoints/pi0_libero/pi0_libero/29999/params")
pi0_libero_params = _model.restore_params(pi0_libero_params_path, restore_type=np.ndarray)

pi0_libero_10000_params_path = download.maybe_download("checkpoints/pi0_libero/pi0_libero/10000/params")
pi0_libero_10000_params = _model.restore_params(pi0_libero_10000_params_path, restore_type=np.ndarray)


def compare_params(params1, params2, name1="params1", name2="params2", base_path=(), rtol=1e-05, atol=1e-08):
    """
    Recursively compares two parameter structures (dictionaries, arrays).

    Args:
        params1: The first parameter structure.
        params2: The second parameter structure.
        name1 (str): Name for the first parameter set (for printing).
        name2 (str): Name for the second parameter set (for printing).
        base_path (tuple): The current path within the nested structure (for printing).
        rtol (float): Relative tolerance for np.allclose.
        atol (float): Absolute tolerance for np.allclose.

    Returns:
        bool: True if all numerical values are close and structures match, False otherwise.
    """
    all_close = True

    # Check if both are dictionary-like
    if isinstance(params1, (dict, flax.core.FrozenDict)) and isinstance(params2, (dict, flax.core.FrozenDict)):
        keys1 = set(params1.keys())
        keys2 = set(params2.keys())

        if keys1 != keys2:
            print(f"  - Key mismatch at path {base_path}:")
            print(f"    Keys only in {name1}: {keys1 - keys2}")
            print(f"    Keys only in {name2}: {keys2 - keys1}")
            return False # Structure differs, cannot compare further down this path

        # print(f"Comparing keys at path: {base_path or '(root)'}") # Optional: Less verbose output
        for key in sorted(keys1): # Sort for consistent output
            current_path = base_path + (key,)
            val1 = params1[key]
            val2 = params2[key]
            if not compare_params(val1, val2, name1, name2, current_path, rtol, atol):
                all_close = False
                # Continue checking other keys even if one difference is found at this level

    # Check if both are NumPy arrays
    elif isinstance(params1, np.ndarray) and isinstance(params2, np.ndarray):
        if params1.shape != params2.shape:
            print(f"  - Shape mismatch for key path {base_path}: {name1}={params1.shape}, {name2}={params2.shape}")
            all_close = False
        elif not np.allclose(params1, params2, rtol=rtol, atol=atol):
            diff = np.abs(params1 - params2).max()
            print(f"  - Values are not close for key path {base_path}. Max absolute difference: {diff:.2e}")
            all_close = False
        # else: # Optional: Print if they are close
            # print(f"  - Values are close for key path: {base_path} (ndarray)")

    # Check for type mismatch
    elif type(params1) != type(params2):
        print(f"  - Type mismatch for key path {base_path}: {name1}={type(params1)}, {name2}={type(params2)}")
        all_close = False

    # Handle other scalar types (int, float, bool, etc.)
    else:
        # Use np.allclose for floating point scalars as well
        if isinstance(params1, (float, np.floating)) and isinstance(params2, (float, np.floating)):
             if not np.allclose(params1, params2, rtol=rtol, atol=atol):
                 print(f"  - Float values differ for key path {base_path}: {name1}={params1}, {name2}={params2}")
                 all_close = False
        elif params1 != params2: # Direct comparison for non-float scalars
            print(f"  - Values differ for key path {base_path}: {name1}={params1}, {name2}={params2}")
            all_close = False
        # else: # Optional: Print if they are identical
            # print(f"  - Values are identical for key path: {base_path} (scalar)")

    return all_close

# --- Comparison Execution ---

# Extract the image parameters
pi0_libero_img_params = pi0_libero_params['PaliGemma']['img']
pi0_libero_img_10000_params = pi0_libero_10000_params['PaliGemma']['img']
PaliGemma_img_params = paligemma_params['PaliGemma']['img']
pi0_img_params = pi0_params['PaliGemma']['img']

print("\n" + "="*20 + " Comparing pi0_libero (29999) vs pi0_libero (10000) Image Params " + "="*20)
are_close_libero_img = compare_params(
    pi0_libero_img_params,
    pi0_libero_img_10000_params,
    name1="pi0_libero_29999_img",
    name2="pi0_libero_10000_img"
)

if are_close_libero_img:
    print(f"\nResult: All compared values in pi0_libero_29999['PaliGemma']['img'] and pi0_libero_10000['PaliGemma']['img'] are close.")
else:
    print(f"\nResult: Differences found between pi0_libero_29999['PaliGemma']['img'] and pi0_libero_10000['PaliGemma']['img'].")


# Compare pi0_base vs pi0_libero (29999)
print("\n" + "="*20 + " Comparing pi0_base vs pi0_libero (29999) Full Params " + "="*20)
are_close_base_vs_libero = compare_params(
    pi0_params,
    pi0_libero_params,
    name1="pi0_base",
    name2="pi0_libero_29999"
)
if are_close_base_vs_libero:
    print(f"\nResult: All compared values in pi0_base and pi0_libero_29999 are close.")
else:
    print(f"\nResult: Differences found between pi0_base and pi0_libero_29999.")


# Compare original PaliGemma vs pi0_base
print("\n" + "="*20 + " Comparing Original PaliGemma vs pi0_base Full Params " + "="*20)
are_close_paligemma_vs_base = compare_params(
    PaliGemma_img_params,
    pi0_img_params,
    name1="paligemma_original",
    name2="pi0_base"
)
if are_close_paligemma_vs_base:
    print(f"\nResult: All compared values in original PaliGemma and pi0_base are close.")
else:
    print(f"\nResult: Differences found between original PaliGemma and pi0_base.")


# Example: Compare the full parameter sets (optional)
# print("\n" + "="*20 + " Comparing Full pi0_libero (29999) vs pi0_libero (10000) Params " + "="*20)
# are_all_params_close = compare_params(
#     pi0_libero_params,
#     pi0_libero_10000_params,
#     name1="pi0_libero_29999_full",
#     name2="pi0_libero_10000_full"
# )
# if are_all_params_close:
#     print(f"\nResult: All compared values in the full parameter sets are close.")
# else:
#     print(f"\nResult: Differences found between the full parameter sets.")



# --- Old comparison code removed ---
