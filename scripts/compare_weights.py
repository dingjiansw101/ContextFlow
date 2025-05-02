import numpy as np
from flax import traverse_util

from openpi.models import model as _model
from openpi.shared import download
from openpi.training import weight_loaders

# Note: restore_params loads np.ndarray by default in the loader
pi0_params_path = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")
pi0_params = _model.restore_params(pi0_params_path, restore_type=np.ndarray)

# Load PaliGemma params
paligemma_loader = weight_loaders.PaliGemmaWeightLoader()
paligemma_npz_path = download.maybe_download(
    "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
)
with paligemma_npz_path.open("rb") as f:
    flat_params = dict(np.load(f, allow_pickle=False))
paligemma_params = {"PaliGemma": traverse_util.unflatten_dict(flat_params, sep="/")["params"]}

# Now compare pi0_params and paligemma_params
# Example: Compare top-level keys
print("pi0 top-level keys:", pi0_params.keys())
print("PaliGemma top-level keys:", paligemma_params.keys())
# pi0_params['PaliGemma']['img']
# paligemma_params['PaliGemma']['img']
# pi0_params['PaliGemma']['img']['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['key']['kernel']
# paligemma_params['PaliGemma']['img']['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['key']['kernel']
import ipdb; ipdb.set_trace()
# Example: Compare shapes of a specific weight if keys match/are known
# Assuming pi0_params also has a 'PaliGemma' key or similar structure
# key_to_compare = ('PaliGemma', 'llm', 'embedder', 'input_embedding')
# if key_to_compare in traverse_util.flatten_dict(pi0_params) and key_to_compare in traverse_util.flatten_dict(paligemma_params):
#    pi0_shape = pi0_params['PaliGemma']['llm']['embedder']['input_embedding'].shape
#    paligemma_shape = paligemma_params['PaliGemma']['llm']['embedder']['input_embedding'].shape
#    print(f"Shape comparison for {key_to_compare}: pi0={pi0_shape}, PaliGemma={paligemma_shape}")
# else:
#    print(f"Could not compare key {key_to_compare}")

# Further comparison logic (iterating through keys, comparing shapes/values) goes here.
