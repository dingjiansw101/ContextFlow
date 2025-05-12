import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
import jax

def make_libero_example() -> dict:
    """Creates a random input example for the Libero incontext policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
        "task_index": 0,
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoIncontextInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0  # We don't mask for pi0-FAST.
        # jax.debug.print("model_type = {}, mask_padding = {}", self.model_type, mask_padding)
        # Get the state. We are padding from 8 to the model action dim.
        # For pi0-FAST, we don't pad the state (action_dim = 7, which is < 8, so pad is skipped).
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            # We are padding from 7 to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "dem_prompt_indexes" in data:
            inputs["dem_prompt_indexes"] = data["dem_prompt_indexes"]
        # if "dem_all_indexes" in data:
        #     inputs["dem_all_indexes"] = data["dem_all_indexes"]
        # if "dem_all_indexes_mask" in data:
        #     inputs["dem_all_indexes_mask"] = data["dem_all_indexes_mask"]
        # if "pos_list" in data:
        #     inputs["pos_list"] = data["pos_list"]
        if "selected_episode" in data:
            inputs["selected_episode"] = data["selected_episode"]
        if "index" in data:
            inputs["index"] = data["index"]
        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoIncontextInputs_refactor(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0_INCONTEXT

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0_INCONTEXT  # We don't mask for pi0-FAST.
        # jax.debug.print("model_type = {}, mask_padding = {}", self.model_type, mask_padding)
        # Get the state. We are padding from 8 to the model action dim.
        # For pi0-FAST, we don't pad the state (action_dim = 7, which is < 8, so pad is skipped).
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            # We are padding from 7 to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "dem_prompt_indexes" in data:
            inputs["dem_prompt_indexes"] = data["dem_prompt_indexes"]
        # if "dem_all_indexes" in data:
        #     inputs["dem_all_indexes"] = data["dem_all_indexes"]
        # if "dem_all_indexes_mask" in data:
        #     inputs["dem_all_indexes_mask"] = data["dem_all_indexes_mask"]
        if "selected_episode" in data:
            inputs["selected_episode"] = data["selected_episode"]
        if "index" in data:
            inputs["index"] = data["index"]
        return inputs

@dataclasses.dataclass(frozen=True)
class LiberoIncontextOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :7])}
