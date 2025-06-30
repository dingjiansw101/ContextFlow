import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

import numpy as np
import jax
import torch
import math



def make_robocasa_mg_three_image_example() -> dict:
    """Creates a random input example for the RLBench policy."""
    return {
        "observation/state": np.random.rand(9),
        "observation/image_left": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/image_right": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
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
class RobocasaMgThreeImageIncontextInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For RLBench, we do need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        
        get_ob = data["observation/state"]
        if len(get_ob.shape) == 2:
            convert_ob = get_ob[:,:9]
        else:
            convert_ob = get_ob[:9]
        
        state = transforms.pad_to_dim(convert_ob, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image_left = _parse_image(data["observation/image_left"])
        base_image_right = _parse_image(data["observation/image_right"])

        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": wrist_image ,
                "left_wrist_0_rgb": base_image_left, 
                "right_wrist_0_rgb": base_image_right, 
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_, 
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).

            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
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
class RobocasaMgThreeImageIncontextOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For RLBench, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}
