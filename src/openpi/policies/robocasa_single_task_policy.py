import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

import numpy as np
import jax
import torch
import math



def make_robocasa_single_task_three_image_example() -> dict:
    """Creates a random input example for the RLBench policy."""
    return {
        "observation/state": np.random.rand(9),
        "observation/image_left": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/image_right": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # quat = np.asarray(quat)
        # Defensive checks
    # if quat.shape != (4,):
    #     raise ValueError(f"Expected single quaternion of shape (4,), got {quat.shape}")

    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


@dataclasses.dataclass(frozen=True)
class RobocasaSingleTaskThreeImageInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    action_dim: int

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0
        
        get_ob = data["observation/state"]
        if len(get_ob.shape) == 2:
            # get components of action
            pos = get_ob[:, 11:14] 
            quat = get_ob[:, 14:18]
            gripper = get_ob[:, 7:9]

                # convert to axis-angle (in batch)
            axis_angle = np.stack([_quat2axisangle(q) for q in quat])

                # concat：pos + axis_angle + gripper
            convert_ob = np.concatenate([pos, axis_angle, gripper], axis=1, dtype=np.float32)
        else:
            pos = get_ob[11:14] 
            quat = get_ob[14:18]
            gripper = get_ob[7:9]

                # convert to axis-angle (in batch)
            axis_angle = _quat2axisangle(quat)
                # concat：pos + axis_angle + gripper
            convert_ob = np.concatenate([pos, axis_angle, gripper], axis=0, dtype=np.float32)

        new_obs = transforms.pad_to_dim(convert_ob, self.action_dim)
        state = transforms.pad_to_dim(new_obs, self.action_dim)

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

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaSingleTaskThreeImageOutputs(transforms.DataTransformFn):
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
        return {"actions": np.asarray(data["actions"][:, :12])}
