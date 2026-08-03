import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaMobileIncontextInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    # adapt_to_pi: bool = True
    adapt_to_pi: bool = False

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # in the current
            # print("actions shape: ", actions.shape)
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "dem_prompt_indexes" in data:
            inputs["dem_prompt_indexes"] = data["dem_prompt_indexes"]
        if "selected_episode" in data:
            inputs["selected_episode"] = data["selected_episode"]
        if "index" in data:
            inputs["index"] = data["index"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CustomLeRobotAlohaMobileIncontextInputs(transforms.DataTransformFn):
    """Process current observations and dem_prompt_* fields emitted by CustomLeRobotDataset.

    Cache-free counterpart of AlohaMobileIncontextInputs: demonstration states/actions/images
    come directly from the dataset (raw columns) instead of the JSON state/action caches, so
    this transform reproduces the value pipeline the legacy caches baked in:
      - demo states are padded to action_dim (matching AlohaMobileIncontextInputs on each frame),
      - demo actions are padded and (optionally) delta-encoded against their own frame's state,
        matching DeltaActions applied per cached frame,
      - normalization of dem_prompt_all_* happens later in Normalize via norm_stats_aliases.
    """

    # The action dimension of the model. Used to pad state, actions, and demo tensors.
    action_dim: int

    # See AlohaMobileIncontextInputs.adapt_to_pi.
    adapt_to_pi: bool = False

    # When set, demo actions are delta-encoded against the demo frame's state over these
    # dimensions, mirroring what DeltaActions did to each frame before the legacy cache was
    # built. Must equal the DeltaActions mask used for current-frame actions.
    delta_action_mask: tuple[bool, ...] | None = None

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    # Demo camera name -> model image key. Demo frames come from the raw dataset as
    # [num_frames, H, W, 3] uint8 stacks keyed by camera name.
    DEMO_CAMERA_TO_MODEL_KEY: ClassVar[dict[str, str]] = {
        "cam_high": "base_0_rgb",
        "cam_left_wrist": "left_wrist_0_rgb",
        "cam_right_wrist": "right_wrist_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_high"]

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Demo images: [num_frames, H, W, 3] uint8 stacks straight from the raw dataset
        # (already HWC; the legacy float round-trip is value-identity for uint8).
        if "dem_prompt_images" in data:
            dem_images_processed = {}
            dem_image_mask = {}
            for camera, stack in data["dem_prompt_images"].items():
                model_key = self.DEMO_CAMERA_TO_MODEL_KEY[camera]
                stack_np = np.asarray(stack)
                dem_images_processed[model_key] = stack_np
                dem_image_mask[model_key] = np.ones(len(stack_np), dtype=bool)
            inputs["dem_prompt_images"] = dem_images_processed
            inputs["dem_prompt_images_mask"] = dem_image_mask

        # Demo states: [num_samples, 14] raw -> padded to action_dim, like each cached frame was.
        dem_states_padded = None
        if "dem_prompt_states" in data:
            dem_states = np.asarray(data["dem_prompt_states"])
            dem_states = _decode_state_batch(dem_states, adapt_to_pi=self.adapt_to_pi)
            dem_states_padded = transforms.pad_to_dim(dem_states, self.action_dim, axis=-1)
            inputs["dem_prompt_all_states"] = dem_states_padded
            inputs["dem_prompt_all_states_mask"] = np.ones(len(dem_states_padded), dtype=bool)

        # Demo actions: [num_samples, 16] raw -> encode -> pad -> per-frame delta (as the
        # legacy cache rows were: DeltaActions ran on each frame before caching row 0).
        if "dem_prompt_actions" in data:
            dem_actions = np.asarray(data["dem_prompt_actions"])
            dem_actions = _encode_actions_inv(dem_actions, adapt_to_pi=self.adapt_to_pi)
            dem_actions_padded = transforms.pad_to_dim(dem_actions, self.action_dim, axis=-1)
            if self.delta_action_mask is not None:
                if dem_states_padded is None:
                    raise ValueError("delta_action_mask requires dem_prompt_states alongside dem_prompt_actions")
                mask = np.asarray(self.delta_action_mask)
                dims = mask.shape[-1]
                dem_actions_padded = dem_actions_padded.copy()
                dem_actions_padded[..., :dims] -= np.where(mask, dem_states_padded[..., :dims], 0)
            inputs["dem_prompt_all_actions"] = dem_actions_padded
            inputs["dem_prompt_all_actions_mask"] = np.ones(len(dem_actions_padded), dtype=bool)

        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "selected_episode" in data:
            inputs["selected_episode"] = data["selected_episode"]
        if "index" in data:
            inputs["index"] = data["index"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaMobileIncontextOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    # adapt_to_pi: bool = True
    adapt_to_pi: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :16])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = _unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]

    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _decode_state_batch(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """Batched variant of _decode_state for [num_samples, state_dim] demo states."""
    if adapt_to_pi:
        state = _joint_flip_mask() * state
        state[..., [6, 13]] = _gripper_to_angular(state[..., [6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
