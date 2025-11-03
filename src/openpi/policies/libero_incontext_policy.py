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


def _parse_image_batch(image_batch) -> np.ndarray:
    """Vectorized image parsing for batch of frames.

    Args:
        image_batch: Shape [N, C, H, W] or [N, H, W, C]

    Returns:
        Shape [N, H, W, C] as uint8
    """
    batch = np.asarray(image_batch)

    # Convert float to uint8 if needed (vectorized)
    if np.issubdtype(batch.dtype, np.floating):
        batch = (255 * batch).astype(np.uint8)

    # Transpose if in CHW format (vectorized)
    if batch.ndim == 4 and batch.shape[1] == 3:  # [N, C, H, W]
        batch = np.transpose(batch, (0, 2, 3, 1))  # [N, H, W, C]

    return batch


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
        # XJ: add additional inputs to enable stage-wise prompt
        '''
        Example:
            "frame_index": 234,
            "episode_index": 47426,
            "index": 15138419,
            "task_index": 320,
        '''
        if "task_index" in data:
            inputs["task_index"] = data["task_index"]
        if "frame_index" in data:
            inputs["frame_index"] = data["frame_index"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
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
        # XJ: add additional inputs to enable stage-wise prompt
        '''
        Example:
            "frame_index": 234,
            "episode_index": 47426,
            "index": 15138419,
            "task_index": 320,
        '''
        if "task_index" in data:
            inputs["task_index"] = data["task_index"]
        if "frame_index" in data:
            inputs["frame_index"] = data["frame_index"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
        return inputs

@dataclasses.dataclass(frozen=True)
class CustomLeRobotLiberoIncontextInputs(transforms.DataTransformFn):
    """Process both current observations and dem_prompt_* fields for CustomLeRobotDataset.

    Modified from LiberoIncontextInputs_refactor to handle demonstration data
    that comes directly from CustomLeRobotDataset instead of from transforms.
    """
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0_INCONTEXT

    def __call__(self, data: dict) -> dict:
        # TODO: check to see if mask_padding is correct
        mask_padding = self.model_type == _model.ModelType.PI0_INCONTEXT  # We don't mask for pi0-FAST.

        # Process current observation (same as LiberoIncontextInputs_refactor)
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)
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

        # Process dem_prompt_images (NEW for CustomLeRobotDataset)
        if "dem_prompt_images" in data:
            dem_images = data["dem_prompt_images"]

            # dem_images is a dict: {"image": torch.Tensor, "wrist_image": torch.Tensor}
            # Each tensor has shape [sample_frames, C, H, W]
            dem_images_processed = {}
            dem_image_mask = {}

            for key, tensor_stack in dem_images.items():
                # Convert torch to numpy and parse entire batch at once (VECTORIZED)
                numpy_stack = np.asarray(tensor_stack)
                stacked = _parse_image_batch(numpy_stack)  # [sample_frames, H, W, 3]

                # Map dataset keys to model keys
                if key == "image":
                    dem_images_processed["base_0_rgb"] = stacked
                    dem_image_mask["base_0_rgb"] = np.ones(len(stacked), dtype=bool)
                elif key == "wrist_image":
                    dem_images_processed["left_wrist_0_rgb"] = stacked
                    dem_image_mask["left_wrist_0_rgb"] = np.ones(len(stacked), dtype=bool)

            # Add right_wrist_0_rgb as zeros (similar to current observation)
            if "base_0_rgb" in dem_images_processed:
                sample_image = dem_images_processed["base_0_rgb"]
                dem_images_processed["right_wrist_0_rgb"] = np.zeros_like(sample_image)
                dem_image_mask["right_wrist_0_rgb"] = np.full(
                    len(sample_image),
                    fill_value=(not mask_padding),  # False if mask_padding, True otherwise
                    dtype=bool
                )

            inputs["dem_prompt_images"] = dem_images_processed
            inputs["dem_prompt_images_mask"] = dem_image_mask

        # Process dem_prompt_states (NEW for CustomLeRobotDataset)
        if "dem_prompt_states" in data:
            # dem_prompt_states: torch.Tensor [sample_actions, D_s]
            dem_states = np.asarray(data["dem_prompt_states"])

            # Pad entire batch at once (VECTORIZED)
            padded_states = transforms.pad_to_dim(dem_states, self.action_dim, axis=-1)

            inputs["dem_prompt_all_states"] = padded_states
            inputs["dem_prompt_all_states_mask"] = np.ones(len(padded_states), dtype=bool)

        # Process dem_prompt_actions (NEW for CustomLeRobotDataset)
        if "dem_prompt_actions" in data:
            # dem_prompt_actions: torch.Tensor [sample_actions, D_a]
            dem_actions = np.asarray(data["dem_prompt_actions"])

            # Pad entire batch at once (VECTORIZED)
            padded_actions = transforms.pad_to_dim(dem_actions, self.action_dim, axis=-1)

            inputs["dem_prompt_all_actions"] = padded_actions
            inputs["dem_prompt_all_actions_mask"] = np.ones(len(padded_actions), dtype=bool)

        # TODO: Process current frames sequence (NEW for CustomLeRobotDatasetv2)

        
        # Pass through optional fields (same as LiberoIncontextInputs_refactor)
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "dem_prompt_indexes" in data:
            inputs["dem_prompt_indexes"] = data["dem_prompt_indexes"]
        if "selected_episode" in data:
            inputs["selected_episode"] = data["selected_episode"]
        if "index" in data:
            inputs["index"] = data["index"]
        if "task_index" in data:
            inputs["task_index"] = data["task_index"]
        if "frame_index" in data:
            inputs["frame_index"] = data["frame_index"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]

        return inputs

@dataclasses.dataclass(frozen=True)
class LiberoIncontextOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :7])}
