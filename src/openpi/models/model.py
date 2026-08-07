import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

ArrayT = TypeVar("ArrayT", at.Array, jax.ShapeDtypeStruct)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI0_INCONTEXT = "pi0_incontext"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


@at.typecheck
@struct.dataclass
class ObservationIncontext(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # In-context data.
    incontext_images: dict[str, at.Float[ArrayT, "*b t h w c"]] | dict[str, at.Float[ArrayT, "*b e t h w c"]] | None = (
        None
    )
    incontext_image_masks: dict[str, at.Bool[ArrayT, "*b t"]] | dict[str, at.Bool[ArrayT, "*b e t"]] | None = None
    # incontext states, q is the max_len of episode
    incontext_states: at.Float[ArrayT, "*b q ds"] | at.Float[ArrayT, "*b e q ds"] | None = None
    incontext_state_masks: at.Bool[ArrayT, "*b q"] | at.Bool[ArrayT, "*b e q"] | None = None
    # incontext actions
    incontext_actions: at.Float[ArrayT, "*b q da"] | at.Float[ArrayT, "*b e q da"] | None = None
    incontext_action_masks: at.Bool[ArrayT, "*b q"] | at.Bool[ArrayT, "*b e q"] | None = None
    # selected episode for incontext prompt
    incontext_selected_episode: at.Int[ArrayT, "*b e"] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    # Future states for state expert (v17)
    future_states: at.Float[ArrayT, "*b fh s"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "ObservationIncontext[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0

        if "dem_prompt_images" in data:
            for key in data["dem_prompt_images"]:
                if data["dem_prompt_images"][key].dtype == np.uint8:
                    data["dem_prompt_images"][key] = (
                        data["dem_prompt_images"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
                    )
        # in the current implementation, the incontext images only sampeld 16 frames
        # for the states and actions, we used the full length of the episode
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            incontext_images=data.get("dem_prompt_images"),
            incontext_image_masks=data.get("dem_prompt_images_mask"),
            incontext_states=data.get("dem_prompt_all_states"),
            incontext_state_masks=data.get("dem_prompt_all_states_mask"),
            incontext_actions=data.get("dem_prompt_all_actions"),
            incontext_action_masks=data.get("dem_prompt_all_actions_mask"),
            incontext_selected_episode=data["selected_episode"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            # Future states for state expert (v17)
            future_states=data.get("future_states"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        result["dem_prompt_images"] = result.pop("incontext_images")
        result["dem_prompt_images_mask"] = result.pop("incontext_image_masks")
        result["dem_prompt_all_states"] = result.pop("incontext_states")
        result["dem_prompt_all_states_mask"] = result.pop("incontext_state_masks")
        result["dem_prompt_all_actions"] = result.pop("incontext_actions")
        result["dem_prompt_all_actions_mask"] = result.pop("incontext_action_masks")

        result["selected_episode"] = result.pop("incontext_selected_episode")

        # Future states for state expert (v17)
        result["future_states"] = result.pop("future_states")
        return result


@at.typecheck
@struct.dataclass
class ObservationFASTIncontext(Generic[ArrayT]):
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    state: at.Float[ArrayT, "*b sf"]

    incontext_images: dict[str, at.Float[ArrayT, "*b t h w c"]] | dict[str, at.Float[ArrayT, "*b e t h w c"]] | None = (
        None
    )
    incontext_image_masks: dict[str, at.Bool[ArrayT, "*b t"]] | dict[str, at.Bool[ArrayT, "*b e t"]] | None = None
    incontext_states: at.Float[ArrayT, "*b q ds"] | at.Float[ArrayT, "*b e q ds"] | None = None
    incontext_state_masks: at.Bool[ArrayT, "*b q"] | at.Bool[ArrayT, "*b e q"] | None = None
    incontext_actions: at.Float[ArrayT, "*b q da"] | at.Float[ArrayT, "*b e q da"] | None = None
    incontext_action_masks: at.Bool[ArrayT, "*b q"] | at.Bool[ArrayT, "*b e q"] | None = None
    incontext_tracks: at.Float[ArrayT, "*b q st"] | at.Float[ArrayT, "*b e q st"] | None = None
    incontext_track_masks: at.Bool[ArrayT, "*b q"] | at.Bool[ArrayT, "*b e q"] | None = None
    incontext_selected_episode: at.Int[ArrayT, "*b e"] | None = None

    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    tokenized_incontext_states: at.Int[ArrayT, "*b ls"] | None = None
    tokenized_incontext_states_mask: at.Bool[ArrayT, "*b ls"] | None = None
    incontext_states_ar_mask: at.Int[ArrayT, "*b ls"] | None = None
    incontext_states_loss_mask: at.Bool[ArrayT, "*b ls"] | None = None

    tokenized_incontext_actions: at.Int[ArrayT, "*b la"] | None = None
    tokenized_incontext_actions_mask: at.Bool[ArrayT, "*b la"] | None = None
    incontext_actions_ar_mask: at.Int[ArrayT, "*b la"] | None = None
    incontext_actions_loss_mask: at.Bool[ArrayT, "*b la"] | None = None

    @staticmethod
    def _ensure_float_image(arr):
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
        return jnp.asarray(arr, dtype=jnp.float32)

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "ObservationFASTIncontext[ArrayT]":
        images = {k: cls._ensure_float_image(v) for k, v in data["image"].items()}
        image_masks = {k: jnp.asarray(v, dtype=jnp.bool_) for k, v in data["image_mask"].items()}
        state = jnp.asarray(data["state"], dtype=jnp.float32)

        inctx_images = None
        if data.get("dem_prompt_images") is not None:
            inctx_images = {k: cls._ensure_float_image(v) for k, v in data["dem_prompt_images"].items()}
        inctx_image_masks = None
        if data.get("dem_prompt_images_mask") is not None:
            inctx_image_masks = {k: jnp.asarray(v, dtype=jnp.bool_) for k, v in data["dem_prompt_images_mask"].items()}

        inctx_states = None
        if data.get("dem_prompt_all_states") is not None:
            inctx_states = jnp.asarray(data["dem_prompt_all_states"], dtype=jnp.float32)
        inctx_states_mask = None
        if data.get("dem_prompt_all_states_mask") is not None:
            inctx_states_mask = jnp.asarray(data["dem_prompt_all_states_mask"], dtype=jnp.bool_)

        inctx_actions = None
        if data.get("dem_prompt_all_actions") is not None:
            inctx_actions = jnp.asarray(data["dem_prompt_all_actions"], dtype=jnp.float32)
        inctx_actions_mask = None
        if data.get("dem_prompt_all_actions_mask") is not None:
            inctx_actions_mask = jnp.asarray(data["dem_prompt_all_actions_mask"], dtype=jnp.bool_)

        inctx_tracks = None
        if data.get("dem_prompt_tracks") is not None:
            inctx_tracks = jnp.asarray(data["dem_prompt_tracks"], dtype=jnp.float32)
        inctx_track_masks = None
        if data.get("dem_prompt_tracks_mask") is not None:
            inctx_track_masks = jnp.asarray(data["dem_prompt_tracks_mask"], dtype=jnp.bool_)

        tokenized_prompt = None
        if data.get("tokenized_prompt") is not None:
            tokenized_prompt = jnp.asarray(data["tokenized_prompt"], dtype=jnp.int32)
        tokenized_prompt_mask = None
        if data.get("tokenized_prompt_mask") is not None:
            tokenized_prompt_mask = jnp.asarray(data["tokenized_prompt_mask"], dtype=jnp.bool_)
        token_ar_mask = None
        if data.get("token_ar_mask") is not None:
            token_ar_mask = jnp.asarray(data["token_ar_mask"], dtype=jnp.int32)
        token_loss_mask = None
        if data.get("token_loss_mask") is not None:
            token_loss_mask = jnp.asarray(data["token_loss_mask"], dtype=jnp.bool_)

        selected_episode = None
        if data.get("selected_episode") is not None:
            selected_episode = jnp.asarray(data["selected_episode"], dtype=jnp.int32)

        tokenized_states = data.get("tokenized_incontext_states")
        if tokenized_states is not None:
            tokenized_states = jnp.asarray(tokenized_states, dtype=jnp.int32)
        tokenized_states_mask = data.get("tokenized_incontext_states_mask")
        if tokenized_states_mask is not None:
            tokenized_states_mask = jnp.asarray(tokenized_states_mask, dtype=jnp.bool_)
        states_ar_mask = data.get("incontext_states_ar_mask")
        if states_ar_mask is not None:
            states_ar_mask = jnp.asarray(states_ar_mask, dtype=jnp.int32)
        states_loss_mask = data.get("incontext_states_loss_mask")
        if states_loss_mask is not None:
            states_loss_mask = jnp.asarray(states_loss_mask, dtype=jnp.bool_)

        tokenized_actions = data.get("tokenized_incontext_actions")
        if tokenized_actions is not None:
            tokenized_actions = jnp.asarray(tokenized_actions, dtype=jnp.int32)
        tokenized_actions_mask = data.get("tokenized_incontext_actions_mask")
        if tokenized_actions_mask is not None:
            tokenized_actions_mask = jnp.asarray(tokenized_actions_mask, dtype=jnp.bool_)
        actions_ar_mask = data.get("incontext_actions_ar_mask")
        if actions_ar_mask is not None:
            actions_ar_mask = jnp.asarray(actions_ar_mask, dtype=jnp.int32)
        actions_loss_mask = data.get("incontext_actions_loss_mask")
        if actions_loss_mask is not None:
            actions_loss_mask = jnp.asarray(actions_loss_mask, dtype=jnp.bool_)

        return cls(
            images=images,
            image_masks=image_masks,
            state=state,
            incontext_images=inctx_images,
            incontext_image_masks=inctx_image_masks,
            incontext_states=inctx_states,
            incontext_state_masks=inctx_states_mask,
            incontext_actions=inctx_actions,
            incontext_action_masks=inctx_actions_mask,
            incontext_tracks=inctx_tracks,
            incontext_track_masks=inctx_track_masks,
            incontext_selected_episode=selected_episode,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
            tokenized_incontext_states=tokenized_states,
            tokenized_incontext_states_mask=tokenized_states_mask,
            incontext_states_ar_mask=states_ar_mask,
            incontext_states_loss_mask=states_loss_mask,
            tokenized_incontext_actions=tokenized_actions,
            tokenized_incontext_actions_mask=tokenized_actions_mask,
            incontext_actions_ar_mask=actions_ar_mask,
            incontext_actions_loss_mask=actions_loss_mask,
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        result["dem_prompt_images"] = result.pop("incontext_images")
        result["dem_prompt_images_mask"] = result.pop("incontext_image_masks")
        result["dem_prompt_all_states"] = result.pop("incontext_states")
        result["dem_prompt_all_states_mask"] = result.pop("incontext_state_masks")
        result["dem_prompt_all_actions"] = result.pop("incontext_actions")
        result["dem_prompt_all_actions_mask"] = result.pop("incontext_action_masks")
        result["dem_prompt_tracks"] = result.pop("incontext_tracks")
        result["dem_prompt_tracks_mask"] = result.pop("incontext_track_masks")
        result["selected_episode"] = result.pop("incontext_selected_episode")
        result["tokenized_incontext_states"] = result.pop("tokenized_incontext_states")
        result["tokenized_incontext_states_mask"] = result.pop("tokenized_incontext_states_mask")
        result["incontext_states_ar_mask"] = result.pop("incontext_states_ar_mask")
        result["incontext_states_loss_mask"] = result.pop("incontext_states_loss_mask")
        result["tokenized_incontext_actions"] = result.pop("tokenized_incontext_actions")
        result["tokenized_incontext_actions_mask"] = result.pop("tokenized_incontext_actions_mask")
        result["incontext_actions_ar_mask"] = result.pop("incontext_actions_ar_mask")
        result["incontext_actions_loss_mask"] = result.pop("incontext_actions_loss_mask")
        return result


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        # print("image_resolution: ", image_resolution)
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


def preprocess_observation_incontext(
    rng: at.KeyArrayLike | None,
    observation: ObservationIncontext,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> ObservationIncontext:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    def process_images(
        observation_images,
        image_keys,
        image_resolution,
        *,
        train: bool,
        rng,
    ):
        """
        Process and optionally augment images in `observation_images`.

        Args:
            observation_images: An object or dict that has an attribute/dict `images`
                        containing images keyed by `image_keys`.
            image_keys (list[str]): Keys to the images you want to process.
            image_resolution (tuple[int, int]): (height, width) to resize images.
            train (bool): If True, apply data augmentation.
            rng: A JAX PRNG key for random operations.

        Returns:
            dict: A dictionary of processed images, keyed by the same keys in `image_keys`.
        """
        out_images = {}

        for key in image_keys:
            image = observation_images[key]
            if image.shape[1:3] != image_resolution:
                logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
                image = image_tools.resize_with_pad(image, *image_resolution)

            if train:
                # Convert from [-1, 1] to [0, 1] for augmax
                image = image / 2.0 + 0.5

                # Build a list of augmentation transforms
                transforms = []
                if "wrist" not in key:
                    height, width = image.shape[1:3]
                    transforms += [
                        augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                        augmax.Resize(width, height),
                        augmax.Rotate((-5, 5)),
                    ]
                transforms += [
                    augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                ]

                # Apply augmentations with vmap
                sub_rngs = jax.random.split(rng, image.shape[0])
                image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

                # Convert back to [-1, 1]
                image = image * 2.0 - 1.0

            out_images[key] = image

        return out_images

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = process_images(observation.images, image_keys, image_resolution, train=train, rng=rng)

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    # reshape incontext images
    out_incontext_images = None
    out_incontext_masks = None
    if observation.incontext_images is not None:
        first = observation.incontext_images[image_keys[0]]
        ndim = first.ndim
        if ndim == 5:
            # B, T, H, W, C
            batch_size, length, height, width, channel = first.shape

            def flatten(x):
                return x.reshape(batch_size * length, height, width, channel)

            def unflatten(x):
                return x.reshape(batch_size, length, height, width, channel)

            mask_shape = (batch_size, length)

        elif ndim == 6:
            # B, E, T, H, W, C
            batch_size, episodes, length, height, width, channel = first.shape

            def flatten(x):
                return x.reshape(batch_size * episodes * length, height, width, channel)

            def unflatten(x):
                return x.reshape(batch_size, episodes, length, height, width, channel)

            mask_shape = (batch_size, episodes, length)

        else:
            raise ValueError(f"incontext_images must be 5-D or 6-D, got ndim={ndim}")

        # flatten all views (create new dict to avoid mutation)
        flattened_incontext_images = {
            key: flatten(observation.incontext_images[key]) for key in observation.incontext_images
        }

        out_incontext_images = process_images(
            flattened_incontext_images, image_keys, image_resolution, train=train, rng=rng
        )

        # reshape back
        for key in out_incontext_images:
            out_incontext_images[key] = unflatten(out_incontext_images[key])

        # build masks
        out_incontext_masks = {}
        for key in out_incontext_images:
            if observation.incontext_image_masks is None or key not in observation.incontext_image_masks:
                out_incontext_masks[key] = jnp.ones(mask_shape, dtype=jnp.bool)
            else:
                out_incontext_masks[key] = jnp.asarray(observation.incontext_image_masks[key])

    return ObservationIncontext(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        incontext_images=out_incontext_images,
        incontext_image_masks=out_incontext_masks,
        incontext_states=observation.incontext_states,
        incontext_state_masks=observation.incontext_state_masks,
        incontext_actions=observation.incontext_actions,
        incontext_action_masks=observation.incontext_action_masks,
        incontext_selected_episode=observation.incontext_selected_episode,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        future_states=observation.future_states,
    )


Actions = at.Float[ArrayT, "*b ah ad"]


@at.typecheck
def preprocess_observation_incontext_fast(
    rng: at.KeyArrayLike | None,
    observation: ObservationFASTIncontext,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> ObservationFASTIncontext:
    """Preprocess FAST in-context observations while preserving token fields."""

    def process_images_dict(
        images: dict[str, at.Array],
        *,
        train: bool,
        rng: at.KeyArrayLike | None,
    ) -> dict[str, at.Array]:
        processed: dict[str, at.Array] = {}
        for key in image_keys:
            image = images[key]
            if image.shape[1:3] != image_resolution:
                logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
                image = image_tools.resize_with_pad(image, *image_resolution)

            if train:
                image01 = image / 2.0 + 0.5
                transforms = []
                if "wrist" not in key:
                    height, width = image01.shape[1:3]
                    transforms += [
                        augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                        augmax.Resize(width, height),
                        augmax.Rotate((-5, 5)),
                    ]
                transforms += [
                    augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                ]
                if rng is None:
                    raise ValueError("rng must be provided when train=True")
                sub_rngs = jax.random.split(rng, image01.shape[0])
                image01 = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image01)
                image = image01 * 2.0 - 1.0

            processed[key] = image
        return processed

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]
    out_images = process_images_dict(observation.images, train=train, rng=rng)

    out_masks: dict[str, at.Array] = {}
    for key in image_keys:
        if key not in observation.image_masks:
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool_)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    out_incontext_images: dict[str, at.Array] | None = None
    out_incontext_masks: dict[str, at.Array] | None = None
    if observation.incontext_images is not None:
        # choose a reference key that is expected in both dicts
        ref_key = next((k for k in image_keys if k in observation.incontext_images), None)
        if ref_key is None:
            raise ValueError(f"incontext_images missing expected keys from {image_keys}")

        first = observation.incontext_images[ref_key]
        ndim = first.ndim
        if ndim == 5:
            batch_size, length, height, width, channel = first.shape

            def flatten(x):
                return x.reshape(batch_size * length, height, width, channel)

            def unflatten(x):
                return x.reshape(batch_size, length, height, width, channel)

            mask_shape = (batch_size, length)
        elif ndim == 6:
            batch_size, episodes, length, height, width, channel = first.shape

            def flatten(x):
                return x.reshape(batch_size * episodes * length, height, width, channel)

            def unflatten(x):
                return x.reshape(batch_size, episodes, length, height, width, channel)

            mask_shape = (batch_size, episodes, length)
        else:
            raise ValueError(f"incontext_images must be 5-D or 6-D, got ndim={ndim}")

        flattened_images = {key: flatten(observation.incontext_images[key]) for key in image_keys}
        processed_flat = process_images_dict(flattened_images, train=train, rng=rng)
        out_incontext_images = {key: unflatten(processed_flat[key]) for key in image_keys}

        out_incontext_masks = {}
        for key in image_keys:
            if observation.incontext_image_masks is None or key not in observation.incontext_image_masks:
                out_incontext_masks[key] = jnp.ones(mask_shape, dtype=jnp.bool_)
            else:
                out_incontext_masks[key] = jnp.asarray(observation.incontext_image_masks[key])

    return ObservationFASTIncontext(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        incontext_images=out_incontext_images,
        incontext_image_masks=out_incontext_masks,
        incontext_states=observation.incontext_states,
        incontext_state_masks=observation.incontext_state_masks,
        incontext_actions=observation.incontext_actions,
        incontext_action_masks=observation.incontext_action_masks,
        incontext_tracks=observation.incontext_tracks,
        incontext_track_masks=observation.incontext_track_masks,
        incontext_selected_episode=observation.incontext_selected_episode,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        tokenized_incontext_states=observation.tokenized_incontext_states,
        tokenized_incontext_states_mask=observation.tokenized_incontext_states_mask,
        incontext_states_ar_mask=observation.incontext_states_ar_mask,
        incontext_states_loss_mask=observation.incontext_states_loss_mask,
        tokenized_incontext_actions=observation.tokenized_incontext_actions,
        tokenized_incontext_actions_mask=observation.tokenized_incontext_actions_mask,
        incontext_actions_ar_mask=observation.incontext_actions_ar_mask,
        incontext_actions_loss_mask=observation.incontext_actions_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    use_action_state_prompts: bool = True

    use_image_prompts: bool = True

    use_text_prompts: bool = True

    sample_episodes: int = 1

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Model params not found at: {params_path}")

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
