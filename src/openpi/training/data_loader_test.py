import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0
import openpi.models.model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.data_loader import create_custom_dataset
from openpi.training.data_loader import create_dataset
from openpi.training.data_loader import transform_dataset


def test_torch_data_loader():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_libero_incontext_dataset():
    config = _config.get_config("pi0_libero_incontext_low_mem_finetune")
    # TODO: add assets_dirs to the config in the future
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_dataset(data_config, config.model)

    # dataset = transform_dataset(dataset, data_config, skip_norm_stats=False)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=True)
    for i in range(len(dataset)):
        print(dataset[i].keys())
        # dict_keys(['state', 'image', 'image_mask', 'actions',
        # 'tokenized_prompt', 'tokenized_prompt_mask'])


def test_custom_lerobot_dataset():
    # config = _config.get_config("pi0_libero_incontext_low_mem_finetune")
    config = _config.get_config("pi0mini_incontext_libero_custom_dataset_debug")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_custom_dataset(data_config, config.model, config.data)
    for i in range(len(dataset)):
        print(dataset[i].keys())


def test_create_custom_incontext_data_loader():
    """Test create_custom_incontext_data_loader with CustomLeRobotDataset."""
    # Setup: Get config using CustomLeRobotLiberoIncontextDataConfig
    config = _config.get_config("pi0mini_incontext_libero_custom_dataset_debug")

    # Create data loader using CustomLeRobotDataset
    data_loader = _data_loader.create_custom_incontext_data_loader(config, skip_norm_stats=False, num_batches=1)

    # Get one batch
    data_iter = iter(data_loader)
    obs, actions = next(data_iter)

    # Verify types
    assert isinstance(obs, _model.ObservationIncontext), "Observation should be ObservationIncontext"
    assert isinstance(actions, np.ndarray | jnp.ndarray), "Actions should be numpy/jax array"

    # Verify standard observation fields exist
    assert hasattr(obs, "images"), "Should have images field"
    assert hasattr(obs, "image_masks"), "Should have image_masks field"
    assert hasattr(obs, "state"), "Should have state field"

    # Verify required camera keys
    assert "base_0_rgb" in obs.images, "Should have base_0_rgb camera"
    assert "left_wrist_0_rgb" in obs.images, "Should have left_wrist_0_rgb camera"
    assert "right_wrist_0_rgb" in obs.images, "Should have right_wrist_0_rgb camera"

    # Verify batch dimensions
    batch_size = config.batch_size
    assert obs.state.shape[0] == batch_size, f"State batch size should be {batch_size}"
    assert actions.shape[0] == batch_size, f"Actions batch size should be {batch_size}"

    # Verify in-context (demo) fields from CustomLeRobotDataset
    assert obs.incontext_images is not None, "Should have incontext_images"
    assert obs.incontext_states is not None, "Should have incontext_states"
    assert obs.incontext_actions is not None, "Should have incontext_actions"
    assert obs.incontext_selected_episode is not None, "Should have incontext_selected_episode"

    # Verify incontext image batch dimensions
    for key, img_batch in obs.incontext_images.items():
        assert img_batch.shape[0] == batch_size, f"Incontext image {key} batch size should be {batch_size}"

    # Verify incontext states/actions shapes [batch_size, num_frames, dim]
    assert obs.incontext_states.shape[0] == batch_size, "Incontext states batch size mismatch"
    assert len(obs.incontext_states.shape) == 3, "Incontext states should be 3D [batch, frames, dim]"

    assert obs.incontext_actions.shape[0] == batch_size, "Incontext actions batch size mismatch"
    assert len(obs.incontext_actions.shape) == 3, "Incontext actions should be 3D [batch, frames, dim]"

    # Verify masks exist and have correct batch size
    assert obs.incontext_state_masks is not None, "Should have incontext_state_masks"
    assert obs.incontext_action_masks is not None, "Should have incontext_action_masks"
    assert obs.incontext_state_masks.shape[0] == batch_size, "State masks batch size mismatch"
    assert obs.incontext_action_masks.shape[0] == batch_size, "Action masks batch size mismatch"

    # Verify action output shape
    expected_action_shape = (batch_size, config.model.action_horizon, config.model.action_dim)
    assert actions.shape == expected_action_shape, f"Actions shape should be {expected_action_shape}"

    # Verify image data ranges (should be float32 in [-1, 1])
    for key, img in obs.images.items():
        assert img.dtype in [np.float32, jnp.float32], f"Image {key} should be float32"
        assert np.all(img >= -1.0) and np.all(img <= 1.0), f"Image {key} should be in [-1, 1] range"

    # Verify incontext image data ranges
    for key, img in obs.incontext_images.items():
        assert img.dtype in [np.float32, jnp.float32], f"Incontext image {key} should be float32"
        assert np.all(img >= -1.0) and np.all(img <= 1.0), f"Incontext image {key} should be in [-1, 1] range"


if __name__ == "__main__":
    # test_libero_incontext_dataset()
    # test_custom_lerobot_dataset()
    # test_create_custom_incontext_data_loader()
    pass
