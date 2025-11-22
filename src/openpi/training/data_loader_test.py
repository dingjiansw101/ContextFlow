import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0
import openpi.models.model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.data_loader import TransformedDataset
from openpi.training.data_loader import create_custom_dataset
from openpi.training.data_loader import create_custom_datasetv2
from openpi.training.data_loader import create_dataset
from openpi.training.data_loader import transform_dataset
import openpi.transforms as _transforms


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
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, frame_sequence_length=6)
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_custom_dataset(data_config, config.model, config.data)
    for i in range(len(dataset)):
        print(dataset[i].keys())

def test_custom_lerobot_datasetv2():
    # config = _config.get_config("pi0_libero_incontext_low_mem_finetune")
    config = _config.get_config("pi0mini_incontext_libero_custom_dataset_v2_debug")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_custom_datasetv2(data_config, config.model, config.data)
    for i in range(len(dataset)):
        print(dataset[i].keys())

def test_libero_incontext_data_loader():
    # config = _config.get_config("pi0_libero_incontext_low_mem_finetune")
    # config = _config.get_config("pi0_libero_incontext_low_mem_finetune_sample2")
    config = _config.get_config("vitb_95m_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1")
    data_loader = _data_loader.create_incontext_data_loader(config, skip_norm_stats=False, num_batches=2)
    data_iter = iter(data_loader)
    next(data_iter)


def test_create_custom_incontext_data_loader():
    """Test create_custom_incontext_data_loader with CustomLeRobotDataset."""
    # Setup: Get config using CustomLeRobotLiberoIncontextDataConfig
    config = _config.get_config("pi0mini_incontext_libero_custom_dataset_debug")

    # Create data loader using CustomLeRobotDataset
    data_loader = _data_loader.create_custom_incontext_data_loader(
        config,
        skip_norm_stats=False,
        num_batches=1
    )

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


def test_custom_lerobot_datasetv2_data_loader():
    """Test create_custom_incontext_data_loaderv2 with CustomLeRobotDatasetv2."""
    # Setup: Get config using CustomLeRobotDatasetv2
    config = _config.get_config("pi0mini_incontext_libero_custom_dataset_v2_debug")

    # Create data loader using CustomLeRobotDatasetv2
    data_loader = _data_loader.create_custom_incontext_data_loaderv2(
        config,
        skip_norm_stats=False,
        num_batches=1
    )

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

    # V2-SPECIFIC: Verify current frame sequence fields exist
    assert hasattr(obs, "current_images_seq"), "Should have current_images_seq field"
    assert hasattr(obs, "current_state_seq"), "Should have current_state_seq field"
    assert hasattr(obs, "actions_seq"), "Should have actions_seq field"
    # assert hasattr(obs, 'actions_padding_seq'), "Should have actions_padding_seq field"

    # V2-SPECIFIC: Verify current_images_seq camera keys (transformed to model format)
    assert "base_0_rgb" in obs.current_images_seq, "current_images_seq should have 'base_0_rgb' key"
    assert "left_wrist_0_rgb" in obs.current_images_seq, "current_images_seq should have 'left_wrist_0_rgb' key"
    assert "right_wrist_0_rgb" in obs.current_images_seq, "current_images_seq should have 'right_wrist_0_rgb' key"

    # V2-SPECIFIC: Verify frame sequence dimensions
    # Config has frame_sequence_length (num_current_frames in dataset)
    # Expected shape: [batch_size, frame_sequence_length, ...]
    frame_seq_len = config.data.frame_sequence_length

    # Verify current_images_seq shapes: [batch_size, frame_seq_len, H, W, 3]
    for key, img_seq in obs.current_images_seq.items():
        assert img_seq.shape[0] == batch_size, f"current_images_seq[{key}] batch size should be {batch_size}"
        assert img_seq.shape[1] == frame_seq_len, f"current_images_seq[{key}] should have {frame_seq_len} frames"
        assert len(img_seq.shape) == 5, f"current_images_seq[{key}] should be 5D [batch, frames, H, W, C]"
        assert img_seq.shape[-1] == 3, f"current_images_seq[{key}] should have 3 color channels"

    # Verify current_state_seq shape: [batch_size, frame_seq_len, state_dim]
    assert obs.current_state_seq.shape[0] == batch_size, "current_state_seq batch size mismatch"
    assert obs.current_state_seq.shape[1] == frame_seq_len, f"current_state_seq should have {frame_seq_len} frames"
    assert len(obs.current_state_seq.shape) == 3, "current_state_seq should be 3D [batch, frames, state_dim]"

    # Verify actions_seq shape: [batch_size, frame_seq_len, action_horizon, action_dim]
    assert obs.actions_seq.shape[0] == batch_size, "actions_seq batch size mismatch"
    assert obs.actions_seq.shape[1] == frame_seq_len, f"actions_seq should have {frame_seq_len} frames"
    assert obs.actions_seq.shape[2] == config.model.action_horizon, "actions_seq action_horizon mismatch"
    assert obs.actions_seq.shape[3] == config.model.action_dim, "actions_seq action_dim mismatch"
    assert len(obs.actions_seq.shape) == 4, "actions_seq should be 4D [batch, frames, horizon, action_dim]"

    # Verify actions_padding_seq shape: [batch_size, frame_seq_len, action_horizon]
    # NOTE: Commented out - actions_padding_seq not currently needed
    # assert obs.actions_padding_seq.shape[0] == batch_size, "actions_padding_seq batch size mismatch"
    # assert obs.actions_padding_seq.shape[1] == frame_seq_len, f"actions_padding_seq should have {frame_seq_len} frames"
    # assert obs.actions_padding_seq.shape[2] == config.model.action_horizon, "actions_padding_seq horizon mismatch"
    # assert len(obs.actions_padding_seq.shape) == 3, "actions_padding_seq should be 3D [batch, frames, horizon]"

    # Verify action output shape (should be same as v1)
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

    # V2-SPECIFIC: Verify current frame sequence image data ranges
    for key, img_seq in obs.current_images_seq.items():
        assert img_seq.dtype in [np.float32, jnp.float32], f"current_images_seq[{key}] should be float32"
        assert np.all(img_seq >= -1.0) and np.all(img_seq <= 1.0), f"current_images_seq[{key}] should be in [-1, 1] range"

def test_custom_lerobot_datasetv2_future_states():
    """Test future_states functionality in CustomLeRobotDatasetv2."""
    # Setup: Get config with use_future_states enabled
    config = _config.get_config("pi0mini_incontext_libero_custom_dataset_v2_future_states_debug")

    # Create data loader using CustomLeRobotDatasetv2 with future_states
    data_loader = _data_loader.create_custom_incontext_data_loaderv2(
        config,
        skip_norm_stats=False,
        num_batches=1
    )

    # Get one batch
    data_iter = iter(data_loader)
    obs, actions = next(data_iter)

    # Verify types
    assert isinstance(obs, _model.ObservationIncontext), "Observation should be ObservationIncontext"
    assert isinstance(actions, np.ndarray | jnp.ndarray), "Actions should be numpy/jax array"

    # Verify batch dimensions
    batch_size = config.batch_size
    assert obs.state.shape[0] == batch_size, f"State batch size should be {batch_size}"
    assert actions.shape[0] == batch_size, f"Actions batch size should be {batch_size}"

    # FUTURE_STATES-SPECIFIC: Verify future_states field exists
    assert hasattr(obs, "future_states"), "Should have future_states field"
    assert obs.future_states is not None, "future_states should not be None"

    # FUTURE_STATES-SPECIFIC: Verify future_states shape
    # Expected: [batch_size, future_state_horizon, state_dim]
    action_horizon = config.model.action_horizon
    future_state_downsample = config.data.future_state_downsample
    expected_future_state_horizon = action_horizon // future_state_downsample
    state_dim = obs.state.shape[-1]  # Get state dimension from current state

    assert obs.future_states.shape[0] == batch_size, f"future_states batch size should be {batch_size}"
    assert obs.future_states.shape[1] == expected_future_state_horizon, \
        f"future_states should have {expected_future_state_horizon} timesteps (action_horizon={action_horizon} / downsample={future_state_downsample})"
    assert obs.future_states.shape[2] == state_dim, \
        f"future_states should have state_dim={state_dim}"
    assert len(obs.future_states.shape) == 3, "future_states should be 3D [batch, future_horizon, state_dim]"

    # FUTURE_STATES-SPECIFIC: Verify data type
    assert obs.future_states.dtype in [np.float32, jnp.float32], "future_states should be float32"

    # FUTURE_STATES-SPECIFIC: Verify normalization (should be in similar range to current state)
    # After normalization, values should typically be in a reasonable range (e.g., [-10, 10] for z-score)
    assert np.all(np.isfinite(obs.future_states)), "future_states should not contain inf/nan"
    future_states_mean = np.abs(np.mean(obs.future_states))
    future_states_std = np.std(obs.future_states)
    print(f"future_states stats: mean={future_states_mean:.4f}, std={future_states_std:.4f}")
    # Normalized values should have reasonable statistics (not too extreme)
    assert future_states_mean < 10.0, "future_states mean should be reasonable after normalization"
    assert future_states_std > 0.0, "future_states should have non-zero variance"

    # Test with different downsample factor
    # Modify config to use different downsample and verify the relationship holds
    config_ds10 = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, future_state_downsample=10)
    )
    data_loader_ds10 = _data_loader.create_custom_incontext_data_loaderv2(
        config_ds10,
        skip_norm_stats=False,
        num_batches=1
    )
    obs_ds10, _ = next(iter(data_loader_ds10))
    expected_horizon_ds10 = action_horizon // 10
    assert obs_ds10.future_states.shape[1] == expected_horizon_ds10, \
        f"With downsample=10, future_state_horizon should be {expected_horizon_ds10}"

    print(f"✓ future_states test passed: shape={obs.future_states.shape}, "
          f"future_state_horizon={expected_future_state_horizon}, downsample={future_state_downsample}")

def test_AddImagePromptTransform():
    # config = _config.get_config("pi0_libero_incontext_low_mem_finetune")
    config = _config.get_config("vitb_95m_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1")
    # TODO: add assets_dirs to the config in the future
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_dataset(data_config, config.model)

    # dataset = transform_dataset(dataset, data_config, skip_norm_stats=False)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=True)
    add_image_transform = _transforms.AddImagePromptTransform(dataset=dataset)
    dataset = TransformedDataset(dataset, [add_image_transform])

    for i in range(len(dataset)):
        print(dataset[i].keys())
        # dict_keys(['state', 'image', 'image_mask', 'actions',
        # 'tokenized_prompt', 'tokenized_prompt_mask'])

if __name__ == "__main__":
    # test_libero_incontext_dataset()
    # test_libero_incontext_data_loader()
    # test_AddImagePromptTransform()
    # test_custom_lerobot_dataset()
    # test_custom_lerobot_datasetv2()
    # test_custom_lerobot_datasetv2_data_loader()
    # test_create_custom_incontext_data_loader()
    test_custom_lerobot_datasetv2_future_states()