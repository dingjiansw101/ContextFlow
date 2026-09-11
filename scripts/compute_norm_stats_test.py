import copy
import dataclasses

import numpy as np
import pytest

from openpi import transforms
from openpi.training import config as _config
from openpi.training import config_libero
from openpi.training import data_loader
from scripts import compute_norm_stats


@pytest.mark.parametrize("delta", [None, True, False])
def test_contextflow_norm_actions_do_not_change_training_config(monkeypatch, delta):
    config = _config.get_config("ContextFlow")
    original_data = config.data
    monkeypatch.setattr(_config, "get_kept_episode_indices", lambda *args: None)
    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda *args: transforms.Group())
    monkeypatch.setattr(
        _config.DataConfigFactory,
        "create_base_config",
        lambda self, assets: dataclasses.replace(self.base_config, repo_id=self.repo_id),
    )
    sample = {
        "image": np.zeros((2, 2, 3), dtype=np.uint8),
        "wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "state": np.arange(8, dtype=np.float32),
        "actions": np.full((2, 7), 10, dtype=np.float32),
        "prompt": "fixture",
        "episode_index": 0,
        "frame_index": 0,
        "index": 0,
        "task_index": 0,
        "dem_prompt_images": {
            "image": np.zeros((2, 2, 2, 3), dtype=np.uint8),
            "wrist_image": np.zeros((2, 2, 2, 3), dtype=np.uint8),
        },
        "dem_prompt_states": np.zeros((2, 8), dtype=np.float32),
        "dem_prompt_actions": np.zeros((2, 7), dtype=np.float32),
        "selected_episode": 0,
    }
    factories = []

    def custom_dataset(data_config, model, factory):
        factories.append(factory)
        return [copy.deepcopy(sample)]

    monkeypatch.setattr(data_loader, "create_custom_dataset", custom_dataset)
    kwargs = {} if delta is None else {"use_delta_joint_actions": delta}
    _, dataset = compute_norm_stats.create_dataset(config, **kwargs)
    result = dataset[0]
    expected = sample["actions"].copy()
    if delta is not False:
        expected[:, :6] -= sample["state"][:6]
    np.testing.assert_array_equal(result["actions"][:, :7], expected)
    np.testing.assert_array_equal(result["state"][:8], sample["state"])
    assert factories[0].use_delta_joint_actions is (delta is not False)
    assert config.data is original_data
    assert _config.get_config("ContextFlow").data.use_delta_joint_actions is False


def test_libero90_configs_are_not_registered():
    assert not any("libero90" in config.name for config in config_libero.build(_config))
