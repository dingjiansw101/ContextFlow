import dataclasses

import jax

from openpi.models import pi0
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
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
        import ipdb

        ipdb.set_trace()
        # dict_keys(['state', 'image', 'image_mask', 'actions',
        # 'tokenized_prompt', 'tokenized_prompt_mask'])
        # import ipdb;
        # ipdb.set_trace()


def test_libero_incontext_data_loader():
    config = _config.get_config("pi0_libero_incontext_low_mem_finetune")
    data_loader = _data_loader.create_incontext_data_loader(config, skip_norm_stats=False, num_batches=2)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    # test_libero_incontext_dataset()
    test_libero_incontext_data_loader()
