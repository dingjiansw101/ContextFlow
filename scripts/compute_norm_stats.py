"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(
    config: _config.TrainConfig, *, use_delta_joint_actions: bool = True
) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    # Override a copy only for statistics; the registered train/eval config stays unchanged.
    if hasattr(config.data, "use_delta_joint_actions"):
        config = dataclasses.replace(
            config, data=dataclasses.replace(config.data, use_delta_joint_actions=use_delta_joint_actions)
        )
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    if config.use_custom_dataloader:
        dataset = _data_loader.create_custom_dataset(data_config, config.model, config.data)
    else:
        dataset = _data_loader.create_dataset(data_config, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def main(
    config_name: str,
    sample_frames: int | None = None,
    *,
    use_delta_joint_actions: bool = True,
    fast: bool = True,
    output_dir: str | None = None,
):
    """Compute statistics, using delta joint actions by default for compatible data configs.

    Args:
        config_name: Registered configuration name.
        sample_frames: Optional number of frames to sample.
        use_delta_joint_actions: Subtract joint state from actions for statistics only.
        fast: Use numeric-only loading for full ContextFlow statistics. Sampling and other configs use the legacy path.
        output_dir: Optional output directory for norm_stats.json; defaults to the config's asset directory.
    """
    config = _config.get_config(config_name)
    if fast and config_name == "ContextFlow" and sample_frames is None:
        from openpi.training import libero_norm_stats

        data_config = config.data.create(config.assets_dirs, config.model)
        dataset = libero_norm_stats.load_dataset(config, data_config, use_delta_joint_actions=use_delta_joint_actions)
        norm_stats = libero_norm_stats.compute(dataset)
        asset_id = data_config.asset_id or data_config.repo_id
        output_path = output_dir or config.assets_dirs / asset_id
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return

    data_config, dataset = create_dataset(config, use_delta_joint_actions=use_delta_joint_actions)

    num_frames = len(dataset)
    shuffle = False

    if sample_frames is not None and sample_frames < num_frames:
        num_frames = sample_frames
        shuffle = True

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_workers=8,
        shuffle=shuffle,
        num_batches=num_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_frames, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # Write to asset_id, not repo_id: asset_id is what DataConfigFactory._load_norm_stats
    # and checkpoints.load_norm_stats read back. It defaults to repo_id, but configs may
    # override it (e.g. ContextFlow uses "libero" rather than "physical-intelligence/libero").
    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("Data config must have an asset_id or repo_id to write norm stats")
    output_path = output_dir or config.assets_dirs / asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
