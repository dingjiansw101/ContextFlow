import dataclasses
import os
import pathlib

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.models import pi0
from openpi.training import config as _config

from . import train


def test_train(tmp_path: pathlib.Path):
    config = _config.TrainConfig(
        name="debug",
        data=_config.FakeDataConfig(),
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        batch_size=2,
        checkpoint_base_dir=tmp_path / "checkpoint",
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
        num_workers=0,
        wandb_enabled=False,
    )
    train.main(config)

    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
