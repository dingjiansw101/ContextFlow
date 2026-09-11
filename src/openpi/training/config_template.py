# config_template.py
"""
An example of children config.
Example child config.
Parent config.py will import this file and call build(api) to collect TrainConfig entries.
"""

from __future__ import annotations
import dataclasses
import tyro
import pathlib
import json
from collections.abc import Sequence
from typing_extensions import override

# If this child only needs certain policies/modules here, import them directly
import openpi.policies.libero_policy as libero_policy

def build(api) -> list["api.TrainConfig"]:
    g = globals()
    g["DataConfig"] = getattr(api, "DataConfig")
    g["BaseModelConfig"] = getattr(api._model, "BaseModelConfig")
    # 1) Define DataConfig subclasses inside this function, inheriting DataConfigFactory via api
    @dataclasses.dataclass(frozen=True)
    class DummyLeRobotLiberoDataConfig(api.DataConfigFactory):
        # Add any extra fields as needed
        # e.g. default_prompt: str | None = None

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform({
                        "observation/image": "image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    })
                ]
            )

            data_transforms = api._transforms.Group(
                inputs=[libero_policy.LiberoInputs(
                    action_dim=model_config.action_dim, model_type=model_config.model_type)],
                outputs=[libero_policy.LiberoOutputs()],
            )

            model_transforms = api.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
            )

    # 2) Return this child's TrainConfig entries directly (can be multiple)
    return [
        api.TrainConfig(
            name="dummy_pi0_libero",
            model=api.pi0.Pi0Config(
                action_dim=7,
                action_horizon=8,
                max_token_len=512,
            ),
            data=DummyLeRobotLiberoDataConfig(
                assets=api.AssetsConfig(asset_id="libero"),
                # Optional fields from DataConfigFactory can be filled here
            ),
            batch_size=128,
            num_train_steps=30_000,
            exp_name="exp_libero_v1",
        ),
    ]
