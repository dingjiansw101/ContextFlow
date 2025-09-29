# config_template.py
"""
An example of children config.
RoboCasa 实验的 children config。
父 config.py 会 import 本文件，并调用 build(api) 来拿 TrainConfig 列表。
"""

from __future__ import annotations
import dataclasses
import tyro
import pathlib
import json
from collections.abc import Sequence
from typing_extensions import override

# 如果这个 child 只在这里用到某些 policy/模块，可以直接在这里 import
import openpi.policies.robocasa_insertion_policy as robocasa_insertion_policy

def build(api) -> list["api.TrainConfig"]:
    g = globals()
    g["DataConfig"] = getattr(api, "DataConfig")
    g["BaseModelConfig"] = getattr(api._model, "BaseModelConfig")
    # 1) 在函数内定义 DataConfig 子类，继承父里的 DataConfigFactory（通过 api 取）
    @dataclasses.dataclass(frozen=True)
    class DummyLeRobotRobocasaInsertionDataConfig(api.DataConfigFactory):
        # 需要的额外字段可以随便加
        # e.g. default_prompt: str | None = None

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform({
                        "observation/image": "image_left",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    })
                ]
            )

            data_transforms = api._transforms.Group(
                inputs=[robocasa_insertion_policy.RobocasaInsertionInputs(
                    action_dim=model_config.action_dim, model_type=model_config.model_type)],
                outputs=[robocasa_insertion_policy.RobocasaInsertionOutputs()],
            )

            model_transforms = api.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
            )

    # 2) 直接返回本 child 的 TrainConfig 条目（可多个）
    return [
        api.TrainConfig(
            name="dummy_pi0_robocasa_insertion",
            model=api.pi0.Pi0Config(
                action_dim=7,
                action_horizon=8,
                max_token_len=512,
            ),
            data=DummyLeRobotRobocasaInsertionDataConfig(
                assets=api.AssetsConfig(asset_id="robocasa"),
                # 这里可以填 DataConfigFactory 的可选字段
            ),
            batch_size=128,
            num_train_steps=30_000,
            exp_name="exp_robocasa_insertion_v1",
        ),
    ]
