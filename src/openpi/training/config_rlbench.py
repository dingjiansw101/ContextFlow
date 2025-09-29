from __future__ import annotations
import dataclasses
import tyro
import pathlib
import json
from collections.abc import Sequence
from typing_extensions import override

import openpi.policies.rlbench_gripper_policy as rlbench_gripper_policy
import openpi.policies.rlbench_joint_policy as rlbench_joint_policy

def build(api) -> list["api.TrainConfig"]:
    g = globals()
    g["DataConfig"] = getattr(api, "DataConfig")
    g["BaseModelConfig"] = getattr(api._model, "BaseModelConfig")
    @dataclasses.dataclass(frozen=True)
    class LeRobotRLBenchJointDataConfig(api.DataConfigFactory):
        # deprecated
        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            # "action_joint_velocity": "action_joint_velocity",
                            "prompt": "prompt",
                        }
                    )
                ]
            )

            data_transforms = api._transforms.Group(
                inputs=[rlbench_joint_policy.RLBenchJointInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
                outputs=[rlbench_joint_policy.RLBenchJointOutputs()],
            )
            
            model_transforms = api.ModelTransformFactory()(model_config)
            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
            )

    @dataclasses.dataclass(frozen=True)
    class LeRobotRLBenchGripperDataConfig(api.DataConfigFactory):
        # deprecated
        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )

            data_transforms = api._transforms.Group(
                inputs=[rlbench_gripper_policy.RLBenchGripperInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
                outputs=[rlbench_gripper_policy.RLBenchGripperOutputs()],
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
    
        # #
        # # XJ: Fine-tuning RLBench configs
        # #
        # api.TrainConfig(
        #     name="pi0_rlbench_gripper_low_mem_finetune_train",
        #     model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        #     data=LeRobotRLBenchGripperDataConfig(
        #         repo_id="daixianjie/rlbench_lerobot_train",
        #         base_config=DataConfig(
        #             local_files_only=False,  # Set to True for local-only datasets.
        #             prompt_from_task=True,
        #         ),
        #     ),
        #     weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        #     num_train_steps=20_000,
        #     freeze_filter=api.pi0.Pi0Config(
        #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        #     ).get_freeze_filter(),
        #     # Turn off EMA for LoRA finetuning.
        #     ema_decay=None,
        #     num_workers=4,
        #     batch_size=36,
        # ),
        
        # TrainConfig(
        #     name="pi0_rlbench_joint_low_mem_finetune_train",
        #     # Here is an example of loading a pi0 model for LoRA fine-tuning.
        #     model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        #     data=LeRobotRLBenchJointDataConfig(
        #         repo_id="daixianjie/rlbench_joint_vel_action_lerobot_train",
        #         base_config=api.DataConfig(
        #             local_files_only=False,  # Set to True for local-only datasets.
        #             prompt_from_task=True,
        #         ),
        #     ),
        #     weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        #     num_train_steps=20_000,
        #     freeze_filter=api.pi0.Pi0Config(
        #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        #     ).get_freeze_filter(),
        #     # Turn off EMA for LoRA finetuning.
        #     ema_decay=None,
        #     num_workers=4,
        #     batch_size=36,
        # ),
    ]
