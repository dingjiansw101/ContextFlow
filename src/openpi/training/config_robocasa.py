# 同目录下：config_robocasa.py
"""
Children config file for RoboCasa experiments.
Put DataConfigFactory subclasses + corresponding TrainConfig entries here.
父 config.py 会 import 本文件，并调用 build(api) 获取 TrainConfig 列表。
"""

from __future__ import annotations
import dataclasses
import pathlib
import json
from typing_extensions import override


import openpi.policies.robocasa_insertion_policy as robocasa_insertion_policy
import openpi.policies.robocasa_human_policy as robocasa_human_policy
import openpi.policies.robocasa_human_three_image_policy as robocasa_human_three_image_policy
# import openpi.policies.robocasa_human_three_image_base_obs_policy as robocasa_human_three_image_base_obs_policy
import openpi.policies.robocasa_single_task_policy as robocasa_single_task_policy
import openpi.policies.robocasa_human_three_image_incontext_policy as robocasa_human_three_image_incontext_policy
import openpi.policies.robocasa_mg_three_image_policy as robocasa_mg_three_image_policy
import openpi.policies.robocasa_mg_three_image_incontext_policy as robocasa_mg_three_image_incontext_policy


def build(api) -> list["api.TrainConfig"]:
    g = globals()
    g["DataConfig"] = getattr(api, "DataConfig")
    g["BaseModelConfig"] = getattr(api._model, "BaseModelConfig")
    # @dataclasses.dataclass(frozen=True)
    # class LeRobotRobocasaInsertionDataConfig(api.DataConfigFactory):
    #     # deprecated
    #     @override
    #     def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
    #         repack_transform = api._transforms.Group(
    #             inputs=[
    #                 api._transforms.RepackTransform(
    #                     {
    #                         "observation/image": "image_left",
    #                         "observation/wrist_image": "wrist_image",
    #                         "observation/state": "state",
    #                         "actions": "actions",
    #                         "prompt": "prompt",
    #                     }
    #                 )
    #             ]
    #         )

    #         data_transforms = api._transforms.Group(
    #             inputs=[robocasa_insertion_policy.RobocasaInsertionInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
    #             outputs=[robocasa_insertion_policy.RobocasaInsertionOutputs()],
    #         )
    #         model_transforms = api.ModelTransformFactory()(model_config)
    #         return dataclasses.replace(
    #             self.create_base_config(assets_dirs),
    #             repack_transforms=repack_transform,
    #             data_transforms=data_transforms,
    #             model_transforms=model_transforms,
    #         )

    @dataclasses.dataclass(frozen=True)
    class LeRobotRobocasaHumanDataConfig(api.DataConfigFactory):
        # deprecated
        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "observation/image": "image_left",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )

            data_transforms = api._transforms.Group(
                inputs=[robocasa_human_policy.RobocasaHumanInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
                outputs=[robocasa_human_policy.RobocasaHumanOutputs()],
            )
            model_transforms = api.ModelTransformFactory()(model_config)
            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
            )


    @dataclasses.dataclass(frozen=True)
    class LeRobotRobocasaHumanThreeImageDataConfig(api.DataConfigFactory):
        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "observation/image_left": "image_left",
                            "observation/image_right": "image_right",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            
            # XJ: debug (libero use same json for train-test split instead of a list of strings)
            if self.remove_task_list:
                with open(self.remove_task_list[0], "r") as f:
                    remove_test_tasks = json.load(f)["test_tasks"]

                # XJ: calculate training episode indexi first
                train_epi = api.get_kept_episode_indices(self.episode_json_path, remove_test_tasks)
            else:
                train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)

            data_transforms = api._transforms.Group(
                inputs=[robocasa_human_three_image_policy.RobocasaHumanThreeImageInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
                outputs=[robocasa_human_three_image_policy.RobocasaHumanThreeImageOutputs()],
            )
            model_transforms = api.ModelTransformFactory()(model_config)
            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                train_episode=train_epi,
            )
        
    @dataclasses.dataclass(frozen=True)
    class LeRobotRobocasaMgThreeImageDataConfig(api.DataConfigFactory):
        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "observation/image_left": "image_left",
                            "observation/image_right": "image_right",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            
            # XJ: debug (libero use same json for train-test split instead of a list of strings)
            if self.remove_task_list:
                with open(self.remove_task_list[0], "r") as f:
                    remove_test_tasks = json.load(f)["test_tasks"]

                # XJ: calculate training episode indexi first
                train_epi = api.get_kept_episode_indices(self.episode_json_path, remove_test_tasks)
            else:
                train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)

            data_transforms = api._transforms.Group(
                inputs=[robocasa_mg_three_image_policy.RobocasaMgThreeImageInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
                outputs=[robocasa_mg_three_image_policy.RobocasaMgThreeImageOutputs()],
            )
            model_transforms = api.ModelTransformFactory()(model_config)
            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                train_episode=train_epi,
            ) 
        
    @dataclasses.dataclass(frozen=True)
    class LeRobotRobocasaHumanThreeImageIncontextDataConfig(api.DataConfigFactory):
        states_cache_path: str = "metadata/robocasa/episode_states_cache.json"
        actions_cache_path: str = "metadata/robocasa/episode_actions_cache.json"
        task_to_episode: str='metadata/robocasa/task_to_episode.json'
        episode_to_indexes_file: str='metadata/robocasa/episode_to_indexes.json'
        tracks_path: str = "metadata/robocasa/episode_tracks_combined.json"
        # # XJ: deprecated flags
        # use_delta_joint_actions: bool = False
        # robocasa_input_refactor: bool = False
        
        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "observation/image_left": "image_left",
                            "observation/image_right": "image_right",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                            "episode_index": "episode_index",
                            "frame_index": "frame_index", 
                            "index": "index",
                            "task_index": "task_index",
                        }
                    )
                ]
            )
            
            # XJ: debug (libero use same json for train-test split instead of a list of strings)
            if self.remove_task_list:
                with open(self.remove_task_list[0], "r") as f:
                    remove_test_tasks = json.load(f)["test_tasks"]

                # XJ: calculate training episode indexi first
                train_epi = api.get_kept_episode_indices(self.episode_json_path, remove_test_tasks)
            else:
                train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)
            
            # Prepare data for policy training
            # inject the indexes of demo prompt, TODO: provide json file_paths here
            data_transforms = api._transforms.Group(
                inputs=[api._transforms.InjectDemoIndexes(sample_frames=model_config.sample_frames, 
                                                    random_select=model_config.random_select,
                                                    sample_episodes=model_config.sample_episodes,
                                                    task_to_episode=self.task_to_episode,
                                                    episode_to_indexes=self.episode_to_indexes_file,
                                                    train_episode_index_list=train_epi)],
                outputs=[],
            )
            
            data_transforms = data_transforms.push(
                    inputs=[
                        robocasa_human_three_image_incontext_policy.RobocasaHumanThreeImageIncontextInputs(
                            action_dim=model_config.action_dim, model_type=model_config.model_type
                        )
                    ],
                    outputs=[robocasa_human_three_image_incontext_policy.RobocasaHumanThreeImageIncontextOutputs()],
            )
            
            model_transforms = api.ModelTransformFactory()(model_config)
            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                train_episode=train_epi,
            )    
            

    @dataclasses.dataclass(frozen=True)
    class LeRobotRobocasaMgThreeImageIncontextDataConfig(api.DataConfigFactory):
        states_cache_path: str = "metadata/robocasa_mg/episode_states_cache.json"
        actions_cache_path: str = "metadata/robocasa_mg/episode_actions_cache.json"
        task_to_episode: str='metadata/robocasa_mg/task_to_episode.json'
        episode_to_indexes_file: str='metadata/robocasa_mg/episode_to_indexes.json'
        tracks_path: str = "metadata/robocasa_mg/episode_tracks_combined.json"
        # # XJ: deprecated flags
        # use_delta_joint_actions: bool = False
        # robocasa_input_refactor: bool = False
        
        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "observation/image_left": "image_left",
                            "observation/image_right": "image_right",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                            "episode_index": "episode_index",
                            "frame_index": "frame_index", 
                            "index": "index",
                            "task_index": "task_index",
                        }
                    )
                ]
            )
            
            # XJ: debug
            if self.remove_task_list:
                with open(self.remove_task_list[0], "r") as f:
                    remove_test_tasks = json.load(f)["test_tasks"]

                # XJ: calculate training episode indexi first
                train_epi = api.get_kept_episode_indices(self.episode_json_path, remove_test_tasks)
            else:
                train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)
            
            # Prepare data for policy training
            # inject the indexes of demo prompt, TODO: provide json file_paths here
            data_transforms = api._transforms.Group(
                inputs=[api._transforms.InjectDemoIndexes(sample_frames=model_config.sample_frames, 
                                                    random_select=model_config.random_select,
                                                    sample_episodes=model_config.sample_episodes,
                                                    task_to_episode=self.task_to_episode,
                                                    episode_to_indexes=self.episode_to_indexes_file,
                                                    train_episode_index_list=train_epi)],
                outputs=[],
            )
            
            data_transforms = data_transforms.push(
                    inputs=[
                        robocasa_mg_three_image_incontext_policy.RobocasaMgThreeImageIncontextInputs(
                            action_dim=model_config.action_dim, model_type=model_config.model_type
                        )
                    ],
                    outputs=[robocasa_mg_three_image_incontext_policy.RobocasaMgThreeImageIncontextOutputs()],
            )
            
            model_transforms = api.ModelTransformFactory()(model_config)
            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                train_episode=train_epi,
            ) 

    # @dataclasses.dataclass(frozen=True)
    # class LeRobotRobocasaSingleTaskThreeImageDataConfig(api.DataConfigFactory):
    #     @override
    #     def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
    #         repack_transform = api._transforms.Group(
    #             inputs=[
    #                 api._transforms.RepackTransform(
    #                     {
    #                         "observation/image_left": "image_left",
    #                         "observation/image_right": "image_right",
    #                         "observation/wrist_image": "wrist_image",
    #                         "observation/state": "state",
    #                         "actions": "actions",
    #                         "prompt": "prompt",
    #                     }
    #                 )
    #             ]
    #         )

    #         data_transforms = api._transforms.Group(
    #             inputs=[robocasa_single_task_policy.RobocasaSingleTaskThreeImageInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
    #             outputs=[robocasa_single_task_policy.RobocasaSingleTaskThreeImageOutputs()],
    #         )
    #         model_transforms = api.ModelTransformFactory()(model_config)
    #         return dataclasses.replace(
    #             self.create_base_config(assets_dirs),
    #             repack_transforms=repack_transform,
    #             data_transforms=data_transforms,
    #             model_transforms=model_transforms,
    #         )  
        
    return [
        api.TrainConfig(
            # no delta with split
            name="pi0_robocasa_human_low_mem_finetune_train",
            # Here is an example of loading a pi0 model for LoRA fine-tuning.
            model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotRobocasaHumanDataConfig(
                repo_id="daixianjie/robocasa_human_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=20_000,
            # The freeze filter defines which parameters should be frozen during training.
            # We have a convenience function in the model config that returns the default freeze filter
            # for the given model config for LoRA finetuning. Just make sure it matches the model config
            # you chose above.
            freeze_filter=api.pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            # Turn off EMA for LoRA finetuning.
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ),
        api.TrainConfig(
            # no delta with split
            name="pi0_robocasa_human_three_image_low_mem_finetune_train",
            # Here is an example of loading a pi0 model for LoRA fine-tuning.
            model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotRobocasaHumanThreeImageDataConfig(
                repo_id="daixianjie/robocasa_human_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=200_000,
            # The freeze filter defines which parameters should be frozen during training.
            # We have a convenience function in the model config that returns the default freeze filter
            # for the given model config for LoRA finetuning. Just make sure it matches the model config
            # you chose above.
            freeze_filter=api.pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            # Turn off EMA for LoRA finetuning.
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ),
        api.TrainConfig(
            # no delta with split
            name="pi0_robocasa_mg_three_image_low_mem_finetune_train_split",
            # Here is an example of loading a pi0 model for LoRA fine-tuning.
            model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotRobocasaMgThreeImageDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                remove_task_list=api.DEFAULT_ROBOCASA_MG_TEST_TASK,
                episode_json_path=api.DEFAULT_ROBOCASA_MG_EPISODE_JSON,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 500_000,
                decay_lr= 2.5e-6),
            num_train_steps=500_000,
            freeze_filter=api.pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            # Turn off EMA for LoRA finetuning.
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ),
        
        api.TrainConfig(
            # no delta with split
            name="pi0_incontext_robocasa_mg_three_image_low_mem_finetune_train",
            # Here is an example of loading a pi0 model for LoRA fine-tuning.
            model=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
                sample_frames=2, sample_actions=32, random_select=True, 
            ),
            data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                task_to_episode='metadata/robocasa_mg/task_to_episode.json',
                episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
                states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
                actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
                remove_task_list=api.DEFAULT_ROBOCASA_MG_TEST_TASK,
                episode_json_path=api.DEFAULT_ROBOCASA_MG_EPISODE_JSON,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 500_000,
                decay_lr= 2.5e-6),
            num_train_steps=500_000,
            freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
                sample_frames=2, sample_actions=32, random_select=True, 
            ).get_freeze_filter(),
            # Turn off EMA for LoRA finetuning.
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ),
        
        api.TrainConfig(
            # no delta with split
            name="pi0_incontext_robocasa_mg_three_image_low_mem_finetune_inference",
            # Here is an example of loading a pi0 model for LoRA fine-tuning.
            model=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
                sample_frames=2, sample_actions=32, random_select=True, 
            ),
            data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                task_to_episode='metadata/robocasa_mg/task_to_episode.json',
                episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
                states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
                actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 500_000,
                decay_lr= 2.5e-6),
            num_train_steps=500_000,
            freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
                sample_frames=2, sample_actions=32, random_select=True, 
            ).get_freeze_filter(),
            # Turn off EMA for LoRA finetuning.
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ),

        api.TrainConfig(
            name="pi0mini_incontext_robocasa_mg_three_image_low_mem_finetune",
            model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
                prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
                sample_frames=2, sample_actions=32, random_select=True,  
                freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
            data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                task_to_episode='metadata/robocasa_mg/task_to_episode.json',
                episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
                states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
                actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
                remove_task_list=api.DEFAULT_ROBOCASA_MG_TEST_TASK,
                episode_json_path=api.DEFAULT_ROBOCASA_MG_EPISODE_JSON,
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 500_000,
                decay_lr= 2.5e-6),
            num_train_steps=500_000,
            freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
                prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
                sample_frames=2, sample_actions=32, random_select=True, 
                freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ), 

        api.TrainConfig(
            name="pi0mini_incontext_robocasa_mg_three_image_low_mem_finetune_inference",
            model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
                prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
                sample_frames=2, sample_actions=32, random_select=True,  
                freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
            data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                task_to_episode='metadata/robocasa_mg/task_to_episode.json',
                episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
                states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
                actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 500_000,
                decay_lr= 2.5e-6),
            num_train_steps=500_000,
            freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
                prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
                sample_frames=2, sample_actions=32, random_select=True, 
                freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ), 
        
            
        api.TrainConfig(
            # no delta with split
            name="pi0light_robocasa_human_three_image_low_mem_finetune_train",
            # Here is an example of loading a pi0 model for LoRA fine-tuning.
            model=api.pi0Light.Pi0LightConfig(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", siglip_variant="Ti/16"),  # So400m/14, Ti/16, S/32
            data=LeRobotRobocasaHumanThreeImageDataConfig(
                repo_id="daixianjie/robocasa_human_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                # npz_path="gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz", # S/32
                npz_path="gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz", # Ti/16
            ),
            num_train_steps=100_000,
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-4,
                decay_steps= 100_000,
                decay_lr= 2.5e-6),
            # The freeze filter defines which parameters should be frozen during training.
            # We have a convenience function in the model config that returns the default freeze filter
            # for the given model config for LoRA finetuning. Just make sure it matches the model config
            # you chose above.
            freeze_filter=api.pi0Light.Pi0LightConfig(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            # Turn off EMA for LoRA finetuning.
            ema_decay=None,
            num_workers=8,
            batch_size=32,
        ), 
        api.TrainConfig(
            # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini.py pi0mini_robocasa_human_three_image_low_mem_finetune_train --exp-name=pi0mini_robocasa_human_three_image_low_mem_finetune_train_1000k --overwrite
            name="pi0mini_robocasa_human_three_image_low_mem_finetune_train",
            model=api.pi0Light.Pi0LightConfig(paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
            data=LeRobotRobocasaHumanThreeImageDataConfig(
                repo_id="daixianjie/robocasa_human_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 1000_000,
                decay_lr= 2.5e-6),
            num_train_steps=1000_000,
            freeze_filter=api.pi0Light.Pi0LightConfig(
                paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            batch_size=36,
        ),    
        api.TrainConfig(
            # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext.py pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train --exp-name=pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train --project-name=pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train --overwrite
            name="pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train",
            model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
                prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
                sample_frames=2, sample_actions=32, random_select=True,  
                freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
            data=LeRobotRobocasaHumanThreeImageIncontextDataConfig(
                repo_id="daixianjie/robocasa_human_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                task_to_episode='metadata/robocasa/task_to_episode.json',
                episode_to_indexes_file='metadata/robocasa/episode_to_indexes.json',
                states_cache_path="metadata/robocasa/episode_states_cache.json",
                actions_cache_path="metadata/robocasa/episode_actions_cache.json",
                remove_task_list=api.DEFAULT_ROBOCASA_TEST_TASK,
                episode_json_path=api.DEFAULT_ROBOCASA_EPISODE_JSON,
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 500_000,
                decay_lr= 2.5e-6),
            num_train_steps=500_000,
            freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
                prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
                sample_frames=2, sample_actions=32, random_select=True, 
                freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=8,
            batch_size=36,
        ), 
        api.TrainConfig(
            # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext.py pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train --exp-name=pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train --project-name=pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train --overwrite
            name="pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_inference",
            model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
                prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
                sample_frames=2, sample_actions=32, random_select=True,  
                freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
            data=LeRobotRobocasaHumanThreeImageIncontextDataConfig(
                repo_id="daixianjie/robocasa_human_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                task_to_episode='metadata/robocasa/task_to_episode.json',
                episode_to_indexes_file='metadata/robocasa/episode_to_indexes.json',
                states_cache_path="metadata/robocasa/episode_states_cache.json",
                actions_cache_path="metadata/robocasa/episode_actions_cache.json",
                # remove_task_list=api.DEFAULT_ROBOCASA_TEST_TASK,
                # episode_json_path=api.DEFAULT_ROBOCASA_EPISODE_JSON,
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 500_000,
                decay_lr= 2.5e-6),
            num_train_steps=500_000,
            freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
                prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
                sample_frames=2, sample_actions=32, random_select=True, 
                freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=8,
            batch_size=36,
        ), 

        api.TrainConfig(
            name="pi0tiny_robocasa_mg_three_image_train_split_large_lr",
            model=api.pi0Light.Pi0LightConfig(
                vocab_size=50_000, 
                paligemma_variant="gemma_A", action_expert_variant="gemma_B", 
                freeze_llm_embedder=False, freeze_img_encoder=False, 
                siglip_variant="S/16"),
            data=LeRobotRobocasaMgThreeImageDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                remove_task_list=api.DEFAULT_ROBOCASA_MG_TEST_TASK,
                episode_json_path=api.DEFAULT_ROBOCASA_MG_EPISODE_JSON,
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.EmptyLoader(),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-4,
                decay_steps= 500_000,
                decay_lr= 2.5e-5),
            num_train_steps=500_000,
            # freeze_filter=api.pi0Light.Pi0LightConfig(
            #     vocab_size=50_000, paligemma_variant="gemma_A", action_expert_variant="gemma_B", freeze_llm_embedder=False, freeze_img_encoder = False, siglip_variant="S/16",
            # ).get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ),
        api.TrainConfig(
            name="pi0tiny_robocasa_mg_three_image_train_split",
            model=api.pi0Light.Pi0LightConfig(
                vocab_size=50_000, 
                paligemma_variant="gemma_A", action_expert_variant="gemma_B", 
                freeze_llm_embedder=False, freeze_img_encoder=False, 
                siglip_variant="S/16"),
            data=LeRobotRobocasaMgThreeImageDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                remove_task_list=api.DEFAULT_ROBOCASA_MG_TEST_TASK,
                episode_json_path=api.DEFAULT_ROBOCASA_MG_EPISODE_JSON,
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.EmptyLoader(),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 1_000_000,
                decay_lr= 2.5e-6),
            num_train_steps=1_000_000,
            # freeze_filter=api.pi0Light.Pi0LightConfig(
            #     vocab_size=50_000, paligemma_variant="gemma_A", action_expert_variant="gemma_B", freeze_llm_embedder=False, freeze_img_encoder = False, siglip_variant="S/16",
            # ).get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ),
        api.TrainConfig(
            name="pi0tiny_robocasa_mg_three_image_inference",
            model=api.pi0Light.Pi0LightConfig(
                vocab_size=50_000, 
                paligemma_variant="gemma_A", action_expert_variant="gemma_B", 
                freeze_llm_embedder=False, freeze_img_encoder=False, 
                siglip_variant="S/16"),
            data=LeRobotRobocasaMgThreeImageDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.EmptyLoader(),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 500_000,
                decay_lr= 2.5e-6),
            num_train_steps=500_000,
            # freeze_filter=api.pi0Light.Pi0LightConfig(
            #     vocab_size=50_000, paligemma_variant="gemma_A", action_expert_variant="gemma_B", freeze_llm_embedder=False, freeze_img_encoder = False, siglip_variant="S/16",
            # ).get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ),
        api.TrainConfig(
            name="pi0tiny_incontext_robocasa_mg_three_image_train_split_large_lr",
            model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
                vocab_size=50_000, 
                prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
                sample_frames=2, sample_actions=32, random_select=True,  
                freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16"),
            data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                task_to_episode='metadata/robocasa_mg/task_to_episode.json',
                episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
                states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
                actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
                remove_task_list=api.DEFAULT_ROBOCASA_MG_TEST_TASK,
                episode_json_path=api.DEFAULT_ROBOCASA_MG_EPISODE_JSON,
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.EmptyLoader(),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-4,
                decay_steps= 500_000,
                decay_lr= 2.5e-5),
            num_train_steps=500_000,
            # freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            #     vocab_size=50_000, 
            #     prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            #     sample_frames=2, sample_actions=32, random_select=True,  
            #     freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16").get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ), 
        
        api.TrainConfig(
            name="pi0tiny_incontext_robocasa_mg_three_image_train_split",
            model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
                vocab_size=50_000, 
                prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
                sample_frames=2, sample_actions=32, random_select=True,  
                freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16"),
            data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                task_to_episode='metadata/robocasa_mg/task_to_episode.json',
                episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
                states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
                actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
                remove_task_list=api.DEFAULT_ROBOCASA_MG_TEST_TASK,
                episode_json_path=api.DEFAULT_ROBOCASA_MG_EPISODE_JSON,
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.EmptyLoader(),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 500_000,
                decay_lr= 2.5e-6),
            num_train_steps=500_000,
            # freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            #     vocab_size=50_000, 
            #     prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            #     sample_frames=2, sample_actions=32, random_select=True,  
            #     freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16").get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ), 
        api.TrainConfig(
            name="pi0tiny_incontext_robocasa_mg_three_image_inference",
            model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
                vocab_size=50_000, 
                prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
                sample_frames=2, sample_actions=32, random_select=True,  
                freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16"),
            data=LeRobotRobocasaMgThreeImageIncontextDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                task_to_episode='metadata/robocasa_mg/task_to_episode.json',
                episode_to_indexes_file='metadata/robocasa_mg/episode_to_indexes.json',
                states_cache_path="metadata/robocasa_mg/episode_states_cache.json",
                actions_cache_path="metadata/robocasa_mg/episode_actions_cache.json",
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.EmptyLoader(),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 500_000,
                decay_lr= 2.5e-6),
            num_train_steps=500_000,
            # freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            #     vocab_size=50_000, 
            #     prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            #     sample_frames=2, sample_actions=32, random_select=True,  
            #     freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="S/16").get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ), 
        
        api.TrainConfig(
            # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini.py pi0mini_robocasa_mg_three_image_low_mem_finetune_train --exp-name=pi0mini_robocasa_mg_three_image_low_mem_finetune_train --overwrite
            name="pi0mini_robocasa_mg_three_image_low_mem_finetune_train",
            model=api.pi0Light.Pi0LightConfig(paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
            data=LeRobotRobocasaMgThreeImageDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 2_000_000,
                decay_lr= 2.5e-6),
            num_train_steps=2_000_000,
            freeze_filter=api.pi0Light.Pi0LightConfig(
                paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ),  
        api.TrainConfig(
            # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini.py pi0mini_robocasa_mg_three_image_low_mem_finetune_train_split --exp-name=pi0mini_robocasa_mg_three_image_low_mem_finetune_train_split --overwrite
            name="pi0mini_robocasa_mg_three_image_low_mem_finetune_train_split",
            model=api.pi0Light.Pi0LightConfig(paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
            data=LeRobotRobocasaMgThreeImageDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
                remove_task_list=api.DEFAULT_ROBOCASA_MG_TEST_TASK,
                episode_json_path=api.DEFAULT_ROBOCASA_MG_EPISODE_JSON,
            ),
            vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
            weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 2_000_000,
                decay_lr= 2.5e-6),
            num_train_steps=2_000_000,
            freeze_filter=api.pi0Light.Pi0LightConfig(
                paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ), 
            
        api.TrainConfig(
            # no delta with split
            name="pi0_robocasa_mg_three_image_low_mem_finetune_train",
            # Here is an example of loading a pi0 model for LoRA fine-tuning.
            model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotRobocasaMgThreeImageDataConfig(
                repo_id="daixianjie/robocasa_mg_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  
                    prompt_from_task=True,
                ),
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule = api._optimizer.CosineDecaySchedule(
                warmup_steps = 1_000,
                peak_lr= 2.5e-5,
                decay_steps= 500_000,
                decay_lr= 2.5e-6),
            num_train_steps=500_000,
            freeze_filter=api.pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            # Turn off EMA for LoRA finetuning.
            ema_decay=None,
            num_workers=16,
            batch_size=32,
        ),
    
    ]



        

    

