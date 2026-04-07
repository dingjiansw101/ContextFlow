# config_sequence_debug.py
"""
An example of child configs.
Debug experiment child configs.
Parent config.py will import this file and call build(api) to collect TrainConfig entries.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import pathlib

import jsonlines
from typing_extensions import override

# If this child only needs certain policies/modules here, import them directly
import openpi.policies.libero_incontext_policy as libero_incontext_policy


def build(api) -> list[api.TrainConfig]:
    g = globals()
    g["DataConfig"] = api.DataConfig
    g["BaseModelConfig"] = api._model.BaseModelConfig

    # 1) Define DataConfig subclasses inside this function, inheriting DataConfigFactory via api
    @dataclasses.dataclass(frozen=True)
    class SequenceDebugLeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
        use_delta_joint_actions: bool = True
        states_cache_path: str = "metadata/libero/episode_states_cache.json"
        actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
        task_to_episode: str = "metadata/libero/task_to_episode.json"
        episode_to_indexes_file: str = "metadata/libero/episode_to_indexes.json"

        # Padding mode for AddStatesActionsPromptTransform
        padding_mode: str = "keep_all"
        mask_padding_as_valid: bool = False

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: BaseModelConfig) -> DataConfig:
            # Make inputs look like they come from the Libero environment
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "observation/image": "image",
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

            # Xianjie: calculate training episode indexi first
            train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)

            # Prepare data for policy training
            # inject the indexes of demo prompt, TODO: provide json file_paths here
            data_transforms = api._transforms.Group(
                inputs=[
                    api._transforms.InjectDemoIndexes(
                        sample_frames=model_config.sample_frames,
                        random_select=model_config.random_select,
                        sample_episodes=model_config.sample_episodes,
                        task_to_episode=self.task_to_episode,
                        episode_to_indexes=self.episode_to_indexes_file,
                        train_episode_index_list=train_epi,
                    )
                ],
                outputs=[],
            )

            # Convert images to uint8 numpy arrays, add masks
            data_transforms = data_transforms.push(
                inputs=[
                    libero_incontext_policy.LiberoIncontextInputs(
                        action_dim=model_config.action_dim, model_type=model_config.model_type
                    )
                ],
                outputs=[libero_incontext_policy.LiberoIncontextOutputs()],
            )

            # TODO: fix the bug of libero actions.
            # fix it and re-train on libero
            # Use delta actions (not for gripper)
            if self.use_delta_joint_actions:
                delta_action_mask = api._transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[api._transforms.DeltaActions(delta_action_mask)],
                    outputs=[api._transforms.AbsoluteActions(delta_action_mask)],
                )
            # else:
            # Model transforms include things like tokenizing the prompt and action targets
            model_transforms = api.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                train_episode=train_epi,
            )
      
    @dataclasses.dataclass(frozen=True)
    class XJCustomLeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
        """Config for CustomLeRobotDataset with in-context learning.

        Unlike LeRobotLiberoIncontextDataConfig which uses transforms to add
        demonstration data, CustomLeRobotLiberoIncontextDataConfig delegates this to
        the dataset itself via CustomLeRobotDataset.
        """

        use_delta_joint_actions: bool = False

        # CustomLeRobotDataset specific parameters
        custom_dataloader_version: str = "v1"  # Version of custom dataloader to use
        frame_sequence_length: int = 1  # Number of consecutive frames for main context
        sample_frames: int = 2  # Number of frames for in-context demonstration
        sample_actions: int = 32  # Number of actions for in-context demonstration
        task_to_episode_path: str = "metadata/libero/task_to_episode.json"
        states_cache_path: str = "metadata/libero/episode_states_cache.json"
        actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
        random_select: bool = True  # If True, randomly select demo episodes; if False, use deterministic selection
        norm_stats_aliases: dict[str, str] | None = dataclasses.field(
            default_factory=lambda: {
                "dem_prompt_all_states": "state",
                "dem_prompt_all_actions": "actions",
            }
        )

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: BaseModelConfig) -> DataConfig:
            # Make inputs look like they come from the Libero environment
            # Pass through dem_prompt_* keys from CustomLeRobotDataset
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                            "episode_index": "episode_index",
                            "frame_index": "frame_index",
                            "index": "index",
                            "task_index": "task_index",
                            # Pass through dem_prompt_* keys from CustomLeRobotDataset
                            # dem_prompt_images is nested, so map the flattened keys
                            "dem_prompt_images": {
                                "image": "dem_prompt_images/image",
                                "wrist_image": "dem_prompt_images/wrist_image",
                            },
                            "dem_prompt_states": "dem_prompt_states",
                            "dem_prompt_actions": "dem_prompt_actions",
                            "selected_episode": "selected_episode",
                        }
                    )
                ]
            )

            # Calculate training episode indices
            train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)

            # Prepare data for policy training
            # NOTE: CustomLeRobotDataset handles demo loading internally,
            # so we DON'T use InjectDemoIndexes
            data_transforms = api._transforms.Group(
                inputs=[
                    libero_incontext_policy.CustomLeRobotLiberoIncontextInputs(
                        action_dim=model_config.action_dim, model_type=model_config.model_type
                    )
                ],
                outputs=[libero_incontext_policy.LiberoIncontextOutputs()],
            )

            # Use delta actions (not for gripper)
            if self.use_delta_joint_actions:
                delta_action_mask = api._transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[api._transforms.DeltaActions(delta_action_mask)],
                    outputs=[api._transforms.AbsoluteActions(delta_action_mask)],
                )

            # Model transforms include things like tokenizing the prompt and action targets
            model_transforms = api.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                train_episode=train_epi,
            )  
            
    @dataclasses.dataclass(frozen=True)
    class XJLeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
        use_delta_joint_actions: bool = True
        states_cache_path: str = "metadata/libero/episode_states_cache.json"
        actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
        task_to_episode: str = "metadata/libero/task_to_episode.json"
        episode_to_indexes_file: str = "metadata/libero/episode_to_indexes.json"

        # Padding strategy for AddStatesActionsPromptTransform
        # "keep_all": Keep all frames when L < max_len, then pad (default, current behavior)
        # "linspace_repeat": Always linspace sample, then repeat last (training behavior)
        padding_mode: str = "keep_all"

        # Whether to mask padded frames as valid (True) or invalid (False)
        # True = training behavior, False = current inference behavior (default)
        mask_padding_as_valid: bool = False

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: BaseModelConfig) -> DataConfig:
            # Make inputs look like they come from the Libero environment
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "observation/image": "image",
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

            # Xianjie: calculate training episode indexi first
            train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)

            # Prepare data for policy training
            # inject the indexes of demo prompt, TODO: provide json file_paths here
            data_transforms = api._transforms.Group(
                inputs=[
                    api._transforms.InjectDemoIndexes(
                        sample_frames=model_config.sample_frames,
                        random_select=model_config.random_select,
                        sample_episodes=model_config.sample_episodes,
                        task_to_episode=self.task_to_episode,
                        episode_to_indexes=self.episode_to_indexes_file,
                        train_episode_index_list=train_epi,
                    )
                ],
                outputs=[],
            )

            # Convert images to uint8 numpy arrays, add masks
            data_transforms = data_transforms.push(
                inputs=[
                    libero_incontext_policy.LiberoIncontextInputs(
                        action_dim=model_config.action_dim, model_type=model_config.model_type
                    )
                ],
                outputs=[libero_incontext_policy.LiberoIncontextOutputs()],
            )

            # TODO: fix the bug of libero actions.
            # fix it and re-train on libero
            # Use delta actions (not for gripper)
            if self.use_delta_joint_actions:
                delta_action_mask = api._transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[api._transforms.DeltaActions(delta_action_mask)],
                    outputs=[api._transforms.AbsoluteActions(delta_action_mask)],
                )
            # else:
            # Model transforms include things like tokenizing the prompt and action targets
            model_transforms = api.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                train_episode=train_epi,
            )

    # 2) Return this child's TrainConfig entries directly (can be multiple)
    return [
        api.TrainConfig(
            name="xj_libero_incontextv18_low_mem_finetune_sample_frames8_actions32",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=32,
                random_select=True,
            ),
            data=XJCustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=32,
                task_to_episode_path="metadata/libero/task_to_episode.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                random_select=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=32,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="xj_libero_incontextv18_low_mem_finetune_sample_frames8_actions32_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=32,
                random_select=True,
            ),
            data=XJLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
                actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
                # remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
                # episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                # Match training behavior: use linspace sampling and mask all frames as valid
                padding_mode="linspace_repeat",
                mask_padding_as_valid=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=32,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=False,  # Inference uses LeRobotDataset + AddStatesActionsPromptTransform
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="xj_libero_incontextv18_low_mem_finetune_sample_frames8_actions8",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=8,
                random_select=True,
            ),
            data=XJCustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=8,
                task_to_episode_path="metadata/libero/task_to_episode.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                random_select=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=8,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="xj_libero_incontextv18_low_mem_finetune_sample_frames8_actions8_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=8,
                random_select=True,
            ),
            data=XJLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
                actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
                # remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
                # episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                # Match training behavior: use linspace sampling and mask all frames as valid
                padding_mode="linspace_repeat",
                mask_padding_as_valid=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=8,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=False,  # Inference uses LeRobotDataset + AddStatesActionsPromptTransform
            # wandb_enabled=False,
        ),
    ]
