from __future__ import annotations

import dataclasses
import pathlib

from typing_extensions import override

import openpi.policies.libero_incontext_policy as libero_incontext_policy
import openpi.policies.libero_policy as libero_policy


def build(api) -> list[api.TrainConfig]:
    g = globals()
    g.setdefault("DataConfig", object)
    g.setdefault("BaseModelConfig", object)

    @dataclasses.dataclass(frozen=True)
    class LeRobotLiberoDataConfig(api.DataConfigFactory):
        use_delta_joint_actions: bool = True

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
                        }
                    )
                ]
            )

            # Prepare data for policy training
            # Convert images to uint8 numpy arrays, add masks
            data_transforms = api._transforms.Group(
                inputs=[
                    libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)
                ],
                outputs=[libero_policy.LiberoOutputs()],
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
                train_episode=api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list),
            )

    @dataclasses.dataclass(frozen=True)
    class LeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
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

    @dataclasses.dataclass(frozen=True)
    class CustomLeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
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
    class LeRobotLiberoStageIncontextDataConfig(api.DataConfigFactory):
        use_delta_joint_actions: bool = True
        states_cache_path: str = "metadata/libero/episode_states_cache.json"
        actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
        task_to_episode: str = "metadata/libero/task_to_episode.json"
        episode_to_indexes_file: str = "metadata/libero/episode_to_indexes.json"

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
            # --- choose train episodes ---
            if self.keep_episode_filename_list is not None:
                # white list
                train_epi = api.get_kept_episode_indices(
                    self.episode_json_path,
                    exclude_task_language=None,
                    include_episode_filenames=self.keep_episode_filename_list,
                )
            else:
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
        #
        # Fine-tuning Libero configs.
        #
        # in 7b, we full finetune the llm
        # in the 8_1 version, we used dense point tracks
        # in the 8_1_1 version, we used all
        # in the 8_2 version, we used video tokens
        # in the 8_2 version, we used all the prompts
        # Xianjie: pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32 with train_test_split
        api.TrainConfig(
            name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
            model=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=32,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
                actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=32,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            # num_workers=16,
            num_workers=1,
            batch_size=32,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
            model=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=32,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=32,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            # num_workers=16,
            num_workers=4,
            batch_size=32,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=32,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=2,
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
            freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=32,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=8,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor_inference",
            model=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=32,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                padding_mode="linspace_repeat",
                mask_padding_as_valid=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
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
            name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor_train_split_v5",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=32,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=2,
                sample_actions=32,
                task_to_episode_path="metadata/libero/task_to_episode.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V5,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                random_select=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=32,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=8,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor_train_split_v5_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=32,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
                actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
                padding_mode="linspace_repeat",
                mask_padding_as_valid=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
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
            name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample128_random_select_without_delta_train_split_dataset_refactor",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=128,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=2,
                sample_actions=128,
                task_to_episode_path="metadata/libero/task_to_episode.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                random_select=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=128,
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
            name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample128_random_select_without_delta_train_split_dataset_refactor_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=128,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                padding_mode="linspace_repeat",
                mask_padding_as_valid=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=128,
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
            name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample64_random_select_without_delta_train_split_dataset_refactor",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=64,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=2,
                sample_actions=64,
                task_to_episode_path="metadata/libero/task_to_episode.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                random_select=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=64,
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
            name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample64_random_select_without_delta_train_split_dataset_refactor_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=64,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                padding_mode="linspace_repeat",
                mask_padding_as_valid=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=2,
                sample_actions=64,
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
            name="pi0_libero_incontextv18_low_mem_finetune",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=4,
                sample_actions=128,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=4,
                sample_actions=128,
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
                sample_frames=4,
                sample_actions=128,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=40,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=4,
                sample_actions=128,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                sample_frames=4,
                sample_actions=128,
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
            name="pi0_libero_incontextv18_low_mem_finetune_without_img",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=1,
                sample_actions=128,
                random_select=True,
                use_image_prompts=False,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=4,
                sample_actions=128,
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
                sample_frames=1,
                sample_actions=128,
                random_select=True,
                use_image_prompts=False,
            ).get_freeze_filter(),
            ema_decay=None,
            # num_workers=40,
            num_workers=2,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_without_img_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=1,
                sample_actions=128,
                random_select=True,
                use_image_prompts=False,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                sample_frames=1,
                sample_actions=128,
                random_select=True,
                use_image_prompts=False,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=False,  # Inference uses LeRobotDataset + AddStatesActionsPromptTransform
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=128,
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
                sample_actions=128,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=2,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                sample_actions=128,
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
            name="pi0_libero_incontextv18_gemma2b_low_mem_finetune_sample_frames8",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                paligemma_variant="gemma_2b",
                action_expert_variant="gemma_300m",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=128,
                task_to_episode_path="metadata/libero/task_to_episode.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                random_select=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=30_000,
            freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
                paligemma_variant="gemma_2b",
                action_expert_variant="gemma_300m",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=80,
            batch_size=32,
            use_custom_dataloader=True,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_gemma2b_low_mem_finetune_sample_frames8_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                paligemma_variant="gemma_2b",
                action_expert_variant="gemma_300m",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
                actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
                padding_mode="linspace_repeat",
                mask_padding_as_valid=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=30_000,
            freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
                paligemma_variant="gemma_2b",
                action_expert_variant="gemma_300m",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            batch_size=32,
            use_custom_dataloader=False,  # Inference uses LeRobotDataset + AddStatesActionsPromptTransform
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_train_split_v5",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=128,
                task_to_episode_path="metadata/libero/task_to_episode.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V5,
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
                sample_actions=128,
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
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_train_split_v5_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
                actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
                # remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V5,
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
                sample_actions=128,
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
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_paligemma_init",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=128,
                task_to_episode_path="metadata/libero/task_to_episode.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                random_select=True,
            ),
            # weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
            # weight_loader=api.weight_loaders.PaliGemmaWeightLoader(),  # This causes shape mismatch with gemma_300m
            weight_loader=api.weight_loaders.VisionEncoderOnlyLoader(verbose=True, include_embedder=True),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m",
                sample_frames=8,
                sample_actions=128,
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
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_paligemma_init_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
            # weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
            weight_loader=api.weight_loaders.VisionEncoderOnlyLoader(verbose=True, include_embedder=True),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m",
                sample_frames=8,
                sample_actions=128,
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
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_selective_init",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=128,
                task_to_episode_path="metadata/libero/task_to_episode.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                random_select=True,
            ),
            weight_loader=api.weight_loaders.SelectiveVisionAndProjectionsLoader(
                params_path="s3://openpi-assets/checkpoints/pi0_base/params", verbose=True
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=8,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_selective_init_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
                actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
                # Match training behavior: use linspace sampling and mask all frames as valid
                padding_mode="linspace_repeat",
                mask_padding_as_valid=True,
            ),
            weight_loader=api.weight_loaders.SelectiveVisionAndProjectionsLoader(
                params_path="s3://openpi-assets/checkpoints/pi0_base/params", verbose=True
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            batch_size=32,
            use_custom_dataloader=False,  # Inference uses LeRobotDataset + AddStatesActionsPromptTransform
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_avg_current_img",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
                avg_current_img=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=128,
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
                sample_actions=128,
                random_select=True,
                avg_current_img=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=2,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_avg_current_img_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
                avg_current_img=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                sample_actions=128,
                random_select=True,
                avg_current_img=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=False,  # Inference uses LeRobotDataset + AddStatesActionsPromptTransform
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_wo_compress_state",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
                compress_state_action_prompts=False,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=128,
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
                sample_actions=128,
                random_select=True,
                compress_state_action_prompts=False,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=2,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_wo_compress_state_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
                compress_state_action_prompts=False,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                sample_actions=128,
                random_select=True,
                compress_state_action_prompts=False,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=False,  # Inference uses LeRobotDataset + AddStatesActionsPromptTransform
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_sample_actions64",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=64,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=64,
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
                sample_actions=64,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=64,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_sample_actions64_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=64,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                sample_actions=64,
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
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_sample_actions256",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=256,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=256,
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
                sample_actions=256,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=64,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_sample_actions256_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=256,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                sample_actions=256,
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
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_sample_actions32",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=32,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
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
            num_workers=40,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_sample_actions32_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=32,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_without_text",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
                use_text_prompts=False,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=128,
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
                sample_actions=128,
                random_select=True,
                use_text_prompts=False,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=8,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_without_text_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
                use_text_prompts=False,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                sample_actions=128,
                random_select=True,
                use_text_prompts=False,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=False,  # Inference uses LeRobotDataset + AddStatesActionsPromptTransform
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_without_state_action",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
                use_action_state_prompts=False,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=128,
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
                sample_actions=128,
                random_select=True,
                use_action_state_prompts=False,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=16,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_without_state_action_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
                use_action_state_prompts=False,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                sample_actions=128,
                random_select=True,
                use_action_state_prompts=False,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=False,  # Inference uses LeRobotDataset + AddStatesActionsPromptTransform
            # wandb_enabled=False,
        ),
        # v3 defined here, corresponds to v2 in google sheet.
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_split_v3",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=128,
                task_to_episode_path="metadata/libero/task_to_episode.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V3,
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
                sample_actions=128,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=2,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_split_v3_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                sample_actions=128,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=False,
            # wandb_enabled=False,
        ),
        # v3 defined here, corresponds to v2 in google sheet.
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_sample_actions64_split_v3",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=64,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=64,
                task_to_episode_path="metadata/libero/task_to_episode.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V3,
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
                sample_actions=64,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=40,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_sample_actions64_split_v3_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=64,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                sample_actions=64,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=False,  # Inference uses LeRobotDataset + AddStatesActionsPromptTransform
            # wandb_enabled=False,
        ),
        # v4 defined here, corresponds to v3 in google sheet.
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_split_v4",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=CustomLeRobotLiberoIncontextDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                frame_sequence_length=1,
                sample_frames=8,
                sample_actions=128,
                task_to_episode_path="metadata/libero/task_to_episode.json",
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V4,
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
                sample_actions=128,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=2,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
            # wandb_enabled=False,
        ),
        api.TrainConfig(
            name="pi0_libero_incontextv18_low_mem_finetune_sample_frames8_split_v4_inference",
            assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
            model=api.pi0_incontextv18.Pi0IncontextConfigv18(
                prompt_expert_variant="gemma_300m_v2",
                action_expert_variant="gemma_300m_lora",
                sample_frames=8,
                sample_actions=128,
                random_select=True,
            ),
            data=LeRobotLiberoIncontextDataConfig(
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
                sample_actions=128,
                random_select=True,
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=False,  # Inference uses LeRobotDataset + AddStatesActionsPromptTransform
            # wandb_enabled=False,
        ),
        # TODO: 12_7 is not finished yet, finish it
        # ablation of causal attention
        api.TrainConfig(
            name="pi0_libero_low_mem_finetune_split_train",
            model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                use_delta_joint_actions=False,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=20_000,
            freeze_filter=api.pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=8,
            batch_size=36,
        ),
        api.TrainConfig(
            name="pi0_libero_low_mem_finetune_split_train_v3",
            model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V3,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                use_delta_joint_actions=False,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=20_000,
            freeze_filter=api.pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=8,
            batch_size=36,
        ),
        api.TrainConfig(
            name="pi0_libero_low_mem_finetune_split_train_v4",
            model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V4,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                use_delta_joint_actions=False,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=20_000,
            freeze_filter=api.pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=8,
            batch_size=36,
        ),
        api.TrainConfig(
            name="pi0_libero_low_mem_finetune_split_inference",
            model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=20_000,
            freeze_filter=api.pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
            batch_size=32,
        ),
        # Xianjie:
        # Xianjie:
        # Xianjie:
        #
        # Fine-tuning Libero configs.
        #
        api.TrainConfig(
            name="pi0_libero",
            model=api.pi0.Pi0Config(),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            # num_train_steps=30_000,
            num_train_steps=10_000,
        ),
        api.TrainConfig(
            name="pi0_libero_without_delta",
            model=api.pi0.Pi0Config(),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            # num_train_steps=10_000,
        ),
        api.TrainConfig(
            name="pi0_libero_zero",
            model=api.pi0.Pi0Config(),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                assets=api.AssetsConfig(
                    assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                    asset_id="droid",
                ),
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            # num_train_steps=30_000,
            num_train_steps=10_000,
        ),
        api.TrainConfig(
            name="pi0_libero_low_mem_finetune",
            model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            # num_train_steps=30_000,
            num_train_steps=10_000,
            freeze_filter=api.pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=4,
        ),
        api.TrainConfig(
            name="pi0_fast_libero",
            model=api.pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader(
                "s3://openpi-assets/checkpoints/pi0_fast_base/params"
            ),
            num_train_steps=20_000,
        ),
        api.TrainConfig(
            name="pi0_fast_libero_low_mem_finetune",
            wandb_enabled=False,
            model=api.pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader(
                "s3://openpi-assets/checkpoints/pi0_fast_base/params"
            ),
            num_train_steps=20_000,
            freeze_filter=api.pi0_fast.Pi0FASTConfig(
                action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        ## rebuttal exp
        # InSpire Training Setting
        # pi0 incontext v12 none lora
        # pi0 incontextv12 lora
        # pi0
        api.TrainConfig(
            name="pi0_libero90_low_mem_finetune",
            model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotLiberoDataConfig(
                repo_id="daixianjie/libero_90_lerobot",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule=api._optimizer.CosineDecaySchedule(
                warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6
            ),
            num_train_steps=30_000,
            freeze_filter=api.pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=8,
            batch_size=32,
        ),
        api.TrainConfig(
            name="pi0_libero90_low_mem_finetune_inference",
            model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=False,  # Set to True for local-only datasets.
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule=api._optimizer.CosineDecaySchedule(
                warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6
            ),
            num_train_steps=30_000,
            freeze_filter=api.pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            num_workers=8,
            batch_size=32,
        ),
    ]
