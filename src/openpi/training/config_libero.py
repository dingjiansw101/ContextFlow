from __future__ import annotations
import dataclasses
import tyro
import pathlib
import json
from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias, Optional, List, Union, Iterable
from pathlib import Path
from typing_extensions import override

import openpi.policies.libero_incontext_policy as libero_incontext_policy
import openpi.policies.libero_policy as libero_policy


def build(api) -> list["api.TrainConfig"]:
    g = globals()
    g.setdefault("DataConfig", object)
    g.setdefault("BaseModelConfig", object)

    @dataclasses.dataclass(frozen=True)
    class LeRobotLiberoDataConfig(api.DataConfigFactory):
        use_delta_joint_actions: bool = True
        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
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
                inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
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
        task_to_episode: str='metadata/libero/task_to_episode.json'
        episode_to_indexes_file: str='metadata/libero/episode_to_indexes.json'
        tracks_path: str = "metadata/libero/episode_tracks_combined.json"
        libero_input_refactor: bool = False

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
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
                inputs=[api._transforms.InjectDemoIndexes(sample_frames=model_config.sample_frames, 
                                                    random_select=model_config.random_select,
                                                    sample_episodes=model_config.sample_episodes,
                                                    task_to_episode=self.task_to_episode,
                                                    episode_to_indexes=self.episode_to_indexes_file,
                                                    train_episode_index_list=train_epi)],
                outputs=[],
            )

            # Convert images to uint8 numpy arrays, add masks
            if self.libero_input_refactor:
                data_transforms = data_transforms.push(
                    inputs=[
                        libero_incontext_policy.LiberoIncontextInputs_refactor(
                            action_dim=model_config.action_dim, model_type=model_config.model_type
                        )
                    ],
                    outputs=[libero_incontext_policy.LiberoIncontextOutputs()],
                )
            else:
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
                # import ipdb; ipdb.set_trace()
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
        custom_dataloader_version: str = "v1"  # Version of custom dataloader to use ("v1" or "v2")
        frame_sequence_length: int = 1  # Number of consecutive frames for main context
        sample_frames: int = 2  # Number of frames for in-context demonstration
        sample_actions: int = 32  # Number of actions for in-context demonstration
        task_to_episode_path: str = "metadata/libero/task_to_episode.json"
        states_cache_path: str = "metadata/libero/episode_states_cache.json"
        actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
        random_select: bool = True  # If True, randomly select demo episodes; if False, use deterministic selection
        norm_stats_aliases: dict[str, str] | None = dataclasses.field(default_factory=lambda: {
            "dem_prompt_all_states": "state",
            "dem_prompt_all_actions": "actions",
        })

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
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
                                "wrist_image": "dem_prompt_images/wrist_image"
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
    class Customv2LeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
        """Config for CustomLeRobotDataset with in-context learning.

        Unlike LeRobotLiberoIncontextDataConfig which uses transforms to add
        demonstration data, Customv2LeRobotLiberoIncontextDataConfig delegates this to
        the dataset itself via CustomLeRobotDatasetv2.
        """
        use_delta_joint_actions: bool = False

        # CustomLeRobotDataset specific parameters
        custom_dataloader_version: str = "v2"  # Version of custom dataloader to use ("v1" or "v2")
        frame_sequence_length: int = 1  # Number of consecutive frames for main context
        sample_frames: int = 2  # Number of frames for in-context demonstration
        sample_actions: int = 32  # Number of actions for in-context demonstration
        task_to_episode_path: str = "metadata/libero/task_to_episode.json"
        states_cache_path: str = "metadata/libero/episode_states_cache.json"
        actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
        random_select: bool = True  # If True, randomly select demo episodes; if False, use deterministic selection
        norm_stats_aliases: dict[str, str] | None = dataclasses.field(default_factory=lambda: {
            "dem_prompt_all_states": "state",
            "dem_prompt_all_actions": "actions",
            "current_state_seq": "state",
            "actions_seq": "actions",
            "future_states": "state",  # Future states use same normalization as current states
        })
        current_frame_sample_mode: str = "random"
        use_future_states: bool = False  # Enable loading future states for supervision
        future_state_downsample: int = 5  # Downsample factor for future states (e.g., 5 means every 5th action timestep)
        multiple_current_frames: bool = True  # Enable loading multiple current frames as a sequence
        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            # Validate parameter constraints
            if self.use_future_states and self.multiple_current_frames:
                raise ValueError(
                    "Cannot use both use_future_states=True and multiple_current_frames=True. "
                    "These features are currently incompatible. Please set one of them to False."
                )

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
                                "wrist_image": "dem_prompt_images/wrist_image"
                            },
                            "dem_prompt_states": "dem_prompt_states",
                            "dem_prompt_actions": "dem_prompt_actions",
                            "selected_episode": "selected_episode",
                            # V2-specific: current frame sequence fields
                            # current_images_seq is nested, so map the flattened keys
                            "current_images_seq": {
                                "image": "current_images_seq/image",
                                "wrist_image": "current_images_seq/wrist_image"
                            },
                            "current_state_seq": "current_state_seq",
                            "actions_seq": "actions_seq",
                            "future_states": "future_states",  # V2-specific: future states for supervision
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
    class Customv2FutureStatesLeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
        """Config for CustomLeRobotDatasetv2 with future_states support.

        Unlike Customv2LeRobotLiberoIncontextDataConfig which supports frame sequences,
        this variant is optimized for future state prediction without multiple current frames.
        """
        use_delta_joint_actions: bool = False

        # CustomLeRobotDataset specific parameters
        custom_dataloader_version: str = "v2"
        frame_sequence_length: int = 1  # Must be 1 for future_states
        sample_frames: int = 2
        sample_actions: int = 32
        task_to_episode_path: str = "metadata/libero/task_to_episode.json"
        states_cache_path: str = "metadata/libero/episode_states_cache.json"
        actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
        random_select: bool = True
        norm_stats_aliases: dict[str, str] | None = dataclasses.field(default_factory=lambda: {
            "dem_prompt_all_states": "state",
            "dem_prompt_all_actions": "actions",
            "future_states": "state",  # Future states use same normalization as current states
        })
        current_frame_sample_mode: str = "random"
        use_future_states: bool = True  # Enable by default
        future_state_downsample: int = 5
        multiple_current_frames: bool = False  # Must be False for future_states

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            # Validate constraints
            if self.use_future_states and self.multiple_current_frames:
                raise ValueError(
                    "Cannot use both use_future_states=True and multiple_current_frames=True. "
                    "These features are currently incompatible. Please set one of them to False."
                )
            if self.frame_sequence_length != 1:
                raise ValueError(
                    "frame_sequence_length must be 1 when use_future_states=True (got {self.frame_sequence_length})"
                )

            # Simplified RepackTransform WITHOUT frame sequence fields
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
                            "dem_prompt_images": {
                                "image": "dem_prompt_images/image",
                                "wrist_image": "dem_prompt_images/wrist_image"
                            },
                            "dem_prompt_states": "dem_prompt_states",
                            "dem_prompt_actions": "dem_prompt_actions",
                            "selected_episode": "selected_episode",
                            # V2-specific: Only future_states, NO frame sequence fields
                            "future_states": "future_states",
                        }
                    )
                ]
            )

            # Calculate training episode indices
            train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)

            # Prepare data for policy training
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
        task_to_episode: str='metadata/libero/task_to_episode.json'
        episode_to_indexes_file: str='metadata/libero/episode_to_indexes.json'
        tracks_path: str = "metadata/libero/episode_tracks_combined.json"
        libero_input_refactor: bool = False
        # white list: a list of training episodes
        all_episode_stage: Optional[Union[str, Path, List[str]]] = None

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
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
                inputs=[api._transforms.InjectDemoIndexes(sample_frames=model_config.sample_frames, 
                                                    random_select=model_config.random_select,
                                                    sample_episodes=model_config.sample_episodes,
                                                    task_to_episode=self.task_to_episode,
                                                    episode_to_indexes=self.episode_to_indexes_file,
                                                    train_episode_index_list=train_epi,
                                                    all_episode_stage = self.all_episode_stage)],
                outputs=[],
            )

            # Convert images to uint8 numpy arrays, add masks
            if self.libero_input_refactor:
                data_transforms = data_transforms.push(
                    inputs=[
                        libero_incontext_policy.LiberoIncontextInputs_refactor(
                            action_dim=model_config.action_dim, model_type=model_config.model_type
                        )
                    ],
                    outputs=[libero_incontext_policy.LiberoIncontextOutputs()],
                )
            else:
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
                # import ipdb; ipdb.set_trace()
            # Model transforms include things like tokenizing the prompt and action targets
            model_transforms = api.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                train_episode=train_epi,
            )


    # 2) 直接返回本 child 的 TrainConfig 条目（可多个）
    return [
#
    # Fine-tuning Libero configs.
    #
    api.TrainConfig(
        name="pi0_libero_incontext_low_mem_finetune",
        model=api.pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        batch_size=36,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_libero_incontext_low_mem_finetune_sample2",
        model=api.pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=8, random_select=False,
            # paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=False, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=api.pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=8, random_select=False,
            # paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_libero_incontext_low_mem_finetune_sample2_random_select",
        model=api.pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=8,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=api.pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=8,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=36,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_libero_incontext_low_mem_finetune_sample2_actionssample32",
        model=api.pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=api.pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),
    
    api.TrainConfig(
        name="pi0_libero_incontext_low_mem_finetune_sample2_actionssample32_without_delta",
        model=api.pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json"
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontext.Pi0IncontextConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64_train_split",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64_train_split_inference",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            # remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample128",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=128, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=128, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64_long",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=400_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=64, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_without_delta",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json"
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv4_low_mem_finetune_sample2_actionssample32_random_select_without_delta",
        model=api.pi0_incontextv4.Pi0IncontextConfigv4(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json"
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv4.Pi0IncontextConfigv4(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv4.Pi0IncontextConfigv4(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv4.Pi0IncontextConfigv4(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv4.Pi0IncontextConfigv4(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv4.Pi0IncontextConfigv4(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv6.Pi0IncontextConfigv6(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv6.Pi0IncontextConfigv6(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv6.Pi0IncontextConfigv6(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv6.Pi0IncontextConfigv6(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        # batch_size=36,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_without_text_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, use_text_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, use_text_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        # batch_size=36,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_without_text_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m",
             action_expert_variant="gemma_300m_lora", sample_frames=2, 
             sample_actions=32, random_select=True, use_text_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, use_text_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7a_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, block_attention=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        # batch_size=36,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7a_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv11_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv11.Pi0IncontextConfigv11(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, block_attention=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        # num_train_steps=22_500,
        freeze_filter=api.pi0_incontextv11.Pi0IncontextConfigv11(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        # batch_size=36,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv11_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv11.Pi0IncontextConfigv11(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        # num_train_steps=22_500,
        freeze_filter=api.pi0_incontextv11.Pi0IncontextConfigv11(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    # in 7b, we full finetune the llm    
    api.TrainConfig(
        name="pi0_libero_incontextv7b_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, block_attention=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=2, # 
        num_workers=8,
        # batch_size=36,
        batch_size=32,# 
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7b_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
        num_train_steps=22_500,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m",
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True, block_attention=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_10_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=32, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=32, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_10_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=32, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=32, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),


    api.TrainConfig(
        name="pi0_libero_incontextv10_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv10_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, 
            sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv10_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=4, 
            sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", 
            sample_frames=4, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv10_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=4, 
            sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv10.Pi0IncontextConfigv10(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=4, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora",
              sample_frames=2, sample_actions=64, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=64, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=64, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora",
              sample_frames=2, sample_actions=64, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, sample_episodes=4,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, sample_episodes=4,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_8_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_8_low_mem_finetune_sample2_actionssample32_random_select_without_delta",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=8,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_8_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),


    api.TrainConfig(
        name="pi0_libero_incontextv7_7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=8, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_debug",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora",
              sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=True,
              use_action_state_prompts=True, sample_episodes=2,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_5_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_5_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv7_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv7.Pi0IncontextConfigv7(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m", action_expert_variant="gemma_300m", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv8_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
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
            tracks_path="metadata/libero/episode_tracks_combined_all.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32,
              random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv8_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32,
              random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True,
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
            tracks_path="metadata/libero/episode_tracks_combined_all.json",
            # remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    # in the 8_1 version, we used dense point tracks
    api.TrainConfig(
        name="pi0_libero_incontextv8_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, point_track_dim=444,
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
            tracks_path="metadata/libero/episode_tracks_combined_grid32_all.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32,
              random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv8_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32,
              random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, point_track_dim=444,
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
            tracks_path="metadata/libero/episode_tracks_combined_grid32_all.json",
            # remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    # in the 8_1_1 version, we used all 
    api.TrainConfig(
        name="pi0_libero_incontextv8_1_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True, point_track_dim=444,
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
            tracks_path="metadata/libero/episode_tracks_combined_grid32_all.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32,
              random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv8_1_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32,
              random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True, point_track_dim=444,
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
            tracks_path="metadata/libero/episode_tracks_combined_grid32_all.json",
            # remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    # in the 8_2 version, we used video tokens
    api.TrainConfig(
        name="pi0_libero_incontextv8_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, point_track_dim=784,
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
            tracks_path="metadata/libero/video_tokens_merged.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20,
              random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, point_track_dim=784,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv8_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20,
              random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, point_track_dim=784,
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
            tracks_path="metadata/libero/video_tokens_merged.json",
            # remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, point_track_dim=784,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),


    # in the 8_2 version, we used all the prompts
    api.TrainConfig(
        name="pi0_libero_incontextv8_2_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20, 
            random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True, point_track_dim=784,
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
            tracks_path="metadata/libero/video_tokens_merged.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20,
              random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True, point_track_dim=784,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv8_2_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20,
              random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True, point_track_dim=784,
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
            tracks_path="metadata/libero/video_tokens_merged.json",
            # remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv8.Pi0IncontextConfigv8(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m", 
            action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=20, 
            random_select=True, use_image_prompts=True, use_action_state_prompts=True,
            use_point_track_prompts=True, point_track_dim=784,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv6_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv6.Pi0IncontextConfigv6(
             sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv6_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv6.Pi0IncontextConfigv6(
             sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),


    api.TrainConfig(
        name="pi0_libero_incontextv3_low_mem_finetune_sample2_actionssample32_random_select_without_delta",
        model=api.pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json"
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv3_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv3_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv3_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv3_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv3.Pi0IncontextConfigv3(
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),


    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_states_cache_debug",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache_debug.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache_debug.json"
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample8",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=8, random_select=False,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=8, random_select=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        wandb_enabled=False,
    ),

    # Xianjie: pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32 with train_test_split
    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            # Xianjie: newly added para for train-test split
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_splitv2",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            # Xianjie: newly added para for train-test split
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv2.Pi0IncontextConfigv2(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=36,
        wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_embedding_train_split",
        model=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_embedding_train_split_inference",
        model=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v3",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V3,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v4",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V4,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=16, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=16, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=100,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_9_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=16, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=16, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_8_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=64,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_8_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_10_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=8,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=8,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=64,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_10_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=8,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, sample_episodes=8,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_11_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=4, sample_actions=32, random_select=True, sample_episodes=4,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=4, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=64,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_11_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=4, sample_actions=32, random_select=True, sample_episodes=4,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=4, sample_actions=32, random_select=True, sample_episodes=4,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),


    api.TrainConfig(
        name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
            libero_input_refactor=True,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
            libero_input_refactor=True,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=4,
        # num_workers=1,
        batch_size=32,
        use_custom_dataloader=True,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True,
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
            libero_input_refactor=True,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=4,
        # num_workers=1,
        batch_size=32,
        use_custom_dataloader=True,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv18_low_mem_finetune",
        assets_repo_override="pi0_libero_refactor_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_dataset_refactor",
        model=api.pi0_incontextv18.Pi0IncontextConfigv18(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=4, sample_actions=128, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=4, sample_actions=128, random_select=True,
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
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=4, sample_actions=128, random_select=True,
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
            libero_input_refactor=True,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=4, sample_actions=128, random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=4,
        # num_workers=1,
        batch_size=32,
        use_custom_dataloader=True,
        # wandb_enabled=False,
    ),

    # TODO: 12_7 is not finished yet, finish it
    api.TrainConfig(
        name="pi0_libero_incontextv12_7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, causal_attention=True,
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
            # libero_input_refactor=True,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, causal_attention=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    # ablation of causal attention
    api.TrainConfig(
        name="pi0_libero_incontextv12_7_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, causal_attention=True,
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
            # libero_input_refactor=True,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, causal_attention=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),


    api.TrainConfig(
        name="pi0_libero_incontextv12_6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, avg_current_img=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, avg_current_img=True,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_5_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False, use_image_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False, use_image_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_5_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False, use_image_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_action_state_prompts=False, use_image_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_3_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_3_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),


    api.TrainConfig(
        name="pi0_libero_incontextv12_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_text_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_text_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_text_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_text_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),


    api.TrainConfig(
        name="pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=8, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=8, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),


    api.TrainConfig(
        name="pi0_libero_incontextv9_low_mem_finetune_sample2_actionssample32_random_select_without_delta",
        model=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_v9_low_mem_finetune_sample2_actionssample32_random_select_without_delta",
        model=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False, use_action_state_prompts=False,
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, use_image_prompts=False, use_action_state_prompts=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),


    api.TrainConfig(
        name="pi0_libero_incontextv9_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=40_000,
        freeze_filter=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=16,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv9_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
        # weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        weight_loader=api.weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=40_000,
        freeze_filter=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv9_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
        weight_loader=api.weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=2,
        batch_size=32,
        # wandb_enabled=False,
    ),

    api.TrainConfig(
        name="pi0_libero_incontextv9_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_debug",
        model=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
        # weight_loader=api.weight_loaders.PaliGemmaWeightLoader(),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=1,
        # num_workers=2,
        batch_size=32,
        # wandb_enabled=False,
    ),


    api.TrainConfig(
        name="pi0_libero_incontextv9_2_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        model=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
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
        # weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        weight_loader=api.weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv9.Pi0IncontextConfigv9(
            paligemma_variant="gemma_2b_lora", prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=4,
        batch_size=32,
        # wandb_enabled=False,
    ),

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
    api.TrainConfig(
        # XIANJIE: no lora; no split; with delta
        name="pi0_libero_incontextv2_sample2_actionssample64",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),
    # Xianjie:
    api.TrainConfig(
        # XIANJIE: no lora; with split; with delta
        name="pi0_libero_incontextv2_sample2_actionssample64_train_split",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ),   
    # Xianjie:
    api.TrainConfig(
        # XIANJIE: no lora; with split; with delta
        name="pi0_libero_incontextv2_sample2_actionssample64_test_split",
        model=api.pi0_incontextv2.Pi0IncontextConfigv2(
            sample_frames=2, sample_actions=64, random_select=True,
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        num_workers=4,
        batch_size=36,
        # wandb_enabled=False,
    ), 
    #
    # Pi0 Light
    #
    # Xianjie
    # api.TrainConfig(
    #     # no delta with split
    #     name="pi0light_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
    #     model=deprecated_pi0light_incontextv12.Pi0LightIncontextConfigv12(
    #         prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
    #         sample_frames=2, sample_actions=32, random_select=True, siglip_variant="Ti/16"
    #     ),  
    #     data=LeRobotLiberoIncontextDataConfig(
    #         repo_id="physical-intelligence/libero",
    #         base_config=api.DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #         use_delta_joint_actions=False,
    #         states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
    #         actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
    #         remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
    #         episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

    #     ),
    #     weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
    #         npz_path="gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz", # Ti/16
    #     ),
    #     num_train_steps=20_000,
    #     freeze_filter=deprecated_pi0light_incontextv12.Pi0LightIncontextConfigv12(
    #         prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
    #         sample_frames=2, sample_actions=32, random_select=True, 
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    #     num_workers=16,
    #     # num_workers=1,
    #     batch_size=32,
    #     # wandb_enabled=False,
    # ),
    # api.TrainConfig(
    #     # no delta with split
    #     name="pi0light_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
    #     model=deprecated_pi0light_incontextv12.Pi0LightIncontextConfigv12(
    #         prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
    #         sample_frames=2, sample_actions=32, random_select=True, siglip_variant="Ti/16"
    #     ),  
    #     data=LeRobotLiberoIncontextDataConfig(
    #         repo_id="physical-intelligence/libero",
    #         base_config=api.DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #         use_delta_joint_actions=False,
    #         states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
    #         actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
    #         # remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
    #         # episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

    #     ),
    #     weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
    #         npz_path="gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz", # Ti/16
    #     ),
    #     num_train_steps=20_000,
    #     freeze_filter=deprecated_pi0light_incontextv12.Pi0LightIncontextConfigv12(
    #         prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
    #         sample_frames=2, sample_actions=32, random_select=True, 
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    #     num_workers=16,
    #     # num_workers=1,
    #     batch_size=32,
    #     # wandb_enabled=False,
    # ),
    
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
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
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
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    ######################
    
    api.TrainConfig(
        # no delta with split
        name="debug_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split",
        model=api.deprecated_pi0light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, siglip_variant="Ti/16"
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
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz", # Ti/16
        ),
        num_train_steps=5_000,
        freeze_filter=api.deprecated_pi0light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    

    api.TrainConfig(
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini.py debug_pi0mini_libero_low_mem_finetune_split_train --exp-name=debug_pi0mini_libero_low_mem_finetune_split_train_ibex --overwrite
        # this exp use customized paligemma and different pre-trained img encoder 
        # which has train_test_split; without delta; language prompt; lora; 20k
        name="debug_pi0mini_libero_low_mem_finetune_split_train",
        model=api.pi0Light.Pi0LightConfig(paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0Light.Pi0LightConfig(
            paligemma_variant="gemma_132m", action_expert_variant="gemma_66m", freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=36,
    ),
    
    # XJ_REBUTAL
    ## XJ: debug pi0mini incontext code on libero
    api.TrainConfig(
        name="pi0mini_incontext_libero_low_mem_finetune_inference",
        model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
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
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
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
        name="pi0mini_incontext_libero_low_mem_finetune_train",
        model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
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
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        # batch_size=32,
        num_workers=2,
        batch_size=2,
    ),

    api.TrainConfig(
        name="pi0mini_incontext_libero_low_mem_finetune_train_debug_baseline",
        model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=False,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
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
            libero_input_refactor=True,
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=False, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        # batch_size=32,
        num_workers=2,
        batch_size=2,
    ),
    # TODO: need to set random_select as True for non debug configs
    api.TrainConfig(
        name="pi0mini_incontext_libero_custom_dataset_debug",
        model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=CustomLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
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
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",  # S/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=api._optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6),
        num_train_steps=20_000,  # Reduced for debugging
        freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=2,  # Reduced for debugging
        batch_size=2,  # Reduced for debugging
        use_custom_dataloader=True, # TODO: refactor this later
    ),


    api.TrainConfig(
        name="pi0mini_incontext_libero_custom_dataset_v2_debug",
        model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16"),
        data=Customv2LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            frame_sequence_length=6,
            sample_frames=2,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",  # S/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=api._optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6),
        num_train_steps=20_000,  # Reduced for debugging
        freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=2,  # Reduced for debugging
        batch_size=2,  # Reduced for debugging
        use_custom_dataloader=True, # TODO: refactor this later
    ),

    api.TrainConfig(
        name="pi0_incontextv17_libero_custom_dataset_v2",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",  # NEW: 3rd expert for future state prediction
            action_expert_variant="gemma_300m_lora",
            future_state_downsample=5,  # Downsample factor (horizon computed automatically)
            state_loss_weight=0.5,
            sample_frames=2,
            sample_actions=32,
            random_select=True,
        ),
        data=Customv2FutureStatesLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            frame_sequence_length=1,  # Required for future_states
            sample_frames=2,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
            use_future_states=True,  # Enabled by default in this config
            future_state_downsample=5,  # Must match model's future_state_downsample
            multiple_current_frames=False,  # Must be False for future_states
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontextV17("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,  # Reduced for debugging
        batch_size=32,  # Reduced for debugging
        use_custom_dataloader=True,
    ),

    api.TrainConfig(
        name="pi0_incontextv17_libero_custom_dataset_v2_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",  # NEW: 3rd expert for future state prediction
            action_expert_variant="gemma_300m_lora",
            future_state_downsample=5,  # Downsample factor (horizon computed automatically)
            state_loss_weight=0.5,
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
            libero_input_refactor=True,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontextV17("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,  # Reduced for debugging
        batch_size=32,  # Reduced for debugging
        use_custom_dataloader=True,
    ),


    api.TrainConfig(
        name="pi0_incontextv17_libero_custom_dataset_v2_seq_mask",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",  # NEW: 3rd expert for future state prediction
            action_expert_variant="gemma_300m_lora",
            future_state_downsample=5,  # Downsample factor (horizon computed automatically)
            state_loss_weight=0.5,
            sample_frames=2,
            sample_actions=32,
            random_select=True,
            future_states_seq_mask_prob=0.5,
        ),
        data=Customv2FutureStatesLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            frame_sequence_length=1,  # Required for future_states
            sample_frames=2,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
            use_future_states=True,  # Enabled by default in this config
            future_state_downsample=5,  # Must match model's future_state_downsample
            multiple_current_frames=False,  # Must be False for future_states
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontextV17("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
            future_states_seq_mask_prob=0.5,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=20,  # Reduced for debugging
        batch_size=32,  # Reduced for debugging
        use_custom_dataloader=True,
    ),

    api.TrainConfig(
        name="pi0_incontextv17_libero_custom_dataset_v2_seq_mask_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",  # NEW: 3rd expert for future state prediction
            action_expert_variant="gemma_300m_lora",
            future_state_downsample=5,  # Downsample factor (horizon computed automatically)
            state_loss_weight=0.5,
            sample_frames=2,
            sample_actions=32,
            random_select=True,
            future_states_seq_mask_prob=0.5,
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
            libero_input_refactor=True,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontextV17("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
            future_states_seq_mask_prob=0.5,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=2,  # Reduced for debugging
        batch_size=32,  # Reduced for debugging
        use_custom_dataloader=True,
    ),

    api.TrainConfig(
        name="pi0_incontextv17_libero_custom_dataset_v2_seq_frame_mask",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",  # NEW: 3rd expert for future state prediction
            action_expert_variant="gemma_300m_lora",
            future_state_downsample=5,  # Downsample factor (horizon computed automatically)
            state_loss_weight=0.5,
            sample_frames=2,
            sample_actions=32,
            random_select=True,
            future_states_seq_mask_prob=0.5,
            future_states_frame_mask_prob=0.2,
            future_states_mask_noise_scale=0.5
        ),
        data=Customv2FutureStatesLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            frame_sequence_length=1,  # Required for future_states
            sample_frames=2,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
            use_future_states=True,  # Enabled by default in this config
            future_state_downsample=5,  # Must match model's future_state_downsample
            multiple_current_frames=False,  # Must be False for future_states
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontextV17("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
            future_states_seq_mask_prob=0.5,
            future_states_frame_mask_prob=0.2,
            future_states_mask_noise_scale=0.5
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=20,  # Reduced for debugging
        batch_size=32,  # Reduced for debugging
        use_custom_dataloader=True,
    ),

    api.TrainConfig(
        name="pi0_incontextv17_libero_custom_dataset_v2_seq_frame_mask_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",  # NEW: 3rd expert for future state prediction
            action_expert_variant="gemma_300m_lora",
            future_state_downsample=5,  # Downsample factor (horizon computed automatically)
            state_loss_weight=0.5,
            sample_frames=2,
            sample_actions=32,
            random_select=True,
            future_states_seq_mask_prob=0.5,
            future_states_frame_mask_prob=0.2,
            future_states_mask_noise_scale=0.5
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
            libero_input_refactor=True,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontextV17("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
            future_states_seq_mask_prob=0.5,
            future_states_frame_mask_prob=0.2,
            future_states_mask_noise_scale=0.5
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=2,  # Reduced for debugging
        batch_size=32,  # Reduced for debugging
        use_custom_dataloader=True,
    ),

    api.TrainConfig(
        name="pi0_incontextv17_libero_custom_dataset_v2_seq_frame_mask_debug",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",  # NEW: 3rd expert for future state prediction
            action_expert_variant="gemma_300m_lora",
            future_state_downsample=5,  # Downsample factor (horizon computed automatically)
            state_loss_weight=0.0,
            sample_frames=2,
            sample_actions=32,
            random_select=True,
            future_states_seq_mask_prob=1.0,
            future_states_frame_mask_prob=0.2,
            future_states_mask_noise_scale=0.5
        ),
        data=Customv2FutureStatesLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            frame_sequence_length=1,  # Required for future_states
            sample_frames=2,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
            use_future_states=True,  # Enabled by default in this config
            future_state_downsample=5,  # Must match model's future_state_downsample
            multiple_current_frames=False,  # Must be False for future_states
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontextV17("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv17.Pi0IncontextConfigv17(
            prompt_expert_variant="gemma_300m_v2",
            state_expert_variant="gemma_100m",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
            future_states_seq_mask_prob=0.5,
            future_states_frame_mask_prob=0.2,
            future_states_mask_noise_scale=0.5
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=2,  # Reduced for debugging
        batch_size=32,  # Reduced for debugging
        use_custom_dataloader=True,
    ),


    api.TrainConfig(
        name="vitb_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=6,
            avg_current_img=True,
            ),
        data=Customv2LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            frame_sequence_length=6,
            sample_frames=2,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=6,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        use_custom_dataloader=True,
    ), 
    api.TrainConfig(
        name="vitb_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=6,
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
            libero_input_refactor=True,
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=6,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        use_custom_dataloader=True,
    ),

    api.TrainConfig(
        name="vitb_16_sample_frames_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=16, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=6,
            avg_current_img=True,
            ),
        data=Customv2LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            frame_sequence_length=6,
            sample_frames=16,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=16, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=6,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        use_custom_dataloader=True,
    ), 
    api.TrainConfig(
        name="vitb_16_sample_frames_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=16, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=6,
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
            libero_input_refactor=True,
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=16, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=6,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        use_custom_dataloader=True,
    ),

    api.TrainConfig(
        name="vitb_16_sample_frames_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_avg_false",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=16, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=False,
            ),
        data=Customv2LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            frame_sequence_length=48,
            sample_frames=16,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=16, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=4,
        use_custom_dataloader=True,
    ), 
    api.TrainConfig(
        name="vitb_16_sample_frames_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_avg_false_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=16, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=False,
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
            libero_input_refactor=True,
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=16, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=4,
        use_custom_dataloader=True,
    ),

    api.TrainConfig(
        name="vitb_16_sample_frames_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=16, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=True,
            ),
        data=Customv2LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            frame_sequence_length=48,
            sample_frames=16,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=16, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=4,
        use_custom_dataloader=True,
    ), 
    api.TrainConfig(
        name="vitb_16_sample_frames_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=16, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
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
            libero_input_refactor=True,
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=16, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=4,
        use_custom_dataloader=True,
    ),
    

    api.TrainConfig(
        name="vitb_16_sample_frames_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_160k",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=16, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=True,
            ),
        data=Customv2LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            frame_sequence_length=48,
            sample_frames=16,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 160_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=16, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=20,
        batch_size=4,
        use_custom_dataloader=True,
    ), 
    api.TrainConfig(
        name="vitb_16_sample_frames_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_160k_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=16, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
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
            libero_input_refactor=True,
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 160_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=16, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=4,
        use_custom_dataloader=True,
    ),


    api.TrainConfig(
        name="vitb_16_sample_frames_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_avg_false_80k",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=16, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=False,
            ),
        data=Customv2LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            frame_sequence_length=48,
            sample_frames=16,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 80_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=16, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=20,
        batch_size=4,
        use_custom_dataloader=True,
    ), 
    api.TrainConfig(
        name="vitb_16_sample_frames_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_avg_false_80k_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=16, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=False,
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
            libero_input_refactor=True,
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 80_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=16, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=4,
        use_custom_dataloader=True,
    ),


    api.TrainConfig(
        name="vitb_4_sample_frames_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_avg_false_40k",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=4, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=False,
            ),
        data=Customv2LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            frame_sequence_length=48,
            sample_frames=4,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 40_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=4, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=20,
        batch_size=4,
        use_custom_dataloader=True,
    ), 
    api.TrainConfig(
        name="vitb_4_sample_frames_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_avg_false_40k_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=4, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=False,
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
            libero_input_refactor=True,
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 40_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=4, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=20,
        batch_size=4,
        use_custom_dataloader=True,
    ),

        api.TrainConfig(
        name="vitb_32_sample_frames_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_avg_false_40k",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=32, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=False,
            ),
        data=Customv2LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            frame_sequence_length=48,
            sample_frames=32,
            sample_actions=32,
            task_to_episode_path="metadata/libero/task_to_episode.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            random_select=True,
            current_frame_sample_mode="random",
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 40_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=32, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=20,
        batch_size=4,
        use_custom_dataloader=True,
    ), 
    api.TrainConfig(
        name="vitb_32_sample_frames_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1_dataset_refactor_avg_false_40k_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=32, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=False,
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
            libero_input_refactor=True,
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 40_000,
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=32, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="B/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=20,
        batch_size=4,
        use_custom_dataloader=True,
    ),
        #
    # XJ libero_with_depth: just to pull newly generated libero dataset with depth image but with more episodes
    #
    # api.TrainConfig(
    #     name="pi0_depth_libero_low_mem_finetune",
    #     model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    #     data=LeRobotLiberoDataConfig(
    #         repo_id="daixianjie/libero_with_depth",
    #         base_config=api.DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #     ),
    #     weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     # num_train_steps=30_000,
    #     num_train_steps=40_000,
    #     freeze_filter=api.pi0.Pi0Config(
    #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    #     num_workers=4,
    #     batch_size=36,
    # ),
    
    ## rebuttal exp
    # InSpire Training Setting
    # pi0 incontext v12 none lora
    api.TrainConfig(
        name="pi0_libero90_incontextv12_finetune",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="daixianjie/libero_90_lerobot",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode='metadata/libero_90/task_to_episode.json',
            episode_to_indexes_file='metadata/libero_90/episode_to_indexes.json',
            states_cache_path="metadata/libero_90/episode_states_cache.json",
            actions_cache_path="metadata/libero_90/episode_actions_cache.json",
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 30_000,
            decay_lr= 2.5e-6),
        num_train_steps=30_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=128,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_libero90_incontextv12_finetune_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode='metadata/libero/task_to_episode.json',
            episode_to_indexes_file='metadata/libero/episode_to_indexes.json',
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 30_000,
            decay_lr= 2.5e-6),
        num_train_steps=30_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=128,
        # wandb_enabled=False,
    ),
    # pi0 incontextv12 lora
    api.TrainConfig(
        name="pi0_libero90_incontextv12_low_mem_finetune",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="daixianjie/libero_90_lerobot",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode='metadata/libero_90/task_to_episode.json',
            episode_to_indexes_file='metadata/libero_90/episode_to_indexes.json',
            states_cache_path="metadata/libero_90/episode_states_cache.json",
            actions_cache_path="metadata/libero_90/episode_actions_cache.json",
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 30_000,
            decay_lr= 2.5e-6),
        num_train_steps=30_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_libero90_incontextv12_low_mem_finetune_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode='metadata/libero/task_to_episode.json',
            episode_to_indexes_file='metadata/libero/episode_to_indexes.json',
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 30_000,
            decay_lr= 2.5e-6),
        num_train_steps=30_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),
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
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 30_000,
            decay_lr= 2.5e-6),
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
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 30_000,
            decay_lr= 2.5e-6),
        num_train_steps=30_000,        
        freeze_filter=api.pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ),
    
    # more incontext images
    api.TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_train_split_v1",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=8, sample_actions=32, random_select=True, 
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
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V2,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=8, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_inference",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=8, sample_actions=32, random_select=True, 
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
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps=20_000,        
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=8, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),
    # video tokens
    api.TrainConfig(
        name="pi0_libero_incontextv12_video_prompt_low_mem_finetune_train_split",
        model=api.pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=784,
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
            tracks_path="metadata/libero/video_tokens_merged.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V2,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=784,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_libero_incontextv12_video_prompt_low_mem_finetune_inference",
        model=api.pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=784,
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
            tracks_path="metadata/libero/video_tokens_merged.json",
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=784,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    # point track tokens
    api.TrainConfig(
        name="pi0_libero_incontextv12_point_track_low_mem_finetune_train_split",
        model=api.pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=256,
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
            tracks_path="metadata/libero/episode_tracks_combined_all.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V2,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=256,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_libero_incontextv12_point_track_low_mem_finetune_inference",
        model=api.pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=256,
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
            tracks_path="metadata/libero/episode_tracks_combined_all.json",
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12_dummy.Pi0IncontextConfigv12Dummy(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",  
            sample_frames=2, sample_actions=32, 
            random_select=True, use_image_prompts=False, use_action_state_prompts=False,
            use_point_track_prompts=True, 
            point_track_dim=256,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),
    # totally initialized ICFM (ViT-B-16) deprecated: shouldn't be using lora since it's random
    api.TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_random_init_train_split",
        model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
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
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V2,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.EmptyLoader(),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps=20_000,
        freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    api.TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_random_init_inference",
        model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
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
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.EmptyLoader(),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps=20_000,
        freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    # total init without lora:
    api.TrainConfig(
        name="pi0_libero_incontextv12_random_init_train_split",
        model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
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
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V2,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.EmptyLoader(),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps=20_000,
        freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    api.TrainConfig(
        name="pi0_libero_incontextv12_random_init_inference",
        model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
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
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
            npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
        ),
        weight_loader=api.weight_loaders.EmptyLoader(),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps=20_000,
        freeze_filter=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=False, freeze_img_encoder=False, siglip_variant="B/16"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),
    
    # stage-wise incontext 
    api.TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_clean_stage_wise_prompt_train_all",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoStageIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            keep_episode_filename_list="examples/libero/all_stage_clean.json",
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            all_episode_stage = "examples/libero/all_segments_summary.json",

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_libero_incontextv12_low_mem_finetune_noisy_stage_wise_prompt_train_all",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotLiberoStageIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            keep_episode_filename_list="/home/dingj0b/dingjian/openpi_explore/project/openpi/examples/libero/all_stage_noisy.json",
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            all_episode_stage = "examples/libero/all_segments_summary.json",

        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        # wandb_enabled=False,
    ),
    ]
