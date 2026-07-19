from __future__ import annotations
import dataclasses
import tyro
import pathlib
import json
from collections.abc import Sequence
from typing_extensions import override

import openpi.policies.aloha_mobile_incontext_policy as aloha_incontext_policy
import openpi.policies.aloha_mobile_policy as aloha_mobile_policy
import openpi.policies.aloha_policy as aloha_policy

# Constants for ALOHA objects dataset
ALOHA_OBJECT_EPISODE_JSON = "/home/dingj0b/.cache/huggingface/lerobot/vo2yager/objects_pickup_place/meta/episodes.jsonl"

ALOHA_OBJECT_TEST_TASK = [
    "pick_up_the_onion_and_place_it_in_the_basket_with_left_hand",
    "pick_up_the_pear_and_place_it_in_the_basket_with_left_hand",
    "pick_up_the_bottle_and_place_it_in_the_basket_with_left_hand",
    "pick_up_the_orange_juice_and_place_it_in_the_basket_with_left_hand",
    "pick_up_the_onion_and_place_it_in_the_basket_with_right_hand",
    "pick_up_the_kiwi_and_place_it_in_the_basket_with_left_hand",
    "pick_up_the_kiwi_and_place_it_in_the_basket_with_right_hand",
    "pick_up_the_bottle_and_place_it_in_the_basket_with_right_hand",
    "pick_up_the_gluten_flour_and_place_it_in_the_basket_with_right_hand",
]

ALOHA_DATA_UNIQUE_TEST_TASK = [
    # Explicitly selected test tasks
    "pen_uncap_red_right_b5",
    "pen_uncap_blue_left_b5",
    "put_red_egg_close_box",
    "separate_cups_big_right",
    # All pick-up-and-place tasks with 1 demonstration
    "pick_up_the_gluten_flour_and_place_it_in_the_basket_with_left_hand",
    "pick_up_the_orange_juice_and_place_it_in_the_basket_with_left_hand",
    "pick_up_the_cucumber_and_place_it_in_the_basket_with_left_hand",
    "pick_up_the_kiwi_and_place_it_in_the_basket_with_left_hand",
    "pick_up_the_gluten_flour_and_place_it_in_the_basket_with_right_hand",
    "pick_up_the_pear_and_place_it_in_the_basket_with_left_hand",
    "pick_up_the_apple_and_place_it_in_the_basket_with_right_hand",
    "pick_up_the_onion_and_place_it_in_the_basket_with_left_hand",
    "pick_up_the_bottle_and_place_it_in_the_basket_with_left_hand",
    "pick_up_the_blue_milk_and_place_it_in_the_basket_with_left_hand",
]

def build(api) -> list["api.TrainConfig"]:
    g = globals()
    g["DataConfig"] = getattr(api, "DataConfig")
    g["BaseModelConfig"] = getattr(api._model, "BaseModelConfig")
    g["Group"] = getattr(api, "_transforms").Group
    g["TrainConfig"] = getattr(api, "TrainConfig")
    
    # 1) Define DataConfig subclasses inside this function, inheriting DataConfigFactory via api
    @dataclasses.dataclass(frozen=True)
    class LeRobotAlohaDataConfig(api.DataConfigFactory):
        # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
        # Gripper dimensions will remain in absolute values.
        use_delta_joint_actions: bool = True
        # If provided, will be injected into the input data if the "prompt" key is not present.
        default_prompt: str | None = None
        # If true, this will convert the joint and gripper values from the standard Aloha space to
        # the space used by the pi internal runtime which was used to train the base model. People who
        # use standard Aloha data should set this to true.
        adapt_to_pi: bool = True

        # Repack transforms.
        repack_transforms: tyro.conf.Suppress["Group"] = dataclasses.field(
            default=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {"cam_high": "observation.images.top"},
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            )
        )
        # Action keys that will be used to read the action sequence from the dataset.
        action_sequence_keys: Sequence[str] = ("action",)

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            data_transforms = api._transforms.Group(
                inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
                outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
            )
            if self.use_delta_joint_actions:
                delta_action_mask = api._transforms.make_bool_mask(6, -1, 6, -1)
                data_transforms = data_transforms.push(
                    inputs=[api._transforms.DeltaActions(delta_action_mask)],
                    outputs=[api._transforms.AbsoluteActions(delta_action_mask)],
                )

            model_transforms = api.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=self.repack_transforms,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=self.action_sequence_keys,
                train_episode=api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list),
            )
            
            
    @dataclasses.dataclass(frozen=True)
    class LeRobotAlohaMobileDataConfig(api.DataConfigFactory):
        # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
        # Gripper dimensions will remain in absolute values.
        use_delta_joint_actions: bool = True
        # If provided, will be injected into the input data if the "prompt" key is not present.
        default_prompt: str | None = None
        # If true, this will convert the joint and gripper values from the standard Aloha space to
        # the space used by the pi internal runtime which was used to train the base model. People who
        # use standard Aloha data should set this to true.
        # adapt_to_pi: bool = True
        adapt_to_pi: bool = False

        # Repack transforms.
        repack_transforms: tyro.conf.Suppress["Group"] = dataclasses.field(
            default=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {"cam_high": "observation.images.top"},
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            )
        )
        # Action keys that will be used to read the action sequence from the dataset.
        action_sequence_keys: Sequence[str] = ("action",)

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            # assert model_config.action_dim == 16
            data_transforms = api._transforms.Group(
                inputs=[
                    aloha_mobile_policy.AlohaMobileInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)
                ],
                outputs=[aloha_mobile_policy.AlohaMobileOutputs(adapt_to_pi=self.adapt_to_pi)],
            )
            if self.use_delta_joint_actions:
                # TODO: for base action, is it delta?
                delta_action_mask = api._transforms.make_bool_mask(6, -1, 6, -1, -1, -1)
                data_transforms = data_transforms.push(
                    inputs=[api._transforms.DeltaActions(delta_action_mask)],
                    outputs=[api._transforms.AbsoluteActions(delta_action_mask)],
                )

            model_transforms = api.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=self.repack_transforms,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=self.action_sequence_keys,
                train_episode=api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list),
            )
        
    @dataclasses.dataclass(frozen=True)
    class LeRobotAlohaMobileIncontextDataConfig(api.DataConfigFactory):
        states_cache_path: str = "metadata/aloha_pen_uncap/episode_states_cache.json"
        actions_cache_path: str = "metadata/aloha_pen_uncap/episode_actions_first_cache.json"
        task_to_episode: str = "metadata/aloha_pen_uncap/task_to_episode.json"
        episode_to_indexes_file: str = "metadata/aloha_pen_uncap/episode_to_indexes.json"
        # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
        # Gripper dimensions will remain in absolute values.
        use_delta_joint_actions: bool = True
        # If provided, will be injected into the input data if the "prompt" key is not present.
        # TODO: check the issue of default prompt
        default_prompt: str | None = None
        # If true, this will convert the joint and gripper values from the standard Aloha space to
        # the space used by the pi internal runtime which was used to train the base model. People who
        # use standard Aloha data should set this to true.
        # adapt_to_pi: bool = True
        adapt_to_pi: bool = False
        multi_process: bool = False

        # Padding mode for AddStatesActionsPromptTransform
        # "keep_all": Keep all L frames when L < max_len, then pad with last frame
        # "linspace_repeat": Always use linspace sampling, then repeat last sample if needed
        padding_mode: str = "keep_all"
        # Whether to mask padded frames as valid (True) or invalid (False)
        mask_padding_as_valid: bool = False

        # Repack transforms.
        repack_transforms: tyro.conf.Suppress["Group"] = dataclasses.field(
            default=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                            "episode_index": "episode_index",
                            "index": "index",
                            "task_index": "task_index",
                        }
                    )
                ]
            )
        )
        # Action keys that will be used to read the action sequence from the dataset.
        action_sequence_keys: Sequence[str] = ("action",)

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            # assert model_config.action_dim == 16

            # TODO: generate the indexes for aloha mobile data
            train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)

            data_transforms = api._transforms.Group(
                inputs=[api._transforms.InjectDemoIndexes(
                                                    task_to_episode=self.task_to_episode,
                                                    episode_to_indexes=self.episode_to_indexes_file,
                                                    sample_frames=model_config.sample_frames,
                                                    random_select=model_config.random_select,
                                                    sample_episodes=model_config.sample_episodes,
                                                    train_episode_index_list=train_epi,
                                                    seed_base=self.seed_base)],
                outputs=[],
            )

            data_transforms = data_transforms.push(
                inputs=[
                    aloha_incontext_policy.AlohaMobileIncontextInputs(
                        action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi
                    )
                ],
                outputs=[aloha_mobile_policy.AlohaMobileOutputs(adapt_to_pi=self.adapt_to_pi)],

            )
            # data_transforms = _transforms.Group(
            #     inputs=[
            #         aloha_mobile_policy.AlohaMobileInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)
            #     ],
            #     outputs=[aloha_mobile_policy.AlohaMobileOutputs(adapt_to_pi=self.adapt_to_pi)],
            # )
            if self.use_delta_joint_actions:
                # TODO: for base action, is it delta?
                delta_action_mask = api._transforms.make_bool_mask(6, -1, 6, -1, -1, -1)
                data_transforms = data_transforms.push(
                    inputs=[api._transforms.DeltaActions(delta_action_mask)],
                    outputs=[api._transforms.AbsoluteActions(delta_action_mask)],
                )
            # TODO: change it to support multi-task?
            # model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
            model_transforms = api.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=self.repack_transforms,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=self.action_sequence_keys,
                train_episode=api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list),
            )
        
        

    @dataclasses.dataclass(frozen=True)
    class LeRobotAlohaMobileFASTIncontextDataConfig(api.DataConfigFactory):
        states_cache_path: str = "metadata/aloha_data_unique/episode_states_cache.json"
        actions_cache_path: str = "metadata/aloha_data_unique/episode_actions_cache.json"
        task_to_episode: str = "metadata/aloha_data_unique/task_to_episode.json"
        episode_to_indexes_file: str = "metadata/aloha_data_unique/episode_to_indexes.json"
        use_delta_joint_actions: bool = True
        adapt_to_pi: bool = False
        # Action keys that will be used to read the action sequence from the dataset.
        action_sequence_keys: Sequence[str] = ("action",)

        # Padding mode for AddStatesActionsPromptTransform
        padding_mode: str = "keep_all"
        mask_padding_as_valid: bool = False
        demo_state_dim: int | None = 32

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                            "episode_index": "episode_index",
                            "index": "index",
                            "task_index": "task_index",
                        }
                    )
                ]
            )

            train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)

            data_transforms = api._transforms.Group(
                inputs=[api._transforms.InjectDemoIndexes(
                    task_to_episode=self.task_to_episode,
                    episode_to_indexes=self.episode_to_indexes_file,
                    sample_frames=model_config.sample_frames,
                    random_select=model_config.random_select,
                    sample_episodes=model_config.sample_episodes,
                    train_episode_index_list=train_epi)],
                outputs=[],
            )

            data_transforms = data_transforms.push(
                inputs=[
                    aloha_incontext_policy.AlohaMobileIncontextInputs(
                        action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi
                    )
                ],
                outputs=[aloha_mobile_policy.AlohaMobileOutputs(adapt_to_pi=self.adapt_to_pi)],
            )

            if self.use_delta_joint_actions:
                delta_action_mask = api._transforms.make_bool_mask(6, -1, 6, -1, -1, -1)
                data_transforms = data_transforms.push(
                    inputs=[api._transforms.DeltaActions(delta_action_mask)],
                    outputs=[api._transforms.AbsoluteActions(delta_action_mask)],
                )

            model_transforms = api.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=self.action_sequence_keys,
                train_episode=train_epi,
                demo_state_dim=self.demo_state_dim,
            )

    @dataclasses.dataclass(frozen=True)
    class CustomLeRobotAlohaMobileIncontextDataConfig(api.DataConfigFactory):
        """Cache-free aloha in-context config.

        Demonstrations (frames/states/actions) come directly from CustomLeRobotDataset
        instead of the legacy InjectDemoIndexes + AddImagePromptTransform +
        AddStatesActionsPromptTransform pipeline, which depended on the JSON state/action
        caches under metadata/aloha_data_unique/. Training must route through
        create_custom_incontext_data_loader (TrainConfig.use_custom_dataloader=True).
        """

        use_delta_joint_actions: bool = True
        default_prompt: str | None = None
        adapt_to_pi: bool = False

        # CustomLeRobotDataset parameters; sample_frames / sample_actions must match the
        # model config (create_custom_dataset reads them from this factory, not the model).
        sample_frames: int = 8
        sample_actions: int = 128
        task_to_episode_path: str = "metadata/aloha_data_unique/task_to_episode.json"
        random_select: bool = True
        policy_local_files_only: bool = True

        # Aloha LeRobot column layout (differs from the LIBERO defaults in CustomLeRobotDataset).
        state_key: str = "observation.state"
        actions_key: str = "action"
        demo_image_keys: dict[str, str] = dataclasses.field(
            default_factory=lambda: {
                "cam_high": "observation.images.cam_high",
                "cam_left_wrist": "observation.images.cam_left_wrist",
                "cam_right_wrist": "observation.images.cam_right_wrist",
            }
        )
        # Reproduce InjectDemoIndexes' seeded demo selection whenever seed_base is set, so
        # deterministic consistency checks line up with the legacy cache-based loader.
        demo_selection_seed_compat: bool = True

        # Demo states/actions are raw dataset values, normalized with this config's own
        # state/action stats: the default aliases point the demo fields at "state" /
        # "actions", so demos are normalized exactly like the current frame. (The legacy
        # caches stored post-Normalize values, so this reproduces that normalization.)
        norm_stats_aliases: dict[str, str] | None = dataclasses.field(
            default_factory=lambda: {
                "dem_prompt_all_states": "state",
                "dem_prompt_all_actions": "actions",
            }
        )

        action_sequence_keys: Sequence[str] = ("action",)

        def _delta_action_mask(self):
            if not self.use_delta_joint_actions:
                return None
            return api._transforms.make_bool_mask(6, -1, 6, -1, -1, -1)

        def _data_transforms(self, model_config) -> "Group":
            mask = self._delta_action_mask()
            group = api._transforms.Group(
                inputs=[
                    aloha_incontext_policy.CustomLeRobotAlohaMobileIncontextInputs(
                        action_dim=model_config.action_dim,
                        adapt_to_pi=self.adapt_to_pi,
                        delta_action_mask=tuple(mask) if mask is not None else None,
                    )
                ],
                outputs=[aloha_mobile_policy.AlohaMobileOutputs(adapt_to_pi=self.adapt_to_pi)],
            )
            if mask is not None:
                group = group.push(
                    inputs=[api._transforms.DeltaActions(mask)],
                    outputs=[api._transforms.AbsoluteActions(mask)],
                )
            return group

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                            "episode_index": "episode_index",
                            "index": "index",
                            "task_index": "task_index",
                            # dem_prompt_* keys emitted by CustomLeRobotDataset
                            # (dem_prompt_images is nested, so map the flattened keys).
                            "dem_prompt_images": {
                                "cam_high": "dem_prompt_images/cam_high",
                                "cam_left_wrist": "dem_prompt_images/cam_left_wrist",
                                "cam_right_wrist": "dem_prompt_images/cam_right_wrist",
                            },
                            "dem_prompt_states": "dem_prompt_states",
                            "dem_prompt_actions": "dem_prompt_actions",
                            "selected_episode": "selected_episode",
                        }
                    )
                ]
            )

            train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)
            model_transforms = api.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=self._data_transforms(model_config),
                model_transforms=model_transforms,
                action_sequence_keys=tuple(self.action_sequence_keys),
                train_episode=train_epi,
            )

        @override
        def create_policy(self, assets_dirs: pathlib.Path, model_config):
            # Cache-free eval path: pull in-context demos directly from CustomLeRobotDataset
            # instead of the older cache-based transforms. Lazy imports avoid a circular
            # import (config -> data_loader -> config).
            from lerobot.common.datasets import lerobot_dataset as lerobot_dataset_mod

            from openpi.training.custom_dataset import CustomLeRobotDataset

            base_config = self.create_base_config(assets_dirs)
            if self.policy_local_files_only:
                base_config = dataclasses.replace(base_config, local_files_only=True)

            dataset_meta = lerobot_dataset_mod.LeRobotDatasetMetadata(
                base_config.repo_id, local_files_only=base_config.local_files_only
            )
            custom_dataset = CustomLeRobotDataset(
                base_config.repo_id,
                episodes=None,  # policy spans all episodes
                delta_timestamps={
                    key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
                    for key in self.action_sequence_keys
                },
                local_files_only=base_config.local_files_only,
                num_sample_frames=self.sample_frames,
                num_sample_actions=self.sample_actions,
                task_to_episode_path=self.task_to_episode_path,
                random_select=self.random_select,
                seed_base=self.seed_base,
                state_key=self.state_key,
                actions_key=self.actions_key,
                demo_image_keys=dict(self.demo_image_keys),
                demo_selection_seed_compat=self.demo_selection_seed_compat,
            )

            data_transforms = self._data_transforms(model_config)
            data_transforms = api._transforms.Group(
                inputs=[
                    api._transforms.InjectDemoFromCustomDataset(dataset=custom_dataset),
                    *data_transforms.inputs,
                ],
                outputs=tuple(data_transforms.outputs),
            )

            model_transforms = api.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                base_config,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=tuple(self.action_sequence_keys),
                train_episode=None,
                provides_incontext_demos=True,
            )

    # 2) Return this child's api.TrainConfig entries directly (can be multiple)
    return [
        #
    # In`fe`rence Aloha configs.
    #
    api.TrainConfig(
        name="pi0_aloha",
        model=api.pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=api.AssetsConfig(asset_id="trossen"),
        ),
    ),
    api.TrainConfig(
        name="pi0_aloha_mobile",
        model=api.pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="mobile_trossen",
            ),
        ),
    ),
    api.TrainConfig(
        name="pi0_aloha_towel",
        model=api.pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=api.AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
    ),
    api.TrainConfig(
        name="pi0_aloha_tupperware",
        model=api.pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=api.AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
    ),
    api.TrainConfig(
        name="pi0_aloha_pen_uncap_b5_low_mem_finetune",
        model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            default_prompt="uncap the pen",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=10_000,
        freeze_filter=api.pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    api.TrainConfig(
        name="pi0_aloha_pen_uncap_b5_low_mem_finetune_trossen_norm",
        model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_fast_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="uncap the pen",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=10_000,
        freeze_filter=api.pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    api.TrainConfig(
        name="pi0_aloha_pen_uncap_b5_low_mem_finetune_trossen_normv2",
        model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="uncap the pen",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    api.TrainConfig(
        name="pi0_aloha_pen_uncap_b5_trossen_norm",
        model=api.pi0.Pi0Config(),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="uncap the pen",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_workers=16,
        num_train_steps=20_000,
    ),

    # TODO: check the prompt
    api.TrainConfig(
        name="pi0_aloha_pen_uncap_incontextv12_low_mem_finetune_sample2_actionssample32_random_select",
        model=api.contextflow_plain.ContextFlowPlainConfig(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ),
        data=LeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            # default_prompt="uncap the pen",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/aloha_pen_uncap/episode_states_cache.json",
            actions_cache_path="metadata/aloha_pen_uncap/episode_actions_cache.json",
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.contextflow_plain.ContextFlowPlainConfig(
            prompt_expert_variant="gemma_300m_v2", action_expert_variant="gemma_300m_lora", 
            sample_frames=2, sample_actions=32, random_select=True, 
        ).get_freeze_filter(),
        ema_decay=None,
        # num_workers=16,
        num_workers=1,
        batch_size=32,
        # wandb_enabled=False,
    ),

    # Objects pickup/place configs
    api.TrainConfig(
        name="pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select",
        model=api.contextflow_plain.ContextFlowPlainConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
        ),
        data=LeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/objects_pickup_place",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="pick up object and place in basket",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode="metadata/objects_pickup_place/task_to_episode.json",
            episode_to_indexes_file="metadata/objects_pickup_place/episode_to_indexes.json",
            states_cache_path="metadata/objects_pickup_place/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/objects_pickup_place/episode_actions_without_delta_cache.json",
            remove_task_list=ALOHA_OBJECT_TEST_TASK,
            episode_json_path=ALOHA_OBJECT_EPISODE_JSON,
            multi_process=False,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.contextflow_plain.ContextFlowPlainConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=2,
        batch_size=32,
    ),

    # Inference variant (no test task filtering)
    api.TrainConfig(
        name="pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_inference",
        model=api.contextflow_plain.ContextFlowPlainConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
        ),
        data=LeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/objects_pickup_place",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="pick up object and place in basket",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode="metadata/objects_pickup_place/task_to_episode.json",
            episode_to_indexes_file="metadata/objects_pickup_place/episode_to_indexes.json",
            states_cache_path="metadata/objects_pickup_place/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/objects_pickup_place/episode_actions_without_delta_cache.json",
            multi_process=False,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.contextflow_plain.ContextFlowPlainConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),

    api.TrainConfig(
        name="pi0_aloha_objects_all_incontextv18_low_mem_finetune_sample_frames8",
        model=api.contextflow.ContextFlowConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=8,
            sample_actions=128,
            random_select=True,
        ),
        data=LeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/objects_pickup_place",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="pick up object and place in basket",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode="metadata/objects_pickup_place/task_to_episode.json",
            episode_to_indexes_file="metadata/objects_pickup_place/episode_to_indexes.json",
            states_cache_path="metadata/objects_pickup_place/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/objects_pickup_place/episode_actions_without_delta_cache.json",
            remove_task_list=ALOHA_OBJECT_TEST_TASK,
            episode_json_path=ALOHA_OBJECT_EPISODE_JSON,
            multi_process=False,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.contextflow.ContextFlowConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=8,
            sample_actions=128,
            random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=2,
        batch_size=32,
    ),

    # Inference variant (no test task filtering)
    api.TrainConfig(
        name="pi0_aloha_objects_all_incontextv18_low_mem_finetune_sample_frames8_inference",
        model=api.contextflow.ContextFlowConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=8,
            sample_actions=128,
            random_select=True,
        ),
        data=LeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/objects_pickup_place",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="pick up object and place in basket",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode="metadata/objects_pickup_place/task_to_episode.json",
            episode_to_indexes_file="metadata/objects_pickup_place/episode_to_indexes.json",
            states_cache_path="metadata/objects_pickup_place/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/objects_pickup_place/episode_actions_without_delta_cache.json",
            multi_process=False,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.contextflow.ContextFlowConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=8,
            sample_actions=128,
            random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),

    api.TrainConfig(
        name="pi0_aloha_objects_task_suite_incontextv18_low_mem_finetune_sample_frames8",
        model=api.contextflow.ContextFlowConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=8,
            sample_actions=128,
            random_select=True,
        ),
        data=LeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/object_task_suite",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode="metadata/object_task_suite/task_to_episode.json",
            episode_to_indexes_file="metadata/object_task_suite/episode_to_indexes.json",
            states_cache_path="metadata/object_task_suite/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/object_task_suite/episode_actions_without_delta_cache.json",
            multi_process=False,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.contextflow.ContextFlowConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=8,
            sample_actions=128,
            random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=64,
        batch_size=32,
    ),

    # Inference variant (no test task filtering)
    # TODO: need to set a dataset of unseen tasks in test config
    api.TrainConfig(
        name="pi0_aloha_objects_task_suite_incontextv18_low_mem_finetune_sample_frames8_inference",
        model=api.contextflow.ContextFlowConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=8,
            sample_actions=128,
            random_select=True,
        ),
        data=LeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/demonstrations",
            # repo_id="vo2yager/object_task_suite",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode="metadata/demonstrations/task_to_episode.json",
            episode_to_indexes_file="metadata/demonstrations/episode_to_indexes.json",
            states_cache_path="metadata/demonstrations/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/demonstrations/episode_actions_without_delta_cache.json",
            multi_process=False,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.contextflow.ContextFlowConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=8,
            sample_actions=128,
            random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),

    # 40k variant
    api.TrainConfig(
        name="pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_40k",
        model=api.contextflow_plain.ContextFlowPlainConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
        ),
        data=LeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/objects_pickup_place",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="pick up object and place in basket",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode="metadata/objects_pickup_place/task_to_episode.json",
            episode_to_indexes_file="metadata/objects_pickup_place/episode_to_indexes.json",
            states_cache_path="metadata/objects_pickup_place/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/objects_pickup_place/episode_actions_without_delta_cache.json",
            remove_task_list=ALOHA_OBJECT_TEST_TASK,
            episode_json_path=ALOHA_OBJECT_EPISODE_JSON,
            multi_process=False,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=api.contextflow_plain.ContextFlowPlainConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),

    # 80k variant
    api.TrainConfig(
        name="pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_80k",
        model=api.contextflow_plain.ContextFlowPlainConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
        ),
        data=LeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/objects_pickup_place",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="pick up object and place in basket",
            base_config=api.DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            task_to_episode="metadata/objects_pickup_place/task_to_episode.json",
            episode_to_indexes_file="metadata/objects_pickup_place/episode_to_indexes.json",
            states_cache_path="metadata/objects_pickup_place/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/objects_pickup_place/episode_actions_without_delta_cache.json",
            remove_task_list=ALOHA_OBJECT_TEST_TASK,
            episode_json_path=ALOHA_OBJECT_EPISODE_JSON,
            multi_process=False,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=80_000,
        freeze_filter=api.contextflow_plain.ContextFlowPlainConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=2,
            sample_actions=32,
            random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),

    api.TrainConfig(
        name="pi0_fast_aloha_pen_uncap_b5",
        model=api.pi0_fast.Pi0FASTConfig(action_dim=16, action_horizon=50, max_token_len=576),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            default_prompt="uncap the pen",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=10_000,
        wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_fast_aloha_pen_uncap_b5_trossen_norm",
        model=api.pi0_fast.Pi0FASTConfig(action_horizon=50, max_token_len=576),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_fast_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="uncap the pen",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=10_000,
        wandb_enabled=False,
    ),
    api.TrainConfig(
        name="pi0_fast_aloha_pen_uncap_low_mem_finetune_trossen_norm",
        model=api.pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora", action_horizon=50, max_token_len=576),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_fast_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="uncap the pen",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=10_000,
        freeze_filter=api.pi0_fast.Pi0FASTConfig(
            action_dim=16, action_horizon=50, max_token_len=448, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    api.TrainConfig(
        name="pi0_fast_aloha_pen_uncap_low_mem_finetune",
        model=api.pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora", action_horizon=50, max_token_len=576),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            default_prompt="uncap the pen",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=10_000,
        freeze_filter=api.pi0_fast.Pi0FASTConfig(
            action_dim=16, action_horizon=50, max_token_len=448, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    api.TrainConfig(
        name="pi0_fast_aloha_pen_uncap_low_mem_finetune_bs30",
        batch_size=30,
        wandb_enabled=False,
        model=api.pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora", max_token_len=300),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/pen_uncap_b5",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_fast_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="uncap the pen",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0_fast.Pi0FASTConfig(
            action_dim=16, action_horizon=40, max_token_len=400, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
        #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    api.TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=api.pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    # This config is used to demonstrate how to train on a simple simulated environment.
    api.TrainConfig(
        name="pi0_aloha_sim",
        model=api.pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),

    # Transferred from /home/dingj0b/code/openpi/src/openpi/training/config.py
    api.TrainConfig(
        name="pi0_aloha_objects_all_pickup_place_incontext_low_mem_finetune_split_train",
        model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/objects_pickup_place",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            # default_prompt="uncap the pen",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            # remove_task_list=DEFAULT_LIBERO_TEST_TASK,
            # episode_json_path=DEFAULT_LIBERO_EPISODE_JSON,
            remove_task_list=ALOHA_OBJECT_TEST_TASK,
            episode_json_path=ALOHA_OBJECT_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),

    api.TrainConfig(
        name="pi0_aloha_objects_all_pickup_place_incontext_low_mem_finetune_split_train_inference",
        model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/objects_pickup_place",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            # default_prompt="uncap the pen",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),


    api.TrainConfig(
        name="pi0_aloha_objects_task_suite_low_mem_finetune_split_train",
        assets_repo_override="pi0_aloha_objects_task_suite_incontextv18_low_mem_finetune_sample_frames8",
        model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/object_task_suite",
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
    ),

    #
    # aloha_data_unique in-context (v18, gemma_300m_v2) -- ported from the `aloha` branch.
    # Training excludes the held-out test tasks (ALOHA_DATA_UNIQUE_TEST_TASK); the
    # _inference variant does no task filtering (used for eval/serving).
    #
    # aloha_data_unique: test tasks excluded, delta joint actions
    api.TrainConfig(
        # Paper method name (ALOHA variant of ContextFlow). The assets key matches the
        # config name; the on-disk ./assets/<key> dir must be renamed to match (see
        # CONFIG_NAME_MAPPING.md). assets_repo_override is kept explicit so the inference
        # sibling can share this key.
        name="ContextFlow_Aloha",
        assets_repo_override="ContextFlow_Aloha",
        model=api.contextflow.ContextFlowConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=8,
            sample_actions=128,
            random_select=True,
        ),
        data=CustomLeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/aloha_data_unique",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="perform the task",
            base_config=api.DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=True,
            sample_frames=8,
            sample_actions=128,
            episode_json_path="/home/dingj0b/.cache/huggingface/lerobot/vo2yager/aloha_data_unique/meta/episodes.jsonl",
            remove_task_list=ALOHA_DATA_UNIQUE_TEST_TASK,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.contextflow.ContextFlowConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=8,
            sample_actions=128,
            random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=80,
        batch_size=32,
        # Route training through CustomLeRobotDataset (cache-free demo loading).
        use_custom_dataloader=True,
    ),
    # aloha_data_unique inference variant (no task filtering)
    api.TrainConfig(
        # Paper method name (ALOHA variant of ContextFlow, inference). Assets key matches
        # the config name; rename the on-disk ./assets/<key> dir to match.
        name="ContextFlow_Aloha_Inference",
        assets_repo_override="ContextFlow_Aloha_Inference",
        model=api.contextflow.ContextFlowConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=8,
            sample_actions=128,
            random_select=True,
        ),
        data=CustomLeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/aloha_data_unique",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            default_prompt="perform the task",
            base_config=api.DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=True,
            sample_frames=8,
            sample_actions=128,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.contextflow.ContextFlowConfig(
            prompt_expert_variant="gemma_300m_v2",
            action_expert_variant="gemma_300m_lora",
            sample_frames=8,
            sample_actions=128,
            random_select=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=16,
        batch_size=32,
        # Route training through CustomLeRobotDataset (cache-free demo loading).
        use_custom_dataloader=True,
    ),


    ####### ported from origin/aloha-dev: aloha_data_unique baselines + ContextAR #######

    # aloha_data_unique pi0: test tasks excluded
    api.TrainConfig(
        name="Pi0_Aloha",
        model=api.pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaMobileDataConfig(
            repo_id="vo2yager/aloha_data_unique",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen_mobile",
            ),
            repack_transforms=api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=api.DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
            episode_json_path="/home/dingj0b/.cache/huggingface/lerobot/vo2yager/aloha_data_unique/meta/episodes.jsonl",
            remove_task_list=ALOHA_DATA_UNIQUE_TEST_TASK,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        freeze_filter=api.pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=80,
        batch_size=32,
    ),

    # Training config with test tasks excluded
    api.TrainConfig(
        # Paper method name (ALOHA variant of ContextAR). Assets key matches the config
        # name; the inference sibling below shares this same key. Rename the on-disk
        # ./assets/<key> dir to match (see CONFIG_NAME_MAPPING.md).
        name="ContextAR_Aloha",
        assets_repo_override="ContextAR_Aloha",
        model=api.contextar.ContextARConfig(
            action_dim=32, action_horizon=10, max_token_len=256,
            sample_frames=2, sample_actions=4, random_select=True,
        ),
        data=CustomLeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/aloha_data_unique",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_fast_base/assets",
                asset_id="trossen_mobile",
            ),
            base_config=api.DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=True,
            sample_frames=2,
            sample_actions=4,
            remove_task_list=ALOHA_DATA_UNIQUE_TEST_TASK,
            episode_json_path="/home/dingj0b/.cache/huggingface/lerobot/vo2yager/aloha_data_unique/meta/episodes.jsonl",
            # Demo states/actions use this config's own pi0_fast_base trossen_mobile stats
            # (factory default aliases dem_prompt_all_* -> state/actions), i.e. demos are
            # normalized identically to the current frame.
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        ema_decay=None,
        num_workers=32,
        batch_size=4,
        # Route training through CustomLeRobotDataset (cache-free demo loading).
        use_custom_dataloader=True,
    ),

    # Inference config (no test task filtering)
    api.TrainConfig(
        # Paper method name (ALOHA variant of ContextAR, inference). Shares the train
        # config's assets key (ContextAR_Aloha), matching the pre-rename behavior.
        name="ContextAR_Aloha_Inference",
        assets_repo_override="ContextAR_Aloha",
        model=api.contextar.ContextARConfig(
            action_dim=32, action_horizon=10, max_token_len=256,
            sample_frames=2, sample_actions=4, random_select=True,
        ),
        data=CustomLeRobotAlohaMobileIncontextDataConfig(
            repo_id="vo2yager/aloha_data_unique",
            assets=api.AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_fast_base/assets",
                asset_id="trossen_mobile",
            ),
            base_config=api.DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
            use_delta_joint_actions=True,
            sample_frames=2,
            sample_actions=4,
            # Demo states/actions use this config's own pi0_fast_base stats (factory
            # default aliases), i.e. normalized identically to the current frame.
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        ema_decay=None,
        num_workers=8,
        batch_size=4,
        # Route training through CustomLeRobotDataset (cache-free demo loading).
        use_custom_dataloader=True,
    ),

    ]
