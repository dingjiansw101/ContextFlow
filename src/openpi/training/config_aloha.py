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
        libero_input_refactor: bool = False
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
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
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

    # Objects pickup/place configs
    api.TrainConfig(
        name="pi0_aloha_objects_all_incontextv12_low_mem_finetune_sample2_actionssample32_random_select",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
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
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
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
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
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
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
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
        model=api.pi0_incontextv18.Pi0IncontextConfigv18(
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
        freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
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
        model=api.pi0_incontextv18.Pi0IncontextConfigv18(
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
        freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
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
        model=api.pi0_incontextv18.Pi0IncontextConfigv18(
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
        freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
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
        model=api.pi0_incontextv18.Pi0IncontextConfigv18(
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
        freeze_filter=api.pi0_incontextv18.Pi0IncontextConfigv18(
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
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
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
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
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
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
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
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
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

    ]
