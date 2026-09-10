from __future__ import annotations

import dataclasses
import pathlib

LIBERO_UNSEEN_TASKS = [
    # libero_10
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the tomato sauce in the basket",
    # libero_goal
    "put the bowl on the plate",
    "put the bowl on the stove",
    # libero_object
    "pick up the milk and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    # libero_spatial
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
]


def build(api) -> list[api.TrainConfig]:
    # Eval clients import the task split under Python 3.8 without the training dependencies.
    from typing_extensions import override  # noqa: PLC0415

    import openpi.policies.libero_incontext_policy as libero_incontext_policy  # noqa: PLC0415
    import openpi.policies.libero_policy as libero_policy  # noqa: PLC0415
    from openpi.training.dataset_spec import DatasetSpec  # noqa: PLC0415

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
    class CustomLeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
        """Config for in-context learning with CustomLeRobotDataset."""

        use_delta_joint_actions: bool = False

        # CustomLeRobotDataset specific parameters
        sample_frames: int = 2  # Number of frames for in-context demonstration
        sample_actions: int = 32  # Number of actions for in-context demonstration
        policy_local_files_only: bool = False
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

            # CustomLeRobotDataset supplies the demonstration data.
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

        @override
        def create_policy(self, assets_dirs: pathlib.Path, model_config):
            # Pull in-context demos directly from CustomLeRobotDataset.
            # Lazy imports avoid a circular import (config -> data_loader -> config).
            from lerobot.common.datasets import lerobot_dataset as lerobot_dataset_mod

            from openpi.training.custom_dataset import CustomLeRobotDataset

            base_config = self.create_base_config(assets_dirs)
            if self.policy_local_files_only:
                base_config = dataclasses.replace(base_config, local_files_only=True)

            # Construct the demo source as a bare CustomLeRobotDataset (no
            # PromptFromLeRobotTask wrapper — the live env supplies the prompt directly).
            # load_incontext_demonstration is defined on CustomLeRobotDataset itself, so
            # the new transform needs the bare instance, not a TransformedDataset wrapper.
            dataset_meta = lerobot_dataset_mod.LeRobotDatasetMetadata(
                base_config.repo_id, local_files_only=base_config.local_files_only
            )
            custom_dataset = CustomLeRobotDataset(
                base_config.repo_id,
                episodes=None,  # policy spans all episodes
                delta_timestamps={
                    key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
                    for key in base_config.action_sequence_keys
                },
                local_files_only=base_config.local_files_only,
                num_sample_frames=self.sample_frames,
                num_sample_actions=self.sample_actions,
                random_select=self.random_select,
            )

            # Eval-only repack: lists only keys that the WebSocket client sends
            # (examples/libero/main_incontext.py). Listing keys absent from the live env
            # input (e.g. "actions", "frame_index", "dem_prompt_*") would crash
            # RepackTransform.__call__ with a KeyError.
            repack_transform = api._transforms.Group(
                inputs=[
                    api._transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "prompt": "prompt",
                            "task_index": "task_index",
                            "split": "split",
                        }
                    )
                ]
            )

            data_transforms = api._transforms.Group(
                inputs=[
                    api._transforms.InjectDemoFromCustomDataset(dataset=custom_dataset),
                    libero_incontext_policy.CustomLeRobotLiberoIncontextInputs(
                        action_dim=model_config.action_dim,
                        model_type=model_config.model_type,
                    ),
                ],
                outputs=[libero_incontext_policy.LiberoIncontextOutputs()],
            )

            if self.use_delta_joint_actions:
                delta_action_mask = api._transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[api._transforms.DeltaActions(delta_action_mask)],
                    outputs=[api._transforms.AbsoluteActions(delta_action_mask)],
                )

            model_transforms = api.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                base_config,
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                train_episode=None,
            )


    # 2) Return this child's TrainConfig entries directly (can be multiple)
    return [
        api.TrainConfig(
            name="ContextFlow",
            assets_repo_override="ContextFlow",
            model=api.contextflow.ContextFlowConfig(
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
                sample_frames=8,
                sample_actions=128,
                remove_task_list=LIBERO_UNSEEN_TASKS,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                random_select=True,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext(
                "s3://openpi-assets/checkpoints/pi0_base/params"
            ),
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
            # num_workers=1,
            batch_size=32,
            use_custom_dataloader=True,
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
                remove_task_list=LIBERO_UNSEEN_TASKS,
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
            assets_repo_override="pi0_libero_heldout",
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
            name="pi0_libero_heldout",
            model=api.pi0.Pi0Config(),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=True,
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                remove_task_list=LIBERO_UNSEEN_TASKS,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=20_000,
            num_workers=16,
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
            name="pi0_fast_libero_heldout",
            model=api.pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=api.DataConfig(
                    local_files_only=True,
                    prompt_from_task=True,
                ),
                use_delta_joint_actions=False,
                remove_task_list=LIBERO_UNSEEN_TASKS,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            ),
            weight_loader=api.weight_loaders.CheckpointWeightLoader(
                "s3://openpi-assets/checkpoints/pi0_fast_base/params"
            ),
            num_train_steps=20_000,
            num_workers=16,
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
    ]
