"""Sequence-debug LIBERO in-context training configs."""

# ruff: noqa: SLF001

from __future__ import annotations

import dataclasses
import pathlib

from typing_extensions import override

import openpi.policies.libero_incontext_policy as libero_incontext_policy
from openpi.training.dataset_spec import DatasetSpec


def _libero90_episode_json_path() -> str:
    return str(pathlib.Path("~/.cache/huggingface/lerobot/vo2yager/libero_90/meta/episodes.jsonl").expanduser())


def build(api):
    g = globals()
    g["DataConfig"] = api.DataConfig
    g["BaseModelConfig"] = api._model.BaseModelConfig

    @dataclasses.dataclass(frozen=True)
    class SequenceDebugLeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
        """FAST in-context LIBERO config backed by CustomLeRobotDataset.

        Both training (``create``) and evaluation (``create_policy``) source the
        in-context demonstration directly from ``CustomLeRobotDataset`` /
        ``InjectDemoFromCustomDataset``, mirroring the proven
        ``CustomLeRobotLiberoIncontextDataConfig`` (config_libero.py). Because the
        dataset spans all episodes (``episodes=None``), demos can be drawn for
        held-out (unseen) tasks too — unlike the older ``InjectDemoIndexes`` path,
        which was filtered to the train split and crashed on unseen tasks.
        """

        use_delta_joint_actions: bool = False
        # CustomLeRobotDataset in-context parameters (read by create_custom_dataset
        # via getattr on this factory, and by create_policy below).
        sample_frames: int = 2  # Number of frames for the in-context demonstration.
        sample_actions: int = 32  # Number of demo states/actions for the in-context prompt.
        task_to_episode_path: str = "metadata/libero/task_to_episode.json"
        random_select: bool = True  # If True, randomly select demo episodes; else deterministic.
        policy_local_files_only: bool = False
        # Demo tensors ARE normalized, matching the pre-migration pipeline: the old
        # JSON caches (episode_states/actions_without_delta_cache.json) were built
        # from transform_dataset(skip_norm_stats=False), i.e. post-Normalize, so the
        # FAST in-context model was trained on z-scored demo tensors with zero pads
        # (e.g. cached action dim3 = 8.465 = (0 + 2.9716) / 0.3511). Demo actions are
        # padded to demo_action_dim=32 before Normalize while the "actions" stats are
        # 7-dim, so the alias is identity-padded to 32 (pads stay zero).
        norm_stats_aliases: dict[str, str] | None = dataclasses.field(
            default_factory=lambda: {
                "dem_prompt_all_states": "state",
                "dem_prompt_all_actions": "actions",
            }
        )
        norm_stats_alias_pad_dims: dict[str, int] | None = dataclasses.field(
            default_factory=lambda: {"dem_prompt_all_actions": 32}
        )
        # Retained for backward compatibility with make_data kwargs; unused by the
        # CustomLeRobotDataset path (kept so existing call sites do not break).
        states_cache_path: str = "metadata/libero/episode_states_cache.json"
        actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
        task_to_episode: str = "metadata/libero/task_to_episode.json"
        episode_to_indexes_file: str = "metadata/libero/episode_to_indexes.json"
        padding_mode: str = "keep_all"
        mask_padding_as_valid: bool = False
        demo_state_dim: int | None = 32

        @override
        def create(self, assets_dirs: pathlib.Path, model_config):
            # Pass through dem_prompt_* keys produced by CustomLeRobotDataset.
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
                            # dem_prompt_images is nested, so map the flattened keys.
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

            train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)
            # CustomLeRobotDataset handles demo loading internally, so we do NOT use
            # InjectDemoIndexes / AddImagePromptTransform / AddStatesActionsPromptTransform.
            data_transforms = api._transforms.Group(
                inputs=[
                    libero_incontext_policy.CustomLeRobotLiberoIncontextInputs(
                        action_dim=model_config.action_dim,
                        model_type=model_config.model_type,
                        # FAST in-context: action_dim (7) != demo dims; pad demos to
                        # the model's demo_action_proj / demo_state_proj input dims.
                        demo_action_dim=model_config.demo_action_dim,
                        demo_state_dim=model_config.demo_state_dim,
                    )
                ],
                outputs=[libero_incontext_policy.LiberoIncontextOutputs()],
            )

            if self.use_delta_joint_actions:
                delta_action_mask = api._transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[api._transforms.DeltaActions(delta_action_mask)],
                    outputs=[api._transforms.AbsoluteActions(delta_action_mask)],
                )

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=api.ModelTransformFactory()(model_config),
                train_episode=train_epi,
            )

        @override
        def create_policy(self, assets_dirs: pathlib.Path, model_config):
            # Cache-free eval path: pull in-context demos directly from
            # CustomLeRobotDataset (spanning all episodes, so unseen tasks work)
            # instead of the train-split-filtered InjectDemoIndexes pipeline.
            # Lazy imports avoid a circular import (config -> data_loader -> config).
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
                episodes=None,  # policy spans all episodes (incl. unseen tasks)
                delta_timestamps={
                    key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
                    for key in base_config.action_sequence_keys
                },
                local_files_only=base_config.local_files_only,
                num_sample_frames=self.sample_frames,
                num_sample_actions=self.sample_actions,
                task_to_episode_path=self.task_to_episode_path,
                random_select=self.random_select,
            )

            # Eval-only repack: only the keys the WebSocket client sends.
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
                        demo_action_dim=model_config.demo_action_dim,
                        demo_state_dim=model_config.demo_state_dim,
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
                provides_incontext_demos=True,
            )

    @dataclasses.dataclass(frozen=True)
    class MultiCustomSequenceDebugLeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
        """Multi-dataset variant of SequenceDebugLeRobotLiberoIncontextDataConfig.

        Inherited DataConfigFactory.repo_id is set to the FIRST spec's repo_id and
        serves only as a placeholder; norm stats are loaded via the TrainConfig's
        assets_repo_override. The DataConfig returned has train_episode=None;
        per-dataset filtering happens inside create_custom_dataset using each
        spec's episode_json_path + remove_task_list.

        Demo normalization uses the FIXED semantics (demo states AND actions
        z-scored, action stats identity-padded to 32) — deliberately NOT
        replicating the refactor_fast_incontext branch, which trained its
        plus_libero90 runs with raw (unnormalized) demo actions due to a
        states-only alias (the bug fixed in the demo-norm fix).
        """

        dataset_specs: tuple[DatasetSpec, ...] = ()
        use_delta_joint_actions: bool = False
        sample_frames: int = 2
        sample_actions: int = 32
        random_select: bool = True
        policy_local_files_only: bool = False
        norm_stats_aliases: dict[str, str] | None = dataclasses.field(
            default_factory=lambda: {
                "dem_prompt_all_states": "state",
                "dem_prompt_all_actions": "actions",
            }
        )
        norm_stats_alias_pad_dims: dict[str, int] | None = dataclasses.field(
            default_factory=lambda: {"dem_prompt_all_actions": 32}
        )

        @override
        def create(self, assets_dirs: pathlib.Path, model_config):
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

            data_transforms = api._transforms.Group(
                inputs=[
                    libero_incontext_policy.CustomLeRobotLiberoIncontextInputs(
                        action_dim=model_config.action_dim,
                        model_type=model_config.model_type,
                        demo_action_dim=model_config.demo_action_dim,
                        demo_state_dim=model_config.demo_state_dim,
                    )
                ],
                outputs=[libero_incontext_policy.LiberoIncontextOutputs()],
            )

            if self.use_delta_joint_actions:
                delta_action_mask = api._transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[api._transforms.DeltaActions(delta_action_mask)],
                    outputs=[api._transforms.AbsoluteActions(delta_action_mask)],
                )

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=api.ModelTransformFactory()(model_config),
                train_episode=None,
            )

        @override
        def create_policy(self, assets_dirs: pathlib.Path, model_config):
            # Cache-free eval path mirroring SequenceDebugLeRobotLiberoIncontextDataConfig.
            # Demos come from the FIRST dataset spec (libero): unseen-eval tasks are
            # libero tasks, so libero_90 is never needed at eval time.
            from lerobot.common.datasets import lerobot_dataset as lerobot_dataset_mod

            from openpi.training.custom_dataset import CustomLeRobotDataset

            if not self.dataset_specs:
                raise ValueError("dataset_specs must be non-empty to create a policy.")
            demo_spec = self.dataset_specs[0]

            base_config = self.create_base_config(assets_dirs)
            if self.policy_local_files_only:
                base_config = dataclasses.replace(base_config, local_files_only=True)

            dataset_meta = lerobot_dataset_mod.LeRobotDatasetMetadata(
                demo_spec.repo_id, local_files_only=base_config.local_files_only
            )
            custom_dataset = CustomLeRobotDataset(
                demo_spec.repo_id,
                episodes=None,  # policy spans all episodes (incl. unseen tasks)
                delta_timestamps={
                    key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
                    for key in base_config.action_sequence_keys
                },
                local_files_only=base_config.local_files_only,
                num_sample_frames=self.sample_frames,
                num_sample_actions=self.sample_actions,
                task_to_episode_path=demo_spec.task_to_episode_path,
                random_select=self.random_select,
            )

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
                        demo_action_dim=model_config.demo_action_dim,
                        demo_state_dim=model_config.demo_state_dim,
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
                provides_incontext_demos=True,
            )

    def make_model(*, paligemma_variant: str | None = None, avg_incontext_image_tokens: bool = False):
        kwargs = {}
        if paligemma_variant is not None:
            kwargs["paligemma_variant"] = paligemma_variant
        return api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            **kwargs,
            action_dim=7,
            action_horizon=10,
            max_token_len=128,
            demo_action_dim=32,
            demo_state_dim=8,
            sample_frames=2,
            sample_actions=32,
            random_select=True,
            avg_incontext_image_tokens=avg_incontext_image_tokens,
        )

    def make_data(*, remove_task_list: list[str] | None = None):
        kwargs = {}
        if remove_task_list is not None:
            kwargs.update(
                remove_task_list=remove_task_list,
                episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            )
        return SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            **kwargs,
        )

    def make_config(
        *,
        name: str,
        remove_task_list: list[str] | None = None,
        paligemma_variant: str | None = None,
        shape_flexible_loader: bool = False,
        save_interval: int = 5000,
        avg_incontext_image_tokens: bool = False,
    ) -> api.TrainConfig:
        weight_loader_cls = (
            api.weight_loaders.CheckpointWeightLoaderShapeFlexible
            if shape_flexible_loader
            else api.weight_loaders.CheckpointWeightLoader
        )
        return api.TrainConfig(
            name=name,
            assets_repo_override="debug_pi0_fast_libero_incontext_inference",
            model=make_model(
                paligemma_variant=paligemma_variant,
                avg_incontext_image_tokens=avg_incontext_image_tokens,
            ),
            data=make_data(remove_task_list=remove_task_list),
            weight_loader=weight_loader_cls("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
            num_train_steps=20_000,
            ema_decay=None,
            num_workers=8,
            batch_size=32,
            save_interval=save_interval,
            # Route training through CustomLeRobotDataset (scripts/train.py keys on this).
            use_custom_dataloader=True,
        )

    def make_plus_libero90_data():
        return MultiCustomSequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(local_files_only=False, prompt_from_task=True),
            use_delta_joint_actions=False,
            sample_frames=2,
            sample_actions=32,
            random_select=True,
        )

    def _fast_split_plus_libero90_config(
        name: str,
        remove_task_list: list[str],
        *,
        paligemma_variant: str | None = None,
        shape_flexible_loader: bool = False,
        avg_incontext_image_tokens: bool = False,
    ) -> api.TrainConfig:
        weight_loader_cls = (
            api.weight_loaders.CheckpointWeightLoaderShapeFlexible
            if shape_flexible_loader
            else api.weight_loaders.CheckpointWeightLoader
        )
        return api.TrainConfig(
            name=name,
            assets_repo_override="debug_pi0_fast_libero_incontext_inference",
            model=make_model(
                paligemma_variant=paligemma_variant,
                avg_incontext_image_tokens=avg_incontext_image_tokens,
            ),
            data=dataclasses.replace(
                make_plus_libero90_data(),
                dataset_specs=(
                    DatasetSpec(
                        repo_id="physical-intelligence/libero",
                        episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
                        task_to_episode_path="metadata/libero/task_to_episode.json",
                        remove_task_list=remove_task_list,
                        local_files_only=False,
                    ),
                    DatasetSpec(
                        repo_id="vo2yager/libero_90",
                        episode_json_path=_libero90_episode_json_path(),
                        task_to_episode_path="metadata/libero_90/task_to_episode.json",
                        remove_task_list=None,
                        local_files_only=True,
                    ),
                ),
            ),
            weight_loader=weight_loader_cls("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
            lr_schedule=api._optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=69_000,
                decay_lr=2.5e-6,
            ),
            num_train_steps=70_000,
            ema_decay=None,
            num_workers=8,
            batch_size=32,
            save_interval=1000,
            use_custom_dataloader=True,
        )

    split_variants = [
        ("", api.DEFAULT_LIBERO_TEST_TASK),
        ("2", api.DEFAULT_LIBERO_TEST_TASK_V2),
        ("3", api.DEFAULT_LIBERO_TEST_TASK_V3),
        ("4", api.DEFAULT_LIBERO_TEST_TASK_V4),
        ("5", api.DEFAULT_LIBERO_TEST_TASK_V5),
        ("6", api.DEFAULT_LIBERO_TEST_TASK_V6),
        ("7", api.DEFAULT_LIBERO_TEST_TASK_V7),
        ("8", api.DEFAULT_LIBERO_TEST_TASK_V8),
    ]

    configs = [
        make_config(
            name="pi0_fast_incontext_prompt_action_7_state_8_inference",
        ),
        make_config(
            name="pi0_fast_incontext_prompt_action_7_state_8_inference_900m",
            paligemma_variant="gemma_900m",
            shape_flexible_loader=True,
        ),
    ]

    for suffix, remove_task_list in split_variants:
        configs.append(
            make_config(
                name=f"pi0_fast_incontext_prompt_action_7_state_8_train_split{suffix}",
                remove_task_list=remove_task_list,
                save_interval=1000,
            )
        )

    for suffix, remove_task_list in split_variants:
        name_suffix = f"{suffix}_900m" if suffix else "_900m"
        configs.append(
            make_config(
                name=f"pi0_fast_incontext_prompt_action_7_state_8_train_split{name_suffix}",
                remove_task_list=remove_task_list,
                paligemma_variant="gemma_900m",
                shape_flexible_loader=True,
                save_interval=1000,
            )
        )

    # Multi-dataset (libero train split + libero_90) FAST in-context configs with
    # averaged in-context image tokens, 900M PaliGemma, fixed 70k LR schedule.
    # ECCV rebuttal row-73 architecture (but with FIXED demo normalization).
    configs.extend(
        [
            _fast_split_plus_libero90_config(
                "pi0_fast_incontext_prompt_action_7_state_8_train_split_900m_avg_demo_img_plus_libero90",
                api.DEFAULT_LIBERO_TEST_TASK,
                paligemma_variant="gemma_900m",
                shape_flexible_loader=True,
                avg_incontext_image_tokens=True,
            ),
            # NOTE: the "split2" plus_libero90 family uses TEST_TASK_V3, matching the
            # refactor_fast_incontext branch (verified — not a typo).
            _fast_split_plus_libero90_config(
                "pi0_fast_incontext_prompt_action_7_state_8_train_split2_900m_avg_demo_img_plus_libero90",
                api.DEFAULT_LIBERO_TEST_TASK_V3,
                paligemma_variant="gemma_900m",
                shape_flexible_loader=True,
                avg_incontext_image_tokens=True,
            ),
            # Eval policy config for avg-demo-img checkpoints: the plain
            # inference_900m has avg_incontext_image_tokens=False, which changes the
            # compute graph and would mis-evaluate avg checkpoints.
            make_config(
                name="pi0_fast_incontext_prompt_action_7_state_8_inference_900m_avg_demo_img",
                paligemma_variant="gemma_900m",
                shape_flexible_loader=True,
                avg_incontext_image_tokens=True,
            ),
        ]
    )

    return configs
