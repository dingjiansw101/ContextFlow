"""Sequence-debug LIBERO in-context training configs."""

# ruff: noqa: SLF001

from __future__ import annotations

import dataclasses
import pathlib

from typing_extensions import override

import openpi.policies.libero_incontext_policy as libero_incontext_policy


def build(api):
    g = globals()
    g["DataConfig"] = api.DataConfig
    g["BaseModelConfig"] = api._model.BaseModelConfig

    @dataclasses.dataclass(frozen=True)
    class SequenceDebugLeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
        use_delta_joint_actions: bool = True
        states_cache_path: str = "metadata/libero/episode_states_cache.json"
        actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
        task_to_episode: str = "metadata/libero/task_to_episode.json"
        episode_to_indexes_file: str = "metadata/libero/episode_to_indexes.json"
        padding_mode: str = "keep_all"
        mask_padding_as_valid: bool = False
        demo_state_dim: int | None = 32

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
                        }
                    )
                ]
            )

            train_epi = api.get_kept_episode_indices(self.episode_json_path, self.remove_task_list)
            data_transforms = api._transforms.Group(
                inputs=[
                    api._transforms.InjectDemoIndexes(
                        sample_frames=model_config.sample_frames,
                        random_select=model_config.random_select,
                        sample_episodes=model_config.sample_episodes,
                        task_to_episode=self.task_to_episode,
                        episode_to_indexes=self.episode_to_indexes_file,
                        train_episode_index_list=train_epi,
                        seed_base=self.seed_base,
                    )
                ],
                outputs=[],
            )
            data_transforms = data_transforms.push(
                inputs=[
                    libero_incontext_policy.LiberoIncontextInputs(
                        action_dim=model_config.action_dim, model_type=model_config.model_type
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
                demo_state_dim=self.demo_state_dim,
            )

    def make_model(*, paligemma_variant: str | None = None):
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
    ) -> api.TrainConfig:
        weight_loader_cls = (
            api.weight_loaders.CheckpointWeightLoaderShapeFlexible
            if shape_flexible_loader
            else api.weight_loaders.CheckpointWeightLoader
        )
        return api.TrainConfig(
            name=name,
            assets_repo_override="debug_pi0_fast_libero_incontext_inference",
            model=make_model(paligemma_variant=paligemma_variant),
            data=make_data(remove_task_list=remove_task_list),
            weight_loader=weight_loader_cls("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
            num_train_steps=20_000,
            ema_decay=None,
            num_workers=8,
            batch_size=32,
            save_interval=save_interval,
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

    return configs
