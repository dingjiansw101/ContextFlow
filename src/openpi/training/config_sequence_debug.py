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
    class SequenceDebugLiberoRoboSSMDataConfig(api.DataConfigFactory):
        """Data config for the RoboSSM kitchen split with explicit train/test tasks."""

        use_delta_joint_actions: bool = False
        states_cache_path: str = "metadata/libero_90/episode_states_cache.json"
        actions_cache_path: str = "metadata/libero_90/episode_actions_cache.json"
        task_to_episode: str = "metadata/libero_90/task_to_episode.json"
        episode_to_indexes_file: str = "metadata/libero_90/episode_to_indexes.json"
        tasks_split_path: str = "examples/libero_90/libero_robossm_kitchen_tasks.json"
        split: str = "train"  # "train" or "test"
        episode_json_path: str | None = None
        debug_prompt_cache: bool = False

        def _load_split(self) -> tuple[list[str], list[str]]:
            split_path = pathlib.Path(self.tasks_split_path)
            if not split_path.exists():
                raise FileNotFoundError(f"Task split file not found at: {split_path}")
            with split_path.open("r") as f:
                spec = json.load(f)
            train_tasks = spec.get("train_tasks") or []
            test_tasks = spec.get("test_tasks") or []
            return list(train_tasks), list(test_tasks)

        def _select_episode_indices(self, allowed_tasks: Sequence[str]) -> list[int]:
            if self.episode_json_path is None:
                raise ValueError("episode_json_path must be set for RoboSSM kitchen split.")
            ep_path = pathlib.Path(self.episode_json_path).expanduser()
            if not ep_path.exists():
                raise FileNotFoundError(f"episodes.jsonl file not found at: {ep_path}")

            allowed = set(allowed_tasks)
            selected: list[int] = []
            seen: set[int] = set()
            with jsonlines.open(ep_path, mode="r") as reader:
                for entry in reader:
                    tasks = set(entry.get("tasks", []))
                    if not tasks or not allowed.intersection(tasks):
                        continue
                    idx = entry.get("episode_index")
                    if idx is not None:
                        idx = int(idx)
                        if idx not in seen:
                            selected.append(idx)
                            seen.add(idx)
            return selected

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: BaseModelConfig) -> DataConfig:
            train_tasks, test_tasks = self._load_split()
            if self.split == "train":
                target_tasks = train_tasks
            elif self.split == "test":
                target_tasks = test_tasks
            else:
                raise ValueError(f"Unknown split '{self.split}'. Expected 'train' or 'test'.")

            if not target_tasks:
                raise ValueError(f"No tasks defined for split '{self.split}' in {self.tasks_split_path}.")

            kept_indices = self._select_episode_indices(target_tasks)

            # try:
            #     kept_indices = self._select_episode_indices(target_tasks)
            #     if self.split == "test":
            #         kept_indices = None
            # except (FileNotFoundError, ValueError) as exc:
            #     import warnings
            #     warnings.warn(f"[Warning]: SequenceDebugLiberoRoboSSMDataConfig: Skipping episode selection: {exc}")
            #     kept_indices = None

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

            data_transforms = api._transforms.Group(
                inputs=[
                    api._transforms.InjectDemoIndexes(
                        sample_frames=model_config.sample_frames,
                        random_select=model_config.random_select,
                        sample_episodes=model_config.sample_episodes,
                        task_to_episode=self.task_to_episode,
                        episode_to_indexes=self.episode_to_indexes_file,
                        train_episode_index_list=kept_indices,
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

            model_transforms = api.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                train_episode=kept_indices,
            )

    # 2) Return this child's TrainConfig entries directly (can be multiple)
    return []
