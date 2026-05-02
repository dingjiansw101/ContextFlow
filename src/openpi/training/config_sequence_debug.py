# config_sequence_debug.py
"""
An example of child configs.
Debug experiment child configs.
Parent config.py will import this file and call build(api) to collect TrainConfig entries.
"""

from __future__ import annotations
import dataclasses
import tyro
import pathlib
import json
import jsonlines
from collections.abc import Sequence
from typing_extensions import override

# If this child only needs certain policies/modules here, import them directly
import openpi.policies.libero_incontext_policy as libero_incontext_policy


def build(api) -> list["api.TrainConfig"]:
    g = globals()
    g["DataConfig"] = getattr(api, "DataConfig")
    g["BaseModelConfig"] = getattr(api._model, "BaseModelConfig")
    # 1) Define DataConfig subclasses inside this function, inheriting DataConfigFactory via api
    @dataclasses.dataclass(frozen=True)
    class SequenceDebugLeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
        use_delta_joint_actions: bool = True
        states_cache_path: str = "metadata/libero/episode_states_cache.json"
        actions_cache_path: str = "metadata/libero/episode_actions_first_cache.json"
        task_to_episode: str='metadata/libero/task_to_episode.json'
        episode_to_indexes_file: str='metadata/libero/episode_to_indexes.json'

        # Padding mode for AddStatesActionsPromptTransform
        padding_mode: str = "keep_all"
        mask_padding_as_valid: bool = False
        demo_state_dim: int | None = 32

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
                demo_state_dim=self.demo_state_dim,
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
        def create(self, assets_dirs: pathlib.Path, model_config: "BaseModelConfig") -> "DataConfig":
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
    return [
    ####################
    ####################
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    # no sequence training frames + pi0mini + without avg current img tokens + current img tokens in action expert
    # no sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # sequence training frames + pi0mini + without avg current img tokens + current img tokens in action expert
    # V12: no sequence training frames + pi0mini + without avg current img tokens + current img tokens in PROMPT expert
    ####################
    #####seq_avg########
    ####################
    ####################
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    ####################
    #####seq_no_avg#####
    ####################
    ####################
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    ####################
    #####seq_avg########
    #####prompt_img#####
    ####################
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    #### total_bs=384 variants; seq_avg
    ####################
    #####seq_avg########
    #####train##########
    #####longer##########
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    ####################
    #####seq_avg########
    #####large##########
    #####ViT############
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    ###############
    ####RoboSSM####
    ###############
    ###############
    ###############
    ####Split######
    ###############
    # api.TrainConfig(
    #     name="vitb_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1",
    #     assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
    #     model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
    #         prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
    #         sample_frames=2, sample_actions=32, random_select=True,  
    #         freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
    #         use_frame_sequence_transform=True, 
    #         frame_sequence_length=6,
    #         avg_current_img=True,
    #         ),
    #     data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
    #     vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
    #         npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
    #     ),
    #     weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     lr_schedule = api._optimizer.CosineDecaySchedule(
    #         warmup_steps = 1_000,
    #         peak_lr= 2.5e-5,
    #         decay_steps= 20_000,
    #         decay_lr= 2.5e-6),
    #     num_train_steps= 20_000,
    #     freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
    #         prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
    #         sample_frames=2, sample_actions=32, random_select=True,  
    #         freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
    #         use_frame_sequence_transform=True, 
    #         frame_sequence_length=6,
    #         avg_current_img=True,
    #         ).get_freeze_filter(),
    #     ema_decay=None,
    #     num_workers=8,
    #     batch_size=32,
    # ), 
    # api.TrainConfig(
    #     name="vitb_6_sequence_avg_pi0mini_libero_incontextv14_inference",
    #     assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
    #     model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
    #         prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
    #         sample_frames=2, sample_actions=32, random_select=True,  
    #         freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
    #         use_frame_sequence_transform=True, 
    #         frame_sequence_length=6,
    #         avg_current_img=True,
    #         ),
    #     data=SequenceDebugLeRobotLiberoIncontextDataConfig(
    #         repo_id="physical-intelligence/libero",
    #         base_config=api.DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #         use_delta_joint_actions=False,
    #         states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
    #         actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
    #     ),
    #     vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
    #         npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
    #     ),
    #     weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     lr_schedule = api._optimizer.CosineDecaySchedule(
    #         warmup_steps = 1_000,
    #         peak_lr= 2.5e-5,
    #         decay_steps= 20_000,
    #         decay_lr= 2.5e-6),
    #     num_train_steps= 20_000,
    #     freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
    #         prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
    #         sample_frames=2, sample_actions=32, random_select=True,  
    #         freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
    #         use_frame_sequence_transform=True, 
    #         frame_sequence_length=6,
    #         avg_current_img=True,
    #         ).get_freeze_filter(),
    #     ema_decay=None,
    #     num_workers=8,
    #     batch_size=32,
    # ), 
    ##############
    ##v12 model###
    ####no seq####
    ###############
    ####pi0-fast###
    ###############
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py trivial --project-name=ddd --exp-name=trivial --overwrite
    api.TrainConfig(
        name="pi0_fast_incontext_action_7_state_8_example_config",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api.pi0_fast_incontext.Pi0FASTIncontextConfig(
            action_dim=7, action_horizon=10, max_token_len=256, 
            sample_frames=2, sample_actions=32, random_select=True,
             
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache_state_8.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache_action_7.json",
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ),
    api.TrainConfig(
        name="debug_pi0_fast_libero_incontext_inference",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api.pi0_fast_incontext.Pi0FASTIncontextConfig(
            action_dim=7, action_horizon=10, max_token_len=128, 
            sample_frames=2, sample_actions=4, random_select=True, 
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        ema_decay=None,
        num_workers=8,
        batch_size=4,
    ),
    api.TrainConfig(
        name="debug_pi0_fast_libero_incontext_train_split_v1",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api.pi0_fast_incontext.Pi0FASTIncontextConfig(
            action_dim=7, action_horizon=10, max_token_len=128, 
            sample_frames=2, sample_actions=4, random_select=True, 
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        ema_decay=None,
        num_workers=8,
        batch_size=4,
    ),
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py debug_pi0_fast_prompt_libero_incontext_inference --project-name=ddd --exp-name=debug_pi0_fast_prompt_libero_incontext_inference --overwrite
    api.TrainConfig(
        name="debug_pi0_fast_prompt_libero_incontext_train_split_v1",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128, 
            demo_action_dim = 32,
            sample_frames=2, sample_actions=32, random_select=True, 
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        ema_decay=None,
        num_workers=8,
        batch_size=4,
    ),
    api.TrainConfig(
        name="debug_pi0_fast_prompt_libero_incontext_inference",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128, 
            demo_action_dim = 32,
            sample_frames=2, sample_actions=32, random_select=True, 
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        ema_decay=None,
        num_workers=8,
        batch_size=4,
    ),
    api.TrainConfig(
        name="debug_pi0_libero_incontextv12_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_2b", action_expert_variant="gemma_300m",
            sample_frames=2, sample_actions=32, random_select=True, 
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        ema_decay=None,
        num_workers=8,
        batch_size=4,
    ),
    api.TrainConfig(
        name="debug_pi0_libero_incontextv12_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_2b", action_expert_variant="gemma_300m",
            sample_frames=2, sample_actions=32, random_select=True, 
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        ema_decay=None,
        num_workers=8,
        batch_size=4,
    ),
    ###############
    ####pi0-fast###
    #####lora#######
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py debug_pi0_fast_libero_incontextv_inference --project-name=ddd --exp-name=debug_pi0_fast_libero_incontextv_inference --overwrite
    api.TrainConfig(
        name="debug_pi0_fast_libero_incontext_low_mem_inference",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api.pi0_fast_incontext.Pi0FASTIncontextConfig(
            action_dim=7, action_horizon=10, max_token_len=128, 
            sample_frames=2, sample_actions=4, random_select=True,
            paligemma_variant="gemma_2b_lora",
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        freeze_filter=api.pi0_fast_incontext.Pi0FASTIncontextConfig(
            action_dim=7, action_horizon=10, max_token_len=128, 
            sample_frames=2, sample_actions=4, random_select=True,
            paligemma_variant="gemma_2b_lora",
            ).get_freeze_filter(),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ),
    api.TrainConfig(
        name="debug_pi0_fast_libero_incontext_low_mem_train_split_v1",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api.pi0_fast_incontext.Pi0FASTIncontextConfig(
            action_dim=7, action_horizon=10, max_token_len=128, 
            sample_frames=2, sample_actions=4, random_select=True,
            paligemma_variant="gemma_2b_lora",
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        freeze_filter=api.pi0_fast_incontext.Pi0FASTIncontextConfig(
            action_dim=7, action_horizon=10, max_token_len=128, 
            sample_frames=2, sample_actions=4, random_select=True,
            paligemma_variant="gemma_2b_lora",
            ).get_freeze_filter(),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ),
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py debug_pi0_fast_prompt_libero_incontext_inference --project-name=ddd --exp-name=debug_pi0_fast_prompt_libero_incontext_inference --overwrite
    api.TrainConfig(
        name="debug_pi0_fast_prompt_libero_incontext_low_mem_train_split_v1",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128, 
            demo_action_dim = 32,
            sample_frames=2, sample_actions=32, random_select=True, 
            paligemma_variant="gemma_2b_lora",
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        freeze_filter=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128, 
            demo_action_dim = 32,
            sample_frames=2, sample_actions=32, random_select=True, 
            paligemma_variant="gemma_2b_lora",
            ).get_freeze_filter(),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ),
    api.TrainConfig(
        name="debug_pi0_fast_prompt_libero_incontext_low_mem_inference",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128, 
            demo_action_dim = 32,
            sample_frames=2, sample_actions=32, random_select=True, 
            paligemma_variant="gemma_2b_lora",
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        freeze_filter=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128, 
            demo_action_dim = 32,
            sample_frames=2, sample_actions=32, random_select=True, 
            paligemma_variant="gemma_2b_lora",
            ).get_freeze_filter(),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ),
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py pi0_fast_incontext_prompt_action_7_state_8_train_split --project-name=ddd --exp-name=pi0_fast_incontext_prompt_action_7_state_8_train_split --overwrite
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_inference",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split2",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V2,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    # Size-matched ContextAR variant: PaliGemma LLM swapped from gemma_2b -> gemma_900m
    # (~934M layer params, total ~1.86B) to match ContextFlow non-2B (~1.86B) on Libero V1 split.
    # Vision + embedder transfer from pi0_fast_base; LLM trunk trains from scratch.
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split2_900m",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            paligemma_variant="gemma_900m",
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V2,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split3",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V3,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split4",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V4,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split5",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V5,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split6",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V6,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split7",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V7,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split8",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V8,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    # === gemma_900m size-matched variants (paired with the splits above) ===
    # Each config is identical to its sibling except paligemma_variant="gemma_900m".
    # Total ~1.86B params (size-matched to ContextFlow non-2B). split2_900m is registered earlier
    # near its base config; the remaining 7 splits are grouped here for easy iteration.
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split_900m",  # V0
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            paligemma_variant="gemma_900m",
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split3_900m",  # V2
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            paligemma_variant="gemma_900m",
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V3,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split4_900m",  # V3
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            paligemma_variant="gemma_900m",
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V4,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split5_900m",  # V4
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            paligemma_variant="gemma_900m",
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V5,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split6_900m",  # V5
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            paligemma_variant="gemma_900m",
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V6,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split7_900m",  # V6
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            paligemma_variant="gemma_900m",
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V7,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="pi0_fast_incontext_prompt_action_7_state_8_train_split8_900m",  # V7
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api._pi0_fast_incontext_seq.Pi0FASTIncontextSeqConfig(
            paligemma_variant="gemma_900m",
            action_dim=7, action_horizon=10, max_token_len=128,
            demo_action_dim=32, demo_state_dim=8,
            sample_frames=2, sample_actions=32, random_select=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            demo_state_dim=8,
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V8,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
        save_interval=1000,
    ),
    api.TrainConfig(
        name="debug_pi0_libero_incontextv12_low_mem_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True, 
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True, 
            ).get_freeze_filter(),
        weight_loader=api.weight_loaders.CheckpointWeightLoaderIncontext("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ),
    api.TrainConfig(
        name="debug_pi0_libero_incontextv12_low_mem_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True, 
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
        ),
        freeze_filter=api.pi0_incontextv12.Pi0IncontextConfigv12(
            prompt_expert_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
            sample_frames=2, sample_actions=32, random_select=True, 
            ).get_freeze_filter(),
        weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ),
    
    
    
    
    # api.TrainConfig(
    #     name="debug_prompt_pi0_libero_low_mem_finetune_incontextv14_inference",
    #     assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
    #     model=api.pi0_light_incontextv14_prompt.Pi0LightIncontextConfigv14Prompt(
    #         prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
    #         sample_frames=2, sample_actions=32, random_select=True,  
    #         freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
    #         use_frame_sequence_transform=True, 
    #         frame_sequence_length=6,
    #         avg_current_img=True,
    #         ),
    #     data=SequenceDebugLeRobotLiberoIncontextDataConfig(
    #         repo_id="physical-intelligence/libero",
    #         base_config=api.DataConfig(
    #             local_files_only=False,  # Set to True for local-only datasets.
    #             prompt_from_task=True,
    #         ),
    #         use_delta_joint_actions=False,
    #         states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
    #         actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
    #     ),
    #     weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    #     vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
    #         npz_path="gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", # B/16
    #     ),
    #     num_train_steps=20_000,
    #     freeze_filter=api.pi0_light_incontextv14_prompt.Pi0LightIncontextConfigv14Prompt(
    #         prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
    #         sample_frames=2, sample_actions=32, random_select=True,  
    #         freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="B/16",
    #         use_frame_sequence_transform=True, 
    #         frame_sequence_length=6,
    #         avg_current_img=True,
    #         ).get_freeze_filter(),
    #     ema_decay=None,
    #     num_workers=8,
    #     batch_size=32,
    # ),

    api.TrainConfig(
        name="sup_pi0_fast_incontextv12_train_split_v1",
        model_summary_json="fast_incontextv12_model_summary.json",
        assets_repo_override="debug_pi0_fast_libero_incontext_inference",
        model=api.pi0_fast_incontext.Pi0FASTIncontextConfig(
            paligemma_variant="gemma_incontextv12_fast",
            action_dim=7, action_horizon=10, max_token_len=256, 
            sample_frames=2, sample_actions=32, random_select=True, 
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
            demo_state_dim=8,
        ),
        weight_loader=api.weight_loaders.InputEmbedderAndSiglipLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        # weight_loader=api.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ),
    ]
