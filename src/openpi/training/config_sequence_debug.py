# config_template.py
"""
An example of children config.
RoboCasa 实验的 children config。
父 config.py 会 import 本文件，并调用 build(api) 来拿 TrainConfig 列表。
"""

from __future__ import annotations
import dataclasses
import tyro
import pathlib
import json
from collections.abc import Sequence
from typing_extensions import override

# 如果这个 child 只在这里用到某些 policy/模块，可以直接在这里 import
import openpi.policies.libero_incontext_policy as libero_incontext_policy
import openpi.policies.libero_policy as libero_policy

def build(api) -> list["api.TrainConfig"]:
    g = globals()
    g["DataConfig"] = getattr(api, "DataConfig")
    g["BaseModelConfig"] = getattr(api._model, "BaseModelConfig")
    # 1) 在函数内定义 DataConfig 子类，继承父里的 DataConfigFactory（通过 api 取）
    @dataclasses.dataclass(frozen=True)
    class SequenceDebugLeRobotLiberoIncontextDataConfig(api.DataConfigFactory):
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

    # 2) 直接返回本 child 的 TrainConfig 条目（可多个）
    return [
    api.TrainConfig(
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext.py sequence_compare_pi0_libero_incontextv12_train_split_v3 --project-name=ddd --exp-name=ddd --overwrite
        name="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            freeze_llm_embedder=False, freeze_img_encoder=False, 
            siglip_variant="S/16",
            sample_frames=2, sample_actions=32, random_select=True, 
            #use_frame_sequence_transform = True, 
            #frame_sequence_length = 6,
            avg_current_img=False,
        ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
        weight_loader=api.weight_loaders.EmptyLoader(),        
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=72,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0_libero_incontextv14_train_split_v3 --project-name=ddd --exp-name=ddd --overwrite
        name="sequence_debug_pi0_libero_incontextv14_train_split_v3",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            freeze_llm_embedder=False, freeze_img_encoder=False, 
            siglip_variant="S/16",
            sample_frames=2, sample_actions=32, random_select=True, 
            use_frame_sequence_transform = True, 
            frame_sequence_length=36,
            avg_current_img=True,
        ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            remove_task_list=api.DEFAULT_LIBERO_TEST_TASK,#_V3,
            episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,

        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
        weight_loader=api.weight_loaders.EmptyLoader(),        
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=8,
        # wandb_enabled=False,
    ),
    api.TrainConfig(
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0_libero_incontextv14_train_split_v3 --project-name=ddd --exp-name=ddd --overwrite
        name="sequence_debug_pi0_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            vocab_size=50_000, 
            prompt_expert_variant="gemma_A", action_expert_variant="gemma_B",
            freeze_llm_embedder=False, freeze_img_encoder=False, 
            siglip_variant="S/16",
            sample_frames=2, sample_actions=32, random_select=True, 
            use_frame_sequence_transform = False, 
            # frame_sequence_length=36,
            avg_current_img=True,
        ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=api.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            use_delta_joint_actions=False,
            states_cache_path="metadata/libero/episode_states_without_delta_cache.json",
            actions_cache_path="metadata/libero/episode_actions_without_delta_cache.json",
            # remove_task_list=api.DEFAULT_LIBERO_TEST_TASK_V3,
            # episode_json_path=api.DEFAULT_LIBERO_EPISODE_JSON,
        ),
        vision_weight_loader=api.weight_loaders.RemapSigLIPPrefixLoader(
                npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
            ),
        weight_loader=api.weight_loaders.EmptyLoader(),        
        num_train_steps=20_000,
        ema_decay=None,
        num_workers=8,
        # num_workers=1,
        batch_size=8,
        # wandb_enabled=False,
    ),
    ####################
    ####################
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    api.TrainConfig(
        name="sequence_avg_cur_img_debug_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=24,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=24,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=16,
    ), 
    api.TrainConfig(
        name="sequence_avg_cur_img_debug_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=24,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
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
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=24,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=16,
    ), 
    # no sequence training frames + pi0mini + without avg current img tokens + current img tokens in action expert
    api.TrainConfig(
        name="no_sequence_no_avg_debug_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=False, 
            frame_sequence_length=0,
            avg_current_img=False,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = False, 
            frame_sequence_length=0,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32*6*2,
    ), 
    api.TrainConfig(
        name="no_sequence_no_avg_debug_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=False, 
            frame_sequence_length=0,
            avg_current_img=False,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
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
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = False, 
            frame_sequence_length=0,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32*6*2,
    ), 
    # no sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    api.TrainConfig(
        name="no_sequence_avg_cur_img_debug_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=False, 
            frame_sequence_length=0,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = False, 
            frame_sequence_length=0,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32*6*2,
    ), 
    api.TrainConfig(
        name="no_sequence_avg_cur_img_debug_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=False, 
            frame_sequence_length=0,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
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
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = False, 
            frame_sequence_length=0,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32*6*2,
    ), 
    # sequence training frames + pi0mini + without avg current img tokens + current img tokens in action expert
    api.TrainConfig(
        name="sequence_no_avg_debug_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=6,
            avg_current_img=False,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = False, 
            frame_sequence_length=24,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ), 
    api.TrainConfig(
        name="sequence_no_avg_debug_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=6,
            avg_current_img=False,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
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
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = False, 
            frame_sequence_length=24,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ), 
    
    ##################
    # V12: no sequence training frames + pi0mini + without avg current img tokens + current img tokens in action expert
    api.TrainConfig(
        name="no_sequence_no_avg_debug_pi0mini_libero_incontextv12_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv12_separate_img_proj.Pi0LightIncontextConfigv12SepImgProj(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv12_separate_img_proj.Pi0LightIncontextConfigv12SepImgProj(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32*6*2,
    ), 
    api.TrainConfig(
        name="no_sequence_no_avg_debug_pi0mini_libero_incontextv12_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv12_separate_img_proj.Pi0LightIncontextConfigv12SepImgProj(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
        ),
        weight_loader=api.weight_loaders.InputEmbedderLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule = api._optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr= 2.5e-5,
            decay_steps= 20_000,
            decay_lr= 2.5e-6),
        num_train_steps= 20_000,
        freeze_filter=api.pi0_light_incontextv12_separate_img_proj.Pi0LightIncontextConfigv12SepImgProj(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32*6*2,
    ), 
    # V12: no sequence training frames + pi0mini + without avg current img tokens + current img tokens in PROMPT expert
    api.TrainConfig(
        name="no_sequence_no_avg_prompt_token_debug_pi0mini_libero_incontextv12_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        num_workers=8,
        batch_size=32*6*2,
    ), 
    api.TrainConfig(
        name="no_sequence_no_avg_prompt_token_debug_pi0mini_libero_incontextv12_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv12.Pi0LightIncontextConfigv12(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        num_workers=8,
        batch_size=32*6*2,
    ),
    ####################
    #####seq_avg########
    ####################
    ####################
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    api.TrainConfig(
        name="12_sequence_avg_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=12,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=12,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=16,
    ), 
    api.TrainConfig(
        name="12_sequence_avg_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=12,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
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
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=12,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=16,
    ), 
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    api.TrainConfig(
        name="24_sequence_avg_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=24,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=24,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=8,
    ), 
    api.TrainConfig(
        name="24_sequence_avg_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=24,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
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
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=24,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=8,
    ), 
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    api.TrainConfig(
        name="48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=4,
    ), 
    api.TrainConfig(
        name="48_sequence_avg_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
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
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=4,
    ), 
    ####################
    #####seq_no_avg#####
    ####################
    ####################
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    api.TrainConfig(
        name="12_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=12,
            avg_current_img=False,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=12,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=16,
    ), 
    api.TrainConfig(
        name="12_sequence_no_avg_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=12,
            avg_current_img=False,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
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
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=12,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=16,
    ), 
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    api.TrainConfig(
        name="24_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=24,
            avg_current_img=False,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=24,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=8,
    ), 
    api.TrainConfig(
        name="24_sequence_no_avg_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=24,
            avg_current_img=False,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
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
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=24,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=8,
    ), 
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    api.TrainConfig(
        name="48_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=False,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=4,
    ), 
    api.TrainConfig(
        name="48_sequence_no_avg_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=False,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
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
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=48,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=4,
    ), 
    api.TrainConfig(
        name="96_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=96,
            avg_current_img=False,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=2, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=96,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=2,
    ), 
    api.TrainConfig(
        name="96_sequence_no_avg_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=96,
            avg_current_img=False,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
            npz_path="gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz", # S/16
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
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=96,
            avg_current_img=False,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=2,
    ), 
    ####################
    #####seq_avg########
    #####prompt_img#####
    ####################
    # sequence training frames + pi0mini + with avg current img tokens + current img tokens in action expert
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_v14.py sequence_debug_pi0mini_libero_incontextv14_train_split_v1 --project-name=ddd --exp-name=ddd --overwrite
    api.TrainConfig(
        name="8_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=8, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=6,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=8, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=12,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ), 
    api.TrainConfig(
        name="8_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=8, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=6,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=8, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=6,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ), 
    api.TrainConfig(
        name="16_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=16, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=6,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=16, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=6,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ), 
    api.TrainConfig(
        name="16_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=16, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=6,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=16, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=6,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ), 
    api.TrainConfig(
        name="32_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=32, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=6,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=32, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=12,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ), 
    api.TrainConfig(
        name="32_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=32, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=6,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m", 
            sample_frames=32, sample_actions=32, random_select=True, 
            freeze_llm_embedder=True, freeze_img_encoder = False, siglip_variant="S/16",
            use_frame_sequence_transform = True, 
            frame_sequence_length=6,
            avg_current_img=True,
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=32,
    ), 
    #### total_bs=384 variants; seq_avg
    api.TrainConfig(
        name="bs_8_seq_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=True,
            ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=8,
    ), 
    api.TrainConfig(
        name="bs_8_seq_48_sequence_avg_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=48,
            avg_current_img=True,
            ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=8,
    ), 
    api.TrainConfig(
        name="bs_4_seq_96_sequence_avg_pi0mini_libero_incontextv14_train_split_v1",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=96,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=96,
            avg_current_img=True,
            ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=4,
    ), 
    api.TrainConfig(
        name="bs_4_seq_96_sequence_avg_pi0mini_libero_incontextv14_inference",
        assets_repo_override="sequence_compare_pi0_libero_incontextv12_train_split_v3",
        model=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=96,
            avg_current_img=True,
            ),
        data=SequenceDebugLeRobotLiberoIncontextDataConfig(
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
        freeze_filter=api.pi0_light_incontextv14.Pi0LightIncontextConfigv14(
            prompt_expert_variant="gemma_132m", action_expert_variant="gemma_66m",
            sample_frames=2, sample_actions=32, random_select=True,  
            freeze_llm_embedder=True, freeze_img_encoder=False, siglip_variant="S/16",
            use_frame_sequence_transform=True, 
            frame_sequence_length=96,
            avg_current_img=True,
            ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=4,
    ), 
    ]
