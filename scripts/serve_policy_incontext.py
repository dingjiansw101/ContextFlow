import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    LIBERO_FM_LORA_INCONTEXT = "pi0_libero_incontext_low_mem_finetune"
    LIBERO_FM_LORA_INCONTEXT_SAMPLE2 = "pi0_libero_incontext_low_mem_finetune_sample2"
    TEST1 = "pi0_libero_incontext_low_mem_finetune_sample2_actionssample32"
    TEST2 = "pi0_libero_incontext_low_mem_finetune_sample2_random_select"
    TEST3 = "pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32"
    TEST4 = "pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample8"
    TEST5 = "pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select"
    TEST6 = "pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64"

    XJ_TEST3_TRAIN = "pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split"
    XJ_TEST6_WITHOUT_LORA = "pi0_libero_incontextv2_sample2_actionssample64"
    XJ_TEST6_WITHOUT_LORA_WITH_SPLIT = "pi0_libero_incontextv2_sample2_actionssample64_train_split"
    XJ_TEST6_WITHOUT_LORA_WITH_SPLIT_20K = "pi0_libero_incontextv2_sample2_actionssample64_train_split"
    XJ_PI0LIGHT_V12_20K = "pi0light_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference"
    
    # supplementary
    XJ_V12_V3 = "pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v3"
    XJ_V12_V4 = "pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v4"
    XJ_ROBOCASA_INFERENCE = "pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train"
    XJ_PI0MINI_ROBOCASA_MG_INCONTEXT = 'pi0mini_incontext_robocasa_mg_three_image_low_mem_finetune'
    XJ_PI0MINI_INCONTEXT_ROBOCASA_MG_INFERENCE = 'pi0mini_incontext_robocasa_mg_three_image_low_mem_finetune_inference'

    # rebutal
    XJ_PI0_MINI_LIBERO_INCONTEXT = "pi0mini_incontext_libero_low_mem_finetune_inference"
    XJ_PI0_MINI_LIBERO_INCONTEXT_ALL = "pi0mini_incontext_libero_low_mem_finetune_inference"
    XJ_PI0_INCONTEXT_ROBOCASA_MG = "pi0_incontext_robocasa_mg_three_image_low_mem_finetune_train"
    XJ_PI0MINI_INCONTEXTV12_1_ROBOCASA_MG_TRAIN = "pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_train"
    XJ_PI0MINI_INCONTEXTV12_1_ROBOCASA_MG_INFERENCE = "pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_inference"
    XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_TRAIN = "pi0tiny_incontext_robocasa_mg_three_image_train_split"
    
    ###
    XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_TRAIN_1M = "pi0tiny_incontext_robocasa_mg_three_image_train_split"
    XJ_PI0MINI_INCONTEXTV12_1_ROBOCASA_MG_TRAIN_1M = "pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_train"
    
    # rebuttal
    XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_TRAIN_LR = "pi0tiny_incontext_robocasa_mg_three_image_train_split_large_lr"
    XJ_PI0MINI_INCONTEXTV12_1_ROBOCASA_MG_TRAIN_LR = "pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_train_large_lr"
    
    # debug piotiny incontext
    DEBUG_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_TRAIN_SPLIT = "debug_pi0tiny_incontext_robocasa_mg_three_image_train_split"
    DEBUG_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_INFERENCE = "debug_pi0tiny_incontext_robocasa_mg_three_image_inference"
    DEBUG_PROMPT_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_INFERENCE = "debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference"
    DEBUG_PROMPT_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_TRAIN_SPLIT = "debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_train_split"
    DEBUG_PROMPT_NO_RANDOM_SELECT_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_INFERENCE = "debug_prompt_no_random_select_pi0tiny_incontext_robocasa_mg_three_image_inference"
    DEBUG_PROPRIO_PROMPT_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_TRAIN_SPLIT = "debug_proprio_prompt_pi0tiny_incontext_robocasa_mg_three_image_train_split"
    DEBUG_PROPRIO_PROMPT_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_INFERENCE = "debug_proprio_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference"
    ## final
    FINAL_PI0TINY_BOOST_IMG_PROMPT_INCONTEXT_LARGE_LR_1M_TRAIN_SPLIT_500K_CKP = "final_boost_img_prompt_pi0tiny_incontext_robocasa_mg_three_image_large_lr_1M_train_split"
    # final_2
    FINAL_PI0TINY_INCONTEXT_LARGEST_LR_TRAIN_SPLIT = "final_pi0tiny_incontext_robocasa_mg_three_image_largest_lr_train_split"
    FINAL_PI0TINY_INCONTEXT_1M_TRAIN_SPLIT_1M_CKP = "pi0tiny_incontext_robocasa_mg_three_image_large_lr_train_split_1M_dummy"
    FINAL_PI0TINY_INCONTEXT_1M_TRAIN_SPLIT_500K_CKP = "pi0tiny_incontext_robocasa_mg_three_image_large_lr_train_split_1M"
    DEBUG_IMG_ENCODER = "debug_img_encoder"
    DEBUG_LOW_ACTION_HORIZON = "pi0tiny_incontext_robocasa_mg_three_image_large_lr_train_split_low_action_horizon"
    DEBUG_TRAIN_WITHOUT_OPENDOUBLEDOOR = "debug_train_without_open_double_door"
    
    # final_3
    XJ_PI0_LIBERO90_INCONTEXTV12_LOW_MEM_FINETUNE = "pi0_libero90_incontextv12_low_mem_finetune"
    PI0_LIBERO_MORE_SAMPLE_FRAMES = "pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_train_split_v1"
    POINT_TRACK = "pi0_libero_incontextv12_point_track_low_mem_finetune_train_split"
    VIDEO_PROMPT = "pi0_libero_incontextv12_video_prompt_low_mem_finetune_train_split"
    RANDOM_INIT = "pi0_libero_incontextv12_low_mem_finetune_random_init_train_split"
    PI0_LIBERO_90_NONE_LORA = "pi0_libero90_incontextv12_finetune_x"
    PI0_LIBERO_90_NONE_LORA_LIBERO90 = "pi0_libero90_incontextv12_finetune"

    DEBUG_PAPER_V12 = "pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split"
    GET_VIDEO = "pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split"
    RANDOM_INIT_NO_LORA = "pi0_libero_incontextv12_random_init_train_split"
    
    # new
    CLEAN_INFERENCE = "pi0_libero_incontextv12_low_mem_finetune_clean_stage_wise_prompt_train_all"
    NOISY_INFERENCE = "pi0_libero_incontextv12_low_mem_finetune_noisy_stage_wise_prompt_train_all"

    
@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
    EnvMode.LIBERO_FM_LORA_INCONTEXT: Checkpoint(
        config="pi0_libero_incontext_low_mem_finetune",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune/pi0_libero_incontext_low_mem_finetune/10000"
    ),
    EnvMode.LIBERO_FM_LORA_INCONTEXT_SAMPLE2: Checkpoint(
        config="pi0_libero_incontext_low_mem_finetune_sample2",
        # dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune_sample2/pi0_libero_incontext_low_mem_finetune_sample2/19999"
        # dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune_sample2/pi0_libero_incontext_low_mem_finetune_sample2_actionsample_32/39999"
        # dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune_sample2/pi0_libero_incontext_low_mem_finetune_sample2_actionsample_32/30000"
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune_sample2/pi0_libero_incontext_low_mem_finetune_sample2_actionsample_8_random_select_false/39999"
    ),
    EnvMode.TEST1: Checkpoint(
        config="pi0_libero_incontext_low_mem_finetune_sample2_actionssample32",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune_sample2_actionssample32/pi0_libero_incontext_low_mem_finetune_sample2_actionssample32/39999"
    ),
    EnvMode.TEST2: Checkpoint(
        config="pi0_libero_incontext_low_mem_finetune_sample2_random_select",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune_sample2_random_select/pi0_libero_incontext_low_mem_finetune_sample2_random_select/30000"
    ),
    EnvMode.TEST3: Checkpoint(
        config="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32/39999"
    ),
    EnvMode.TEST4: Checkpoint(
        config="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample8",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample8/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample8/30000"
    ),
    EnvMode.TEST5: Checkpoint(
        config="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select/19999"
    ),
    
    EnvMode.XJ_TEST3_TRAIN: Checkpoint(
        config="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split",
        dir="/ibex/tmp/c2090/openpi_explore_storage//checkpoints/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split/19999"
    ),
    EnvMode.XJ_TEST6_WITHOUT_LORA: Checkpoint(
        config="pi0_libero_incontextv2_sample2_actionssample64",
        dir="/ibex/tmp/c2090/openpi_explore_storage//checkpoints/pi0_libero_incontextv2_sample2_actionssample64/pi0_libero_incontextv2_sample2_actionssample64/39999"
    ),

    EnvMode.TEST6: Checkpoint(
        config="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64/39999"
    ),
    EnvMode.XJ_TEST6_WITHOUT_LORA_WITH_SPLIT: Checkpoint(
        config="pi0_libero_incontextv2_sample2_actionssample64_test_split",
        dir="/ibex/tmp/c2090/openpi_explore_storage/checkpoints/pi0_libero_incontextv2_sample2_actionssample64_train_split/pi0_libero_incontextv2_sample2_actionssample64_train_split/39999"
    ),
    EnvMode.XJ_TEST6_WITHOUT_LORA_WITH_SPLIT_20K: Checkpoint(
        config="pi0_libero_incontextv2_sample2_actionssample64_test_split",
        dir="/ibex/tmp/c2090/openpi_explore_storage/checkpoints/pi0_libero_incontextv2_sample2_actionssample64_train_split/pi0_libero_incontextv2_sample2_actionssample64_train_split_20k/19999"
    ),
    EnvMode.XJ_PI0LIGHT_V12_20K: Checkpoint(
        config="pi0light_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        dir="/ibex/tmp/c2090/openpi_explore_storage/checkpoints/pi0light_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/pi0light_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/19999"
    ),
    # supplementary
    EnvMode.XJ_V12_V3: Checkpoint(
        config="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v3/pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v3/19999"
    ),
    EnvMode.XJ_V12_V4: Checkpoint(
        config="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v4/pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v4/19999"
    ),   
    EnvMode.XJ_ROBOCASA_INFERENCE: Checkpoint(
        config="pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train/pi0mini_incontext_robocasa_human_three_image_low_mem_finetune_train/499999"
    ),  
    EnvMode.XJ_PI0MINI_INCONTEXT_ROBOCASA_MG_INFERENCE: Checkpoint(
        config="pi0mini_incontext_robocasa_mg_three_image_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontext_robocasa_mg_three_image_low_mem_finetune_inference/pi0mini_incontext_robocasa_mg_three_image_low_mem_finetune_inference/499999"
    ),  
    # rebutal
    EnvMode.XJ_PI0MINI_ROBOCASA_MG_INCONTEXT: Checkpoint(
        config="pi0mini_incontext_robocasa_mg_three_image_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontext_robocasa_mg_three_image_low_mem_finetune/pi0mini_incontext_robocasa_mg_three_image_low_mem_finetune/499999"
    ),  
    EnvMode.XJ_PI0_MINI_LIBERO_INCONTEXT: Checkpoint(
        config="pi0mini_incontext_libero_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontext_libero_low_mem_finetune_train/pi0mini_incontext_libero_low_mem_finetune_train/19999"
    ), 
    EnvMode.XJ_PI0_MINI_LIBERO_INCONTEXT_ALL: Checkpoint(
        config="pi0mini_incontext_libero_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontext_libero_low_mem_finetune_inference/pi0mini_incontext_libero_low_mem_finetune_inference/19999"
    ), 
    EnvMode.XJ_PI0_INCONTEXT_ROBOCASA_MG: Checkpoint(
        config="pi0_incontext_robocasa_mg_three_image_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_incontext_robocasa_mg_three_image_low_mem_finetune_train/pi0_incontext_robocasa_mg_three_image_low_mem_finetune_train/499999"
    ), 
    EnvMode.XJ_PI0MINI_INCONTEXTV12_1_ROBOCASA_MG_INFERENCE: Checkpoint(
        config="pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_inference/pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_inference/999999"
    ), 
    EnvMode.XJ_PI0MINI_INCONTEXTV12_1_ROBOCASA_MG_TRAIN: Checkpoint(
        config="pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_train/pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_train/499999"
    ), 
    EnvMode.XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_TRAIN: Checkpoint(
        config="pi0tiny_incontext_robocasa_mg_three_image_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0tiny_incontext_robocasa_mg_three_image_train_split/pi0tiny_incontext_robocasa_mg_three_image_train_split/499999"
    ), 
    ####
    EnvMode.XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_TRAIN_1M: Checkpoint(
        config="pi0tiny_incontext_robocasa_mg_three_image_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0tiny_incontext_robocasa_mg_three_image_train_split/pi0tiny_incontext_robocasa_mg_three_image_train_split_1M/500000"
    ), 
    EnvMode.XJ_PI0MINI_INCONTEXTV12_1_ROBOCASA_MG_TRAIN_1M: Checkpoint(
        config="pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_train/pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_train_1M/500000"
    ), 
    # rebuttal
    EnvMode.XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_TRAIN_LR: Checkpoint(
        config="pi0tiny_incontext_robocasa_mg_three_image_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0tiny_incontext_robocasa_mg_three_image_train_split_large_lr/pi0tiny_incontext_robocasa_mg_three_image_train_split_large_lr/499999"
    ), 
    EnvMode.XJ_PI0MINI_INCONTEXTV12_1_ROBOCASA_MG_TRAIN_LR: Checkpoint(
        config="pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_train_large_lr/pi0mini_incontextv12_1_robocasa_mg_three_image_low_mem_finetune_train_large_lr/499999"
    ), 
    
    ## debug pi0mini incontext robocasa mg
    # no prompt at all = non-incontext 
    EnvMode.DEBUG_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_TRAIN_SPLIT: Checkpoint(
        config="debug_pi0tiny_incontext_robocasa_mg_three_image_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/debug_pi0tiny_incontext_robocasa_mg_three_image_train_split/debug_pi0tiny_incontext_robocasa_mg_three_image_train_split/499999"
    ), 
    EnvMode.DEBUG_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_INFERENCE: Checkpoint(
        config="debug_pi0tiny_incontext_robocasa_mg_three_image_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/debug_pi0tiny_incontext_robocasa_mg_three_image_inference/debug_pi0tiny_incontext_robocasa_mg_three_image_inference/499999"
    ), 
    # image + action/state prompts
    EnvMode.DEBUG_PROMPT_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_INFERENCE: Checkpoint(
        config="debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference/debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference/499999"
    ),  
    EnvMode.DEBUG_PROMPT_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_TRAIN_SPLIT: Checkpoint(
        config="debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_train_split/debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_train_split/499999"
    ),
    # image + action/state prompts without random select (train on all tasks)
    EnvMode.DEBUG_PROMPT_NO_RANDOM_SELECT_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_INFERENCE: Checkpoint(
        config="debug_prompt_no_random_select_pi0tiny_incontext_robocasa_mg_three_image_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/debug_prompt_no_random_select_pi0tiny_incontext_robocasa_mg_three_image_inference/debug_prompt_no_random_select_pi0tiny_incontext_robocasa_mg_three_image_inference/499999"
    ),
    # proprio: only action/state prompt
    EnvMode.DEBUG_PROPRIO_PROMPT_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_TRAIN_SPLIT: Checkpoint(
        config="debug_proprio_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/debug_proprio_prompt_pi0tiny_incontext_robocasa_mg_three_image_train_split/debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_train_split/499999"
    ),
    EnvMode.DEBUG_PROPRIO_PROMPT_XJ_PI0TINY_INCONTEXTV_ROBOCASA_MG_INFERENCE: Checkpoint(
        config="debug_proprio_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/debug_proprio_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference/debug_proprio_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference/499999"
    ),
    
    # final
    EnvMode.FINAL_PI0TINY_BOOST_IMG_PROMPT_INCONTEXT_LARGE_LR_1M_TRAIN_SPLIT_500K_CKP: Checkpoint(
        config="final_boost_img_prompt_pi0tiny_incontext_robocasa_mg_three_image_large_lr_1M_all",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/final_boost_img_prompt_pi0tiny_incontext_robocasa_mg_three_image_large_lr_1M_train_split/final_boost_img_prompt_pi0tiny_incontext_robocasa_mg_three_image_large_lr_1M_train_split/499999"
    ),
    
    # final_2
    EnvMode.FINAL_PI0TINY_INCONTEXT_LARGEST_LR_TRAIN_SPLIT: Checkpoint(
        config="final_pi0tiny_incontext_robocasa_mg_three_image_largest_lr_train_all",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/final_pi0tiny_incontext_robocasa_mg_three_image_largest_lr_train_split/final_pi0tiny_incontext_robocasa_mg_three_image_largest_lr_train_split/499999"
    ),
    EnvMode.FINAL_PI0TINY_INCONTEXT_1M_TRAIN_SPLIT_1M_CKP: Checkpoint(
        config="final_pi0tiny_incontext_robocasa_mg_three_image_largest_lr_train_all",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0tiny_incontext_robocasa_mg_three_image_large_lr_train_split_1M/pi0tiny_incontext_robocasa_mg_three_image_large_lr_train_split_1M/999999"
    ),
    EnvMode.FINAL_PI0TINY_INCONTEXT_1M_TRAIN_SPLIT_500K_CKP: Checkpoint(
        config="final_pi0tiny_incontext_robocasa_mg_three_image_largest_lr_train_all",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0tiny_incontext_robocasa_mg_three_image_large_lr_train_split_1M/pi0tiny_incontext_robocasa_mg_three_image_large_lr_train_split_1M/500000"
    ),
    EnvMode.DEBUG_IMG_ENCODER: Checkpoint(
        config="debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/debug_img_encoder/debug_img_encoder/499999"
    ), 
    EnvMode.DEBUG_LOW_ACTION_HORIZON: Checkpoint(
        config="debug_low_action_horizon_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/debug_low_action_horizon/debug_low_action_horizon/499999"
    ),
    EnvMode.DEBUG_TRAIN_WITHOUT_OPENDOUBLEDOOR: Checkpoint(
        config="debug_prompt_pi0tiny_incontext_robocasa_mg_three_image_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/debug_train_without_open_double_door/debug_train_without_open_double_door/499999"
    ),
    
    # final_3
    EnvMode.XJ_PI0_LIBERO90_INCONTEXTV12_LOW_MEM_FINETUNE: Checkpoint(
        config="pi0_libero90_incontextv12_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero90_incontextv12_low_mem_finetune/pi0_libero90_incontextv12_low_mem_finetune/29999"
    ),
    EnvMode.PI0_LIBERO_MORE_SAMPLE_FRAMES: Checkpoint(
        config="pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_train_split_v1/pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_train_split_v1/19999"
    ),
    EnvMode.POINT_TRACK: Checkpoint(
        config="pi0_libero_incontextv12_point_track_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_point_track_low_mem_finetune_train_split/pi0_libero_incontextv12_point_track_low_mem_finetune_train_split/19999"
    ),
    EnvMode.VIDEO_PROMPT: Checkpoint(
        config="pi0_libero_incontextv12_video_prompt_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_video_prompt_low_mem_finetune_train_split/pi0_libero_incontextv12_video_prompt_low_mem_finetune_train_split/19999"
    ),
    EnvMode.RANDOM_INIT: Checkpoint(
        config="pi0_libero_incontextv12_low_mem_finetune_random_init_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_low_mem_finetune_random_init_train_split/pi0_libero_incontextv12_low_mem_finetune_random_init_train_split/19999"
    ),
    EnvMode.PI0_LIBERO_90_NONE_LORA: Checkpoint(
        config="pi0_libero90_incontextv12_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero90_incontextv12_finetune/pi0_libero90_incontextv12_finetune/29999"
    ),
    EnvMode.PI0_LIBERO_90_NONE_LORA_LIBERO90: Checkpoint(
        config="pi0_libero90_incontextv12_finetune",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero90_incontextv12_finetune/pi0_libero90_incontextv12_finetune/29999"
    ),
    EnvMode.DEBUG_PAPER_V12: Checkpoint(
        config="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/google-cloud-exp/pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/19999"
    ),
    EnvMode.GET_VIDEO: Checkpoint(
        config="pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/19999"
    ),
    
    EnvMode.RANDOM_INIT_NO_LORA: Checkpoint(
        config="pi0_libero_incontextv12_random_init_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_random_init_train_split/pi0_libero_incontextv12_random_init_train_split/19999"
    ),
    
    ## new
    EnvMode.CLEAN_INFERENCE: Checkpoint(
        config="pi0_libero_incontextv12_low_mem_finetune_clean_stage_wise_prompt_train_all",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_low_mem_finetune_clean_stage_wise_prompt_train_all/pi0_libero_incontextv12_low_mem_finetune_clean_stage_wise_prompt_train_all/19999"
    ),
    EnvMode.NOISY_INFERENCE: Checkpoint(
        config="pi0_libero_incontextv12_low_mem_finetune_noisy_stage_wise_prompt_train_all",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_low_mem_finetune_noisy_stage_wise_prompt_train_all/pi0_libero_incontextv12_low_mem_finetune_noisy_stage_wise_prompt_train_all/19999"
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy_incontext(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy_incontext(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy_incontext(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy_incontext(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
