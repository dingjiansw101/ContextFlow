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

    # rebutal
    XJ_PI0_MINI_LIBERO_INCONTEXT = "pi0mini_incontext_libero_low_mem_finetune_inference"
    XJ_PI0_MINI_LIBERO_INCONTEXT_ALL = "pi0mini_incontext_libero_low_mem_finetune_inference"

    # final_3
    XJ_PI0_LIBERO90_INCONTEXTV12_LOW_MEM_FINETUNE = "pi0_libero90_incontextv12_low_mem_finetune"
    PI0_LIBERO_MORE_SAMPLE_FRAMES = "pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_train_split_v1"
    RANDOM_INIT = "pi0_libero_incontextv12_low_mem_finetune_random_init_train_split"
    PI0_LIBERO_90_NONE_LORA = "pi0_libero90_incontextv12_finetune_x"
    PI0_LIBERO_90_NONE_LORA_LIBERO90 = "pi0_libero90_incontextv12_finetune"

    GET_VIDEO = "pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split"
    RANDOM_INIT_NO_LORA = "pi0_libero_incontextv12_random_init_train_split"

    # new
    CLEAN_INFERENCE = "pi0_libero_incontextv12_low_mem_finetune_clean_stage_wise_prompt_train_all"
    NOISY_INFERENCE = "pi0_libero_incontextv12_low_mem_finetune_noisy_stage_wise_prompt_train_all"

    # cvpr
    SEQUENCE_DEBUG = "sequence_debug_pi0_libero_incontextv14_train_split_v3"
    SEQUENCE_DEBUG_PI0MINI_INCONTEXTV14_LIBEROV1 = "sequence_debug_pi0mini_libero_incontextv14_train_split_v1"
    NO_SEQ_AVG = "no_sequence_avg_cur_img_debug_pi0mini_libero_incontextv14_train_split_v1"
    NO_SEQ_NO_AVG = "no_sequence_no_avg_debug_pi0mini_libero_incontextv14_train_split_v1"
    SEQ_AVG = "sequence_avg_cur_img_debug_pi0mini_libero_incontextv14_train_split_v1"
    SEQ_NO_AVG = "sequence_no_avg_debug_pi0mini_libero_incontextv14_train_split_v1"
    NO_SEQ_NO_AVG_V12_ACTION_TOKEN = "no_sequence_no_avg_debug_pi0mini_libero_incontextv12_train_split_v1"
    NO_SEQ_NO_AVG_V12_PROMPT_TOKEN = "no_sequence_no_avg_prompt_token_debug_pi0mini_libero_incontextv12_train_split_v1"
    # improve
    SEQ_AVG_36_SEQ = "sequence_avg_cur_img_36_seq_pi0mini_libero_incontextv14_train_split_v1"
    
    # v14
    seq_avg_12 = "12_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_avg_24 = "24_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_avg_48 = "48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_avg_96 = "96_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_no_avg_12 = "12_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_no_avg_24 = "24_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_no_avg_48 = "48_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_avg_6_prompt_img_8 = "8_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_avg_6_prompt_img_16 = "16_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_avg_6_prompt_img_32 = "32_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"

    seq_avg_bs_8_seq_48 = "bs_8_seq_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_avg_bs_4_seq_96 = "bs_4_seq_96_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    
    seq_avg_6_40k = "40k_ite_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_avg_6_60k = "60k_ite_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_avg_6_80k = "80k_ite_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_avg_6_vitb = "vitb_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_avg_6_vitb_95m = "vitb_95m_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_avg_6_vits_95m = "vits_95m_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"

    seq_avg_6_vitb_living_95m = "sequence_avg_pi0mini_libero90_living_incontextv14_train"
    seq_avg_6_vitb_kitchen_95m = "sequence_avg_pi0mini_libero90_kitchen_incontextv14_train"
    seq_avg_6_vitb_study_95m = "sequence_avg_pi0mini_libero90_study_incontextv14_train"
    seq_avg_6_vitb_object_95m = "sequence_avg_pi0mini_libero90_object_incontextv14_train"
    
    seq_no_avg_12_vitb_95m_split1 = "vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1"
    seq_no_avg_12_vitb_95m_split2 = "vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v2"
    seq_no_avg_12_vitb_95m_split3 = "vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v3"
    seq_no_avg_12_vitb_95m_split4 = "vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v4"

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # Optional: Override inference dtype (e.g., "float32", "bfloat16"). Defaults to bfloat16 if not specified.
    inference_dtype: str | None = None


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
    EnvMode.XJ_PI0_MINI_LIBERO_INCONTEXT: Checkpoint(
        config="pi0mini_incontext_libero_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontext_libero_low_mem_finetune_train/pi0mini_incontext_libero_low_mem_finetune_train/19999"
    ),
    EnvMode.XJ_PI0_MINI_LIBERO_INCONTEXT_ALL: Checkpoint(
        config="pi0mini_incontext_libero_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontext_libero_low_mem_finetune_inference/pi0mini_incontext_libero_low_mem_finetune_inference/19999"
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
    
    # cvpr
    EnvMode.SEQUENCE_DEBUG: Checkpoint(
        config="sequence_debug_pi0_libero_incontextv14_inference",
        # dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/sequence_debug_pi0_libero_incontextv14_train_split_v3/V3/19999"
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/sequence_debug_pi0_libero_incontextv14_train_split_v3/v1_same_proj/19999"

    ),
    # SEQ_NO_AVG
    EnvMode.SEQUENCE_DEBUG_PI0MINI_INCONTEXTV14_LIBEROV1: Checkpoint(
        config="sequence_debug_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/sequence_debug_pi0mini_libero_incontextv14_train_split_v1/sequence_debug_pi0mini_libero_incontextv14_train_split_v1/19999"
    ),
    EnvMode.NO_SEQ_AVG: Checkpoint(
        config="no_sequence_avg_cur_img_debug_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/no_sequence_avg_cur_img_debug_pi0mini_libero_incontextv14_train_split_v1/no_sequence_avg_cur_img_debug_pi0mini_libero_incontextv14_train_split_v1/19999"
    ),
    EnvMode.NO_SEQ_NO_AVG: Checkpoint(
        config="no_sequence_no_avg_debug_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/no_sequence_no_avg_debug_pi0mini_libero_incontextv14_train_split_v1/no_sequence_no_avg_debug_pi0mini_libero_incontextv14_train_split_v1/19999"
    ),
    EnvMode.SEQ_AVG: Checkpoint(
        config="sequence_avg_cur_img_debug_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/sequence_avg_cur_img_debug_pi0mini_libero_incontextv14_train_split_v1/sequence_avg_cur_img_debug_pi0mini_libero_incontextv14_train_split_v1/19999"
    ),
    EnvMode.SEQ_NO_AVG: Checkpoint(
        config="sequence_no_avg_debug_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/sequence_no_avg_debug_pi0mini_libero_incontextv14_train_split_v1/sequence_no_avg_debug_pi0mini_libero_incontextv14_train_split_v1/19999"
    ),
    EnvMode.NO_SEQ_NO_AVG_V12_ACTION_TOKEN: Checkpoint(
        config="no_sequence_no_avg_debug_pi0mini_libero_incontextv12_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/no_sequence_no_avg_debug_pi0mini_libero_incontextv12_train_split_v1/no_sequence_no_avg_debug_pi0mini_libero_incontextv12_train_split_v1/19999"
    ),
    EnvMode.NO_SEQ_NO_AVG_V12_PROMPT_TOKEN: Checkpoint(
        config="no_sequence_no_avg_prompt_token_debug_pi0mini_libero_incontextv12_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/no_sequence_no_avg_prompt_token_debug_pi0mini_libero_incontextv12_train_split_v1/no_sequence_no_avg_prompt_token_debug_pi0mini_libero_incontextv12_train_split_v1/19999"
    ),
    
    # improve
    EnvMode.SEQ_AVG_36_SEQ: Checkpoint(
        config="sequence_avg_cur_img_36_seq_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/sequence_avg_cur_img_36_seq_pi0mini_libero_incontextv14_train_split_v1/sequence_avg_cur_img_36_seq_pi0mini_libero_incontextv14_train_split_v1/39999"
    ),
    
    # v14
    #seq_avg_12 = "12_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    #seq_avg_24 = "24_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    #seq_avg_48 = "48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    #seq_avg_96 = "96_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    #seq_no_avg_12 = "12_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1"
    #seq_no_avg_24 = "24_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1"
    #seq_no_avg_48 = "48_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1"
    #seq_avg_6_prompt_img_8 = "8_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"

    EnvMode.seq_avg_12: Checkpoint(
        config="12_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/12_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_12/19999"
    ),
    EnvMode.seq_avg_24: Checkpoint(
        config="24_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/24_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_24/19999"
    ),
    EnvMode.seq_avg_48: Checkpoint(
        config="48_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_48/19999"
    ),
    EnvMode.seq_avg_96: Checkpoint(
        config="96_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/96_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_96/19999"
    ),
    EnvMode.seq_no_avg_12: Checkpoint(
        config="12_sequence_no_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/12_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1/seq_no_avg_12/19999"
    ),
    EnvMode.seq_no_avg_24: Checkpoint(
        config="24_sequence_no_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/24_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1/seq_no_avg_24/19999"
    ),
    EnvMode.seq_no_avg_48: Checkpoint(
        config="48_sequence_no_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/48_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1/seq_no_avg_48/19999"
    ),
    EnvMode.seq_avg_6_prompt_img_8: Checkpoint(
        config="8_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/8_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_6_prompt_img_8/19999"
    ),
    EnvMode.seq_avg_6_prompt_img_16: Checkpoint(
        config="16_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/16_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_6_prompt_img_16/19999"
    ),
    EnvMode.seq_avg_6_prompt_img_32: Checkpoint(
        config="32_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/32_prompt_img_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_6_prompt_img_32/19999"
    ),

    ### total_bs=384 variants; seq_avg
    EnvMode.seq_avg_bs_8_seq_48: Checkpoint(
        config="bs_8_seq_48_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/bs_8_seq_48_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_bs_8_seq_48/19999"
    ),
    EnvMode.seq_avg_bs_4_seq_96: Checkpoint(
        config="bs_4_seq_96_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/bs_4_seq_96_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_bs_4_seq_96/19999"
    ),
    # seq_avg_6_40k = "40k_ite_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    # seq_avg_6_60k = "60k_ite_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    # seq_avg_6_80k = "80k_ite_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    EnvMode.seq_avg_6_40k: Checkpoint(
        config="40k_ite_6_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/40k_ite_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_6_40k/39999"
    ),
    EnvMode.seq_avg_6_60k: Checkpoint(
        config="60k_ite_6_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/60k_ite_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_6_60k/59999"
    ),
    EnvMode.seq_avg_6_80k: Checkpoint(
        config="80k_ite_6_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/80k_ite_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_6_80k/79999"
    ),
    # seq_avg_6_vitb = "vitb_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    # seq_avg_6_vitb_95m = "vitb_95m_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    # seq_avg_6_vits_95m = "vits_95m_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1"
    EnvMode.seq_avg_6_vitb: Checkpoint(
        config="vitb_6_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/vitb_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_6_vitb/19999"
    ),
    EnvMode.seq_avg_6_vitb_95m: Checkpoint(
        config="vitb_95m_6_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/vitb_95m_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_6_vitb_95m/19999"
    ),
    EnvMode.seq_avg_6_vits_95m: Checkpoint(
        config="vits_95m_6_sequence_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/vits_95m_6_sequence_avg_pi0mini_libero_incontextv14_train_split_v1/seq_avg_6_vits_95m/19999"
    ),
    # seq_avg_6_vitb_living_95m = "sequence_avg_pi0mini_libero90_living_incontextv14_train"
    # seq_avg_6_vitb_kitchen_95m = "sequence_avg_pi0mini_libero90_kitchen_incontextv14_train"
    # seq_avg_6_vitb_study_95m = "sequence_avg_pi0mini_libero90_study_incontextv14_train"
    # seq_avg_6_vitb_object_95m = "sequence_avg_pi0mini_libero90_object_incontextv14_train"
    EnvMode.seq_avg_6_vitb_living_95m: Checkpoint(
        config="sequence_avg_pi0mini_libero90_living_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/sequence_avg_pi0mini_libero90_living_incontextv14_train/seq_avg_6_vitb_living_95m/19999"
    ),
    EnvMode.seq_avg_6_vitb_study_95m: Checkpoint(
        config="sequence_avg_pi0mini_libero90_study_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/sequence_avg_pi0mini_libero90_study_incontextv14_train/seq_avg_6_vitb_study_95m/19999"
    ),
    EnvMode.seq_avg_6_vitb_kitchen_95m: Checkpoint(
        config="sequence_avg_pi0mini_libero90_kitchen_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/sequence_avg_pi0mini_libero90_kitchen_incontextv14_train/seq_avg_6_vitb_kitchen_95m/19999"
    ),
    EnvMode.seq_avg_6_vitb_object_95m: Checkpoint(
        config="sequence_avg_pi0mini_libero_object_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/sequence_avg_pi0mini_libero_object_incontextv14_train/seq_avg_6_vitb_object_95m/19999"
    ),
    # seq_no_avg_12_vitb_95m_split1 = "vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1"
    # seq_no_avg_12_vitb_95m_split2 = "vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v2"
    # seq_no_avg_12_vitb_95m_split3 = "vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v3"
    # seq_no_avg_12_vitb_95m_split4 = "vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v4"
    EnvMode.seq_no_avg_12_vitb_95m_split1: Checkpoint(
        config="vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v1/seq_no_avg_12_vitb_95m_split1/19999"
    ),
    EnvMode.seq_no_avg_12_vitb_95m_split2: Checkpoint(
        config="vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v2/seq_no_avg_12_vitb_95m_split2/19999"
    ),
    EnvMode.seq_no_avg_12_vitb_95m_split3: Checkpoint(
        config="vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v3/seq_no_avg_12_vitb_95m_split3/19999"
    ),
    EnvMode.seq_no_avg_12_vitb_95m_split4: Checkpoint(
        config="vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/vitb_95m_6_sequence_no_avg_pi0mini_libero_incontextv14_train_split_v4/seq_no_avg_12_vitb_95m_split4/19999"
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy_incontext(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt, inference_dtype=checkpoint.inference_dtype
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy_incontext(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy_incontext(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt, inference_dtype=args.policy.inference_dtype
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
