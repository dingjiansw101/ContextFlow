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
    LIBERO_FM = "pi0_libero"
    LIBERO_ZERO = "pi0_libero_zero"
    LIBERO_FM_LORA = "pi0_libero_low_mem_finetune"
    ALOHA_HANDOVER = "aloha_handover"
    ALOHA_MOBILE = "trossen_mobile"
    ALOHA_HANDOVER_LOW_MEM = "aloha_handover_low_mem"
    FAST_ALOHA_HANDOVER = "pi0_fast_aloha_handover"
    ALOHA_PEN_UNCAP_LOW_MEM = "pi0_aloha_pen_uncap_low_mem_finetune"
    ALOHA_PEN_UNCAP_B5_LOW_MEM = "pi0_aloha_pen_uncap_b5_low_mem_finetune"
    ALOHA_FOLD_TSHIRT_LOW_MEM = "pi0_aloha_fold_tshirt_low_mem_finetune"
    FAST_ALOHA_PEN_UNCAP_B5 = "pi0_fast_aloha_pen_uncap_b5"
    FAST_ALOHA_PEN_UNCAP_LOW_MEM = "pi0_fast_aloha_pen_uncap_low_mem_finetune_bs30"
    FAST_ALOHA_PEN_UNCAP_LOW_MEM_CUSTOM_NORM = "pi0_fast_aloha_pen_uncap_low_mem_finetune"
    ALOHA_PEN_UNCAP_B5_LOW_MEM_TROSSWN_NORM = "pi0_aloha_pen_uncap_b5_low_mem_finetune_trossen_norm"
    FAST_ALOHA_PEN_UNCAP_B5_TROSSEN_NORM = "pi0_fast_aloha_pen_uncap_b5_trossen_norm"
    XJ_BASE_WITH_DELTA_WITH_SPLIT_LORA = "pi0_libero_low_mem_finetune_split_train"

    # supplementary non-in-context
    XJ_PI0_V3 = "pi0_libero_low_mem_finetune_split_train_v3"
    XJ_PI0_V4 = "pi0_libero_low_mem_finetune_split_train_v4"

    # final 3 non-in-context
    PI0_LIBERO_90_NONE_INCONTEXT = "pi0_libero90_finetune"
    TEST_V100 = "pi0_libero_low_mem_finetune_without_delta_train_split_v2"

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
    XJ_TEST6_WITHOUT_LORA_WITH_SPLIT_20K = "pi0_libero_incontextv2_sample2_actionssample64_train_split"  # noqa: PIE796
    XJ_PI0LIGHT_V12_20K = "pi0light_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference"

    # supplementary
    XJ_V12_V3 = (
        "pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v3"
    )
    XJ_V12_V4 = (
        "pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v4"
    )

    # rebutal
    XJ_PI0_MINI_LIBERO_INCONTEXT = "pi0mini_incontext_libero_low_mem_finetune_inference"
    XJ_PI0_MINI_LIBERO_INCONTEXT_ALL = "pi0mini_incontext_libero_low_mem_finetune_inference"  # noqa: PIE796

    # final_3
    XJ_PI0_LIBERO90_INCONTEXTV12_LOW_MEM_FINETUNE = "pi0_libero90_incontextv12_low_mem_finetune"
    PI0_LIBERO_MORE_SAMPLE_FRAMES = "pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_train_split_v1"
    RANDOM_INIT = "pi0_libero_incontextv12_low_mem_finetune_random_init_train_split"
    PI0_LIBERO_90_NONE_LORA = "pi0_libero90_incontextv12_finetune_x"
    PI0_LIBERO_90_NONE_LORA_LIBERO90 = "pi0_libero90_incontextv12_finetune"

    GET_VIDEO = (
        "pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split"
    )
    RANDOM_INIT_NO_LORA = "pi0_libero_incontextv12_random_init_train_split"


class LoaderMode(enum.Enum):
    """Policy construction mode."""

    AUTO = "auto"
    NORMAL = "normal"
    INCONTEXT = "incontext"


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

    # Policy loader to use. Auto chooses the in-context loader for in-context model configs.
    loader: LoaderMode = LoaderMode.AUTO

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
    EnvMode.LIBERO_FM: Checkpoint(
        config="pi0_libero",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.LIBERO_ZERO: Checkpoint(
        config="pi0_libero_zero",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.LIBERO_FM_LORA: Checkpoint(
        config="pi0_libero_low_mem_finetune",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_low_mem_finetune/pi0_libero_low_mem_finetune/20000",
    ),
    EnvMode.ALOHA_HANDOVER: Checkpoint(
        config="pi0_aloha_handover",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_aloha_handover/pi0_aloha_handover/19999",
    ),
    EnvMode.ALOHA_MOBILE: Checkpoint(
        config="pi0_aloha_mobile",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_HANDOVER_LOW_MEM: Checkpoint(
        config="pi0_aloha_handover_low_mem_finetune",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_aloha_handover_low_mem_finetune/pi0_aloha_handover_low_mem_finetune/10000",
    ),
    EnvMode.FAST_ALOHA_HANDOVER: Checkpoint(
        config="pi0_fast_aloha_handover",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_fast_aloha_handover/pi0_fast_aloha_handover/19999",
    ),
    EnvMode.ALOHA_PEN_UNCAP_LOW_MEM: Checkpoint(
        config="pi0_aloha_pen_uncap_low_mem_finetune",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_aloha_pen_uncap_low_mem_finetune/pi0_aloha_pen_uncap_low_mem_finetune/10000",
    ),
    EnvMode.ALOHA_FOLD_TSHIRT_LOW_MEM: Checkpoint(
        config="pi0_aloha_fold_tshirt_low_mem_finetune",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_aloha_fold_tshirt_low_mem_finetune/pi0_aloha_fold_tshirt_low_mem_finetune/10000",
    ),
    EnvMode.FAST_ALOHA_PEN_UNCAP_LOW_MEM: Checkpoint(
        config="pi0_fast_aloha_pen_uncap_low_mem_finetune_bs30",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_fast_aloha_pen_uncap_low_mem_finetune_bs30/pi0_fast_aloha_pen_uncap_low_mem_finetune_bs30/19999",
    ),
    EnvMode.ALOHA_PEN_UNCAP_B5_LOW_MEM: Checkpoint(
        config="pi0_aloha_pen_uncap_b5_low_mem_finetune",
        dir="checkpoints/pi0_aloha_pen_uncap_b5_low_mem_finetune/pi0_aloha_pen_uncap_b5_low_mem_finetune/9999",
    ),
    EnvMode.FAST_ALOHA_PEN_UNCAP_LOW_MEM_CUSTOM_NORM: Checkpoint(
        config="pi0_fast_aloha_pen_uncap_low_mem_finetune",
        dir="checkpoints/pi0_fast_aloha_pen_uncap_low_mem_finetune/pi0_fast_aloha_pen_uncap_low_mem_finetune/9999",
    ),
    EnvMode.ALOHA_PEN_UNCAP_B5_LOW_MEM_TROSSWN_NORM: Checkpoint(
        config="pi0_aloha_pen_uncap_b5_low_mem_finetune_trossen_norm",
        dir="checkpoints/pi0_aloha_pen_uncap_b5_low_mem_finetune_trossen_norm/pi0_aloha_pen_uncap_b5_low_mem_finetune_trossen_norm/10000",
    ),
    EnvMode.FAST_ALOHA_PEN_UNCAP_B5: Checkpoint(
        config="pi0_fast_aloha_pen_uncap_b5",
        dir="checkpoints/pi0_fast_aloha_pen_uncap_b5/pi0_fast_aloha_pen_uncap_b5/9999",
    ),
    EnvMode.FAST_ALOHA_PEN_UNCAP_B5_TROSSEN_NORM: Checkpoint(
        config="pi0_fast_aloha_pen_uncap_b5_trossen_norm",
        dir="checkpoints/pi0_fast_aloha_pen_uncap_b5_trossen_norm/pi0_fast_aloha_pen_uncap_b5_trossen_norm/9999",
    ),
    EnvMode.XJ_BASE_WITH_DELTA_WITH_SPLIT_LORA: Checkpoint(
        config="pi0_libero_low_mem_finetune_split_train",
        dir="./checkpoints/pi0_libero_low_mem_finetune_split_train/pi0_libero_low_mem_finetune_split_train/19999",
    ),
    EnvMode.XJ_PI0_V3: Checkpoint(
        config="pi0_libero_low_mem_finetune_split_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_low_mem_finetune_split_train_v3/pi0_libero_low_mem_finetune_split_train_v3/19999",
    ),
    EnvMode.XJ_PI0_V4: Checkpoint(
        config="pi0_libero_low_mem_finetune_split_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_low_mem_finetune_split_train_v4/pi0_libero_low_mem_finetune_split_train_v4/19999",
    ),
    EnvMode.PI0_LIBERO_90_NONE_INCONTEXT: Checkpoint(
        config="pi0_libero90_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero90_low_mem_finetune/pi0_libero90_low_mem_finetune/29999",
    ),
    EnvMode.TEST_V100: Checkpoint(
        config="pi0_libero_low_mem_finetune_split_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/google-cloud-exp/pi0_libero_low_mem_finetune_without_delta_train_split_v2/pi0_libero_low_mem_finetune_without_delta_train_split_v2/19999",
    ),
    EnvMode.LIBERO_FM_LORA_INCONTEXT: Checkpoint(
        config="pi0_libero_incontext_low_mem_finetune",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune/pi0_libero_incontext_low_mem_finetune/10000",
    ),
    EnvMode.LIBERO_FM_LORA_INCONTEXT_SAMPLE2: Checkpoint(
        config="pi0_libero_incontext_low_mem_finetune_sample2",
        # dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune_sample2/pi0_libero_incontext_low_mem_finetune_sample2/19999"
        # dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune_sample2/pi0_libero_incontext_low_mem_finetune_sample2_actionsample_32/39999"
        # dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune_sample2/pi0_libero_incontext_low_mem_finetune_sample2_actionsample_32/30000"
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune_sample2/pi0_libero_incontext_low_mem_finetune_sample2_actionsample_8_random_select_false/39999",
    ),
    EnvMode.TEST1: Checkpoint(
        config="pi0_libero_incontext_low_mem_finetune_sample2_actionssample32",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune_sample2_actionssample32/pi0_libero_incontext_low_mem_finetune_sample2_actionssample32/39999",
    ),
    EnvMode.TEST2: Checkpoint(
        config="pi0_libero_incontext_low_mem_finetune_sample2_random_select",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune_sample2_random_select/pi0_libero_incontext_low_mem_finetune_sample2_random_select/30000",
    ),
    EnvMode.TEST3: Checkpoint(
        config="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32/39999",
    ),
    EnvMode.TEST4: Checkpoint(
        config="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample8",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample8/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample8/30000",
    ),
    EnvMode.TEST5: Checkpoint(
        config="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select/19999",
    ),
    EnvMode.XJ_TEST3_TRAIN: Checkpoint(
        config="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split",
        dir="/ibex/tmp/c2090/openpi_explore_storage//checkpoints/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample32_random_select_train_split/19999",
    ),
    EnvMode.XJ_TEST6_WITHOUT_LORA: Checkpoint(
        config="pi0_libero_incontextv2_sample2_actionssample64",
        dir="/ibex/tmp/c2090/openpi_explore_storage//checkpoints/pi0_libero_incontextv2_sample2_actionssample64/pi0_libero_incontextv2_sample2_actionssample64/39999",
    ),
    EnvMode.TEST6: Checkpoint(
        config="pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64/pi0_libero_incontextv2_low_mem_finetune_sample2_actionssample64/39999",
    ),
    EnvMode.XJ_TEST6_WITHOUT_LORA_WITH_SPLIT: Checkpoint(
        config="pi0_libero_incontextv2_sample2_actionssample64_test_split",
        dir="/ibex/tmp/c2090/openpi_explore_storage/checkpoints/pi0_libero_incontextv2_sample2_actionssample64_train_split/pi0_libero_incontextv2_sample2_actionssample64_train_split/39999",
    ),
    EnvMode.XJ_TEST6_WITHOUT_LORA_WITH_SPLIT_20K: Checkpoint(
        config="pi0_libero_incontextv2_sample2_actionssample64_test_split",
        dir="/ibex/tmp/c2090/openpi_explore_storage/checkpoints/pi0_libero_incontextv2_sample2_actionssample64_train_split/pi0_libero_incontextv2_sample2_actionssample64_train_split_20k/19999",
    ),
    EnvMode.XJ_PI0LIGHT_V12_20K: Checkpoint(
        config="pi0light_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        dir="/ibex/tmp/c2090/openpi_explore_storage/checkpoints/pi0light_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/pi0light_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/19999",
    ),
    # supplementary
    EnvMode.XJ_V12_V3: Checkpoint(
        config="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v3/pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v3/19999",
    ),
    EnvMode.XJ_V12_V4: Checkpoint(
        config="pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v4/pi0_libero_incontextv12_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_v4/19999",
    ),
    EnvMode.XJ_PI0_MINI_LIBERO_INCONTEXT: Checkpoint(
        config="pi0mini_incontext_libero_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontext_libero_low_mem_finetune_train/pi0mini_incontext_libero_low_mem_finetune_train/19999",
    ),
    EnvMode.XJ_PI0_MINI_LIBERO_INCONTEXT_ALL: Checkpoint(
        config="pi0mini_incontext_libero_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0mini_incontext_libero_low_mem_finetune_inference/pi0mini_incontext_libero_low_mem_finetune_inference/19999",
    ),
    # final_3
    EnvMode.XJ_PI0_LIBERO90_INCONTEXTV12_LOW_MEM_FINETUNE: Checkpoint(
        config="pi0_libero90_incontextv12_low_mem_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero90_incontextv12_low_mem_finetune/pi0_libero90_incontextv12_low_mem_finetune/29999",
    ),
    EnvMode.PI0_LIBERO_MORE_SAMPLE_FRAMES: Checkpoint(
        config="pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_train_split_v1/pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_train_split_v1/19999",
    ),
    EnvMode.RANDOM_INIT: Checkpoint(
        config="pi0_libero_incontextv12_low_mem_finetune_random_init_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_low_mem_finetune_random_init_train_split/pi0_libero_incontextv12_low_mem_finetune_random_init_train_split/19999",
    ),
    EnvMode.PI0_LIBERO_90_NONE_LORA: Checkpoint(
        config="pi0_libero90_incontextv12_finetune_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero90_incontextv12_finetune/pi0_libero90_incontextv12_finetune/29999",
    ),
    EnvMode.PI0_LIBERO_90_NONE_LORA_LIBERO90: Checkpoint(
        config="pi0_libero90_incontextv12_finetune",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero90_incontextv12_finetune/pi0_libero90_incontextv12_finetune/29999",
    ),
    EnvMode.GET_VIDEO: Checkpoint(
        config="pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/pi0_libero_incontextv12_1_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/19999",
    ),
    EnvMode.RANDOM_INIT_NO_LORA: Checkpoint(
        config="pi0_libero_incontextv12_random_init_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_incontextv12_random_init_train_split/pi0_libero_incontextv12_random_init_train_split/19999",
    ),
}


def _is_incontext_config(train_config: _config.TrainConfig) -> bool:
    model_config = train_config.model
    model_type = type(model_config)
    if "incontext" in model_type.__name__.lower() or "incontext" in model_type.__module__.lower():
        return True
    return bool(
        getattr(model_config, "use_image_prompts", False) or getattr(model_config, "use_action_state_prompts", False)
    )


def create_policy_from_checkpoint(
    checkpoint: Checkpoint,
    *,
    default_prompt: str | None = None,
    loader: LoaderMode = LoaderMode.AUTO,
) -> _policy.Policy:
    train_config = _config.get_config(checkpoint.config)
    use_incontext = loader == LoaderMode.INCONTEXT or (loader == LoaderMode.AUTO and _is_incontext_config(train_config))

    if use_incontext:
        return _policy_config.create_trained_policy_incontext(
            train_config,
            checkpoint.dir,
            default_prompt=default_prompt,
            inference_dtype=checkpoint.inference_dtype,
        )

    return _policy_config.create_trained_policy(
        train_config,
        checkpoint.dir,
        default_prompt=default_prompt,
    )


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    loader: LoaderMode = LoaderMode.AUTO,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return create_policy_from_checkpoint(checkpoint, default_prompt=default_prompt, loader=loader)
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return create_policy_from_checkpoint(
                args.policy,
                default_prompt=args.default_prompt,
                loader=args.loader,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt, loader=args.loader)


def main(args: Args) -> None:
    policy = create_policy(args)
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
