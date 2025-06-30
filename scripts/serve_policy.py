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
    LIBERO_FM_LORA_INCONTEXT = "pi0_libero_incontext_low_mem_finetune"
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
    XJ_DEBUG_FROZEN_IMG_ENCODER_PI0 = "debug_libero_low_mem_finetune_split_train_inference"

    ## XJ: supplementary
    ROBOCASA_HUMAN_THREE_IMAGE = 'pi0_robocasa_human_three_image_low_mem_finetune_train'
    XJ_PI0MINI_ROBOCASA_HUMAN_THREE_IMAGE = "pi0mini_robocasa_human_three_image_low_mem_finetune_train"

    XJ_PI0_ROBOCASA_TURNONMICROWAVE_THREE_IMAGE = "pi0_robocasa_turnonmicrowave_three_image_low_mem_finetune_train"
    XJ_PI0MINI_ROBOCASA_TURNONMICROWAVE_THREE_IMAGE="pi0mini_robocasa_turnonmicrowave_three_image_low_mem_finetune_train"
    
    # supplementary
    XJ_PI0_V3 = "pi0_libero_low_mem_finetune_split_train_v3"
    XJ_PI0_V4 = "pi0_libero_low_mem_finetune_split_train_v4"
    
    # rebuttal
    XJ_PI0_ROBOCASA_MG_ALL = "pi0_robocasa_mg_three_image_low_mem_finetune_train"
    XJ_PI0_ROBOCASA_MG_TRAIN = "pi0_robocasa_mg_three_image_low_mem_finetune_train_split"


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
    # TODO: add mobile_trossen
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
        # dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero/pi0_libero/19999",
        # dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero/pi0_libero/29999",
        # dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero/pi0_libero/10000",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.LIBERO_ZERO: Checkpoint(
        config="pi0_libero_zero",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.LIBERO_FM_LORA: Checkpoint(
        config="pi0_libero_low_mem_finetune",
        # dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_low_mem_finetune/pi0_libero_low_mem_finetune/10000",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_low_mem_finetune/pi0_libero_low_mem_finetune/20000",
        # dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_low_mem_finetune/pi0_libero_low_mem_finetune/29999",
    ),
    EnvMode.LIBERO_FM_LORA_INCONTEXT: Checkpoint(
        config="pi0_libero_incontext_low_mem_finetune",
        dir="/home/dingj0b/code/openpi/checkpoints/pi0_libero_incontext_low_mem_finetune/pi0_libero_incontext_low_mem_finetune/10000"
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
        dir="./checkpoints/pi0_libero_low_mem_finetune_split_train/pi0_libero_low_mem_finetune_split_train/19999"
    ),
    EnvMode.XJ_DEBUG_FROZEN_IMG_ENCODER_PI0: Checkpoint(
        config="debug_libero_low_mem_finetune_split_train_inference",
        dir="./checkpoints/debug_libero_low_mem_finetune_split_train/debug_libero_low_mem_finetune_split_train/19999"
    ),
    
    # robocasa related
    EnvMode.ROBOCASA_HUMAN_THREE_IMAGE: Checkpoint(
        config="pi0_robocasa_human_three_image_low_mem_finetune_train",
        dir="./checkpoints/pi0_robocasa_human_three_image_low_mem_finetune_train/pi0_robocasa_human_three_image_low_mem_finetune_train/100000",
    ),
    EnvMode.XJ_PI0MINI_ROBOCASA_HUMAN_THREE_IMAGE: Checkpoint(
        config="pi0mini_robocasa_human_three_image_low_mem_finetune_train",
        dir="./checkpoints/pi0mini_robocasa_human_three_image_low_mem_finetune_train/pi0mini_robocasa_human_three_image_low_mem_finetune_train/499999"
    ),
    EnvMode.XJ_PI0_ROBOCASA_TURNONMICROWAVE_THREE_IMAGE: Checkpoint(
        config="pi0_robocasa_turnonmicrowave_three_image_low_mem_finetune_train",
        dir="/home/xianjid/project/pi0/openpi/checkpoints/pi0_robocasa_turnonmicrowave_three_image_low_mem_finetune_train/pi0_robocasa_turnonmicrowave_three_image_low_mem_finetune_train/19999"
    ),
    EnvMode.XJ_PI0MINI_ROBOCASA_TURNONMICROWAVE_THREE_IMAGE: Checkpoint(
        config="pi0mini_robocasa_turnonmicrowave_three_image_low_mem_finetune_train",
        dir="./checkpoints/pi0mini_robocasa_human_three_image_low_mem_finetune_train/pi0mini_robocasa_human_three_image_low_mem_finetune_train/499999"
    ),
    
    # supplementary
    EnvMode.XJ_PI0_V3: Checkpoint(
        config="pi0_libero_low_mem_finetune_split_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_low_mem_finetune_split_train_v3/pi0_libero_low_mem_finetune_split_train_v3/19999"
    ),
    EnvMode.XJ_PI0_V4: Checkpoint(
        config="pi0_libero_low_mem_finetune_split_inference",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_libero_low_mem_finetune_split_train_v4/pi0_libero_low_mem_finetune_split_train_v4/19999"
    ),  
    
    # rebuttal
    EnvMode.XJ_PI0_ROBOCASA_MG_ALL: Checkpoint(
        config="pi0_robocasa_mg_three_image_low_mem_finetune_train",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_robocasa_mg_three_image_low_mem_finetune_train/pi0_robocasa_mg_three_image_low_mem_finetune_train/499999"
    ),  
    EnvMode.XJ_PI0_ROBOCASA_MG_TRAIN: Checkpoint(
        config="pi0_robocasa_mg_three_image_low_mem_finetune_train",
        dir="/home/dingj0b/dingjian/openpi_explore/project/openpi/checkpoints/pi0_robocasa_mg_three_image_low_mem_finetune_train_split/pi0_robocasa_mg_three_image_low_mem_finetune_train_split/499999"
    ), 

}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


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
