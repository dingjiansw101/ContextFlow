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
