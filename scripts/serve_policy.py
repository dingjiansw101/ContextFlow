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
