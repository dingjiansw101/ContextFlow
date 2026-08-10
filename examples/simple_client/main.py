import dataclasses
import enum
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


class EnvMode(enum.Enum):
    """Supported environments."""

    LIBERO = "libero"
    ALOHA_MOBILE = "aloha_mobile"


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    env: EnvMode = EnvMode.LIBERO
    num_steps: int = 10


def main(args: Args) -> None:
    obs_fn = {
        EnvMode.LIBERO: _random_observation_libero,
        EnvMode.ALOHA_MOBILE: _random_observation_aloha_mobile,
    }[args.env]

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {policy.get_server_metadata()}")

    # Send 1 observation to make sure the model is loaded.
    policy.infer(obs_fn())
    start = time.time()
    for _ in range(args.num_steps):
        policy.infer(obs_fn())
    end = time.time()

    print(f"Total time taken: {end - start:.2f} s")
    print(f"Average inference time: {1000 * (end - start) / args.num_steps:.2f} ms")


def _random_observation_aloha_mobile() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
