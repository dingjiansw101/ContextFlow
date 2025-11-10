import dataclasses
import logging
import pathlib
import sys

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

sys.path.append("/home/aloha/workspace/openpi")
sys.path.append("/home/dingj0b/code/openpi")
# from examples.aloha_mobile_real import env as _env

from examples.aloha_mobile_real import env_incontext as _env
from examples.aloha_mobile_real import trajectory_recorder as _trajectory_recorder

"""
ALOHA In-Context Policy Evaluation Script

This script runs the ALOHA robot with in-context learning policy, optionally recording
trajectories to HDF5 files.

Usage Examples:
    # Basic run (no recording)
    python main_incontext.py

    # Enable recording with default settings
    python main_incontext.py --record

    # Custom task with recording
    python main_incontext.py --record --prompt "pick up the apple"

    # Custom output directory
    python main_incontext.py --record --record-dir ./my_data

    # Disable image recording (save only states/actions)
    python main_incontext.py --record --no-record-images

    # Multiple episodes
    python main_incontext.py --record --num-episodes 10

    # Custom server and multiple episodes
    python main_incontext.py --host 10.68.106.82 --port 8000 --num-episodes 5

Output Structure:
    Trajectories saved to: {record_dir}/{task_name}/episode_XXX.hdf5
    Example: ./trajectories/pick_up_the_cucumber_and_place_it_in_the_basket/episode_000.hdf5

    Each HDF5 file contains:
        /observations/states - Robot states
        /observations/images/ - Camera images (if record_images=True)
        /actions - Actions taken
        /metadata/timestamps - Timestep information
"""

@dataclasses.dataclass
class Args:
    # host: str = "10.68.106.146"
    # host: str = "10.68.106.44"
    # host: str = "10.68.106.188"
    # host: str = "10.67.24.146"
    host: str = "10.68.106.197"
    # host: str = "10.68.107.117"
    port: int = 8001

    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 1000  # TODO: Set this to a reasonable value

    # Performance metrics notes:
    # v12: task: pen_uncap_gray_right, prompt: pen_uncap_gray_right, success: 0/3
    # v12: task: pen_uncap_gray_right, prompt: pen_uncap_gray_left, success: 1/1
    # v12: task: pen_uncap_gray_left: 7/10
    # pi0: task: pen_uncap_gray_right, prompt: pen_uncap_gray_right, success: 6/10
    # v12: task: pen_uncap_blue_rightv2, prompt: pen_uncap_gray_right, success: 7/10
    # v12: task: pen_uncap_blue_rightv2, prompt: pen_uncap_blue_right, success: 7/10

    # Recording options
    record: bool = False
    record_dir: str = "./trajectories"
    record_images: bool = True
    record_compression: str = "gzip"

    # Task configuration
    prompt: str = "pick_up_the_cucumber_and_place_it_in_the_basket"
    task_json: str = "metadata/objects_pickup_place_right_hand/tasks.jsonl"


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    # Create subscribers list
    subscribers = []
    if args.record:
        # Create task-specific subdirectory
        task_name = args.prompt.replace(" ", "_")
        task_record_dir = pathlib.Path(args.record_dir) / task_name

        recorder = _trajectory_recorder.TrajectoryRecorder(
            output_dir=task_record_dir,
            record_images=args.record_images,
            compression=args.record_compression,
        )
        subscribers.append(recorder)
        logging.info(f"Trajectory recording enabled. Output directory: {task_record_dir}")

    # metadata = ws_client_policy.get_server_metadata()
    runtime = _runtime.Runtime(
        environment=_env.AlohaRealEnvironment(
            render_height=480,
            render_width=640,
            prompt=args.prompt,
            task_json=args.task_json
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=subscribers,
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
