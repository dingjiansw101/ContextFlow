import dataclasses
import logging
import pathlib
import math
import numpy as np
import tqdm
import tyro
import collections  

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from pyrep.const import RenderMode
from rlbench.backend.utils import task_file_to_task_class
from rlbench.observation_config import ObservationConfig, CameraConfig

RLBENCH_DUMMY_ACTION = [0.0] * 7 + [1.0]
RLBENCH_TASKS = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "put_money_in_safe",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "sweep_to_dustpan_of_size",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
]

@dataclasses.dataclass
class Args:
    host: str = "10.68.106.188"
    port: int = 8000
    resize_size: int = 224
    task_name: str = "reach_and_drag"
    variation: int = -1
    num_episodes: int = 50
    num_steps_wait: int = 10
    max_steps: int = 839
    replan_steps: int = 5  # 
    image_size: tuple = (128, 128)
    renderer: str = "opengl"
    arm_max_velocity: float = 1.0
    arm_max_acceleration: float = 4.0
    seed: int = 7


def rlbench_obs_config(camera_names, camera_resolution, args: Args):
    unused = CameraConfig(); unused.set_all(False)
    used = CameraConfig(
        rgb=True, depth=False, mask=False, point_cloud=False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL3 if args.renderer == "opengl3" else RenderMode.OPENGL,
    )
    return ObservationConfig(
        front_camera=used if "front" in camera_names else unused,
        wrist_camera=used if "wrist" in camera_names else unused,
        left_shoulder_camera=unused,
        right_shoulder_camera=unused,
        overhead_camera=unused,
        joint_forces=False,
        joint_positions=True,
        joint_velocities=False,
        gripper_touch_forces=False,
        gripper_pose=False,
        gripper_open=True,
        gripper_matrix=False,
        gripper_joint_positions=True,
    )


def eval_rlbench(args: Args):
    np.random.seed(args.seed)
    logging.info(f"Task: {args.task_name}, Variation: {args.variation}")

    obs_config = rlbench_obs_config(["front", "wrist"], list(args.image_size), args)
    env = Environment(
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
        obs_config=obs_config,
        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
        headless=True
    )
    env.launch()

    task_cls = task_file_to_task_class(args.task_name)
    task = env.get_task(task_cls)

    if args.variation == -1:
        variation_list = list(range(task.variation_count()))
        logging.info(f"Running all {len(variation_list)} variations.")
    else:
        variation_list = [args.variation]

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    total_episodes, total_successes = 0, 0

    for variation in variation_list:
        task.set_variation(variation)
        descriptions, _ = task.reset()
        task_description = descriptions[0]
        logging.info(f"\n▶ Running Variation {variation}...")

        variation_success = 0

        for episode_idx in tqdm.trange(args.num_episodes):
            obs = task.get_observation()
            t = 0
            action_plan = collections.deque()  

            # for _ in range(args.num_steps_wait):
            #     obs, _, _ = task.step(RLBENCH_DUMMY_ACTION)

            done = False
            while t < args.max_steps:
                try:
                    front = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(obs.front_rgb[::-1, ::-1], args.resize_size, args.resize_size)
                    )
                    wrist = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(obs.wrist_rgb[::-1, ::-1], args.resize_size, args.resize_size)
                    )
                    state = np.concatenate([obs.joint_positions, obs.gripper_joint_positions])

                    if not action_plan:
                        obs_dict = {
                            "observation/image": front,
                            "observation/wrist_image": wrist,
                            "observation/state": state,
                            "prompt": task_description,
                        }
                        actions = client.infer(obs_dict)["actions"]
                        assert len(actions) >= args.replan_steps
                        action_plan.extend(actions[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done = task.step(action)

                    if done:
                        variation_success += 1
                        break

                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception in episode {episode_idx}: {e}")
                    break

            logging.info(f"[Variation {variation} | Episode {episode_idx}] Success = {done}")
        
        sr = variation_success / args.num_episodes
        logging.info(f"SR (Variation {variation}) = {variation_success}/{args.num_episodes} ({sr:.2%})")

        total_successes += variation_success
        total_episodes += args.num_episodes

    logging.info(f"\n Final SR across {len(variation_list)} variations: "
                 f"{total_successes}/{total_episodes} ({total_successes / total_episodes:.2%})")
    env.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_rlbench)
