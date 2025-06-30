import collections
import dataclasses
import logging
import math
import pathlib

import imageio

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils  import create_env_from_metadata
import robomimic.utils.obs_utils as ObsUtils
import robocasa
import json

ROBOCASA_DUMMY_ACTION = [0.0] * 6 + [-1.0] + [0.0] * 4 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
DEFAULT_EVAL_UPDATE_KWARGS =  {
    "generative_textures": None,
    "randomize_cameras": False,
    "obj_instance_split": "B",
    "layout_ids": None,
    "style_ids": None,
    "scene_split": None,
    "layout_and_style_ids": [
                    [
                        1,
                        1
                    ],
                    [
                        2,
                        2
                    ],
                    [
                        4,
                        4
                    ],
                    [
                        6,
                        9
                    ],
                    [
                        7,
                        10
                    ]
                ],
    "camera_heights": 256,
    "camera_widths": 256,
}


SHAPE_META = {
    "obs": {
        "robot0_agentview_left_image": {
            "shape": [256, 256,3],
            "type": "rgb"
        },
        "robot0_eye_in_hand_image": {
            "shape": [256, 256, 3],
            "type": "rgb"
        },
        "robot0_agentview_right_image": {
            "shape": [256, 256, 3],
            "type": "rgb"
        },
        "robot0_eef_pos": {
            "shape": [3]
            # type default: low_dim (not explicitly listed)
        },
        "robot0_eef_quat": {
            "shape": [4]
        },
        "robot0_base_to_eef_pos": {
            "shape": [3]
            # type default: low_dim (not explicitly listed)
        },
        "robot0_base_to_eef_quat": {
            "shape": [4]
        },
        "robot0_gripper_qpos": {
            "shape": [2]
        },
        "robot0_gripper_qvel": {
            "shape": [2]
        },
    },
    "action": {
        "shape": [12]
    }
}

def get_task_to_index_mapping(file_path: pathlib.Path) -> dict:
    mapping = {}
    with file_path.open('r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task_description = record.get("task")
            task_index = record.get("task_index")
            if task_description is not None and task_index is not None:
                mapping[task_description] = task_index
    return mapping


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials: int = 50  # Number of rollouts per task
    
    env_name: str = "OpenSingleDoor"
    dataset_path: str = "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo.hdf5"  # Path to the dataset
    # XJ_PI0MINI_ROBOCASA_HUMAN_THREE_IMAGE(500k): 0.38 (19/50)
    # ROBOCASA_HUMAN_THREE_IMAGE(100k): 0.18 (9/50)
    # LIGHT_ROBOCASA_HUMAN_THREE_IMAGE (100k): 0.12 (6/50)
    
    # env_name: str = "TurnOnStove"
    # dataset_path: str = "/home/xianjid/project/robocasa_xj/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo_gentex_im128_randcams.hdf5"  # Path to the dataset
    # # XJ_PI0MINI_ROBOCASA_HUMAN_THREE_IMAGE(500k): 0.04
    
    # env_name: str = "TurnOnMicrowave"
    # dataset_path: str = "/home/xianjid/project/robocasa_xj/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5"  # Path to the dataset
    # # XJ_PI0MINI_ROBOCASA_HUMAN_THREE_IMAGE(500k): 
    
    # env_name: str = "TurnOffSinkFaucet"
    # dataset_path: str = "/home/xianjid/project/robocasa_xj/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5"  # Path to the dataset
    # # XJ_PI0MINI_ROBOCASA_HUMAN_THREE_IMAGE(500k): 0.06
    
    # env_name: str = "TurnOnMicrowave"
    # dataset_path: str = "/home/xianjid/project/robocasa_xj/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/mg/2024-05-04-22-40-00/demo_gentex_im128_randcams.hdf5"  # Path to the dataset
    
    
    horizon: int = 500  # Number of steps to run in each episode

    #################################################################################################################
    # Utils
    #################################################################################################################
    # video_out_path: str = f"data/robocasa/{env_name}/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_robocasa(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    filename = pathlib.Path("/home/xianjie.dai/.cache/huggingface/lerobot/daixianjie/robocasa_human_lerobot/meta/tasks.jsonl")
    task_description2index = get_task_to_index_mapping(filename)
    
    video_out_path = f"data/robocasa/{args.env_name}/videos"
    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)


    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    # Get task

    # Get default LIBERO initial states

    # Initialize LIBERO environment and task description
   
    env_meta = get_env_metadata_from_dataset(
            args.dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = False
    env_meta['env_kwargs'].update(DEFAULT_EVAL_UPDATE_KWARGS)
    env_meta['env_kwargs']["camera_names"] = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    env = _get_robocasa_env(env_meta, SHAPE_META, enable_render=True)
    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(args.num_trials)):

        # Reset environment
        env.reset()
        task_lang = env._ep_lang_str
        logging.info(f"task_lang: {task_lang}")
        action_plan = collections.deque()

        # Setup
        t = 0
        replay_images = []

        logging.info(f"Starting episode {task_episodes+1}...")
        while t < args.horizon + args.num_steps_wait:
            # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(ROBOCASA_DUMMY_ACTION)
                t += 1
                continue

            # Get preprocessed image
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            # img = np.ascontiguousarray(np.transpose(obs["robot0_agentview_left_image"], (1,2,0)))
            # wrist_img = np.ascontiguousarray(np.transpose(obs["robot0_eye_in_hand_image"], (1,2,0)))
            # img = image_tools.convert_to_uint8(
            #     image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            # )
            # wrist_img = image_tools.convert_to_uint8(
            #     image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            # )
            
            # logging.info(f"ascontiguousarrayrobot0_agentview_left_image shape: {obs['robot0_agentview_left_image'].shape}")

            img_left = np.ascontiguousarray(np.transpose(obs["robot0_agentview_left_image"], (1,2,0)))
            # logging.info(f"ascontiguousarrayrobot0_agentview_left_image shape: {img_left.shape}")

            img_right = np.ascontiguousarray(np.transpose(obs["robot0_agentview_right_image"], (1,2,0)))

            wrist_img = np.ascontiguousarray(np.transpose(obs["robot0_eye_in_hand_image"], (1,2,0)))
            img_left = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img_left, args.resize_size, args.resize_size)
                    )
            img_right = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img_right, args.resize_size, args.resize_size)
                    )
            wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
            # Save preprocessed image for replay video
            replay_images.append(img_right)

            if not action_plan:
                # logging.info(f"robot0_base_to_eef_quat : {obs['robot0_base_to_eef_quat']}")
                # logging.info(f"robot0_base_to_eef_quat to axis-angle: {_quat2axisangle(obs['robot0_base_to_eef_quat'])}")

                state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            obs["robot0_eef_quat"],
                            obs["robot0_gripper_qpos"],
                        ), axis=0
                    )
                # state = np.concatenate(
                #         (
                #             obs["robot0_eef_pos"],
                #             obs["robot0_eef_quat"],
                #             obs["robot0_gripper_qpos"],
                #             obs["robot0_gripper_qvel"],
                #             obs["robot0_base_to_eef_pos"],
                #             obs["robot0_base_to_eef_quat"],
                #         ), axis=0
                #     )
                
                # state = np.ascontiguousarray(state)
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                element = {
                    "observation/image_left": img_left,
                    "observation/image_right": img_right,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": str(task_lang),
                    "task_index": task_description2index[task_lang],
                }

                # Query model to get action
                action_chunk = client.infer(element)["actions"]
                append_values = np.array([0.0, 0.0, 0.0, 0.0, -1.0]) 
                action_chunk = np.concatenate([action_chunk, append_values[None, :].repeat(action_chunk.shape[0], axis=0)], axis=1)
                assert (
                    len(action_chunk) >= args.replan_steps
                ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            success = env.is_success()["task"]

            # replay_img = env.render(mode="rgb_array", height = 512, width = 512, camera_name="robot0_agentview_center")
            # replay_img = np.ascontiguousarray(replay_img)
            # replay_img = image_tools.convert_to_uint8(
            #     replay_img
            # )
            # replay_images.append(replay_img)
            if done or success:
                task_successes += 1
                total_successes += 1
                break
            t += 1

            # except Exception as e:
            #     logging.error(f"Caught exception: {e}")
            #     break

        task_episodes += 1
        total_episodes += 1

        # Save a replay video of the episode
        suffix = "success" if (done or success) else "failure"
        imageio.mimwrite(
            pathlib.Path(video_out_path) / f"rollout_{episode_idx}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=10,
        )

        # Log current results
        logging.info(f"Success: {done or success}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_robocasa_env(env_meta, shape_meta, enable_render=True):
    """Initializes and returns the LIBERO environment, along with the task description."""
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
    env_name = env_meta['env_name']
    env_name = env_name[3:] if env_name.startswith('MG_') else env_name
    env = create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_name,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_robocasa)