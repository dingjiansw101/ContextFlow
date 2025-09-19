import collections
import dataclasses
import logging
import math
import pathlib
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import json
from collections import Counter
from scripts.train_or_infer_stage_classifier import StagePredictor

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

LIBERO_TEST_TASK_DICT = {
    "libero_spatial": [3,8],
    "libero_object":[5,7],
    "libero_goal": [0,7],
    "libero_10": [5,4],
    "libero_90":[0],
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
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero_incontext/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)
    
    use_stage_head: bool = True
    stage_head_dir: str = "checkpoints/stage_head"           # 训练产出的目录（含 stage_head.pt）
    stage_norm_stats: str | None = "assets/pi0_libero_incontextv12_low_mem_finetune_more_sample_frame_train_split_v4/physical-intelligence/libero/norm_stats.json"  # 如果传 RAW state，需要与训练一致的 norm_stats.json
    stage_print_probs: bool = True   # 调试用：是否打印概率向量


def _pack_state32_from_obs(obs: dict) -> np.ndarray:
    """
    将环境观测打包为与训练时一致的 32 维 state：
    [eef_pos(3), axisangle(3), gripper_qpos(2)] + zeros(24)
    注意：如果你的训练 cache 中 32 维恰好就是这 8 维 + 全 0，这里就对齐了；
          若你训练时 state 定义不同，请在这里同步修改顺序与维度。
    """
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)           # 3
    aa  = np.asarray(_quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float32)     # 3
    g   = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)      # ?(1 or 2)

    if g.size == 0:
        g = np.zeros(2, dtype=np.float32)
    elif g.size == 1:
        g = np.repeat(g, 2)
    else:
        g = g[:2]

    s = np.concatenate([eef, aa, g], axis=0)  # 8
    return s

def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.task_suite_name == "libero_90":
        filename = pathlib.Path("metadata/libero_90/tasks.jsonl")
    else:
        filename = pathlib.Path("metadata/libero/tasks.jsonl")
    task_description2index = get_task_to_index_mapping(filename)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    
    # —— 可选：加载阶段分类器 —— 
    stage_pred = None
    stage_hist = None
    if args.use_stage_head:
        stage_pred = StagePredictor(args.stage_head_dir, args.stage_norm_stats)
        stage_hist = collections.deque(maxlen=stage_pred.H)  # H 为训练时的 history
        logging.info(f"[stage] loaded head: classes={len(stage_pred.id2name)} history={stage_pred.H} "
                     f"labels={stage_pred.id2name}")

    # Xianjie: get the test task list based on the task suite name
    # assigned_task_list = LIBERO_TEST_TASK_DICT[args.task_suite_name]

    # Start evaluation
    # Track per-task episode & success counts
    per_task_episodes  = Counter()
    per_task_successes = Counter()
    total_episodes, total_successes = 0, 0

    unseen_ids = set(LIBERO_TEST_TASK_DICT[args.task_suite_name])

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
    # for task_id in assigned_task_list:
        # Get task
        
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        if stage_pred is not None:
                            s32 = _pack_state32_from_obs(obs)         # RAW state（未归一化）
                            stage_hist.append(s32)
                            # 直接用历史窗口进行预测（内部自动左侧补齐）
                            if args.stage_print_probs:
                                idx, name, probs = stage_pred.predict(np.stack(stage_hist, axis=0), return_probs=True)
                                logging.info(f"[stage] pred={idx} name={name} probs={probs.tolist()}")
                            else:
                                idx, name = stage_pred.predict(np.stack(stage_hist, axis=0), return_probs=False)
                                logging.info(f"[stage] pred={idx} name={name}")
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "task_index": task_description2index[task_description],
                            "split": "test",
                            # 可选：把阶段 rank 发给服务端（若那边支持）
                            "stage_rank": int(idx),
                            "stage_name": name if name is not None else "",
                        }
                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        
        per_task_episodes[task_id] = task_episodes
        per_task_successes[task_id] = task_successes
        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    seen_rates, unseen_rates = [], []
    for tid in range(num_tasks_in_suite):
        rate = per_task_successes[tid] / per_task_episodes[tid]
        if tid in unseen_ids:
            unseen_rates.append(rate)
        else:
            seen_rates.append(rate)

    avg_unseen = sum(unseen_rates) / len(unseen_rates) if unseen_rates else 0.0
    avg_seen   = sum(seen_rates)   / len(seen_rates)   if seen_rates   else 0.0
    
    logging.info(f"Average success on UNSEEN tasks {sorted(unseen_ids)}: {avg_unseen:.3f}")
    logging.info(
        f"Average success on SEEN tasks   {sorted(set(range(num_tasks_in_suite)) - unseen_ids)}: "
        f"{avg_seen:.3f}"
    )

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


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
    tyro.cli(eval_libero)
