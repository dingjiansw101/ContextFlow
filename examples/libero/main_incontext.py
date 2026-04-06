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

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def load_task_splits(task_splits_dir: str, split: str):
    """Load seen and unseen task lists from JSON files.

    Returns:
        tuple: (seen_tasks_set, unseen_tasks_set) - sets of task descriptions with spaces
    """
    split_path = pathlib.Path(task_splits_dir) / split

    with open(split_path / "seen_tasks.json", "r") as f:
        seen_tasks = json.load(f)

    with open(split_path / "unseen_tasks.json", "r") as f:
        unseen_tasks = json.load(f)

    # Normalize task names: replace underscores with spaces to match task_description format
    seen_tasks_set = {task.replace("_", " ") for task in seen_tasks}
    unseen_tasks_set = {task.replace("_", " ") for task in unseen_tasks}

    return seen_tasks_set, unseen_tasks_set


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
    results_out_path: str = ""  # Path to save JSON results (default: logs/eval_results/<task_suite_name>_results.json)
    task_split: str = "split0"  # Which task split to use for seen/unseen task lists
    task_splits_dir: str = "libero_task_splits"  # Directory containing task splits

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    if not args.results_out_path:
        args.results_out_path = str(pathlib.Path("logs") / "eval_results" / f"{args.task_suite_name}_incontext_results.json")

    # Load seen and unseen task splits
    seen_tasks, unseen_tasks = load_task_splits(args.task_splits_dir, args.task_split)
    logging.info(f"Loaded task split: {args.task_split}")
    logging.info(f"  Seen tasks: {len(seen_tasks)}")
    logging.info(f"  Unseen tasks: {len(unseen_tasks)}")

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

    # Start evaluation
    # Track per-task episode & success counts
    per_task_episodes  = Counter()
    per_task_successes = Counter()
    total_episodes, total_successes = 0, 0
    per_task_results = []

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
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

        # Track per-task results using task_description as key
        per_task_episodes[task_description] = task_episodes
        per_task_successes[task_description] = task_successes

        if task_description in seen_tasks:
            category = "seen"
        elif task_description in unseen_tasks:
            category = "unseen"
        else:
            category = None

        per_task_results.append({
            "task_id": task_id,
            "task_description": task_description,
            "episodes": task_episodes,
            "successes": task_successes,
            "success_rate": float(task_successes) / float(task_episodes),
            "category": category,
        })

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # Compute success rates for seen vs unseen tasks
    seen_rates, unseen_rates = [], []
    for task_desc in per_task_episodes.keys():
        rate = per_task_successes[task_desc] / per_task_episodes[task_desc]
        if task_desc in unseen_tasks:
            unseen_rates.append(rate)
        elif task_desc in seen_tasks:
            seen_rates.append(rate)
        else:
            logging.warning(f"Task '{task_desc}' not found in seen or unseen lists")

    avg_seen = sum(seen_rates) / len(seen_rates) if seen_rates else 0.0
    avg_unseen = sum(unseen_rates) / len(unseen_rates) if unseen_rates else 0.0

    logging.info(f"\nAverage success on SEEN tasks: {avg_seen:.3f} ({len(seen_rates)} tasks)")
    logging.info(f"Average success on UNSEEN tasks: {avg_unseen:.3f} ({len(unseen_rates)} tasks)")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    results = {
        "config": {
            "task_suite_name": args.task_suite_name,
            "num_trials_per_task": args.num_trials_per_task,
            "seed": args.seed,
            "task_split": args.task_split,
        },
        "per_task_results": per_task_results,
        "summary": {
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "total_success_rate": float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0,
            "seen_success_rate": avg_seen,
            "unseen_success_rate": avg_unseen,
            "num_seen_tasks": len(seen_rates),
            "num_unseen_tasks": len(unseen_rates),
            "num_tasks_evaluated": len(per_task_results),
        },
    }
    results_path = pathlib.Path(args.results_out_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_path}")


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
