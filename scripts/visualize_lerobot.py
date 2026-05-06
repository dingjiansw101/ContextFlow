"""Visualize in-context demonstration trajectories from the LeRobot dataset.

Usage:
    uv run scripts/visualize_lerobot.py --task-index 17
    uv run scripts/visualize_lerobot.py --task-index 17 --episode-id 612
    uv run scripts/visualize_lerobot.py --task-index 17 --num-frames 8 --out-dir data/demo_visualizations
"""

import argparse
import json
from pathlib import Path

import imageio
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def load_task_name_mapping(tasks_jsonl: Path) -> dict[int, str]:
    mapping = {}
    with tasks_jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            mapping[record["task_index"]] = record["task"]
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Visualize in-context demo trajectories")
    parser.add_argument("--task-index", type=int, required=True, help="Task index to visualize")
    parser.add_argument("--episode-id", type=int, default=None, help="Episode ID (default: first available)")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample (default: 8)")
    parser.add_argument("--out-dir", type=str, default="data/demo_visualizations", help="Output directory")
    parser.add_argument("--repo-id", type=str, default="physical-intelligence/libero", help="Dataset repo ID")
    parser.add_argument("--metadata-dir", type=str, default="metadata/libero", help="Metadata directory")
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with (metadata_dir / "task_to_episode.json").open("r") as f:
        task_to_episode = {int(k): v for k, v in json.load(f).items()}

    with (metadata_dir / "episode_to_indexes.json").open("r") as f:
        episode_to_indexes = {int(k): v for k, v in json.load(f).items()}

    task_names = load_task_name_mapping(metadata_dir / "tasks.jsonl")

    # Get task info
    task_name = task_names.get(args.task_index, f"unknown_task_{args.task_index}")
    episodes = task_to_episode.get(args.task_index, [])
    if not episodes:
        print(f"No episodes found for task_index={args.task_index}")
        return

    episode_id = args.episode_id if args.episode_id is not None else episodes[0]
    if episode_id not in episodes:
        print(f"Episode {episode_id} not in task {args.task_index}. Available: {episodes}")
        return

    frame_indices = episode_to_indexes.get(episode_id, [])
    if not frame_indices:
        print(f"No frames found for episode {episode_id}")
        return

    # Sample frames uniformly (same as InjectDemoIndexes)
    n = len(frame_indices)
    num_frames = min(args.num_frames, n)
    positions = np.linspace(0, n - 1, num=num_frames, dtype=int)
    sampled_indices = [frame_indices[p] for p in positions]

    print(f"Task: '{task_name}' (index={args.task_index})")
    print(f"Episode: {episode_id}")
    print(f"Total frames in episode: {n}")
    print(f"Sampled {num_frames} frames at positions: {positions.tolist()}")
    print(f"Global dataset indices: {sampled_indices}")

    # Load dataset
    print(f"\nLoading dataset '{args.repo_id}'...")
    dataset = LeRobotDataset(args.repo_id)
    print(f"Dataset loaded ({len(dataset)} frames)")

    # Read frames and build image grids per camera
    camera_frames: dict[str, list[np.ndarray]] = {}
    for i, idx in enumerate(sampled_indices):
        item = dataset[idx]
        # Collect all image keys
        for key in item:
            if "image" in key:
                img = item[key]
                # Convert from torch tensor (C, H, W) to numpy (H, W, C) uint8
                if hasattr(img, "numpy"):
                    img = img.numpy()
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    else:
                        img = np.clip(img, 0, 255).astype(np.uint8)
                camera_frames.setdefault(key, []).append(img)

    # Save grids
    task_slug = task_name.replace(" ", "_")[:80]
    print(f"\nSaving visualizations to {out_dir}/")
    for cam_name, frames in camera_frames.items():
        grid = np.concatenate(frames, axis=1)
        cam_slug = cam_name.replace("/", "_")
        out_path = out_dir / f"demo_task{args.task_index}_ep{episode_id}_{cam_slug}.png"
        imageio.imwrite(str(out_path), grid)
        print(f"  {out_path} ({grid.shape[1]}x{grid.shape[0]})")

    print("\nDone!")


if __name__ == "__main__":
    main()
