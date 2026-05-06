"""Visualize LeRobot dataset frames and in-context demonstration trajectories.

Usage:
    uv run scripts/visualize_lerobot.py --task-index 17
    uv run scripts/visualize_lerobot.py --task-index 17 --episode-id 612
    uv run scripts/visualize_lerobot.py --task-index 17 --num-frames 8 --out-dir data/demo_visualizations
    uv run scripts/visualize_lerobot.py --repo-id vo2yager/objects_pickup_place --frame-indexes 5600 5628 5656
"""

import argparse
import io
import json
from pathlib import Path

import imageio
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image


def load_task_name_mapping(tasks_jsonl: Path) -> dict[int, str]:
    mapping = {}
    with tasks_jsonl.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            mapping[record["task_index"]] = record["task"]
    return mapping


def sanitize_path_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def scalar_to_int(value) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def convert_image_to_uint8_hwc(image) -> np.ndarray:
    """Convert LeRobot image values to uint8 HWC/RGB-compatible arrays."""
    if hasattr(image, "numpy"):
        image = image.numpy()
    image = np.asarray(image)

    # Some LeRobot datasets store compressed image bytes as (1, 1, N).
    if image.ndim == 3 and image.shape[0] == 1 and image.shape[1] == 1:
        decoded = Image.open(io.BytesIO(bytes(image.flatten()))).convert("RGB")
        return np.asarray(decoded)

    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if image.min() < 0:
                image = (image + 1.0) * 127.5
            elif image.max() <= 1.0:
                image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def image_keys(frame) -> list[str]:
    return [key for key in frame if "image" in key]


def task_name_from_frame(dataset: LeRobotDataset, frame) -> str:
    task_index = frame.get("task_index")
    if task_index is None:
        return "unknown_task"

    task_index = scalar_to_int(task_index)
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "tasks"):
        task_name = dataset.meta.tasks.get(task_index, f"task_{task_index}")
    else:
        task_name = f"task_{task_index}"
    return sanitize_path_name(str(task_name))


def save_frame_images(dataset: LeRobotDataset, frame_idx: int, out_dir: Path) -> int:
    frame = dataset[frame_idx]
    task_name = task_name_from_frame(dataset, frame)
    frame_dir = out_dir / task_name / f"frame_{frame_idx:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = frame_dir / "metadata.txt"
    with metadata_file.open("w") as f:
        f.write(f"Frame index: {frame_idx}\n")
        f.write(f"Task name: {task_name}\n")
        if "task_index" in frame:
            f.write(f"Task index: {frame['task_index']}\n")
        if "episode_index" in frame:
            f.write(f"Episode index: {frame['episode_index']}\n")
        if "frame_index" in frame:
            f.write(f"Frame in episode: {frame['frame_index']}\n")
        if "state" in frame:
            f.write(f"State shape: {frame['state'].shape}\n")
        if "actions" in frame:
            f.write(f"Actions shape: {frame['actions'].shape}\n")

    keys = image_keys(frame)
    if not keys:
        print("  Warning: no images found in this frame")
        return 0

    saved_count = 0
    for key in keys:
        camera_name = sanitize_path_name(key.split(".")[-1])
        try:
            image = convert_image_to_uint8_hwc(frame[key])
        except Exception as exc:
            print(f"  Warning: skipping {key}: {exc}")
            continue

        image_path = frame_dir / f"{camera_name}.png"
        Image.fromarray(image).save(image_path)
        saved_count += 1
        print(f"  Saved {image_path}")

    return saved_count


def save_explicit_frame_indexes(dataset: LeRobotDataset, frame_indexes: list[int], out_dir: Path) -> None:
    print(f"Dataset loaded ({len(dataset)} frames)")
    print(f"Frame indexes to save: {frame_indexes}")

    total_images = 0
    tasks_saved = set()
    for frame_idx in frame_indexes:
        if frame_idx < 0 or frame_idx >= len(dataset):
            print(f"Warning: frame index {frame_idx} outside dataset range [0, {len(dataset) - 1}], skipping")
            continue

        frame = dataset[frame_idx]
        task_name = task_name_from_frame(dataset, frame)
        tasks_saved.add(task_name)
        print(f"\nProcessing frame {frame_idx} (task: {task_name})...")
        num_images = save_frame_images(dataset, frame_idx, out_dir)
        total_images += num_images
        print(f"  Saved {num_images} images")

    print(f"\nDone! Saved {total_images} total images to: {out_dir.absolute()}")
    if tasks_saved:
        print(f"Tasks saved: {', '.join(sorted(tasks_saved))}")


def visualize_task_episode(args, dataset: LeRobotDataset, out_dir: Path) -> None:
    if args.task_index is None:
        raise ValueError("--task-index is required unless --frame-indexes is provided")

    metadata_dir = Path(args.metadata_dir)

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

    # Read frames and build image grids per camera
    camera_frames: dict[str, list[np.ndarray]] = {}
    for idx in sampled_indices:
        item = dataset[idx]
        for key in image_keys(item):
            try:
                image = convert_image_to_uint8_hwc(item[key])
            except Exception as exc:
                print(f"Warning: skipping {key} at dataset index {idx}: {exc}")
                continue
            camera_frames.setdefault(key, []).append(image)

    # Save grids
    print(f"\nSaving visualizations to {out_dir}/")
    for cam_name, frames in camera_frames.items():
        grid = np.concatenate(frames, axis=1)
        cam_slug = sanitize_path_name(cam_name)
        out_path = out_dir / f"demo_task{args.task_index}_ep{episode_id}_{cam_slug}.png"
        imageio.imwrite(str(out_path), grid)
        print(f"  {out_path} ({grid.shape[1]}x{grid.shape[0]})")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Visualize in-context demo trajectories")
    parser.add_argument("--task-index", type=int, default=None, help="Task index to visualize")
    parser.add_argument("--episode-id", type=int, default=None, help="Episode ID (default: first available)")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample (default: 8)")
    parser.add_argument("--out-dir", type=str, default="data/demo_visualizations", help="Output directory")
    parser.add_argument("--repo-id", type=str, default="physical-intelligence/libero", help="Dataset repo ID")
    parser.add_argument("--metadata-dir", type=str, default="metadata/libero", help="Metadata directory")
    parser.add_argument("--frame-indexes", type=int, nargs="+", default=None, help="Global dataset frame indexes to save")
    parser.add_argument("--local-files-only", action="store_true", help="Use only local cached files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset '{args.repo_id}'...")
    dataset = LeRobotDataset(args.repo_id, local_files_only=args.local_files_only)

    if args.frame_indexes is not None:
        save_explicit_frame_indexes(dataset, args.frame_indexes, out_dir)
    else:
        print(f"Dataset loaded ({len(dataset)} frames)")
        visualize_task_episode(args, dataset, out_dir)


if __name__ == "__main__":
    main()
