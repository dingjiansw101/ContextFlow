#!/usr/bin/env python3
"""Script to load specific frames from a dataset and save their images."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def save_frame_images(dataset, frame_idx: int, save_dir: Path):
    """Load a frame and save all its camera images."""
    # Get the frame data
    frame = dataset[frame_idx]

    # Create subdirectory for this frame
    frame_dir = save_dir / f"frame_{frame_idx:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata_file = frame_dir / "metadata.txt"
    with open(metadata_file, "w") as f:
        f.write(f"Frame index: {frame_idx}\n")
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

    # Get all image keys (usually in format "observation.images.<camera_name>")
    image_data = {}

    # Extract images from frame - they can be in different formats
    for key in frame.keys():
        if "observation.images" in key:
            # Extract camera name from key like "observation.images.cam_high"
            camera_name = key.split(".")[-1]
            image_data[camera_name] = frame[key]
            print(f"  Found image: {key}")

    if not image_data:
        print("  ⚠ No images found in this frame")
        return 0

    # Save each camera image
    saved_count = 0
    for camera_name, img_array in image_data.items():
        import io

        # Convert to numpy if needed
        if hasattr(img_array, 'numpy'):
            img_array = img_array.numpy()

        # Check if image is compressed (LeRobot format)
        # Compressed images have shape (1, 1, N) where N is compressed size
        if len(img_array.shape) == 3 and img_array.shape[0] == 1 and img_array.shape[1] == 1:
            # Compressed image - extract bytes and decode
            compressed_bytes = bytes(img_array.flatten())
            img = Image.open(io.BytesIO(compressed_bytes))
        elif len(img_array.shape) == 3 and img_array.shape[0] == 3:
            # CHW format (Channel, Height, Width) - transpose to HWC
            img_array = np.transpose(img_array, (1, 2, 0))

            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                # Assume range [0, 1] or [-1, 1]
                if img_array.min() >= 0:
                    img_uint8 = (img_array * 255).clip(0, 255).astype(np.uint8)
                else:
                    # Range [-1, 1]
                    img_uint8 = ((img_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
            else:
                img_uint8 = img_array
            img = Image.fromarray(img_uint8)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Regular HWC format RGB image
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                # Assume range [0, 1] or [-1, 1]
                if img_array.min() >= 0:
                    img_uint8 = (img_array * 255).clip(0, 255).astype(np.uint8)
                else:
                    # Range [-1, 1]
                    img_uint8 = ((img_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
            else:
                img_uint8 = img_array
            img = Image.fromarray(img_uint8)
        else:
            print(f"    ⚠ Unknown image format: shape={img_array.shape}, dtype={img_array.dtype}")
            continue

        # Save image
        img_path = frame_dir / f"{camera_name}.png"
        img.save(img_path)
        saved_count += 1
        print(f"    Saved {camera_name}: {img.size}")

    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Save images from specific dataset frames")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="vo2yager/objects_pickup_place",
        help="HuggingFace dataset repository ID"
    )
    parser.add_argument(
        "--frame-indexes",
        type=int,
        nargs="+",
        default=[5600, 5628, 5656, 5685, 5713, 5742, 5770, 5799],
        help="Frame indexes to save"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="saved_demo_frames",
        help="Directory to save images"
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local cached files"
    )

    args = parser.parse_args()

    print(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        local_files_only=args.local_files_only,
    )

    print(f"Dataset loaded. Total frames: {len(dataset)}")
    print(f"Frame indexes to save: {args.frame_indexes}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    for idx in args.frame_indexes:
        if idx >= len(dataset):
            print(f"⚠ Warning: Frame index {idx} exceeds dataset size ({len(dataset)}), skipping")
            continue

        print(f"\nProcessing frame {idx}...")
        try:
            num_images = save_frame_images(dataset, idx, save_dir)
            total_images += num_images
            print(f"  ✓ Saved {num_images} images")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n{'='*80}")
    print(f"Done! Saved {total_images} total images to: {save_dir.absolute()}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
