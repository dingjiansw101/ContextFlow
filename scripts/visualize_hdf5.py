#!/usr/bin/env python3
"""Script to visualize frames from object_task_suite HDF5 files."""

import argparse
import pathlib

import cv2
import h5py
import numpy as np


def decode_jpeg_image(jpeg_bytes: np.ndarray) -> np.ndarray:
    """Decode JPEG-compressed image bytes to BGR numpy array using OpenCV."""
    jpeg_array = np.asarray(jpeg_bytes, dtype=np.uint8).reshape(-1)
    frame = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG image bytes")
    return frame


def ensure_uint8_image(frame: np.ndarray) -> np.ndarray:
    """Ensure the frame is uint8 in [0, 255]."""
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)
    return frame


def frame_array_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert raw numpy frame (HWC, HW, or CHW) into BGR format."""
    if frame.ndim == 2:
        frame = frame[..., np.newaxis]
    elif frame.ndim != 3:
        raise ValueError(f"Unsupported frame ndim {frame.ndim}")

    # If data is CHW (C, H, W), transpose to HWC.
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    frame = ensure_uint8_image(frame)
    channels = frame.shape[-1]

    if channels == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if channels == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if channels == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported channel count {channels}")


def get_frame_bgr(camera_dataset: h5py.Dataset, frame_index: int) -> np.ndarray:
    """Return a single frame in BGR format, handling encoded and raw layouts."""
    frame = camera_dataset[frame_index]
    if frame.ndim == 1:
        # JPEG-compressed bytes
        return decode_jpeg_image(frame)
    if frame.ndim in (2, 3):
        # Raw frame, can be HWC, HW, or CHW
        return frame_array_to_bgr(frame)
    if frame.ndim == 4:
        # Occasionally frames are stored as (1, H, W, C) or (C, H, W, 1)
        if frame.shape[0] in (1, 3, 4):
            squeezed = np.squeeze(frame, axis=0)
        else:
            squeezed = frame
        return frame_array_to_bgr(squeezed)

    raise ValueError(
        f"Unsupported frame shape {frame.shape} for dataset {camera_dataset.name}"
    )


def extract_evenly_spaced_frames(total_frames: int, num_frames: int = 16) -> list[int]:
    """Extract evenly spaced frame indices from the episode."""
    if total_frames <= num_frames:
        return list(range(total_frames))

    # Calculate step size to get evenly spaced frames
    step = total_frames / num_frames
    return [int(i * step) for i in range(num_frames)]


def process_episode(
    hdf5_path: pathlib.Path,
    output_dir: pathlib.Path,
    num_frames: int = 16,
    cameras: list[str] = None,
):
    """Process a single episode and save frames."""
    if cameras is None:
        cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    episode_name = hdf5_path.stem  # e.g., "episode_0"

    print(f"Processing {hdf5_path.name}...")

    with h5py.File(hdf5_path, "r") as f:
        # Get total number of frames
        images_group = f["observations/images"]
        total_frames = images_group[cameras[0]].shape[0]

        # Get frame indices to extract
        frame_indices = extract_evenly_spaced_frames(total_frames, num_frames)

        # Process each camera
        for camera in cameras:
            if camera not in images_group:
                print(f"  Warning: Camera {camera} not found in {hdf5_path.name}")
                continue

            camera_data = images_group[camera]

            # Extract and save each frame
            for frame_idx, actual_idx in enumerate(frame_indices):
                frame_bgr = get_frame_bgr(camera_data, actual_idx)
                # Match reference behavior: swap B and R channels before saving
                frame_to_save = frame_bgr[:, :, [2, 1, 0]]

                # Save frame using OpenCV
                output_filename = f"{episode_name}_frame_{frame_idx:02d}_{camera}.png"
                output_path = output_dir / output_filename
                cv2.imwrite(str(output_path), frame_to_save)

            print(f"  Saved {len(frame_indices)} frames for {camera}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize frames from object_task_suite HDF5 files"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/ibex/project/c2090/jian/object_task_suite",
        help="Input directory containing task subdirectories with HDF5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/ibex/tmp/c2090/jian/visualizations/object_task_suite",
        help="Output directory for saved frames",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to extract per episode",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=5,
        help="Number of episodes to process per task",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        help="Camera views to extract",
    )

    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each task directory
    task_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    print(f"Found {len(task_dirs)} task directories")
    print(f"Will process {args.episodes_per_task} episodes per task")
    print(f"Extracting {args.num_frames} frames per episode")
    print(f"Cameras: {args.cameras}")
    print()

    total_processed = 0

    for task_dir in task_dirs:
        task_name = task_dir.name
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        # Create output subdirectory for this task
        task_output_dir = output_dir / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Find HDF5 files for this task
        hdf5_files = sorted(task_dir.glob("episode_*.hdf5"))

        # Process only the first N episodes
        hdf5_files_to_process = hdf5_files[: args.episodes_per_task]

        print(f"Found {len(hdf5_files)} episodes, processing {len(hdf5_files_to_process)}")

        for hdf5_path in hdf5_files_to_process:
            process_episode(
                hdf5_path,
                task_output_dir,
                num_frames=args.num_frames,
                cameras=args.cameras,
            )
            total_processed += 1

    print(f"\n{'='*60}")
    print(f"Completed!")
    print(f"Processed {total_processed} episodes")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
