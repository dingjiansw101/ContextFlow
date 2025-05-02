import os
from pathlib import Path
import cv2

def sample_frames_from_videos(
    input_dir: str,
    output_dir: str,
    frames_per_video: int,
    video_extensions=None
) -> None:
    """
    Read all video files from the specified folder, sample a fixed number of frames evenly from each video,
    and save them as PNG images.

    Args:
        input_dir (str): Path to the folder containing video files.
        output_dir (str): Path to the folder where extracted frames will be saved.
        frames_per_video (int): Number of frames to sample from each video.
        video_extensions (list, optional): List of video file extensions to process. Defaults to common formats.
    """
    # Default video formats if not provided
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Iterate through all video files in the input directory
    for video_file in input_path.iterdir():
        # Skip files that do not match the given extensions
        if video_file.suffix.lower() not in video_extensions:
            continue

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Failed to open video file: {video_file}")
            continue

        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Unable to retrieve frame count for: {video_file}")
            cap.release()
            continue

        # Compute frame indices to sample evenly across the video
        indices = [int(i * total_frames / frames_per_video) for i in range(frames_per_video)]
        indices = [min(idx, total_frames - 1) for idx in indices]

        # Create an output subfolder named after the video file (without extension)
        video_out_dir = output_path / video_file.stem
        video_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {video_file.name}: total_frames={total_frames}, sampling {frames_per_video} frames")

        # Extract and save each sampled frame
        for count, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame at index {frame_idx} in {video_file.name}")
                continue

            output_file = video_out_dir / f"frame_{count:04d}.png"
            cv2.imwrite(str(output_file), frame)

        cap.release()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Sample N frames evenly from each video in a folder and save them as PNG images.'
    )
    parser.add_argument('input_dir', type=str, help='Directory containing video files')
    parser.add_argument('output_dir', type=str, help='Directory to save sampled frames')
    parser.add_argument('N', type=int, help='Number of frames to sample per video')

    args = parser.parse_args()
    sample_frames_from_videos(args.input_dir, args.output_dir, args.N)
