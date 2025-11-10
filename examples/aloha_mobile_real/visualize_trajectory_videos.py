#!/usr/bin/env python3
"""Script to visualize video data from trajectory HDF5 files."""

import argparse
import pathlib
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Optional, Tuple
import cv2


class TrajectoryVideoVisualizer:
    """Visualizes video data from trajectory HDF5 files."""

    def __init__(self, trajectory_path: pathlib.Path):
        self.trajectory_path = trajectory_path
        self.cameras = {}
        self.timestamps = None
        self.episode_length = 0

        # Load trajectory data
        self._load_trajectory()

    def _load_trajectory(self):
        """Load trajectory data from HDF5 file."""
        if not self.trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.trajectory_path}")

        print(f"Loading trajectory: {self.trajectory_path}")

        with h5py.File(self.trajectory_path, "r") as f:
            # Load metadata
            if "metadata" in f:
                metadata = f["metadata"]
                self.episode_length = metadata.attrs.get("episode_length", 0)
                if "timestamps" in metadata:
                    self.timestamps = metadata["timestamps"][:]
                print(f"Episode length: {self.episode_length} steps")

            # Load camera data
            if "observations/images" in f:
                images_group = f["observations/images"]
                for cam_name in images_group.keys():
                    cam_data = images_group[cam_name][:]  # Shape: (N, C, H, W)
                    # Convert from (N, C, H, W) to (N, H, W, C) for display
                    self.cameras[cam_name] = np.transpose(cam_data, (0, 2, 3, 1))
                    print(f"Loaded {cam_name}: {self.cameras[cam_name].shape}")

                if not self.cameras:
                    raise ValueError("No camera data found in trajectory file")
            else:
                raise ValueError("No image data found in trajectory file")

    def play_videos(self, fps: float = 10.0, save_path: Optional[str] = None):
        """Play videos from all cameras simultaneously."""
        num_cameras = len(self.cameras)
        if num_cameras == 0:
            print("No camera data to display")
            return

        # Set up the figure and subplots
        if num_cameras == 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            axes = [ax]
        elif num_cameras == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        elif num_cameras == 3:
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        else:
            # For more cameras, use a 2x2 or larger grid
            cols = min(4, num_cameras)
            rows = (num_cameras + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        # Initialize plots for each camera
        camera_names = list(self.cameras.keys())
        im_objects = []

        for i, (cam_name, cam_data) in enumerate(self.cameras.items()):
            ax = axes[i]
            ax.set_title(f"{cam_name}")
            ax.axis('off')

            # Initialize with first frame
            first_frame = cam_data[0]
            # Ensure values are in [0, 1] range for matplotlib
            if first_frame.max() > 1.0:
                first_frame = first_frame / 255.0

            im = ax.imshow(first_frame, animated=True)
            im_objects.append(im)

        # Hide unused subplots
        for i in range(num_cameras, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        def animate(frame_idx):
            """Animation function to update frames."""
            for i, (cam_name, cam_data) in enumerate(self.cameras.items()):
                if frame_idx < len(cam_data):
                    frame = cam_data[frame_idx]
                    # Ensure values are in [0, 1] range
                    if frame.max() > 1.0:
                        frame = frame / 255.0
                    im_objects[i].set_array(frame)

            # Update title with current frame info
            if self.timestamps is not None and frame_idx < len(self.timestamps):
                fig.suptitle(f"Frame {frame_idx}/{self.episode_length-1} | "
                           f"Time: {self.timestamps[frame_idx]:.2f}s", fontsize=14)
            else:
                fig.suptitle(f"Frame {frame_idx}/{self.episode_length-1}", fontsize=14)

            return im_objects

        # Create animation
        interval = 1000.0 / fps  # milliseconds per frame
        anim = animation.FuncAnimation(
            fig, animate, frames=self.episode_length,
            interval=interval, blit=False, repeat=True
        )

        # Save animation if requested
        if save_path:
            print(f"Saving animation to {save_path}...")
            writer = animation.PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")

        plt.show()
        return anim

    def export_frames(self, output_dir: pathlib.Path, camera_name: Optional[str] = None):
        """Export video frames as individual images."""
        output_dir.mkdir(parents=True, exist_ok=True)

        cameras_to_export = [camera_name] if camera_name else list(self.cameras.keys())

        for cam_name in cameras_to_export:
            if cam_name not in self.cameras:
                print(f"Camera {cam_name} not found. Available: {list(self.cameras.keys())}")
                continue

            cam_dir = output_dir / cam_name
            cam_dir.mkdir(exist_ok=True)

            cam_data = self.cameras[cam_name]
            print(f"Exporting {len(cam_data)} frames for {cam_name}...")

            for i, frame in enumerate(cam_data):
                # Convert to BGR for OpenCV (if needed)
                if frame.shape[-1] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame

                # Ensure frame is in correct format
                if frame_bgr.max() <= 1.0:
                    frame_bgr = (frame_bgr * 255).astype(np.uint8)

                frame_path = cam_dir / f"frame_{i:06d}.png"
                cv2.imwrite(str(frame_path), frame_bgr)

            print(f"Frames exported to {cam_dir}")

    def create_video_file(self, output_path: pathlib.Path, camera_name: Optional[str] = None, fps: float = 10.0):
        """Create a video file from trajectory frames."""
        if camera_name is None:
            camera_name = list(self.cameras.keys())[0]

        if camera_name not in self.cameras:
            print(f"Camera {camera_name} not found. Available: {list(self.cameras.keys())}")
            return

        cam_data = self.cameras[camera_name]
        print(f"Creating video from {len(cam_data)} frames of {camera_name}...")

        # Get video dimensions
        height, width = cam_data.shape[1:3]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame in cam_data:
            # Convert to BGR for OpenCV
            if frame.shape[-1] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame

            # Ensure frame is in correct format
            if frame_bgr.max() <= 1.0:
                frame_bgr = (frame_bgr * 255).astype(np.uint8)

            video_writer.write(frame_bgr)

        video_writer.release()
        print(f"Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize video data from trajectory HDF5 files")
    parser.add_argument("trajectory_path", type=str, help="Path to the HDF5 trajectory file")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback frame rate (default: 10.0)")
    parser.add_argument("--save-gif", type=str, help="Save animation as GIF to this path")
    parser.add_argument("--save-video", type=str, help="Save video file to this path")
    parser.add_argument("--camera", type=str, help="Camera name for single-camera operations")
    parser.add_argument("--export-frames", type=str, help="Export frames to this directory")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras and exit")

    args = parser.parse_args()
    trajectory_path = pathlib.Path(args.trajectory_path)

    # Create visualizer
    try:
        visualizer = TrajectoryVideoVisualizer(trajectory_path)
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return

    # List cameras if requested
    if args.list_cameras:
        print("Available cameras:")
        for cam_name, cam_data in visualizer.cameras.items():
            print(f"  {cam_name}: {cam_data.shape}")
        return

    # Export frames if requested
    if args.export_frames:
        output_dir = pathlib.Path(args.export_frames)
        visualizer.export_frames(output_dir, args.camera)
        return

    # Create video file if requested
    if args.save_video:
        output_path = pathlib.Path(args.save_video)
        visualizer.create_video_file(output_path, args.camera, args.fps)
        return

    # Play videos
    print(f"Playing videos at {args.fps} FPS...")
    print("Close the window to stop playback")
    visualizer.play_videos(fps=args.fps, save_path=args.save_gif)


if __name__ == "__main__":
    main()