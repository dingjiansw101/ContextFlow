import datetime
import logging
import pathlib
from typing import Any, Dict, List

import h5py
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class TrajectoryRecorder(_subscriber.Subscriber):
    """Records trajectory data (observations and actions) to HDF5 files."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        record_images: bool = True,
        compression: str = "gzip",
        compression_level: int = 4,
    ) -> None:
        """Initialize the trajectory recorder.

        Args:
            output_dir: Directory to save trajectory files
            record_images: Whether to record camera images
            compression: HDF5 compression type ('gzip', 'lzf', or None)
            compression_level: Compression level (1-9, only for gzip)
        """
        self._output_dir = pathlib.Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._record_images = record_images
        self._compression = compression
        self._compression_opts = compression_level if compression == "gzip" else None

        # Episode data buffers
        self._states: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._timestamps: List[float] = []
        self._images: Dict[str, List[np.ndarray]] = {}

        # Track episode start time
        self._episode_start_time: float = 0.0

        logging.info(f"TrajectoryRecorder initialized. Output directory: {self._output_dir}")

    @override
    def on_episode_start(self) -> None:
        """Reset buffers at the start of a new episode."""
        self._states = []
        self._actions = []
        self._timestamps = []
        self._images = {}
        self._episode_start_time = datetime.datetime.now().timestamp()

        logging.info("Starting trajectory recording for new episode")

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        """Record a single step of observation and action data."""
        # Record state
        if "state" in observation:
            self._states.append(observation["state"].copy())

        # Record action
        if "actions" in action:
            self._actions.append(action["actions"].copy())

        # Record timestamp
        current_time = datetime.datetime.now().timestamp()
        self._timestamps.append(current_time - self._episode_start_time)

        # Record images if enabled
        if self._record_images and "images" in observation:
            for cam_name, image in observation["images"].items():
                if cam_name not in self._images:
                    self._images[cam_name] = []
                self._images[cam_name].append(image.copy())

    @override
    def on_episode_end(self) -> None:
        """Save the recorded trajectory to an HDF5 file."""
        if len(self._states) == 0:
            logging.warning("No data recorded for this episode, skipping save")
            return

        # Find the next available episode number
        existing_files = list(self._output_dir.glob("episode_*.hdf5"))
        next_idx = 0
        if existing_files:
            indices = []
            for f in existing_files:
                try:
                    idx = int(f.stem.split("_")[1])
                    indices.append(idx)
                except (ValueError, IndexError):
                    continue
            if indices:
                next_idx = max(indices) + 1

        output_path = self._output_dir / f"episode_{next_idx:03d}.hdf5"

        # Save to HDF5
        with h5py.File(output_path, "w") as f:
            # Create main groups
            obs_group = f.create_group("observations")
            metadata_group = f.create_group("metadata")

            # Save states
            if self._states:
                states_array = np.array(self._states)
                obs_group.create_dataset(
                    "states",
                    data=states_array,
                    compression=self._compression,
                    compression_opts=self._compression_opts,
                )
                logging.info(f"Saved states with shape: {states_array.shape}")

            # Save images
            if self._images:
                images_group = obs_group.create_group("images")
                for cam_name, image_list in self._images.items():
                    images_array = np.array(image_list)
                    images_group.create_dataset(
                        cam_name,
                        data=images_array,
                        compression=self._compression,
                        compression_opts=self._compression_opts,
                    )
                    logging.info(f"Saved {cam_name} images with shape: {images_array.shape}")

            # Save actions
            if self._actions:
                actions_array = np.array(self._actions)
                f.create_dataset(
                    "actions",
                    data=actions_array,
                    compression=self._compression,
                    compression_opts=self._compression_opts,
                )
                logging.info(f"Saved actions with shape: {actions_array.shape}")

            # Save metadata
            metadata_group.create_dataset("timestamps", data=np.array(self._timestamps))
            metadata_group.attrs["episode_length"] = len(self._states)
            metadata_group.attrs["recording_date"] = datetime.datetime.now().isoformat()
            metadata_group.attrs["episode_start_timestamp"] = self._episode_start_time

        logging.info(f"Trajectory saved to {output_path}")
        logging.info(f"Episode length: {len(self._states)} steps")