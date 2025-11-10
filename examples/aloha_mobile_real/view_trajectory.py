#!/usr/bin/env python3
"""Script to view trajectory data from HDF5 files."""

import argparse
import pathlib
import h5py
import numpy as np


def view_trajectory(trajectory_path: pathlib.Path):
    """View the contents of a trajectory HDF5 file."""
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    print(f"Loading trajectory: {trajectory_path}")

    with h5py.File(trajectory_path, "r") as f:
        print("\n" + "="*60)
        print("TRAJECTORY SUMMARY")
        print("="*60)

        # Episode metadata
        if "metadata" in f:
            metadata = f["metadata"]
            episode_length = metadata.attrs.get("episode_length", "Unknown")
            recording_date = metadata.attrs.get("recording_date", "Unknown")
            print(f"Episode length: {episode_length} steps")
            print(f"Recording date: {recording_date}")

            if "timestamps" in metadata:
                timestamps = metadata["timestamps"][:]
                duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
                print(f"Episode duration: {duration:.2f} seconds")

        print("\n" + "="*60)
        print("DATA STRUCTURE")
        print("="*60)

        def print_structure(name, obj, indent=0):
            prefix = "  " * indent
            if isinstance(obj, h5py.Dataset):
                print(f"{prefix}{name}: {obj.shape} {obj.dtype}")
                if obj.size < 100:  # Only show values for small arrays
                    if len(obj.shape) == 1:
                        print(f"{prefix}  Values: {obj[:]}")
            elif isinstance(obj, h5py.Group):
                print(f"{prefix}{name}/ (group)")
                for key in obj.keys():
                    print_structure(key, obj[key], indent + 1)

        for key in f.keys():
            print_structure(key, f[key])

        # Show some statistics
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)

        # States statistics
        if "observations/states" in f:
            states = f["observations/states"][:]
            print(f"states0: {states[0]}")

            print(f"states: {states}")
            print(f"States shape: {states.shape}")
            print(f"States range: [{states.min():.3f}, {states.max():.3f}]")
            print(f"States mean: {states.mean():.3f}")

        # Actions statistics
        if "actions" in f:
            actions = f["actions"][:]
            print(f"actions: {actions}")
            print(f"Actions shape: {actions.shape}")
            print(f"Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
            print(f"Actions mean: {actions.mean():.3f}")

        # Images info
        if "observations/images" in f:
            images_group = f["observations/images"]
            print(f"Available cameras: {list(images_group.keys())}")
            for cam_name in images_group.keys():
                img_data = images_group[cam_name]
                print(f"  {cam_name}: {img_data.shape} {img_data.dtype}")


def main():
    parser = argparse.ArgumentParser(description="View trajectory data from HDF5 files")
    parser.add_argument("trajectory_path", type=str, help="Path to the HDF5 trajectory file")

    args = parser.parse_args()
    trajectory_path = pathlib.Path(args.trajectory_path)

    view_trajectory(trajectory_path)


if __name__ == "__main__":
    main()