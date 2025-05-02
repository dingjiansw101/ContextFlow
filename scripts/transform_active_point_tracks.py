#!/usr/bin/env python3
"""
Merge two raw episode-track JSONs, then combine per-episode point-track data
keeping only active points and padding every episode to the global max-active-point count.

Workflow:
 1. Merge raw JSON files: seen + unseen → merged_raw.json
 2. Process merged_raw.json → combined_tracks.json

Output per episode:
    shape (num_frames, max_active_points * 3)   # flattened
    channels per point: [x, y, visibility]
"""
import json
import os
from typing import Dict, Any

import numpy as np
from tqdm import tqdm


def _get_max_active_points(point_tracks: Dict[str, Any]) -> int:
    """One pass to find the maximum number of active points in the dataset."""
    return max(len(item["active_ids"]) for item in point_tracks.values())


def merge_raw_episode_tracks(
    raw_path1: str,
    raw_path2: str,
    merged_path: str,
) -> None:
    """
    Reads two raw episode-track JSONs, merges their entries,
    and writes the result to merged_path.
    If a key exists in both inputs, the entry from raw_path2 wins.
    """
    with open(raw_path1, 'r') as f1:
        data1 = json.load(f1)
    with open(raw_path2, 'r') as f2:
        data2 = json.load(f2)

    merged = {**data1, **data2}

    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, 'w') as f:
        json.dump(merged, f)

    print(f"[✓] Merged raw JSON: {len(data1)} + {len(data2)} → {len(merged)} episodes")


def process_episode_tracks(
    input_json_path: str,
    output_json_path: str,
    normalize_factor: float = 224.0,
) -> None:
    """
    Loads a raw merged JSON, keeps only active points per episode,
    pads to the global max-active-point count, and writes combined output.
    """
    # --- load merged raw
    with open(input_json_path, 'r') as f:
        point_tracks = json.load(f)

    max_pts = _get_max_active_points(point_tracks)
    print(f"Found max active points: {max_pts}")
    output = {}

    for ep_id, item in tqdm(
        point_tracks.items(),
        desc=f"Processing {os.path.basename(input_json_path)}",
        total=len(point_tracks),
        unit="episode",
    ):
        track = np.asarray(item["track"], dtype=float) / normalize_factor  # (T, P, 2)
        visibility = np.asarray(item["visibility"], dtype=int)          # (T, P)
        active_ids = np.asarray(item["active_ids"], dtype=int)          # (K,)

        # slice to active points only
        track = track[:, active_ids, :]                                   # (T, K, 2)
        visibility = visibility[:, active_ids]                            # (T, K)

        num_frames, num_active, _ = track.shape

        # visibility → (..., 1)
        vis_ch = visibility[..., np.newaxis]                              # (T, K, 1)

        # concatenate [x, y, vis]
        combined = np.concatenate([track, vis_ch], axis=-1)               # (T, K, 3)

        # zero-pad to max_pts
        if num_active < max_pts:
            pad_pts = max_pts - num_active
            pad_shape = (num_frames, pad_pts, 3)
            combined = np.concatenate(
                [combined, np.zeros(pad_shape, dtype=combined.dtype)],
                axis=1,
            )

        # flatten to (T, max_pts*3)
        combined_flat = combined.reshape(num_frames, -1).astype(np.float32)
        output[ep_id] = combined_flat.tolist()

    # save combined output
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as fw:
        json.dump(output, fw, indent=2)

    print(f"[✓] Saved combined tracks ({max_pts} points × 3 channels) → {output_json_path}")


if __name__ == "__main__":
    # --- raw inputs
    raw_seen = "/ibex/tmp/c2090/jinjie/shared/episode_tracks_w_active_id_seen_grid32.json"
    raw_unseen = "/ibex/tmp/c2090/jinjie/shared/episode_tracks_w_active_id_unseen_grid32.json"
    merged_raw = "/home/dingj0b/code/openpi/metadata/libero/episode_tracks_grid32_raw_merged.json"

    # --- merge raw first
    merge_raw_episode_tracks(raw_seen, raw_unseen, merged_raw)

    # --- process merged into final combined
    final_out = "/home/dingj0b/code/openpi/metadata/libero/episode_tracks_combined_grid32_all.json"
    process_episode_tracks(merged_raw, final_out)
