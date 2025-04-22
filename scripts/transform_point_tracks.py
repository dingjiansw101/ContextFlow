import numpy as np
import json
import os
from tqdm import tqdm

def process_episode_tracks(input_json_path: str,
                           output_json_path: str,
                           normalize_factor: float = 224.0) -> None:
    """
    Reads per‐episode track data (with active IDs) from `input_json_path`,
    concatenates track coords, visibility, and active mask into a flattened
    float32 array for each episode, and writes the result to `output_json_path`,
    displaying a progress bar.
    """
    # Load the raw point‐track data
    with open(input_json_path, "r") as f:
        point_tracks = json.load(f)

    output = {}

    # Wrap the iteration in tqdm for a progress bar
    for episode_id, item in tqdm(
        point_tracks.items(),
        desc="Processing episodes",
        total=len(point_tracks),
        unit="episode"
    ):
        # 1) Load & normalize
        track = np.array(item["track"], dtype=float) / normalize_factor
        visibility = np.array(item["visibility"], dtype=int)
        active_ids = np.array(item["active_ids"], dtype=int)

        num_frames, num_points, _ = track.shape

        # 2) Build a per‐point binary mask of active IDs
        active_binary = np.zeros(num_points, dtype=int)
        active_binary[active_ids] = 1

        # 3) Expand to time‐aligned mask array: (num_frames, num_points, 1)
        active_mask = np.tile(
            active_binary[np.newaxis, :, np.newaxis],
            (num_frames, 1, 1)
        )

        # 4) Expand visibility to a channel: (num_frames, num_points, 1)
        vis_channel = visibility[..., np.newaxis]

        # 5) Concatenate into (num_frames, num_points, 4): [x, y, vis, active]
        combined = np.concatenate([track, vis_channel, active_mask], axis=-1)

        # 6) Flatten to (num_frames, num_points*4) and cast to float32
        combined_flat = combined.reshape(num_frames, -1).astype(np.float32)

        # 7) Store as regular Python list for JSON serialization
        output[episode_id] = combined_flat.tolist()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Write out the combined data
    with open(output_json_path, "w") as fw:
        json.dump(output, fw)

    print(f"Saved combined tracks to {output_json_path}")

def merge_episode_jsons(json_path1: str,
                        json_path2: str,
                        output_json_path: str) -> None:
    """
    Reads two episode‐track JSON files, merges their top‐level keys,
    and writes the combined dict to output_json_path.
    If a key exists in both inputs, the value from json_path2 wins.
    """
    # 1) Load both files
    with open(json_path1, 'r') as f1:
        data1 = json.load(f1)
    with open(json_path2, 'r') as f2:
        data2 = json.load(f2)

    # 2) Merge: entries in data2 will overwrite duplicates from data1
    merged = {**data1, **data2}

    # 3) Ensure output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # 4) Write merged JSON
    with open(output_json_path, 'w') as fout:
        json.dump(merged, fout, indent=2)
    print(f"Merged {len(data1)} + {len(data2)} → {len(merged)} episodes into\n  {output_json_path}")


if __name__ == "__main__":
    # src = "/ibex/tmp/c2090/jinjie/shared/episode_tracks_w_active_id.json"
    # dst = "/home/dingj0b/code/openpi/metadata/libero/episode_tracks_combined.json"
    # process_episode_tracks(src, dst)

    # src = "/ibex/tmp/c2090/jinjie/shared/episode_tracks_w_active_id_unseen.json"
    # dst = "/home/dingj0b/code/openpi/metadata/libero/episode_tracks_combined_unseen.json"
    # process_episode_tracks(src, dst)

    src1 = "/home/dingj0b/code/openpi/metadata/libero/episode_tracks_combined.json"
    src2 = "/home/dingj0b/code/openpi/metadata/libero/episode_tracks_combined_unseen.json"
    dst  = "/home/dingj0b/code/openpi/metadata/libero/episode_tracks_combined_all.json"
    merge_episode_jsons(src1, src2, dst)
