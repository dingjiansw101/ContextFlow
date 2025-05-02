import torch
import os
import json
import glob
from tqdm import tqdm
import re # Import regular expressions module

def load_and_patch_merge(pt_path: str, patch_size: int = 7):
    """
    Load a .pt file and apply patch merging.

    Args:
        pt_path (str): Path to the .pt file.
        patch_size (int): Size of the patch to merge.

    Returns:
        torch.Tensor: Merged data tensor of shape (T, H//patch_size, W//patch_size, (patch_size**2)*C).
        Returns None if loading or processing fails.
    """
    try:
        # Load tensor [bs, C, T, H, W] or [C, T, H, W] onto CPU
        data = torch.load(pt_path, map_location='cpu')

        # Remove batch dimension if present
        if data.dim() == 5:
            if data.shape[0] != 1:
                print(f"Warning: Batch size > 1 found in {pt_path}, shape: {data.shape}. Squeezing first dimension.")
            data = data.squeeze(0)  # [C, T, H, W]
        elif data.dim() != 4:
             print(f"Warning: Unexpected tensor dimension {data.dim()} in {pt_path}. Expected 4 or 5. Skipping.")
             return None

        # Permute to [T, H, W, C]
        data = data.permute(1, 2, 3, 0)
        T, H, W, C = data.shape

        # Validate divisibility
        if H % patch_size != 0 or W % patch_size != 0:
             print(f"Warning: H({H}) or W({W}) not divisible by patch_size({patch_size}) in {pt_path}. Skipping.")
             return None

        # Group patches: [T, H//p, p, W//p, p, C]
        data = data.view(T, H // patch_size, patch_size, W // patch_size, patch_size, C)

        # Rearrange to bring patch dims together: [T, H//p, W//p, p, p, C]
        data = data.permute(0, 1, 3, 2, 4, 5)

        # Flatten patches: [T, H//p, W//p, p*p*C]
        data = data.reshape(T, H // patch_size, W // patch_size, (patch_size ** 2) * C)

        data = data.reshape(T*H//patch_size*W//patch_size, (patch_size ** 2) * C)
        return data
    except Exception as e:
        print(f"Error processing file {pt_path}: {e}")
        return None

def process_video_tokens_to_json(input_dirs: list[str],
                                 output_json_path: str,
                                 patch_size: int = 7) -> None:
    """
    Reads all .pt files from input directories, processes them using
    load_and_patch_merge, and saves the results into a single JSON file.
    The keys in the JSON file are the numbers following 'epepisode_' in the filenames.
    """
    output_data = {}
    file_paths = []
    for dir_path in input_dirs:
        search_pattern = os.path.join(dir_path, '*.pt')
        found_files = glob.glob(search_pattern)
        print(f"Found {len(found_files)} files in {dir_path}")
        file_paths.extend(found_files)

    print(f"Processing a total of {len(file_paths)} files...")

    # Regex to find 'epepisode_' followed by digits
    episode_pattern = re.compile(r'epepisode_(\d+)')

    for pt_path in tqdm(file_paths, desc="Processing video tokens", unit="file"):
        filename = os.path.basename(pt_path)
        match = episode_pattern.search(filename)

        if match:
            episode_num_str = match.group(1) # Extract the captured digits string (e.g., "001014")
            episode_key = str(int(episode_num_str)) # Convert to int to remove leading zeros, then back to string (e.g., "1014")
        else:
            # Fallback: use the filename without extension if pattern not found
            print(f"Warning: Could not extract episode number from '{filename}'. Using base filename as key.")
            episode_key = os.path.splitext(filename)[0]

        merged_data = load_and_patch_merge(pt_path, patch_size)

        if merged_data is not None:
            # Convert tensor to float32 before converting to numpy, then list for JSON serialization
            output_data[episode_key] = merged_data.cpu().to(torch.float32).numpy().tolist()
        else:
             print(f"Skipping file due to processing issues: {pt_path}")


    # Ensure the output directory exists
    output_dir = os.path.dirname(output_json_path)
    if output_dir: # Only create if output_json_path includes a directory
        os.makedirs(output_dir, exist_ok=True)

    # Write out the combined data
    print(f"Saving processed data for {len(output_data)} episodes to {output_json_path}...")
    try:
        with open(output_json_path, "w") as fw:
            json.dump(output_data, fw) # Consider indent=2 for readability if files aren't too large
        print(f"Successfully saved combined video tokens to {output_json_path}")
    except Exception as e:
        print(f"Error saving JSON file {output_json_path}: {e}")


if __name__ == "__main__":
    input_directories = [
        '/ibex/tmp/c2090/jian/video_tokens/seen',
        '/ibex/tmp/c2090/jian/video_tokens/unseen'
    ]
    # Define the output path for the JSON file
    output_file = "/home/dingj0b/code/openpi/metadata/libero/video_tokens_merged.json"
    patch_s = 7 # Define the patch size used for merging

    process_video_tokens_to_json(input_directories, output_file, patch_size=patch_s)