# import json
# from pathlib import Path
# import numpy as np
# from tqdm import tqdm
# import multiprocessing as mp

# import openpi.training.config as _config
# import openpi.training.data_loader as data_loader


# def save_episode_states_to_json(episode_to_all_states: dict[int, np.ndarray], filename: str):
#     filename = Path(filename)
#     filename.parent.mkdir(parents=True, exist_ok=True)
#     json_dict = {str(ep): arr.tolist() for ep, arr in episode_to_all_states.items()}
#     with filename.open("w") as f:
#         json.dump(json_dict, f)
#     print(f"[INFO] Saved cache to {filename}")


# def process_episode(args):
#     ep, idxs, dataset = args
#     state_list, action_list = [], []
#     for idx in idxs:
#         item = dataset[int(idx)]
#         state_list.append(item["state"])
#         action_list.append(item["actions"][0])
#     return ep, np.stack(state_list, axis=0), np.stack(action_list, axis=0)


# def build_cache(config: _config.TrainConfig, num_workers: int = 8):
#     data_config = config.data.create(config.assets_dirs, config.model)
#     dataset = data_loader.create_dataset(data_config, config.model)
#     dataset = data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)
#     # base_dataset = dataset._dataset if hasattr(dataset, "_dataset") else dataset

#     episode_to_indexes_file = Path(config.data.episode_to_indexes_file)
#     if not episode_to_indexes_file.exists():
#         raise FileNotFoundError(f"Episode indexes file not found: {episode_to_indexes_file}")

#     with episode_to_indexes_file.open("r") as f:
#         raw = json.load(f)
#     idx_map = {int(k): v for k, v in raw.items()}

#     tasks = [(ep, idxs, dataset) for ep, idxs in idx_map.items()]
#     states, actions = {}, {}

#     # Use tqdm.update() to refresh the progress bar in real time
#     with mp.Pool(processes=num_workers) as pool:
#         with tqdm(total=len(tasks), desc="Building episode caches") as pbar:
#             for ep, state_arr, action_arr in pool.imap_unordered(process_episode, tasks):
#                 states[ep] = state_arr
#                 actions[ep] = action_arr
#                 pbar.update()

#     save_episode_states_to_json(states, data_config.states_cache_path)
#     save_episode_states_to_json(actions, data_config.actions_cache_path)
#     print("[INFO] Cache building completed.")


# if __name__ == "__main__":
#     config = _config.cli()
#     build_cache(config, num_workers=8)

import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

import openpi.training.config as _config
import openpi.training.data_loader as data_loader
from openpi.training import lookup_tables


def save_episode_states_to_json(episode_to_all_states: dict[int, np.ndarray], filename: str):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    json_dict = {str(ep): arr.tolist() for ep, arr in episode_to_all_states.items()}
    with filename.open("w") as f:
        json.dump(json_dict, f)
    print(f"[INFO] Saved cache to {filename}")


def build_cache(config: _config.TrainConfig):
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_dataset(data_config, config.model)
    dataset = data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)

    # base_dataset = dataset._dataset if hasattr(dataset, "_dataset") else dataset

    idx_map = lookup_tables.episode_to_indexes_for_dataset(dataset)

    states, actions = {}, {}
    for ep, idxs in tqdm(idx_map.items(), desc="Building episode caches", total=len(idx_map)):
        state_list, action_list = [], []
        for idx in idxs:
            item = dataset[int(idx)]
            state_list.append(item["state"])
            action_list.append(item["actions"][0])
        states[ep] = np.stack(state_list, axis=0)
        actions[ep] = np.stack(action_list, axis=0)

    save_episode_states_to_json(states, config.data.states_cache_path)
    save_episode_states_to_json(actions, config.data.actions_cache_path)
    print("[INFO] Cache building completed.")


if __name__ == "__main__":
    # Reuse the same CLI as the training script
    config = _config.cli()
    build_cache(config)

# 20 min without multi-proc;
# uv run src/openpi/training/build_episode_cache.py pi0_libero90_incontextv12_low_mem_finetune --exp-name dummy
